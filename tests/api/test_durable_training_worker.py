"""API acknowledgements survive restart and execute in a separate worker."""

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from stateset_agents.api.schemas import TrainingRequest
from stateset_agents.api.services.training_service import TrainingService
from stateset_agents.api.training_worker import work_once


@pytest.mark.asyncio
async def test_idempotent_http_retry_reports_existing_terminal_status(tmp_path):
    from stateset_agents.api.routers.training import start_training

    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
    )
    job_id = await api.start_training(request, "owner", idempotency_key="one")
    claimed, token, snapshot = api.queue.claim()
    assert claimed == job_id
    snapshot.update(status="completed", completed_at=datetime.utcnow())
    api.queue.finish(job_id, token, snapshot)
    response = await start_training(
        request, svc=api, user=SimpleNamespace(user_id="owner"), idempotency_key="one"
    )
    assert response.training_id == job_id and response.status == "completed"
    assert len(api.jobs) == 1


@pytest.mark.asyncio
async def test_ephemeral_service_never_silently_ignores_idempotency_keys():
    api = TrainingService()
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
    )
    with pytest.raises(ValueError, match="API_TRAINING_QUEUE_PATH"):
        await api.start_training(request, "owner", idempotency_key="one")
    assert not api.jobs


@pytest.mark.asyncio
async def test_submission_survives_restart_and_worker_publishes_progress(
    tmp_path, monkeypatch
):
    import importlib

    module = importlib.import_module("stateset_agents.training.train")
    calls = []

    async def train(**kwargs):
        calls.append(kwargs)
        for index in range(2):
            kwargs["callbacks"][0].on_episode_end(
                index, {"optimizer_updates": index + 1}
            )

    monkeypatch.setattr(module, "train", train)
    path = tmp_path / "jobs.sqlite3"
    api = TrainingService(queue_path=path)
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case", "expected_response": "done"}],
        reward_config={},
        num_episodes=2,
        training_config_overrides={"output_dir": str(tmp_path / "artifacts")},
    )
    job_id = await api.start_training(request, "owner", idempotency_key="one")
    assert api._tasks == {} and api.get_training_status(job_id)["status"] == "queued"
    await api.aclose()
    restarted = TrainingService(queue_path=path)
    assert (
        await restarted.start_training(request, "owner", idempotency_key="one")
        == job_id
    )
    assert await work_once(restarted.queue, lease_seconds=1)
    status = restarted.get_training_status(job_id, "owner")
    assert status["status"] == "completed" and status["progress"] == 100
    assert status["completion_scope"] == "final_episode_reported"
    assert len(calls) == 1
    assert calls[0]["config_overrides"]["output_dir"].startswith(
        str(tmp_path / "artifacts" / job_id)
    )
    assert not await work_once(restarted.queue, lease_seconds=1)


@pytest.mark.asyncio
async def test_other_replica_can_cancel_running_worker(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("stateset_agents.training.train")
    queue_module = importlib.import_module("stateset_agents.api.training_queue")
    # Cancellation uses the real heartbeat thread. Lease-expiry fencing is
    # covered separately; slow filesystem commits must not turn this into an
    # accidental one-second disk-performance test.
    now = queue_module.time.time()
    monkeypatch.setattr(queue_module, "time", SimpleNamespace(time=lambda: now))
    started = asyncio.Event()

    async def train(**kwargs):
        started.set()
        callback = kwargs["callbacks"][0]
        while not callback.cancel_event.is_set():
            await asyncio.sleep(0.01)
        raise asyncio.CancelledError

    monkeypatch.setattr(module, "train", train)
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
    )
    job_id = await api.start_training(request, "owner")
    task = asyncio.create_task(work_once(api.queue, lease_seconds=1))
    await asyncio.wait_for(started.wait(), 30)
    replica = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    assert replica.cancel_training(job_id, "owner")
    await asyncio.wait_for(task, 30)
    assert replica.get_training_status(job_id)["status"] == "cancelled"
