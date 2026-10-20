"""API acknowledgements survive restart and execute in a separate worker."""

import asyncio
import importlib
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

from stateset_agents.api.schemas import TrainingRequest
from stateset_agents.api.services.training_service import TrainingService
from stateset_agents.api.training_queue import TrainingLeaseLostError
from stateset_agents.api.training_worker import work_once


@pytest.mark.parametrize("durable", [False, True])
async def test_api_model_revision_reaches_the_agent_before_training(
    tmp_path, monkeypatch, durable
):
    module = importlib.import_module("stateset_agents.training.train")
    revision = "a" * 40
    observed = []

    async def train(**kwargs):
        observed.append(kwargs["agent"].config.model_revision)
        assert observed[-1] == kwargs["config_overrides"]["model_revision"]
        kwargs["callbacks"][0].on_episode_end(0, {"optimizer_updates": 1})

    monkeypatch.setattr(module, "train", train)
    service = TrainingService(queue_path=tmp_path / "jobs.sqlite3" if durable else None)
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
        num_episodes=1,
        training_config_overrides={"model_revision": revision},
    )
    try:
        job_id = await service.start_training(request, "owner")
        if durable:
            assert await work_once(service.queue)
        else:
            await asyncio.gather(*tuple(service._tasks.values()))
        assert service.get_training_status(job_id)["status"] == "completed"
        assert observed == [revision]
    finally:
        await service.aclose()


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("revision", ["", " ", True, 123])
async def test_invalid_model_revision_never_admits_an_api_job(
    tmp_path, durable, revision
):
    service = TrainingService(queue_path=tmp_path / "jobs.sqlite3" if durable else None)
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
        training_config_overrides={"model_revision": revision},
    )
    try:
        with pytest.raises(ValueError, match="model_revision"):
            await service.start_training(request, "owner")
        assert service.jobs == {} and not service._tasks
    finally:
        await service.aclose()


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


@pytest.mark.asyncio
async def test_terminal_lease_loss_does_not_stop_worker_from_processing_next_job(
    tmp_path, monkeypatch
):
    module = importlib.import_module("stateset_agents.training.train")
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
    )
    first = await api.start_training(request, "owner")
    second = await api.start_training(request, "owner")
    calls = []

    async def train(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # An external reaper or long stall can invalidate the lease just
            # before terminal publication, independently of the trainer result.
            with api.queue._transaction() as connection:
                connection.execute(
                    "UPDATE stateset_training_jobs SET lease_until=0 WHERE id=?",
                    (first,),
                )

    monkeypatch.setattr(module, "train", train)
    assert await work_once(api.queue)
    assert await work_once(api.queue)
    assert api.get_training_status(first)["status"] == "failed"
    assert api.get_training_status(second)["status"] == "completed"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_lease_lost_during_startup_never_starts_training(tmp_path, monkeypatch):
    module = importlib.import_module("stateset_agents.training.train")
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
    )
    job_id = await api.start_training(request, "owner")
    calls = []

    async def train(**kwargs):
        calls.append(kwargs)

    def lost_heartbeat(*args, **kwargs):
        with api.queue._transaction() as connection:
            connection.execute(
                "UPDATE stateset_training_jobs SET lease_until=0 WHERE id=?", (job_id,)
            )
        raise TrainingLeaseLostError("Training worker lost its lease")

    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(api.queue, "heartbeat", lost_heartbeat)
    assert await work_once(api.queue)
    assert not calls
    assert api.queue.claim() is None
    assert api.get_training_status(job_id)["status"] == "failed"


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_inflight_heartbeat_cleanup(
    tmp_path, monkeypatch
):
    module = importlib.import_module("stateset_agents.training.train")
    queue_module = importlib.import_module("stateset_agents.api.training_queue")
    now = queue_module.time.time()
    monkeypatch.setattr(queue_module, "time", SimpleNamespace(time=lambda: now))
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    request = TrainingRequest(
        agent_config={"model_name": "test"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
    )
    job_id = await api.start_training(request, "owner")
    entered, release = threading.Event(), threading.Event()
    committed = asyncio.Event()
    heartbeat, finish = api.queue.heartbeat, api.queue.finish

    async def train(**kwargs):
        await asyncio.Event().wait()

    def blocked_heartbeat(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            entered.set()
            assert release.wait(30)
        return heartbeat(*args, **kwargs)

    def observed_finish(*args, **kwargs):
        finish(*args, **kwargs)
        committed.set()

    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(api.queue, "heartbeat", blocked_heartbeat)
    monkeypatch.setattr(api.queue, "finish", observed_finish)
    task = asyncio.create_task(work_once(api.queue, lease_seconds=1))
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        task.cancel()
        await asyncio.wait_for(committed.wait(), 30)
        task.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done(), "Cancellation detached the worker's heartbeat thread"
        assert api.get_training_status(job_id)["status"] == "cancelled"
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
