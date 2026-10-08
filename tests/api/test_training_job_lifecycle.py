"""API job state reflects background execution and cooperative cancellation."""

import asyncio
import builtins
import copy
import importlib
from types import SimpleNamespace

import pytest

from stateset_agents.api.errors import ServiceUnavailableError
from stateset_agents.api.routers.training import cancel_training
from stateset_agents.api.schemas import TrainingRequest
from stateset_agents.api.services import training_service as module


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(module, "MultiTurnAgent", lambda config: object())
    return module.TrainingService(max_concurrent_jobs=2)


@pytest.fixture
def request_data():
    return TrainingRequest(
        agent_config={"model_name": "synthetic-test"},
        environment_scenarios=[],
        reward_config={},
        num_episodes=2,
    )


def set_trainer(monkeypatch, trainer):
    monkeypatch.setattr(
        importlib.import_module("stateset_agents.training.train"), "train", trainer
    )


@pytest.mark.asyncio
async def test_cancellation_stays_pending_until_trainer_exits(
    service, request_data, monkeypatch
):
    started, release = asyncio.Event(), asyncio.Event()

    async def train(**kwargs):
        started.set()
        await release.wait()
        return object()  # Late normal return must not override cancellation.

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(request_data, user_id="owner")
    task = service._tasks[job_id]
    try:
        await started.wait()
        for _ in range(3):
            response = await cancel_training(
                job_id, svc=service, user=SimpleNamespace(user_id="owner")
            )
            assert response.status == "cancelling"
            assert service.jobs[job_id]["completed_at"] is None
            assert service._tasks[job_id] is task and not task.done()
        assert not service.cancel_training(job_id, user_id="other")
    finally:
        release.set()
        await task
    assert service.jobs[job_id]["status"] == "cancelled"
    assert service.jobs[job_id]["progress"] == 0
    assert service.jobs[job_id]["completion_scope"] is None
    assert service.jobs[job_id]["completed_at"] is not None
    assert job_id not in service._tasks and job_id not in service._cancel_events


@pytest.mark.asyncio
async def test_cancel_before_start_never_calls_trainer(
    service, request_data, monkeypatch
):
    async def forbidden(**kwargs):
        pytest.fail("Cancelled job started training")

    set_trainer(monkeypatch, forbidden)
    job_id = await service.start_training(request_data)
    task = service._tasks[job_id]
    assert service.cancel_training(job_id)
    await task
    assert service.jobs[job_id]["status"] == "cancelled"
    assert job_id not in service._tasks and job_id not in service._cancel_events


@pytest.mark.asyncio
async def test_task_cancel_before_coroutine_start_cleans_up(service, request_data):
    job_id = await service.start_training(request_data)
    task = service._tasks[job_id]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)  # Allow the task's ownership callback to finish.
    assert service.jobs[job_id]["status"] == "cancelled"
    assert service.jobs[job_id]["completed_at"] is not None
    assert job_id not in service._tasks and job_id not in service._cancel_events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        KeyError("missing metric"),
        AssertionError("invalid state"),
        ImportError("optional trainer"),
    ],
)
async def test_unexpected_trainer_errors_become_terminal_failures(
    service, request_data, monkeypatch, failure
):
    async def train(**kwargs):
        raise failure

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(request_data)
    await service._tasks[job_id]
    assert service.jobs[job_id]["status"] == "failed"
    assert service.jobs[job_id]["error"] == str(failure)
    assert service.jobs[job_id]["completion_scope"] is None
    assert service.jobs[job_id]["completed_at"] is not None
    assert job_id not in service._tasks and job_id not in service._cancel_events


@pytest.mark.asyncio
async def test_trainer_import_error_is_recorded_and_retrieved(
    service, request_data, monkeypatch
):
    original = builtins.__import__

    def fail(name, *args, **kwargs):
        if name == "stateset_agents.training.train":
            raise ImportError("trainer dependency missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    job_id = await service.start_training(request_data)
    task = service._tasks[job_id]
    await task
    assert task.exception() is None
    assert service.jobs[job_id]["status"] == "failed"
    assert service.jobs[job_id]["error"] == "trainer dependency missing"
    assert job_id not in service._tasks and job_id not in service._cancel_events


@pytest.mark.asyncio
async def test_callback_cancellation_waits_for_trainer_cleanup(
    service, request_data, monkeypatch
):
    started, invoke_callback, cleaning, finish = [asyncio.Event() for _ in range(4)]

    async def train(**kwargs):
        started.set()
        await invoke_callback.wait()
        try:
            kwargs["callbacks"][0].on_episode_end(0, {"loss": 0.5})
        finally:
            cleaning.set()
            await finish.wait()

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(request_data)
    task = service._tasks[job_id]
    try:
        await started.wait()
        service.cancel_training(job_id)
        invoke_callback.set()
        await cleaning.wait()
        assert service.jobs[job_id]["status"] == "cancelling"
        assert service.jobs[job_id]["completed_at"] is None
        assert not task.done()
    finally:
        finish.set()
        invoke_callback.set()
        await task
    assert service.jobs[job_id]["status"] == "cancelled"
    assert service.jobs[job_id]["progress"] == 50


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_repeated_cancel_preserves_terminal_state_and_response(service, status):
    service.jobs["terminal"] = {
        "status": status,
        "completed_at": "original-time",
        "error": "original",
        "user_id": "owner",
    }
    original = copy.deepcopy(service.jobs["terminal"])
    for _ in range(2):
        response = await cancel_training(
            "terminal", svc=service, user=SimpleNamespace(user_id="owner")
        )
        assert response.status == status
        assert service.jobs["terminal"] == original


@pytest.mark.asyncio
async def test_successful_training_still_completes(service, request_data, monkeypatch):
    async def train(**kwargs):
        for i in range(2):
            kwargs["callbacks"][0].on_episode_end(i, {"loss": 0.5})
        return object()

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(request_data)
    await service._tasks[job_id]
    assert service.jobs[job_id]["status"] == "completed"
    assert service.jobs[job_id]["progress"] == 100
    assert service.jobs[job_id]["current_episode"] == 2
    assert service.jobs[job_id]["completion_scope"] == "final_episode_reported"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "episodes,scope,progress",
    [
        ([], "no_episode_progress", 0.0),
        ([0], "partial_episode_progress", 50.0),
        ([0, 1], "final_episode_reported", 100.0),
        ([1], "final_episode_reported", 100.0),  # Resumed or sparse callbacks.
    ],
)
async def test_normal_return_preserves_observed_progress_in_status_and_list(
    service, request_data, monkeypatch, episodes, scope, progress
):
    import httpx
    from fastapi import FastAPI

    from stateset_agents.api.auth import AuthenticatedUser
    from stateset_agents.api.dependencies import get_current_user, get_training_service
    from stateset_agents.api.routers.training import router

    async def train(**kwargs):
        for episode in episodes:
            kwargs["callbacks"][0].on_episode_end(episode, {"loss": 0.5})
        return object()

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(request_data, user_id="owner")
    assert service.get_training_status(job_id)["completion_scope"] is None
    await service._tasks[job_id]
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_training_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id="owner", roles=["trainer"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        detail = await client.get(f"/training/{job_id}")
        listing = await client.get("/training", params={"status": "completed"})
    assert detail.status_code == listing.status_code == 200
    assert listing.json()["items"] == [detail.json()]
    status = detail.json()
    assert status["status"] == "completed"
    assert status["completion_scope"] == scope
    assert status["progress"] == progress
    assert status["current_episode"] == (episodes[-1] + 1 if episodes else 0)
    assert status["completed_at"] is not None and status["error"] is None
    assert job_id not in service._tasks
    # Cancelling a terminal job must preserve its completion evidence.
    original = service.get_training_status(job_id)
    assert service.cancel_training(job_id)
    assert service.get_training_status(job_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [True, False])
async def test_task_owner_handles_runner_exit_without_terminal_state(
    service, request_data, monkeypatch, raises
):
    async def broken_runner(*args):
        if raises:
            raise LookupError("unexpected runner failure")

    monkeypatch.setattr(service, "_run_training", broken_runner)
    job_id = await service.start_training(request_data)
    task = service._tasks[job_id]
    finished = asyncio.Event()
    task.add_done_callback(lambda _: finished.set())
    await finished.wait()
    assert service.jobs[job_id]["status"] == "failed"
    assert service.jobs[job_id]["completed_at"] is not None
    assert service.jobs[job_id]["error"] == (
        "unexpected runner failure"
        if raises
        else "Training exited without a terminal status"
    )
    assert job_id not in service._tasks and job_id not in service._cancel_events


@pytest.mark.asyncio
async def test_shutdown_drains_all_workers_despite_repeated_caller_cancellation(
    service, request_data, monkeypatch
):
    started, release = asyncio.Queue(), asyncio.Event()

    async def train(**kwargs):
        started.put_nowait(True)
        await release.wait()

    set_trainer(monkeypatch, train)
    job_ids = [await service.start_training(request_data) for _ in range(2)]
    await started.get()
    await started.get()
    first = asyncio.create_task(service.aclose())
    second = asyncio.create_task(service.aclose())
    try:
        await asyncio.sleep(0)
        assert service.is_closing
        before = set(service.jobs)
        with pytest.raises(ServiceUnavailableError):
            await service.start_training(request_data)
        assert set(service.jobs) == before
        for _ in range(3):
            first.cancel()
            await asyncio.sleep(0)
            assert not first.done() and not second.done()
            assert len(service._tasks) == 2
            assert all(service.jobs[j]["status"] == "cancelling" for j in job_ids)
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
    assert first.cancelled() and second.exception() is None
    assert not service._tasks and not service._cancel_events
    assert all(service.jobs[j]["status"] == "cancelled" for j in job_ids)
    await service.aclose()


@pytest.mark.asyncio
async def test_shutdown_before_worker_start_skips_training(
    service, request_data, monkeypatch
):
    async def forbidden(**kwargs):
        pytest.fail("Shutdown job reached trainer")

    set_trainer(monkeypatch, forbidden)
    job_id = await service.start_training(request_data)
    await service.aclose()
    assert service.jobs[job_id]["status"] == "cancelled"
    assert not service._tasks and not service._cancel_events


@pytest.mark.asyncio
async def test_closed_training_service_returns_http_503(service, request_data):
    import httpx
    from fastapi import FastAPI

    from stateset_agents.api.auth import AuthenticatedUser
    from stateset_agents.api.dependencies import get_current_user, get_training_service
    from stateset_agents.api.errors import setup_exception_handlers
    from stateset_agents.api.routers.training import router

    await service.aclose()
    app = FastAPI()
    app.include_router(router)
    setup_exception_handlers(app)
    app.dependency_overrides[get_training_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id="owner", roles=["trainer"]
    )
    request_data.environment_scenarios = [{"id": "test"}]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/training", json=request_data.model_dump())
    assert response.status_code == 503, response.text
    assert "shutting down" in response.text
    assert not service.jobs and not service._tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["nan", "episode", "repeated"])
@pytest.mark.parametrize("swallow", [False, True])
async def test_invalid_progress_cannot_complete_or_break_status_json(
    service, request_data, monkeypatch, invalid, swallow
):
    import httpx
    from fastapi import FastAPI

    from stateset_agents.api.auth import AuthenticatedUser
    from stateset_agents.api.dependencies import get_current_user, get_training_service
    from stateset_agents.api.routers.training import router
    from stateset_agents.training.callbacks import notify_episode_end

    async def train(**kwargs):
        callbacks = kwargs["callbacks"]
        await notify_episode_end(callbacks, episode=0, metrics={"loss": 0.5})
        try:
            await notify_episode_end(
                callbacks,
                episode=(
                    2 if invalid == "episode" else (0 if invalid == "repeated" else 1)
                ),
                metrics={"loss": float("nan") if invalid == "nan" else 0.1},
            )
        except ValueError:
            if not swallow:
                raise
        return object()

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(request_data, user_id="owner")
    await service._tasks[job_id]
    assert service.jobs[job_id]["status"] == "failed"
    assert "Invalid training progress" in service.jobs[job_id]["error"]
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_training_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id="owner", roles=["trainer"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/training/{job_id}")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "failed"
    assert result["current_episode"] == 1 and result["progress"] == 50
    assert result["metrics"] == {"loss": 0.5}


@pytest.mark.asyncio
async def test_accepted_configuration_isolated_from_caller_status_and_trainer_mutation(
    service, request_data, monkeypatch
):
    started, release = asyncio.Event(), asyncio.Event()
    request_data.environment_scenarios = [
        {"id": "case", "context": {"values": ["original"]}}
    ]
    request_data.training_config_overrides = {"nested": {"schedule": [1, 2]}}
    request_data.resume_from_checkpoint = "original-checkpoint"
    accepted = copy.deepcopy(request_data.model_dump())
    monkeypatch.setattr(
        module, "MultiTurnAgent", lambda config: SimpleNamespace(config=config)
    )
    observed = {}

    async def train(**kwargs):
        started.set()
        await release.wait()
        observed.update(
            episodes=kwargs["num_episodes"],
            profile=kwargs["profile"],
            checkpoint=kwargs["resume_from_checkpoint"],
            model=kwargs["agent"].config.model_name,
            scenarios=copy.deepcopy(kwargs["environment"].scenarios),
            overrides=copy.deepcopy(kwargs["config_overrides"]),
        )
        kwargs["environment"].scenarios[0]["context"]["values"].append("trainer-only")
        kwargs["config_overrides"]["nested"]["schedule"].append(99)

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(request_data, user_id="owner")
    task = service._tasks[job_id]
    try:
        # The background coroutine has not run yet. These edits must not affect it.
        request_data.num_episodes = 7
        request_data.profile = "caller-changed"
        request_data.resume_from_checkpoint = "caller-checkpoint"
        request_data.agent_config.model_name = "caller-model"
        request_data.environment_scenarios[0]["context"]["values"].append("caller-only")
        request_data.training_config_overrides["nested"]["schedule"].append(3)
        await started.wait()
        snapshot = service.get_training_status(job_id, user_id="owner")
        assert snapshot["config"] == accepted
        snapshot["config"]["environment_scenarios"][0]["context"]["values"].append(
            "reader-only"
        )
        snapshot["config"]["training_config_overrides"]["nested"]["schedule"].append(4)
        snapshot.update(status="completed", user_id="intruder", current_episode=-1)
        snapshot["metrics"]["loss"] = float("nan")
        current = service.get_training_status(job_id, user_id="owner")
        assert current["status"] == "running" and current["current_episode"] == 0
        assert current["config"] == accepted and current["metrics"] == {}
        assert service.get_training_status(job_id, user_id="intruder") is None
    finally:
        release.set()
        await task
    assert observed == {
        "episodes": accepted["num_episodes"],
        "profile": accepted["profile"],
        "checkpoint": accepted["resume_from_checkpoint"],
        "model": accepted["agent_config"]["model_name"],
        "scenarios": accepted["environment_scenarios"],
        "overrides": accepted["training_config_overrides"],
    }
    assert service.get_training_status(job_id, user_id="owner")["config"] == accepted
    assert request_data.environment_scenarios[0]["context"]["values"] == [
        "original",
        "caller-only",
    ]
    assert request_data.training_config_overrides["nested"]["schedule"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_reused_request_does_not_share_mutable_training_inputs(
    service, request_data, monkeypatch
):
    request_data.training_config_overrides = {"nested": {"value": "original"}}
    observed = []

    async def train(**kwargs):
        observed.append(kwargs["config_overrides"]["nested"]["value"])
        kwargs["config_overrides"]["nested"]["value"] = "trainer-changed"

    set_trainer(monkeypatch, train)
    first = await service.start_training(request_data)
    second = await service.start_training(request_data)
    await asyncio.gather(service._tasks[first], service._tasks[second])
    assert observed == ["original", "original"]
    assert request_data.training_config_overrides["nested"]["value"] == "original"
    for job_id in (first, second):
        assert (
            service.get_training_status(job_id)["config"]["training_config_overrides"][
                "nested"
            ]["value"]
            == "original"
        )


@pytest.mark.asyncio
async def test_status_snapshot_does_not_change_when_job_finishes(
    service, request_data, monkeypatch
):
    release = asyncio.Event()

    async def train(**kwargs):
        await release.wait()

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(request_data)
    task = service._tasks[job_id]
    snapshot = service.get_training_status(job_id)
    release.set()
    await task
    assert snapshot["status"] == "running" and snapshot["completed_at"] is None
    assert service.get_training_status(job_id)["status"] == "completed"
