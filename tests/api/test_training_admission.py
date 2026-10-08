"""Training admission stays bounded until owned workers have actually exited."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from stateset_agents.api.auth import AuthenticatedUser
from stateset_agents.api.config import APIConfig
from stateset_agents.api.dependencies import get_current_user, get_training_service
from stateset_agents.api.errors import ResourceExhaustedError, setup_exception_handlers
from stateset_agents.api.routers.training import router
from stateset_agents.api.schemas import TrainingRequest
from stateset_agents.api.services import training_service as module
from tests.api.test_training_job_lifecycle import set_trainer


@pytest.fixture
def submission():
    return TrainingRequest(
        agent_config={"model_name": "synthetic"},
        environment_scenarios=[{"id": "test"}],
        reward_config={},
        num_episodes=2,
    )


def test_capacity_option_preserves_existing_positional_config_arguments():
    legacy_fields = (
        "environment",
        "host",
        "port",
        "api_version",
        "title",
        "cors",
        "rate_limit",
        "security",
        "validation",
        "observability",
        "enable_training_lab",
    )
    baseline = APIConfig()
    arguments = [getattr(baseline, name) for name in legacy_fields]
    default = APIConfig(*arguments)
    configured = APIConfig(*arguments, training_max_concurrent_jobs=3)
    for name, value in zip(legacy_fields, arguments, strict=True):
        assert getattr(default, name) is value
        assert getattr(configured, name) is value
    assert default.training_max_concurrent_jobs == 1
    assert configured.training_max_concurrent_jobs == 3


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2", None])
def test_invalid_capacity_is_rejected(limit):
    with pytest.raises(ValueError, match="positive integer"):
        module.TrainingService(max_concurrent_jobs=limit)


@pytest.mark.parametrize(
    "value,expected", [(None, 1), ("3", 3), ("0", 1), ("-2", 1), ("invalid", 1)]
)
def test_capacity_environment_setting_has_a_bounded_default(
    monkeypatch, value, expected
):
    if value is None:
        monkeypatch.delenv("API_TRAINING_MAX_CONCURRENT_JOBS", raising=False)
    else:
        monkeypatch.setenv("API_TRAINING_MAX_CONCURRENT_JOBS", value)
    assert APIConfig.from_env().training_max_concurrent_jobs == expected


@pytest.mark.asyncio
async def test_lazy_service_uses_application_capacity():
    app = FastAPI()
    app.state.config = APIConfig(training_max_concurrent_jobs=3)
    request = Request({"type": "http", "app": app, "headers": []})
    service = await get_training_service(request)
    assert service.max_concurrent_jobs == 3
    assert await get_training_service(request) is service
    await service.aclose()


@pytest.mark.asyncio
async def test_lifespan_service_uses_application_capacity(monkeypatch):
    from stateset_agents.api import distributed_cache, main, persistence

    async def no_op(*args):
        pass

    config = APIConfig(training_max_concurrent_jobs=3)
    monkeypatch.setattr(main, "get_config", lambda: config)
    monkeypatch.delenv("STATESET_DEFAULT_CHECKPOINT", raising=False)
    monkeypatch.setattr(distributed_cache, "init_cache", no_op)
    monkeypatch.setattr(distributed_cache, "close_cache", no_op)
    monkeypatch.setattr(persistence, "init_database", no_op)
    monkeypatch.setattr(persistence, "close_database", no_op)
    app = FastAPI()
    app.state.config = config
    app.state.agent_service = object()
    app.state.inference_service = SimpleNamespace(
        check_health=lambda: True, aclose=no_op
    )
    async with main.lifespan(app):
        assert app.state.training_service.max_concurrent_jobs == 3
    assert app.state.training_service.is_closing


@pytest.mark.asyncio
async def test_concurrent_submissions_cannot_exceed_capacity(monkeypatch, submission):
    created = []
    release = asyncio.Event()

    def agent(config):
        created.append(config)
        return object()

    async def train(**kwargs):
        await release.wait()

    monkeypatch.setattr(module, "MultiTurnAgent", agent)
    set_trainer(monkeypatch, train)
    service = module.TrainingService(max_concurrent_jobs=2)
    try:
        results = await asyncio.gather(
            *(service.start_training(submission) for _ in range(20)),
            return_exceptions=True,
        )
        accepted = [result for result in results if isinstance(result, str)]
        rejected = [
            result for result in results if isinstance(result, ResourceExhaustedError)
        ]
        assert len(accepted) == 2 and len(rejected) == 18
        assert len(created) == len(service._tasks) == len(service.jobs) == 2
        assert set(accepted) == set(service.jobs)
        assert all(error.status_code == 429 and error.limit == 2 for error in rejected)
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["complete", "failed", "cancelled", "cancel_before_start"]
)
async def test_capacity_released_only_when_worker_exits(
    monkeypatch, submission, outcome
):
    started, release = asyncio.Event(), asyncio.Event()
    created = []

    def agent(config):
        created.append(config)
        return object()

    async def train(**kwargs):
        started.set()
        await release.wait()
        if outcome == "failed":
            raise RuntimeError("training failed")

    monkeypatch.setattr(module, "MultiTurnAgent", agent)
    set_trainer(monkeypatch, train)
    service = module.TrainingService()
    assert service.max_concurrent_jobs == 1
    job_id = await service.start_training(submission)
    task = service._tasks[job_id]
    try:
        if outcome == "cancel_before_start":
            task.cancel()
        else:
            await started.wait()
            if outcome == "cancelled":
                service.cancel_training(job_id)
                assert service.jobs[job_id]["status"] == "cancelling"
        with pytest.raises(ResourceExhaustedError):
            await service.start_training(submission)
        assert len(created) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    assert not service._tasks
    replacement = await service.start_training(submission)
    await service._tasks[replacement]
    assert len(created) == 2


@pytest.mark.asyncio
async def test_constructor_failure_does_not_consume_a_slot(monkeypatch, submission):
    def broken(config):
        raise ValueError("bad agent configuration")

    monkeypatch.setattr(module, "MultiTurnAgent", broken)
    service = module.TrainingService()
    with pytest.raises(ValueError, match="bad agent"):
        await service.start_training(submission)
    assert not service.jobs and not service._tasks
    monkeypatch.setattr(module, "MultiTurnAgent", lambda config: object())

    async def train(**kwargs):
        pass

    set_trainer(monkeypatch, train)
    job_id = await service.start_training(submission)
    await service._tasks[job_id]


@pytest.mark.asyncio
async def test_http_admission_rejects_without_creating_another_job(
    monkeypatch, submission
):
    release = asyncio.Event()
    created = []

    def agent(config):
        created.append(config)
        return object()

    async def train(**kwargs):
        await release.wait()

    monkeypatch.setattr(module, "MultiTurnAgent", agent)
    set_trainer(monkeypatch, train)
    service = module.TrainingService()
    app = FastAPI()
    app.include_router(router)
    setup_exception_handlers(app)
    app.dependency_overrides[get_training_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id="owner", roles=["trainer"]
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = await client.post("/training", json=submission.model_dump())
            rejected = await client.post("/training", json=submission.model_dump())
        assert first.status_code == 202, first.text
        assert rejected.status_code == 429, rejected.text
        assert "training_jobs" in rejected.text
        assert len(created) == len(service.jobs) == 1
    finally:
        release.set()
        await service.aclose()
