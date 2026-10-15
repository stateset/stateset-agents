"""Lifespan cleanup retains shared services until owned training has stopped."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from stateset_agents.api import main
from stateset_agents.api.schemas import TrainingRequest
from stateset_agents.api.services import training_service
from tests.api.test_training_job_lifecycle import set_trainer


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "serving_error", "cancel_shutdown"])
async def test_lifespan_drains_training_before_closing_dependencies(monkeypatch, mode):
    from stateset_agents.api import distributed_cache, persistence

    monkeypatch.setattr(training_service, "MultiTurnAgent", lambda config: object())
    service = training_service.TrainingService()
    request_data = TrainingRequest(
        agent_config={"model_name": "synthetic-test"},
        environment_scenarios=[],
        reward_config={},
        num_episodes=2,
    )
    events = []
    started, release, shutdown_started = [asyncio.Event() for _ in range(3)]
    config = SimpleNamespace(api_version="test", is_production=lambda: False)
    monkeypatch.setattr(main, "get_config", lambda: config)
    monkeypatch.setenv("STATESET_AGENTS_STRICT_STARTUP", "false")
    monkeypatch.delenv("STATESET_DEFAULT_CHECKPOINT", raising=False)

    async def initialize(config):
        pass

    async def close_cache():
        events.append("cache")

    async def close_database():
        events.append("database")

    async def close_inference():
        events.append("inference")

    monkeypatch.setattr(distributed_cache, "init_cache", initialize)
    monkeypatch.setattr(distributed_cache, "close_cache", close_cache)
    monkeypatch.setattr(persistence, "init_database", initialize)
    monkeypatch.setattr(persistence, "close_database", close_database)
    original_close = service.aclose

    async def close_training():
        shutdown_started.set()
        await original_close()

    monkeypatch.setattr(service, "aclose", close_training)

    async def train(**kwargs):
        started.set()
        await release.wait()
        assert not events
        events.append("training")

    set_trainer(monkeypatch, train)
    app = FastAPI()
    app.state.config = config
    app.state.agent_service = object()
    app.state.training_service = service
    app.state.inference_service = SimpleNamespace(
        check_health=lambda: True, aclose=close_inference
    )

    async def serve():
        async with main.lifespan(app):
            await service.start_training(request_data)
            await started.wait()
            if mode == "serving_error":
                raise ValueError("serving failed")

    server = asyncio.create_task(serve())
    try:
        await shutdown_started.wait()
        assert service.is_closing and not events
        assert not server.done()
        if mode == "cancel_shutdown":
            for _ in range(3):
                server.cancel()
                await asyncio.sleep(0)
                assert not server.done() and not events
    finally:
        release.set()
        outcome = (await asyncio.gather(server, return_exceptions=True))[0]
    if mode == "serving_error":
        assert isinstance(outcome, ValueError) and str(outcome) == "serving failed"
    elif mode == "cancel_shutdown":
        assert isinstance(outcome, asyncio.CancelledError)
    else:
        assert outcome is None
    assert events == ["training", "cache", "database", "inference"]
    assert not service._tasks and not service._cancel_events
