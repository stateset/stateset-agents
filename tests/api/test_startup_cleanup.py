"""Failed startup releases owned resources without publishing partial state."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from stateset_agents.api import distributed_cache, main, persistence


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["cache", "database", "agent", "health"])
@pytest.mark.parametrize("cleanup_failure", [None, "cache", "database", "inference"])
async def test_lifespan_failure_closes_previously_initialized_resources(
    monkeypatch, stage, cleanup_failure
):
    from stateset_agents.api.services import agent_service

    events = []
    failure = RuntimeError("startup failed")
    config = SimpleNamespace(api_version="test", is_production=lambda: True)
    monkeypatch.setattr(main, "get_config", lambda: config)
    monkeypatch.delenv("STATESET_DEFAULT_CHECKPOINT", raising=False)

    async def cache_init(config):
        if stage == "cache":
            raise failure

    async def database_init(config):
        if stage == "database":
            raise failure

    async def close(name):
        events.append(name)
        if name == cleanup_failure:
            raise KeyError("secondary cleanup failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(distributed_cache, "init_cache", cache_init)
    monkeypatch.setattr(distributed_cache, "close_cache", lambda: close("cache"))
    monkeypatch.setattr(persistence, "init_database", database_init)
    monkeypatch.setattr(persistence, "close_database", lambda: close("database"))
    app = FastAPI()
    app.state.config = config
    app.state.inference_service = SimpleNamespace(
        aclose=lambda: close("inference"), check_health=lambda: True
    )
    app.state.training_service = SimpleNamespace(aclose=lambda: close("training"))
    if stage == "agent":
        monkeypatch.setattr(agent_service, "AgentService", fail)
    else:
        app.state.agent_service = object()
    if stage == "health":
        app.state.health_checker = SimpleNamespace(add_check=fail)
    with pytest.raises(RuntimeError) as caught:
        async with main.lifespan(app):
            pytest.fail("Failed startup yielded an application")
    assert caught.value is failure
    expected = ["training"]
    if stage != "cache":
        expected.append("cache")
    if stage in ("agent", "health"):
        expected.append("database")
    assert events == expected + ["inference"]


@pytest.mark.asyncio
async def test_cancelled_startup_drains_cleanup_under_repeated_cancellation(
    monkeypatch,
):
    events = []
    connecting, closing, release = [asyncio.Event() for _ in range(3)]
    config = SimpleNamespace(api_version="test", is_production=lambda: True)
    monkeypatch.setattr(main, "get_config", lambda: config)

    async def cache_init(config):
        pass

    async def database_init(config):
        connecting.set()
        await asyncio.Event().wait()

    async def close_cache():
        closing.set()
        await release.wait()
        events.append("cache")

    async def close_inference():
        events.append("inference")

    monkeypatch.setattr(distributed_cache, "init_cache", cache_init)
    monkeypatch.setattr(distributed_cache, "close_cache", close_cache)
    monkeypatch.setattr(persistence, "init_database", database_init)
    app = FastAPI()
    app.state.config = config
    app.state.inference_service = SimpleNamespace(
        aclose=close_inference, check_health=lambda: True
    )

    async def start():
        async with main.lifespan(app):
            pytest.fail("Cancelled startup yielded an application")

    task = asyncio.create_task(start())
    try:
        await connecting.wait()
        task.cancel()
        await closing.wait()
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not events
    finally:
        release.set()
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]
    assert isinstance(outcome, asyncio.CancelledError)
    assert events == ["cache", "inference"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["redis", "hybrid", "database"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_partial_initialization_is_closed_before_error_and_never_published(
    monkeypatch, backend, cancelled
):
    connecting, closing, release = [asyncio.Event() for _ in range(3)]
    failure = RuntimeError("connection failed")
    original = object()
    instances = []

    class Resource:
        def __init__(self, config):
            self.closed = False
            self.close_calls = 0
            instances.append(self)

        async def connect(self):
            connecting.set()
            if cancelled:
                await asyncio.Event().wait()
            raise failure

        async def close(self):
            self.close_calls += 1
            closing.set()
            await release.wait()
            self.closed = True

    if backend == "database":
        monkeypatch.setattr(persistence, "_database", original)
        monkeypatch.setattr(persistence, "UnitOfWork", Resource)
        operation = persistence.init_database(persistence.DatabaseConfig())
        getter = persistence.get_database
    else:
        monkeypatch.setattr(distributed_cache, "_cache_instance", original)
        monkeypatch.setattr(
            distributed_cache,
            "RedisCache" if backend == "redis" else "HybridCache",
            Resource,
        )
        operation = distributed_cache.init_cache(
            distributed_cache.CacheConfig(
                backend=distributed_cache.CacheBackend(backend)
            )
        )
        getter = distributed_cache.get_cache
    task = asyncio.create_task(operation)
    try:
        await connecting.wait()
        if cancelled:
            task.cancel()
        await closing.wait()
        assert getter() is original
        assert not task.done() and not instances[0].closed
        if cancelled:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
    finally:
        release.set()
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]
    assert (
        isinstance(outcome, asyncio.CancelledError) if cancelled else outcome is failure
    )
    assert getter() is original
    assert instances[0].closed and instances[0].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["redis", "hybrid", "database"])
async def test_cleanup_error_does_not_hide_initialization_failure(monkeypatch, backend):
    failure = RuntimeError("primary startup error")

    class Resource:
        def __init__(self, config):
            pass

        async def connect(self):
            raise failure

        async def close(self):
            raise ValueError("secondary cleanup error")

    if backend == "database":
        monkeypatch.setattr(persistence, "UnitOfWork", Resource)
        operation = persistence.init_database(persistence.DatabaseConfig())
    else:
        monkeypatch.setattr(
            distributed_cache,
            "RedisCache" if backend == "redis" else "HybridCache",
            Resource,
        )
        operation = distributed_cache.create_cache(
            distributed_cache.CacheConfig(
                backend=distributed_cache.CacheBackend(backend)
            )
        )
    with pytest.raises(RuntimeError) as caught:
        await operation
    assert caught.value is failure


@pytest.mark.asyncio
async def test_database_close_attempts_all_repositories_after_a_failure():
    closed = []
    failure = RuntimeError("first close failed")

    async def close(name):
        closed.append(name)
        if name == "agents":
            raise failure

    database = persistence.UnitOfWork(persistence.DatabaseConfig())
    for name in ("agents", "conversations", "training_jobs", "api_keys"):
        setattr(
            database, "_" + name, SimpleNamespace(close=lambda name=name: close(name))
        )
    with pytest.raises(RuntimeError) as caught:
        await database.close()
    assert caught.value is failure
    assert closed == ["agents", "conversations", "training_jobs", "api_keys"]
