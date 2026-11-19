"""Shared API resources survive overlapping owners and retire exactly once."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from stateset_agents.api import distributed_cache, main, persistence
from stateset_agents.api.resource_lifecycle import (
    ResourceOwnershipError,
    SharedResource,
)


@pytest.fixture(autouse=True)
def isolated_globals(monkeypatch):
    monkeypatch.setattr(distributed_cache, "_cache_instance", None)
    monkeypatch.setattr(
        distributed_cache, "_cache_registry", distributed_cache._new_registry()
    )
    monkeypatch.setattr(persistence, "_database", None)
    monkeypatch.setattr(persistence, "_database_registry", persistence._new_registry())


@dataclass
class Config:
    value: int = 1


class Harness:
    def __init__(self):
        self.current = None
        self.created = []
        self.prepared = []
        self.closed = []
        self.on_prepare = None
        self.on_close = None
        self.registry = SharedResource(
            "test",
            lambda: self.current,
            self.publish,
            self.construct,
            self.prepare,
            self.close,
        )

    def publish(self, resource):
        self.current = resource

    def construct(self, config):
        resource = SimpleNamespace(config=config, closed=False)
        self.created.append(resource)
        return resource

    async def prepare(self, resource):
        self.prepared.append(resource)
        if self.on_prepare:
            await self.on_prepare(resource)

    async def close(self, resource):
        self.closed.append(resource)
        if self.on_close:
            await self.on_close(resource)
        resource.closed = True


async def entered(event):
    await asyncio.wait_for(event.wait(), 2)


@pytest.mark.asyncio
async def test_matching_initialization_preserves_real_memory_records_and_cache():
    database = await persistence.init_database()
    record = persistence.Agent(name="kept")
    await database.agents.create(record)
    assert await persistence.init_database(persistence.DatabaseConfig()) is database
    assert (await database.agents.get(record.id)).name == "kept"
    cache = await distributed_cache.init_cache()
    await cache.set("kept", "value")
    assert await distributed_cache.init_cache(distributed_cache.CacheConfig()) is cache
    assert await cache.get("kept") == "value"
    await persistence.close_database()
    await distributed_cache.close_cache()


@pytest.mark.asyncio
async def test_sqlite_leases_share_data_and_last_release_closes_connection():
    config = persistence.DatabaseConfig(
        backend=persistence.DatabaseBackend.SQLITE, connection_url=":memory:"
    )
    first = await persistence.acquire_database(config)
    second = await persistence.acquire_database(config)
    database = first.resource
    assert second.resource is database
    await database.agents.create(persistence.Agent(id="kept", name="same database"))
    for operation in (
        persistence.close_database(),
        persistence.init_database(persistence.DatabaseConfig()),
    ):
        with pytest.raises(ResourceOwnershipError, match="leases"):
            await operation
    await first.aclose()
    assert persistence.get_database_status()["leases"] == 1
    assert (await second.resource.agents.get("kept")).name == "same database"
    await second.aclose()
    await first.aclose()
    assert persistence.get_database() is None
    assert database.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_simultaneous_acquisition_prepares_once_and_captures_mutable_config():
    h = Harness()
    started, release = asyncio.Event(), asyncio.Event()

    async def prepare(resource):
        started.set()
        await release.wait()

    h.on_prepare = prepare
    config = Config()
    first = asyncio.create_task(h.registry.acquire(config))
    await entered(started)
    second = asyncio.create_task(h.registry.acquire(Config()))
    config.value = 99
    await asyncio.sleep(0)
    assert h.registry.status()["state"] == "initializing"
    release.set()
    a, b = await asyncio.gather(first, second)
    assert a.resource is b.resource
    assert a.resource.config.value == 1
    assert len(h.created) == len(h.prepared) == 1
    await a.aclose()
    assert not h.closed
    await b.aclose()
    assert h.closed == [a.resource]


@pytest.mark.asyncio
async def test_cancelled_waiting_acquisition_does_not_release_another_owner():
    h = Harness()
    started, release = asyncio.Event(), asyncio.Event()

    async def prepare(resource):
        started.set()
        await release.wait()

    h.on_prepare = prepare
    first = asyncio.create_task(h.registry.acquire(Config()))
    await entered(started)
    waiting = asyncio.create_task(h.registry.acquire(Config()))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    release.set()
    lease = await first
    assert len(h.created) == 1 and not h.closed
    assert h.registry.status()["leases"] == 1
    await lease.aclose()


@pytest.mark.asyncio
async def test_cancelled_preparation_drains_callback_then_unpublished_resource():
    h = Harness()
    started, cleaning, release = (asyncio.Event() for _ in range(3))

    async def prepare(resource):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()

    h.on_prepare = prepare
    task = asyncio.create_task(h.registry.acquire(Config()))
    await entered(started)
    task.cancel()
    await entered(cleaning)
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not h.closed
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.current is None and h.closed == h.created
    assert h.registry.status()["leases"] == 0
    assert h.registry.status()["state"] == "empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_replacement_drains_old_owner_before_publishing_new(cancel):
    h = Harness()
    old = await h.registry.initialize(Config())
    started, release = asyncio.Event(), asyncio.Event()

    async def close(resource):
        if resource is old:
            started.set()
            await release.wait()

    h.on_close = close
    task = asyncio.create_task(h.registry.initialize(Config(2)))
    await entered(started)
    assert h.current is old
    with pytest.raises(ResourceOwnershipError):
        h.registry.get()
    if cancel:
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
    release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert h.current is None
        assert h.closed == h.created
    else:
        new = await task
        assert h.current is new and old.closed
        assert h.closed == [old]
        await h.registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["prepare", "retire"])
async def test_failed_replacement_preserves_failure_and_never_retries_cleanup(stage):
    h = Harness()
    old = await h.registry.initialize(Config())
    primary = ValueError("private provider details")
    cleanup = OSError("private cleanup details")

    async def prepare(resource):
        if stage == "prepare":
            raise primary

    async def close(resource):
        if stage == "retire" and resource is old:
            raise primary
        if stage == "prepare":
            raise cleanup

    h.on_prepare, h.on_close = prepare, close
    with pytest.raises(ValueError) as caught:
        await h.registry.initialize(Config(2))
    assert caught.value is primary
    assert h.current is old
    assert h.registry.status()["state"] == "failed"
    assert h.registry.status()["unreleased_resources"] == 1
    assert "private" not in str(h.registry.status())
    calls = list(h.closed)
    for operation in (h.registry.initialize(Config(3)), h.registry.close()):
        with pytest.raises(ResourceOwnershipError):
            await operation
    with pytest.raises(ResourceOwnershipError):
        h.registry.get()
    assert h.closed == calls


@pytest.mark.asyncio
async def test_failed_preparation_leaves_previous_owner_usable_after_clean_rollback():
    h = Harness()
    old = await h.registry.initialize(Config())

    async def fail(resource):
        raise ValueError("cannot connect")

    h.on_prepare = fail
    with pytest.raises(ValueError):
        await h.registry.initialize(Config(2))
    assert h.registry.get() is old and not old.closed
    assert h.registry.status()["state"] == "ready"
    h.on_prepare = None
    await h.registry.close()


@pytest.mark.asyncio
async def test_last_lease_cleanup_is_joined_once_despite_multiple_cancelled_observers():
    h = Harness()
    lease = await h.registry.acquire(Config())
    started, release = asyncio.Event(), asyncio.Event()

    async def close(resource):
        started.set()
        await release.wait()

    h.on_close = close
    first = asyncio.create_task(lease.aclose())
    await entered(started)
    second = asyncio.create_task(lease.aclose())
    await asyncio.sleep(0)  # Both observers have entered the owned close boundary.
    for _ in range(3):
        first.cancel()
        second.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    await lease.aclose()
    assert h.closed == [lease.resource] and h.current is None


@pytest.mark.asyncio
async def test_close_waits_for_initialization_and_prevents_late_publication():
    h = Harness()
    started, release = asyncio.Event(), asyncio.Event()

    async def prepare(resource):
        started.set()
        await release.wait()

    h.on_prepare = prepare
    opening = asyncio.create_task(h.registry.initialize(Config()))
    await entered(started)
    closing = asyncio.create_task(h.registry.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    resource, _ = await asyncio.gather(opening, closing)
    assert resource.closed and h.current is None
    assert h.registry.status()["state"] == "empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("callback", ["prepare", "close"])
async def test_inherited_callback_cannot_reenter_its_owner(callback):
    h = Harness()
    errors = []

    async def reenter(resource):
        # Child tasks inherit callback ownership, rather than waiting forever.
        task = asyncio.create_task(h.registry.close())
        with pytest.raises(ResourceOwnershipError) as caught:
            await asyncio.wait_for(task, 1)
        errors.append(caught.value)

    if callback == "prepare":
        h.on_prepare = reenter
    else:
        h.on_close = reenter
    await h.registry.initialize(Config())
    await h.registry.close()
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_stale_close_cannot_detach_an_explicitly_installed_replacement():
    h = Harness()
    await h.registry.initialize(Config())
    started, release = asyncio.Event(), asyncio.Event()

    async def close(resource):
        started.set()
        await release.wait()

    h.on_close = close
    task = asyncio.create_task(h.registry.close())
    await entered(started)
    replacement = object()
    h.current = replacement
    release.set()
    await task
    assert h.registry.get() is replacement


def application(monkeypatch):
    config = SimpleNamespace(api_version="test", is_production=lambda: False)
    monkeypatch.setattr(main, "get_config", lambda: config)
    monkeypatch.delenv("STATESET_DEFAULT_CHECKPOINT", raising=False)
    monkeypatch.setenv("CACHE_BACKEND", "memory")
    monkeypatch.setenv("DB_BACKEND", "memory")
    app = FastAPI()
    app.state.config = config
    app.state.agent_service = object()

    async def close():
        pass

    app.state.training_service = SimpleNamespace(aclose=close)
    app.state.inference_service = SimpleNamespace(
        aclose=close, check_health=lambda: True
    )
    return app


@pytest.mark.asyncio
async def test_overlapping_applications_share_resources_until_both_have_shutdown(
    monkeypatch,
):
    first, second = application(monkeypatch), application(monkeypatch)
    a, b = main.lifespan(first), main.lifespan(second)
    await a.__aenter__()
    database, cache = persistence.get_database(), distributed_cache.get_cache()
    await database.agents.create(persistence.Agent(id="a", name="shared"))
    await cache.set("a", "shared")
    await b.__aenter__()
    assert persistence.get_database() is database
    assert distributed_cache.get_cache() is cache
    assert persistence.get_database_status()["leases"] == 2
    await a.__aexit__(None, None, None)
    assert (await persistence.get_database().agents.get("a")).name == "shared"
    assert await distributed_cache.get_cache().get("a") == "shared"
    await b.__aexit__(None, None, None)
    assert persistence.get_database() is None and distributed_cache.get_cache() is None


@pytest.mark.asyncio
async def test_incompatible_startup_fails_even_in_development_and_preserves_serving_app(
    monkeypatch,
):
    first, second = application(monkeypatch), application(monkeypatch)
    async with main.lifespan(first):
        database = persistence.get_database()
        cache = distributed_cache.get_cache()
        monkeypatch.setenv("CACHE_KEY_PREFIX", "different:")
        with pytest.raises(ResourceOwnershipError, match="configuration"):
            async with main.lifespan(second):
                pytest.fail("Incompatible application must not begin serving")
        assert persistence.get_database() is database
        assert distributed_cache.get_cache() is cache
        assert distributed_cache.get_cache_status()["leases"] == 1


@pytest.mark.asyncio
async def test_failed_second_application_releases_only_its_own_leases(monkeypatch):
    first, second = application(monkeypatch), application(monkeypatch)
    async with main.lifespan(first):
        database, cache = persistence.get_database(), distributed_cache.get_cache()

        def fail(*args):
            raise ValueError("health registration failed")

        second.state.health_checker = SimpleNamespace(add_check=fail)
        with pytest.raises(ValueError, match="health registration"):
            async with main.lifespan(second):
                pytest.fail("Failed startup cannot serve")
        assert persistence.get_database() is database
        assert distributed_cache.get_cache() is cache
        assert persistence.get_database_status()["leases"] == 1
        await database.agents.create(persistence.Agent(name="still available"))


@pytest.mark.asyncio
async def test_leased_resource_rejects_foreign_event_loop_before_effects():
    h = Harness()
    lease = await h.registry.acquire(Config())

    def foreign_loop():
        async def run():
            with pytest.raises(ResourceOwnershipError, match="event loop"):
                await h.registry.acquire(Config())
            with pytest.raises(ResourceOwnershipError, match="event loop"):
                await lease.aclose()

        asyncio.run(run())

    await asyncio.to_thread(foreign_loop)
    assert h.registry.status()["leases"] == 1 and not h.closed
    await lease.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["redis", "hybrid"])
async def test_real_cache_wrapper_retains_sdk_cleanup_failure_without_reconnect(
    monkeypatch, backend
):
    client = SimpleNamespace()
    calls = []

    async def ping():
        calls.append("ping")
        return True

    async def close():
        calls.append("close")
        raise OSError("pool cleanup failed")

    client.ping, client.aclose = ping, close
    redis = pytest.importorskip("redis.asyncio")

    monkeypatch.setattr(redis, "from_url", lambda *a, **kw: client)
    lease = await distributed_cache.acquire_cache(
        distributed_cache.CacheConfig(
            backend=distributed_cache.CacheBackend(backend),
            redis_url="redis://unused.invalid",
        )
    )
    with pytest.raises(OSError, match="pool cleanup"):
        await lease.aclose()
    redis_cache = lease.resource if backend == "redis" else lease.resource._redis
    assert redis_cache._client is client
    assert not await redis_cache.health_check()
    with pytest.raises(ResourceOwnershipError):
        await redis_cache.connect()
    with pytest.raises(OSError):
        await redis_cache.close()
    with pytest.raises(ResourceOwnershipError):
        await distributed_cache.close_cache()
    assert calls == ["ping", "close"]
    assert distributed_cache.get_cache_status()["unreleased_resources"] == 1


@pytest.mark.asyncio
async def test_redis_pool_close_drains_cancellation_and_blocks_inherited_self_close():
    cache = distributed_cache.RedisCache(distributed_cache.CacheConfig())
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def close():
        calls.append("close")
        with pytest.raises(ResourceOwnershipError, match="itself"):
            await asyncio.create_task(cache.close())
        started.set()
        await release.wait()

    cache._client = SimpleNamespace(aclose=close)
    closing = asyncio.create_task(cache.close())
    await entered(started)
    for _ in range(3):
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done() and cache._client is not None
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    await cache.close()
    assert calls == ["close"] and cache._client is None


@pytest.mark.asyncio
async def test_last_release_failure_is_retained_for_every_observer():
    h = Harness()
    lease = await h.registry.acquire(Config())
    failure = OSError("close failed")

    async def close(resource):
        raise failure

    h.on_close = close
    for _ in range(2):
        with pytest.raises(OSError) as caught:
            await lease.aclose()
        assert caught.value is failure
    assert h.closed == [lease.resource]
    assert h.registry.status()["state"] == "failed"


@pytest.mark.asyncio
async def test_initialization_replaces_a_directly_closed_database_without_reopening_it():
    first = await persistence.init_database()
    await first.close()
    second = await persistence.init_database()
    assert second is not first
    assert first.get_status()["state"] == "closed"
    await second.agents.create(persistence.Agent(name="new owner"))
    await persistence.close_database()


@pytest.mark.asyncio
async def test_development_startup_cannot_hide_unconfirmed_partial_cleanup(monkeypatch):
    app = application(monkeypatch)
    monkeypatch.setenv("STATESET_AGENTS_STRICT_STARTUP", "false")
    calls = []

    class Database:
        def __init__(self, config):
            pass

        async def connect(self):
            raise ValueError("database startup failed")

        async def close(self):
            calls.append("close")
            raise OSError("cleanup unconfirmed")

    monkeypatch.setattr(persistence, "UnitOfWork", Database)
    with pytest.raises(ValueError, match="startup failed"):
        async with main.lifespan(app):
            pytest.fail("Unconfirmed cleanup must abort startup")
    assert persistence.get_database_status()["state"] == "failed"
    assert persistence.get_database_status()["unreleased_resources"] == 1
    assert distributed_cache.get_cache() is None
    assert calls == ["close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_rejected_global_close_keeps_active_transaction_and_later_close_usable(
    backend,
):
    database = await persistence.init_database(
        persistence.DatabaseConfig(
            backend=persistence.DatabaseBackend(backend), connection_url=":memory:"
        )
    )
    async with database.transaction():
        await database.agents.create(persistence.Agent(id="a", name="committed"))
        with pytest.raises(persistence.TransactionConflict, match="active transaction"):
            await persistence.close_database()
        assert persistence.get_database() is database
        assert persistence.get_database_status()["cleanup_error_type"] is None
        assert database._close_task is None
    assert (await database.agents.get("a")).name == "committed"
    await persistence.close_database()
    assert persistence.get_database() is None


@pytest.mark.asyncio
async def test_rejected_replacement_closes_candidate_and_preserves_published_database():
    database = await persistence.init_database()
    async with database.transaction():
        with pytest.raises(persistence.TransactionConflict):
            await persistence.init_database(
                persistence.DatabaseConfig(
                    backend=persistence.DatabaseBackend.SQLITE,
                    connection_url=":memory:",
                )
            )
        assert persistence.get_database() is database
        assert persistence.get_database_status()["unreleased_resources"] == 0
        assert persistence.get_database_status()["state"] == "ready"
    await persistence.close_database()


@pytest.mark.asyncio
async def test_transaction_conflict_after_cleanup_started_remains_a_retained_failure(
    monkeypatch,
):
    database = await persistence.init_database()
    calls = []

    async def close():
        calls.append("close")
        raise persistence.TransactionConflict("repository cleanup failed")

    monkeypatch.setattr(database._agents, "close", close, raising=False)
    with pytest.raises(persistence.TransactionConflict):
        await persistence.close_database()
    assert persistence.get_database_status()["state"] == "failed"
    assert persistence.get_database_status()["unreleased_resources"] == 1
    with pytest.raises(ResourceOwnershipError):
        await persistence.close_database()
    assert calls == ["close"]
