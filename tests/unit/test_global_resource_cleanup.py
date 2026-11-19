"""Global cleanup retains owners and drains consumers before their dependencies."""

import asyncio
from types import SimpleNamespace

import pytest

from stateset_agents.core import async_pool
from stateset_agents.core.async_pool import AsyncResourceFactory


@pytest.fixture(autouse=True)
def isolated_globals(monkeypatch):
    monkeypatch.setattr(async_pool, "_http_pool", None)
    monkeypatch.setattr(async_pool, "_task_manager", None)
    monkeypatch.setattr(async_pool, "_global_cleanup_task", None, raising=False)
    if hasattr(async_pool, "GlobalResourceState"):
        monkeypatch.setattr(
            async_pool, "_global_resource_state", async_pool.GlobalResourceState.READY
        )


@pytest.mark.asyncio
async def test_global_cleanup_retains_owners_and_orders_dependency_cleanup(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def shutdown():
        entered.set()
        await release.wait()
        calls.append("tasks stopped")

    async def close():
        calls.append("pool closed")

    manager = SimpleNamespace(
        shutdown=shutdown, close_admission=lambda: None, _operations=set()
    )
    pool = SimpleNamespace(close=close, cleanup_failures=(), factory=object())
    monkeypatch.setattr(async_pool, "_task_manager", manager)
    monkeypatch.setattr(async_pool, "_http_pool", pool)
    first = asyncio.create_task(async_pool.cleanup_global_resources())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert async_pool._task_manager is manager
        assert async_pool._http_pool is pool
        assert not calls
        second = asyncio.create_task(async_pool.cleanup_global_resources())
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await async_pool.cleanup_global_resources()
        assert calls == ["tasks stopped", "pool closed"]
        assert async_pool._task_manager is None
        assert async_pool._http_pool is None
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_failed_task_drain_preserves_both_owners_and_does_not_replay(monkeypatch):
    calls = []
    error = OSError("cannot confirm worker shutdown")

    async def shutdown():
        calls.append("tasks")
        raise error

    async def close():
        calls.append("pool")

    manager = SimpleNamespace(
        shutdown=shutdown, close_admission=lambda: None, _operations=set()
    )
    pool = SimpleNamespace(close=close, cleanup_failures=(), factory=object())
    monkeypatch.setattr(async_pool, "_task_manager", manager)
    monkeypatch.setattr(async_pool, "_http_pool", pool)
    for _ in range(2):
        with pytest.raises(OSError) as caught:
            await async_pool.cleanup_global_resources()
        assert caught.value is error
    assert calls == ["tasks"]
    assert async_pool._task_manager is manager
    assert async_pool._http_pool is pool


class Factory(AsyncResourceFactory):
    def __init__(self):
        self.created = []
        self.cleaned = []

    async def create_resource(self, **kwargs):
        resource = SimpleNamespace(closed=False)
        self.created.append(resource)
        return resource

    async def cleanup_resource(self, resource):
        self.cleaned.append(resource)
        resource.closed = True


def install_local_http(monkeypatch, factory):
    monkeypatch.setattr(
        async_pool, "_SharedHTTPSessionFactory", lambda **kwargs: factory
    )
    monkeypatch.setattr(
        async_pool,
        "_require_aiohttp",
        lambda: SimpleNamespace(ClientTimeout=lambda **kwargs: None),
    )


@pytest.mark.asyncio
async def test_draining_tasks_can_finish_using_http_before_pool_is_closed(monkeypatch):
    factory = Factory()
    install_local_http(monkeypatch, factory)
    manager = await async_pool.get_task_manager()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        entered.set()
        await release.wait()
        pool = await async_pool.get_http_pool()
        async with pool.acquire() as resource:
            assert not resource.closed
            calls.append("used HTTP")
        return 42

    submitted = asyncio.create_task(manager.submit_task(operation()))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        closing = asyncio.create_task(async_pool.cleanup_global_resources())
        await asyncio.sleep(0)
        assert async_pool.get_global_resource_status()["state"] == "draining_tasks"
        with pytest.raises(RuntimeError, match="shutting down"):
            await async_pool.get_task_manager()
        release.set()
        assert await submitted == 42
        await closing
        assert calls == ["used HTTP"]
        assert factory.cleaned == factory.created
        assert all(resource.closed for resource in factory.created)
        assert async_pool.get_global_resource_status()["state"] == "closed"
        replacement = await async_pool.get_task_manager()
        assert replacement is not manager
        assert async_pool.get_global_resource_status()["state"] == "ready"
        await async_pool.cleanup_global_resources()
    finally:
        release.set()
        await asyncio.gather(
            submitted, *([closing] if closing else []), return_exceptions=True
        )
        await async_pool.cleanup_global_resources()


@pytest.mark.asyncio
async def test_cancelled_waiter_cannot_detach_http_cleanup_or_allow_new_getters(
    monkeypatch,
):
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingFactory(Factory):
        async def cleanup_resource(self, resource):
            entered.set()
            await release.wait()
            await super().cleanup_resource(resource)

    factory = BlockingFactory()
    install_local_http(monkeypatch, factory)
    pool = await async_pool.get_http_pool()
    first = asyncio.create_task(async_pool.cleanup_global_resources())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert async_pool._http_pool is pool
        assert async_pool.get_global_resource_status()["state"] == "draining_http"
        with pytest.raises(RuntimeError, match="shutting down"):
            await async_pool.get_http_pool()
        with pytest.raises(RuntimeError, match="shutting down"):
            await async_pool.get_task_manager()
        second = asyncio.create_task(async_pool.cleanup_global_resources())
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        assert not first.done()
        assert not second.done()
        assert not factory.cleaned
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert factory.cleaned == factory.created
        assert async_pool._http_pool is None
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("global_cleanup", [False, True])
async def test_failed_pool_resource_cleanup_retains_handle_without_retry(
    monkeypatch, global_cleanup
):
    class FailingFactory(Factory):
        async def cleanup_resource(self, resource):
            self.cleaned.append(resource)
            raise OSError("close unconfirmed")

    factory = FailingFactory()
    install_local_http(monkeypatch, factory)
    pool = await async_pool.get_http_pool()
    if global_cleanup:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            await async_pool.cleanup_global_resources()
        assert async_pool.get_global_resource_status()["state"] == "failed"
    else:
        await pool.close()
    assert async_pool._http_pool is pool
    assert len(pool.cleanup_failures) == 2
    assert [handle.resource for handle in pool.cleanup_failures] == factory.created
    assert all(
        handle.info.metadata["cleanup_error_type"] == "OSError"
        for handle in pool.cleanup_failures
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="retained"):
            await async_pool.get_http_pool()
        with pytest.raises(RuntimeError, match="unconfirmed"):
            await async_pool.cleanup_global_resources()
    assert factory.cleaned == factory.created


@pytest.mark.asyncio
async def test_http_getter_in_progress_cannot_return_pool_closed_by_global_cleanup(
    monkeypatch,
):
    entered, release = asyncio.Event(), asyncio.Event()

    class InitializingFactory(Factory):
        async def create_resource(self, **kwargs):
            entered.set()
            await release.wait()
            return await super().create_resource(**kwargs)

    factory = InitializingFactory()
    install_local_http(monkeypatch, factory)
    getter = asyncio.create_task(async_pool.get_http_pool())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        owner = async_pool._http_pool
        closing = asyncio.create_task(async_pool.cleanup_global_resources())
        await asyncio.sleep(0)
        assert async_pool._http_pool is owner
        release.set()
        with pytest.raises(RuntimeError, match="shutting down|closed during"):
            await getter
        await closing
        assert async_pool._http_pool is None
        assert factory.cleaned == factory.created
    finally:
        release.set()
        await asyncio.gather(
            getter, *([closing] if closing else []), return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_session_creation", [False, True])
async def test_failed_shared_connector_cleanup_retains_factory_handle(
    monkeypatch, fail_session_creation
):
    calls = []

    class Connector:
        closed = False

        def __init__(self, **kwargs):
            calls.append("connector created")

        async def close(self):
            calls.append("connector close")
            raise OSError("connector cleanup unconfirmed")

    class Session:
        closed = False

        def __init__(self, **kwargs):
            if fail_session_creation:
                raise ValueError("session creation failed")

        async def close(self):
            self.closed = True

    fake_aiohttp = SimpleNamespace(
        TCPConnector=Connector,
        ClientSession=Session,
        ClientTimeout=lambda **kwargs: None,
    )
    monkeypatch.setattr(async_pool, "_require_aiohttp", lambda: fake_aiohttp)
    if fail_session_creation:
        with pytest.raises(async_pool.ResourceException):
            await async_pool.get_http_pool()
        pool = async_pool._http_pool
    else:
        pool = await async_pool.get_http_pool()
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await async_pool.cleanup_global_resources()
    assert async_pool._http_pool is pool
    assert pool.factory._connector is not None
    assert pool.factory.cleanup_pending
    assert async_pool.get_global_resource_status()["http_cleanup_pending"] is True
    with pytest.raises(RuntimeError, match="failed"):
        await async_pool.get_http_pool()
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await async_pool.cleanup_global_resources()
    assert calls == ["connector created", "connector close"]


@pytest.mark.asyncio
async def test_managed_operation_cannot_deadlock_by_cleaning_up_its_own_globals():
    manager = await async_pool.get_task_manager()
    try:
        with pytest.raises(RuntimeError, match="outside managed tasks"):
            await manager.submit_task(async_pool.cleanup_global_resources())
        assert async_pool.get_global_resource_status()["state"] == "ready"
    finally:
        await async_pool.cleanup_global_resources()


@pytest.mark.asyncio
async def test_default_factory_reports_close_failure_to_pool_owner(monkeypatch):
    calls = []

    class Resource:
        async def close(self):
            calls.append("close")
            raise OSError("resource close failed")

    class DefaultCleanupFactory(AsyncResourceFactory):
        async def create_resource(self, **kwargs):
            return Resource()

    install_local_http(monkeypatch, DefaultCleanupFactory())
    pool = await async_pool.get_http_pool()
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await async_pool.cleanup_global_resources()
    assert async_pool._http_pool is pool
    assert len(pool.cleanup_failures) == 2
    assert calls == ["close", "close"]
