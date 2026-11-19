"""State-service cleanup must join maintenance before touching shared cache state."""

import asyncio

import pytest

from stateset_agents.core import enhanced_state_management as state_module
from stateset_agents.core.enhanced_state_management import DistributedStateService


@pytest.mark.asyncio
async def test_shutdown_joins_maintenance_before_final_sync_and_clear(monkeypatch):
    entered, unwinding, release = [asyncio.Event() for _ in range(3)]
    calls = []

    async def maintenance(self):
        try:
            entered.set()
            await asyncio.Future()
        finally:
            unwinding.set()
            await release.wait()
            calls.append("maintenance stopped")

    monkeypatch.setattr(DistributedStateService, "_periodic_sync", maintenance)
    service = DistributedStateService()

    async def final_sync():
        calls.append("sync")
        return {}

    async def clear():
        calls.append("clear")
        return True

    monkeypatch.setattr(service.state_manager, "sync_with_cache", final_sync)
    monkeypatch.setattr(service.cache_backend, "clear", clear)
    first = second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        first = asyncio.create_task(service.shutdown())
        await asyncio.wait_for(unwinding.wait(), timeout=2)
        second = asyncio.create_task(service.shutdown())
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        assert not calls
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await service.shutdown()
        assert calls == ["maintenance stopped", "sync", "clear"]
    finally:
        release.set()
        await asyncio.gather(
            *([first] if first else []),
            *([second] if second else []),
            *service._background_tasks,
            return_exceptions=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "false"])
async def test_failed_final_sync_retains_cache_and_is_not_replayed(
    monkeypatch, failure
):
    monkeypatch.setattr(
        DistributedStateService, "_start_background_tasks", lambda self: None
    )
    service = DistributedStateService()
    await service.state_manager.set("saved", {"value": 1})
    calls = []

    async def sync():
        calls.append("sync")
        if failure == "raise":
            raise OSError("cache unavailable")
        return {"saved": False}

    monkeypatch.setattr(service.state_manager, "sync_with_cache", sync)
    for _ in range(2):
        with pytest.raises((OSError, RuntimeError)):
            await service.shutdown()
    assert calls == ["sync"]
    assert await service.cache_backend.get("saved") == {"value": 1}
    assert "saved" in service.state_manager.dirty_keys


@pytest.mark.asyncio
async def test_closed_service_health_check_does_not_touch_cache(monkeypatch):
    service = DistributedStateService()
    await service.shutdown()

    async def forbidden(*args, **kwargs):
        pytest.fail("A closed service probed the cache")

    monkeypatch.setattr(service.cache_backend, "set", forbidden)
    health = await service.health_check()
    assert health["status"] == "unhealthy"
    assert health["lifecycle_state"] == "closed"


@pytest.mark.asyncio
async def test_global_getter_replaces_only_successfully_closed_service(monkeypatch):
    service = DistributedStateService()
    monkeypatch.setattr(state_module, "_state_service", service)
    await service.shutdown()
    replacement = state_module.get_state_service()
    try:
        assert replacement is not service
    finally:
        await replacement.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_probe", [False, True])
async def test_shutdown_joins_admitted_health_probe_before_cache_clear(
    monkeypatch, cancel_probe
):
    service = DistributedStateService()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    cache_set = service.cache_backend.set

    async def set_value(key, value, ttl=None):
        entered.set()
        await release.wait()
        calls.append("probe wrote")
        return await cache_set(key, value, ttl)

    async def clear():
        calls.append("clear")
        return True

    monkeypatch.setattr(service.cache_backend, "set", set_value)
    monkeypatch.setattr(service.cache_backend, "clear", clear)
    probe = asyncio.create_task(service.health_check())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        closing = asyncio.create_task(service.shutdown())
        await asyncio.sleep(0)
        if cancel_probe:
            probe.cancel()
            await asyncio.sleep(0)
            probe.cancel()
        assert not calls
        assert not closing.done()
        assert service.get_status()["state"] == "closing"
        assert (await service.health_check())["status"] == "unhealthy"
        release.set()
        if cancel_probe:
            with pytest.raises(asyncio.CancelledError):
                await probe
        else:
            assert (await probe)["status"] == "unhealthy"
        await closing
        assert calls == ["probe wrote", "clear"]
        assert service.get_status() == {
            "state": "closed",
            "background_tasks": 0,
            "failed_background_tasks": 0,
            "health_checks": 0,
        }
    finally:
        release.set()
        await asyncio.gather(
            probe, *([closing] if closing else []), return_exceptions=True
        )
        await service.shutdown()


@pytest.mark.asyncio
async def test_dead_maintenance_is_visible_and_blocks_cache_clearing(monkeypatch):
    error = KeyError("maintenance failed")

    async def fail(self):
        raise error

    monkeypatch.setattr(DistributedStateService, "_periodic_sync", fail)
    service = DistributedStateService()
    await asyncio.sleep(0)
    assert service.get_status()["failed_background_tasks"] == 1
    assert (await service.health_check())["status"] == "degraded"
    await service.state_manager.set("saved", 42)
    with pytest.raises(KeyError) as caught:
        await service.shutdown()
    assert caught.value is error
    assert service.get_status()["state"] == "failed"
    assert await service.cache_backend.get("saved") == 42
    monkeypatch.setattr(state_module, "_state_service", service)
    with pytest.raises(RuntimeError, match="failed"):
        state_module.get_state_service()
    with pytest.raises(RuntimeError, match="failed"):
        service._start_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_failed_clear_preserves_owner_and_shared_failure(monkeypatch, raises):
    service = DistributedStateService()
    monkeypatch.setattr(state_module, "_state_service", service)
    calls = []

    async def clear():
        calls.append("clear")
        if raises:
            raise OSError("clear failed")
        return False

    monkeypatch.setattr(service.cache_backend, "clear", clear)
    first_error = None
    for _ in range(2):
        with pytest.raises((OSError, RuntimeError)) as caught:
            await service.shutdown()
        if first_error is None:
            first_error = caught.value
        assert caught.value is first_error
    assert calls == ["clear"]
    assert state_module._state_service is service
    assert service.get_status()["state"] == "failed"


def test_service_constructed_without_loop_starts_once_on_later_global_access(
    monkeypatch,
):
    service = DistributedStateService()
    monkeypatch.setattr(state_module, "_state_service", service)
    assert service.get_status()["state"] == "new"

    async def use():
        assert state_module.get_state_service() is service
        tasks = tuple(service._background_tasks)
        assert len(tasks) == 2
        assert state_module.get_state_service() is service
        assert tuple(service._background_tasks) == tasks
        await service.shutdown()
        assert all(task.done() for task in tasks)

    asyncio.run(use())


@pytest.mark.asyncio
async def test_concurrent_health_probes_are_isolated_from_each_other_and_user_data(
    monkeypatch,
):
    service = DistributedStateService()
    await service.state_manager.set("health_check_test", {"user": "data"})
    entered, release = asyncio.Event(), asyncio.Event()
    keys = []
    get = service.cache_backend.get

    async def delayed_get(key):
        keys.append(key)
        if len(keys) == 2:
            entered.set()
        await release.wait()
        return await get(key)

    monkeypatch.setattr(service.cache_backend, "get", delayed_get)
    probes = [asyncio.create_task(service.health_check()) for _ in range(2)]
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        release.set()
        results = await asyncio.gather(*probes)
        assert all(result["status"] == "healthy" for result in results)
        assert len(set(keys)) == 2
        assert await get("health_check_test") == {"user": "data"}
        for key in keys:
            assert not await service.cache_backend.exists(key)
    finally:
        release.set()
        await asyncio.gather(*probes, return_exceptions=True)
        await service.shutdown()
