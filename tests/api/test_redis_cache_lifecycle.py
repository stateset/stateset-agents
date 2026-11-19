"""Redis generations own all callbacks until commands and cleanup settle."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from stateset_agents.api import distributed_cache as module
from stateset_agents.api.resource_lifecycle import ResourceOwnershipError
from tests.api.cache_doubles import ReadPipeline


class Client:
    def __init__(self):
        self.calls = []
        self.behavior = {}
        self.closed = False

    async def call(self, name, *args, **kwargs):
        assert not self.closed, "SDK use after its pool was closed"
        self.calls.append(name)
        if name in self.behavior:
            return await self.behavior[name](*args, **kwargs)
        return {
            "ping": True,
            "get": b'"value"',
            "setex": True,
            "delete": 1,
            "exists": 1,
            "scan": (0, []),
            "aclose": None,
        }[name]

    async def ping(self):
        return await self.call("ping")

    async def get(self, *args):
        return await self.call("get", *args)

    def pipeline(self, *, transaction):
        assert transaction is True
        return ReadPipeline(self)

    async def setex(self, *args):
        return await self.call("setex", *args)

    async def delete(self, *args):
        return await self.call("delete", *args)

    async def exists(self, *args):
        return await self.call("exists", *args)

    async def scan(self, *args, **kwargs):
        return await self.call("scan", *args, **kwargs)

    async def aclose(self):
        await self.call("aclose")
        self.closed = True


@pytest.fixture
def factory(monkeypatch):
    clients = []
    pending = []
    redis = ModuleType("redis")
    redis.__path__ = []
    async_redis = ModuleType("redis.asyncio")

    def construct(*args, **kwargs):
        client = pending.pop(0) if pending else Client()
        clients.append(client)
        return client

    async_redis.from_url = construct
    redis.asyncio = async_redis
    monkeypatch.setitem(sys.modules, "redis", redis)
    monkeypatch.setitem(sys.modules, "redis.asyncio", async_redis)
    return SimpleNamespace(clients=clients, pending=pending)


def cache(*, interval=0):
    return module.RedisCache(
        module.CacheConfig(
            redis_url="redis://unused.invalid", health_check_interval=interval
        )
    )


async def wait(event):
    await asyncio.wait_for(event.wait(), 2)


async def ready(factory):
    owner = cache()
    assert await owner.connect()
    return owner, factory.clients[-1]


@pytest.mark.asyncio
async def test_concurrent_connect_observers_share_one_client_and_one_can_cancel(
    factory,
):
    owner = cache()
    client = Client()
    started, release = asyncio.Event(), asyncio.Event()

    async def ping():
        started.set()
        await release.wait()
        return True

    client.behavior["ping"] = ping
    factory.pending.append(client)
    first = asyncio.create_task(owner.connect())
    await wait(started)
    second = asyncio.create_task(owner.connect())
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not second.done() and not client.closed
    assert owner.get_status()["connection"]["observers"] == 1
    release.set()
    assert await second
    assert factory.clients == [client] and client.calls == ["ping"]
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_interrupted_connection_drains_suppressed_cancellation_before_disposal(
    factory, shutdown
):
    owner = cache()
    client = Client()
    started, cleaning, release = (asyncio.Event() for _ in range(3))

    async def ping():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            return True

    client.behavior["ping"] = ping
    factory.pending.append(client)
    opening = asyncio.create_task(owner.connect())
    await wait(started)
    closing = asyncio.create_task(owner.close()) if shutdown else None
    if not shutdown:
        opening.cancel()
    await wait(cleaning)
    for _ in range(3):
        (closing or opening).cancel()
        await asyncio.sleep(0)
        assert not opening.done() and not client.closed
    if not shutdown:
        assert not await owner.connect()  # Retired setup still owns the slot.
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await opening
    if closing:
        with pytest.raises(asyncio.CancelledError):
            await closing
    assert client.calls == ["ping", "aclose"]
    assert not owner.get_status()["connected"]
    assert owner.get_status()["state"] == ("closed" if shutdown else "unavailable")
    await owner.close()


@pytest.mark.asyncio
async def test_connection_failure_with_failed_cleanup_is_retained_and_redacted(
    factory, caplog
):
    owner = cache()
    client = Client()

    async def ping():
        raise OSError("secret connection detail")

    async def close():
        raise ValueError("secret cleanup detail")

    client.behavior.update(ping=ping, aclose=close)
    factory.pending.append(client)
    with pytest.raises(OSError):
        await owner.connect()
    status = owner.get_status()
    assert status["state"] == "failed"
    assert status["error_type"] == "OSError"
    assert status["cleanup_error_type"] == "ValueError"
    assert "secret" not in str(status) + caplog.text
    with pytest.raises(ResourceOwnershipError):
        await owner.connect()
    for _ in range(2):
        with pytest.raises(ValueError):
            await owner.close()
    assert client.calls == ["ping", "aclose"]
    assert owner._client is client


@pytest.mark.asyncio
async def test_concurrent_health_observers_wait_for_same_probe_instead_of_stale_success(
    factory,
):
    owner, client = await ready(factory)
    owner.config.health_check_interval = 60
    owner._last_health_check = float("-inf")
    started, release = asyncio.Event(), asyncio.Event()

    async def ping():
        started.set()
        await release.wait()
        return True

    client.behavior["ping"] = ping
    first = asyncio.create_task(owner.health_check())
    await wait(started)
    second = asyncio.create_task(owner.health_check())
    await asyncio.sleep(0)
    assert not second.done()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert await second and await owner.health_check()
    assert client.calls == ["ping", "ping"]
    snapshot = owner.get_status()
    snapshot["health"]["observers"] = 100
    assert owner.get_status()["health"]["observers"] == 0
    await owner.close()


@pytest.mark.asyncio
async def test_failed_probe_retires_client_only_after_admitted_read_finishes(factory):
    owner, client = await ready(factory)
    started, release, failed = (asyncio.Event() for _ in range(3))

    async def get(key):
        started.set()
        await release.wait()
        return b'"old response"'

    async def ping():
        failed.set()
        raise OSError("unavailable")

    client.behavior.update(get=get, ping=ping)
    reading = asyncio.create_task(owner.get("key"))
    await wait(started)
    probing = asyncio.create_task(owner.health_check())
    await wait(failed)
    while owner.get_status()["state"] != "retiring":
        await asyncio.sleep(0)
    assert not probing.done() and not client.closed
    assert await owner.get("new") is None
    assert not await owner.connect()
    release.set()
    assert await reading == "old response"
    assert not await probing
    assert client.closed
    assert await owner.connect()
    assert len(factory.clients) == 2 and owner.get_status()["generation"] == 2
    assert await owner.get("key") == "value"
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,sdk,args",
    [
        ("get", "get", ("key",)),
        ("set", "setex", ("key", "value")),
        ("delete", "delete", ("key",)),
        ("exists", "exists", ("key",)),
        ("delete_pattern", "scan", ("*",)),
        ("clear", "scan", ()),
    ],
)
async def test_command_cancellation_retains_pool_through_repeated_cleanup_cancellation(
    factory, method, sdk, args
):
    owner, client = await ready(factory)
    started, cleaning, release = (asyncio.Event() for _ in range(3))

    async def command(*args, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()

    client.behavior[sdk] = command
    request = asyncio.create_task(getattr(owner, method)(*args))
    await wait(started)
    request.cancel()
    await wait(cleaning)
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    for _ in range(3):
        request.cancel()
        closing.cancel()
        await asyncio.sleep(0)
        assert not request.done() and not closing.done() and not client.closed
        assert owner.get_status()["active_operations"] == 1
    assert not await owner.set("blocked", "value")
    assert not await owner.health_check()
    release.set()
    results = await asyncio.gather(request, closing, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert client.calls == ["ping", sdk, "aclose"]
    assert owner.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_close_waits_for_write_without_replaying_or_cancelling_it(factory):
    owner, client = await ready(factory)
    started, release = asyncio.Event(), asyncio.Event()
    writes = []

    async def write(*args):
        started.set()
        await release.wait()
        writes.append(args)
        return True

    client.behavior["setex"] = write
    request = asyncio.create_task(owner.set("key", "value"))
    await wait(started)
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    assert not client.closed and not closing.done()
    release.set()
    assert await request
    await closing
    assert len(writes) == 1 and client.closed
    assert not owner.get_status()["connection"]["cancellation_requested"]
    assert not await owner.set("again", "value")
    with pytest.raises(ResourceOwnershipError):
        await owner.connect()


@pytest.mark.asyncio
async def test_pattern_operation_keeps_one_client_for_all_pages_during_shutdown(
    factory,
):
    owner, client = await ready(factory)
    started, release = asyncio.Event(), asyncio.Event()

    async def scan(cursor, **kwargs):
        if cursor == 0:
            started.set()
            await release.wait()
            return 1, [b"first"]
        return 0, [b"second"]

    client.behavior["scan"] = scan
    deleting = asyncio.create_task(owner.delete_pattern("*"))
    await wait(started)
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    release.set()
    assert await deleting == 2
    await closing
    assert client.calls == ["ping", "scan", "delete", "scan", "delete", "aclose"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["ping", "get", "aclose"])
@pytest.mark.parametrize("eager", [False, True])
async def test_sdk_callback_child_cannot_close_its_own_cache(factory, stage, eager):
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("Eager tasks require Python 3.12+")
    owner = cache()
    client = Client()
    factory.pending.append(client)

    async def callback(*args):
        with pytest.raises(ResourceOwnershipError, match="itself"):
            await asyncio.wait_for(asyncio.create_task(owner.close()), 1)
        return b'"value"' if stage == "get" else True

    client.behavior[stage] = callback
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        assert await owner.connect()
        assert await owner.get("key") == "value"
        await owner.close()
    finally:
        loop.set_task_factory(previous_factory)
        await owner.close()


@pytest.mark.asyncio
async def test_hybrid_shutdown_waits_for_memory_refill_and_rejects_late_calls(
    factory, monkeypatch
):
    owner = module.HybridCache(module.CacheConfig(redis_url="redis://unused.invalid"))
    assert await owner.connect()
    client = factory.clients[0]
    started, release = asyncio.Event(), asyncio.Event()
    original_set = owner._memory._set_until

    async def memory_set(*args):
        started.set()
        await release.wait()
        return await original_set(*args)

    monkeypatch.setattr(owner._memory, "_set_until", memory_set)
    reading = asyncio.create_task(owner.get("key"))
    await wait(started)
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    assert not closing.done() and not client.closed
    assert not await owner.set("late", "ignored")
    assert await owner.get("late") is None
    release.set()
    assert await reading == "value"
    await closing
    assert client.closed
    assert await owner._memory.get("key") == "value"
    assert not await owner.health_check()
    assert owner.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_health_retry_interval_uses_settled_monotonic_observations(
    factory, monkeypatch
):
    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    owner = cache(interval=60)
    client = Client()

    async def unavailable():
        raise OSError("offline")

    client.behavior["ping"] = unavailable
    factory.pending.append(client)
    assert not await owner.health_check()
    assert not await owner.health_check()
    assert len(factory.clients) == 1
    clock[0] += 61
    assert await owner.health_check()
    assert len(factory.clients) == 2
    await owner.close()


@pytest.mark.asyncio
async def test_foreign_event_loop_cannot_borrow_or_close_live_client(factory):
    owner, client = await ready(factory)

    def foreign():
        async def run():
            for operation in (owner.get("key"), owner.close(), owner.connect()):
                with pytest.raises(ResourceOwnershipError, match="event loop"):
                    await operation

        asyncio.run(run())

    await asyncio.to_thread(foreign)
    assert client.calls == ["ping"] and not client.closed
    await owner.close()


@pytest.mark.asyncio
async def test_old_probe_cannot_publish_unavailability_after_a_new_client_connects(
    factory, monkeypatch
):
    owner, client = await ready(factory)
    retired, release = asyncio.Event(), asyncio.Event()
    original = owner._close_client

    async def ping():
        raise OSError("old generation failed")

    async def delayed_retirement():
        await original()
        retired.set()
        await release.wait()

    client.behavior["ping"] = ping
    monkeypatch.setattr(owner, "_close_client", delayed_retirement)
    probing = asyncio.create_task(owner.health_check())
    await wait(retired)
    assert await owner.connect()
    assert owner.get_status()["generation"] == 2
    release.set()
    assert not await probing
    assert owner._health_status and owner.get_status()["state"] == "ready"
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connect", "probe"])
async def test_health_deadline_drains_suppressed_cancellation_without_publishing_success(
    factory, phase
):
    owner = cache()
    client = Client()
    factory.pending.append(client)
    if phase == "probe":
        assert await owner.connect()
    owner.config.redis_connect_timeout_seconds = 0.01
    cleaning, release = asyncio.Event(), asyncio.Event()

    async def ping():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            return True

    client.behavior["ping"] = ping
    operation = owner.connect() if phase == "connect" else owner.health_check()
    task = asyncio.create_task(operation)
    await wait(cleaning)
    assert not task.done() and not client.closed
    release.set()
    assert not await task
    assert client.closed and not owner.get_status()["connected"]
    assert owner.get_status()["error_type"] == "TimeoutError"
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connect", "probe"])
async def test_negative_ping_acknowledgement_never_publishes_health(factory, phase):
    owner = cache()
    client = Client()
    factory.pending.append(client)
    if phase == "probe":
        assert await owner.connect()

    async def ping():
        return False

    client.behavior["ping"] = ping
    assert not await (owner.connect() if phase == "connect" else owner.health_check())
    assert not owner._health_status and client.closed
    assert owner.get_status()["state"] == "unavailable"
    await owner.close()
