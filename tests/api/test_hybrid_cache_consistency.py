"""Hybrid entry ordering, fallback expiry, and mutation acknowledgement contracts."""

import asyncio
import fnmatch
from types import SimpleNamespace

import pytest

from stateset_agents.api import distributed_cache as module
from stateset_agents.api.resource_lifecycle import ResourceOwnershipError
from tests.api.cache_doubles import ReadPipeline


class Client:
    def __init__(self):
        self.values = {}
        self.hooks = {}
        self.calls = []
        self.available = True
        self.ttl_ms = 300_000

    def pipeline(self, *, transaction):
        assert transaction is True
        return ReadPipeline(self)

    async def get(self, key):
        self.calls.append(("get", key))
        if "get" in self.hooks:
            return await self.hooks["get"](key)
        return self.values.get(key)

    async def setex(self, key, ttl, data):
        self.calls.append(("setex", key))
        if "setex" in self.hooks:
            return await self.hooks["setex"](key, ttl, data)
        self.values[key] = data
        return True

    async def delete(self, *keys):
        self.calls.append(("delete", keys))
        return sum(self.values.pop(key, None) is not None for key in keys)

    async def scan(self, cursor, *, match, count):
        self.calls.append(("scan", match))
        return 0, [key for key in self.values if fnmatch.fnmatchcase(key, match)]

    async def exists(self, key):
        return int(key in self.values)

    async def aclose(self):
        pass


@pytest.fixture
def cache(monkeypatch):
    owner = module.HybridCache(module.CacheConfig(redis_url="redis://unused.invalid"))
    client = Client()
    owner._redis._client = client
    owner._redis._connected = True

    async def health():
        return client.available

    monkeypatch.setattr(owner._redis, "health_check", health)
    return owner, client


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=100.0, wall=1000.0)
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(monotonic=lambda: value.now, time=lambda: value.wall),
    )
    return value


async def wait(event):
    await asyncio.wait_for(event.wait(), 2)


async def queued(owner, count=1):
    async def observed():
        while owner.get_status()["entry_ordering"]["waiting_operations"] != count:
            await asyncio.sleep(0)

    await asyncio.wait_for(observed(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["set", "delete", "delete_pattern", "clear"])
async def test_late_read_cannot_overwrite_a_later_mutation(cache, mutation):
    owner, client = cache
    client.values["stateset:key"] = b'"old"'
    entered, release = asyncio.Event(), asyncio.Event()

    async def read(key):
        captured = client.values.get(key)
        entered.set()
        await release.wait()
        return captured

    client.hooks["get"] = read
    reading = asyncio.create_task(owner.get("key"))
    await wait(entered)
    calls = {
        "set": lambda: owner.set("key", "new"),
        "delete": lambda: owner.delete("key"),
        "delete_pattern": lambda: owner.delete_pattern("k*"),
        "clear": owner.clear,
    }
    writing = asyncio.create_task(calls[mutation]())
    await queued(owner)
    assert client.calls == [("get", "stateset:key")]
    release.set()
    assert await reading == "old"
    await writing
    client.available = False
    assert await owner.get("key") == ("new" if mutation == "set" else None)
    assert owner.get_status()["last_mutation"]["redis_acknowledged"] is True
    assert owner.get_status()["entry_ordering"]["waiting_operations"] == 0
    await owner.close()


@pytest.mark.asyncio
async def test_ordered_writes_capture_inputs_before_waiting_and_detach_read_results(
    cache,
):
    owner, client = cache
    entered, release = asyncio.Event(), asyncio.Event()

    async def write(key, ttl, data):
        if not entered.is_set():
            entered.set()
            await release.wait()
        client.values[key] = data
        return True

    client.hooks["setex"] = write
    first = asyncio.create_task(owner.set("key", {"value": [1]}))
    await wait(entered)
    value = {"value": [2]}
    second = asyncio.create_task(owner.set("key", value))
    await queued(owner)
    value["value"].append(3)
    reading = asyncio.create_task(owner.get("key"))
    await queued(owner, 2)
    release.set()
    assert await first and await second
    result = await reading
    assert result == {"value": [2]}
    result["value"].append(4)
    client.available = False
    fallback = await owner.get("key")
    assert fallback == {"value": [2]}
    fallback["value"].append(5)
    assert await owner.get("key") == {"value": [2]}
    await owner.close()


@pytest.mark.asyncio
async def test_unrelated_keys_progress_but_cannot_bypass_a_pending_barrier(cache):
    owner, client = cache
    entered, release = asyncio.Event(), asyncio.Event()

    async def read(key):
        if key == "stateset:held":
            entered.set()
            await release.wait()
        return b'"value"'

    client.hooks["get"] = read
    first = asyncio.create_task(owner.get("held"))
    await wait(entered)
    assert await asyncio.wait_for(owner.get("independent"), 2) == "value"
    barrier = asyncio.create_task(owner.clear())
    await queued(owner)
    later = asyncio.create_task(owner.set("later", "retained"))
    await queued(owner, 2)
    assert not later.done()
    release.set()
    await first
    await barrier
    assert await later
    assert client.calls.index(("scan", "stateset:*")) < client.calls.index(
        ("setex", "stateset:later")
    )
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_waiter_removes_its_barrier_without_dispatch(cache):
    owner, client = cache
    entered, release = asyncio.Event(), asyncio.Event()

    async def read(key):
        if key == "stateset:held":
            entered.set()
            await release.wait()
        return b'"value"'

    client.hooks["get"] = read
    first = asyncio.create_task(owner.get("held"))
    await wait(entered)
    barrier = asyncio.create_task(owner.clear())
    await queued(owner)
    later = asyncio.create_task(owner.set("other", "value"))
    await queued(owner, 2)
    barrier.cancel()
    with pytest.raises(asyncio.CancelledError):
        await barrier
    assert await asyncio.wait_for(later, 2)
    assert not any(name == "scan" for name, _ in client.calls)
    release.set()
    await first
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_write_retains_entry_until_sdk_and_invalidation_finish(cache):
    owner, client = cache
    entered, unwinding, release = (asyncio.Event() for _ in range(3))

    async def write(key, ttl, data):
        client.values[key] = data  # Dispatch can commit before cancellation.
        entered.set()
        try:
            await asyncio.Future()
        finally:
            unwinding.set()
            await release.wait()

    client.hooks["setex"] = write
    writing = asyncio.create_task(owner.set("key", "uncertain"))
    await wait(entered)
    reading = asyncio.create_task(owner.get("key"))
    await queued(owner)
    writing.cancel()
    await wait(unwinding)
    closing = asyncio.create_task(owner.close())
    for _ in range(3):
        writing.cancel()
        await asyncio.sleep(0)
        assert not writing.done() and not reading.done() and not closing.done()
    client.available = False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await writing
    assert await reading is None
    await closing
    assert client.calls.count(("setex", "stateset:key")) == 1
    status = owner.get_status()
    assert status["unconfirmed_mutations"] == 1
    assert status["last_mutation"]["error_type"] == "CancelledError"
    assert not status["last_mutation"]["redis_acknowledged"]
    assert status["active_operations"] == 0


@pytest.mark.asyncio
async def test_inherited_entry_callbacks_reject_reentry_before_waiting(cache):
    owner, client = cache

    async def read(key):
        for operation in (
            owner.get("same"),
            owner.set("other", "value"),
            owner.clear(),
        ):
            with pytest.raises(ResourceOwnershipError):
                await asyncio.create_task(operation)
        return b'"value"'

    client.hooks["get"] = read
    assert await owner.get("key") == "value"
    assert client.calls == [("get", "stateset:key")]
    assert owner.get_status()["entry_ordering"]["waiting_operations"] == 0
    await owner.close()


@pytest.mark.asyncio
async def test_refill_deducts_read_latency_and_keeps_remote_expiry(cache, clock):
    owner, client = cache
    client.ttl_ms = 500

    async def read(key):
        clock.now += 0.2
        return b'"short-lived"'

    client.hooks["get"] = read
    assert await owner.get("key") == "short-lived"
    client.available = False
    clock.now = 100.49
    assert await owner.get("key") == "short-lived"
    clock.now = 100.5
    assert await owner.get("key") is None
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, -2, ValueError("TTL denied"), "invalid", True])
async def test_absent_expired_or_unknown_ttl_cannot_seed_fallback(cache, clock, ttl):
    owner, client = cache
    await owner._memory.set("key", "stale")
    client.values["stateset:key"] = b'"remote"'
    client.ttl_ms = ttl
    assert await owner.get("key") == (None if ttl == -2 else "remote")
    client.available = False
    assert await owner.get("key") is None
    await owner.close()


@pytest.mark.asyncio
async def test_persistent_remote_entry_gets_a_bounded_fallback_ttl(cache, clock):
    owner, client = cache
    client.values["stateset:key"] = b'"persistent"'
    client.ttl_ms = -1
    assert await owner.get("key") == "persistent"
    client.available = False
    clock.now += owner.config.default_ttl_seconds
    assert await owner.get("key") is None
    await owner.close()


@pytest.mark.asyncio
async def test_expiry_during_memory_publication_does_not_restart_ttl(
    cache, clock, monkeypatch
):
    owner, client = cache
    client.values["stateset:key"] = b'"expired during refill"'
    client.ttl_ms = 500
    entered, release = asyncio.Event(), asyncio.Event()
    original = owner._memory._set_until

    async def refill(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(owner._memory, "_set_until", refill)
    reading = asyncio.create_task(owner.get("key"))
    await wait(entered)
    clock.now += 1
    release.set()
    assert await reading == "expired during refill"
    client.available = False
    assert await owner.get("key") is None
    await owner.close()


@pytest.mark.asyncio
async def test_local_expiry_is_immune_to_wall_clock_jumps(clock):
    memory = module.MemoryCache(module.CacheConfig())
    assert await memory.set("key", "value", ttl=1)
    clock.wall += 100_000
    assert await memory.get("key") == "value"
    clock.wall -= 200_000
    clock.now += 1
    assert not await memory.exists("key")
    assert await memory.set("another", "value", ttl=1)
    clock.now += 1
    assert await memory.cleanup_expired() == 1


@pytest.mark.asyncio
async def test_redis_recovery_is_authoritative_without_replaying_local_mutations(cache):
    owner, client = cache
    client.values["stateset:key"] = b'"primary"'
    client.available = False
    assert await owner.set("key", "local-only")
    assert await owner.get("key") == "local-only"
    status = owner.get_status()
    assert status["last_mutation"]["memory_acknowledged"] is True
    assert status["last_mutation"]["redis_acknowledged"] is False
    assert status["unconfirmed_mutations"] == 1
    client.available = True
    assert await owner.get("key") == "primary"
    assert client.calls == [("get", "stateset:key")]
    await owner.close()


@pytest.mark.asyncio
async def test_zero_deletions_still_acknowledge_both_layers(cache):
    owner, _ = cache
    assert not await owner.delete("missing-secret-key")
    status = owner.get_status()
    assert status["last_mutation"]["memory_acknowledged"] is True
    assert status["last_mutation"]["redis_acknowledged"] is True
    assert status["unconfirmed_mutations"] == 0
    assert "missing-secret-key" not in str(status)
    status["last_mutation"]["operation"] = "changed"
    assert owner.get_status()["last_mutation"]["operation"] == "delete"
    await owner.close()


@pytest.mark.asyncio
async def test_negative_redis_write_ack_is_not_reported_as_replication(cache):
    owner, client = cache

    async def declined(*args):
        return False

    client.hooks["setex"] = declined
    assert not await owner._redis.set("direct", "value")
    assert await owner.set("key", "local")
    assert owner.get_status()["last_mutation"]["redis_acknowledged"] is False
    await owner.close()


@pytest.mark.asyncio
async def test_failed_cancellation_invalidation_disables_fallback_until_clear(
    cache, monkeypatch
):
    owner, client = cache

    async def cancelled(*args):
        raise asyncio.CancelledError

    async def invalidation_failed(*args):
        raise ValueError("private details")

    client.hooks["setex"] = cancelled
    monkeypatch.setattr(owner._memory, "delete", invalidation_failed)
    with pytest.raises(asyncio.CancelledError):
        await owner.set("key", "uncertain")
    client.available = False
    assert await owner.get("key") is None and not await owner.exists("key")
    assert not await owner.health_check()
    status = owner.get_status()
    assert not status["fallback_safe"]
    assert "private details" not in str(status)
    assert status["last_mutation"]["invalidation_error_type"] == "ValueError"
    await owner.clear()
    assert owner.get_status()["fallback_safe"] and await owner.health_check()
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["delete", "delete_pattern", "clear"])
async def test_failed_remote_invalidation_is_distinct_from_an_acknowledged_miss(
    cache, monkeypatch, mutation
):
    owner, client = cache
    assert await owner.set("key", "value")

    async def fail(*args, **kwargs):
        raise OSError("private server details")

    monkeypatch.setattr(client, "delete" if mutation == "delete" else "scan", fail)
    if mutation == "delete":
        assert await owner.delete("key")
    elif mutation == "delete_pattern":
        assert await owner.delete_pattern("k*") == 1
    else:
        assert await owner.clear() == 1
    status = owner.get_status()
    assert status["last_mutation"]["completed"]
    assert status["last_mutation"]["memory_acknowledged"] is True
    assert status["last_mutation"]["redis_acknowledged"] is False
    assert status["unconfirmed_mutations"] == 1
    assert "private server details" not in str(status)
    client.available = False
    assert await owner.get("key") is None
    await owner.close()


@pytest.mark.asyncio
async def test_uncopyable_input_rejects_before_any_write_or_entry_admission(cache):
    owner, client = cache

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise TypeError("cannot own this value")

    with pytest.raises(TypeError, match="cannot own"):
        await owner.set("key", Uncopyable())
    assert client.calls == []
    assert await owner._memory.get("key") is None
    assert owner.get_status()["active_operations"] == 0
    assert owner.get_status()["last_mutation"] is None
    await owner.close()
