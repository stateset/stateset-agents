"""Bulkhead capacity belongs to admitted work through cancellation cleanup."""

import asyncio
import contextvars
import inspect
from types import SimpleNamespace

import pytest

from stateset_agents.api import resilience
from stateset_agents.api.resilience import Bulkhead, BulkheadConfig, BulkheadFullError


async def wait_for_waiters(bulkhead, count):
    async def wait():
        while bulkhead.stats["waiting"] != count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.asyncio
async def test_zero_waiting_capacity_still_allows_idle_capacity():
    bulkhead = Bulkhead("no queue", BulkheadConfig(max_concurrent=1, max_waiting=0))

    async def operation():
        return "result"

    assert await bulkhead.execute(operation) == "result"
    assert bulkhead.stats["active"] == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_releases_queue_capacity():
    bulkhead = Bulkhead("queue", BulkheadConfig(max_concurrent=1, max_waiting=1))
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def owner():
        entered.set()
        await release.wait()

    async def queued():
        calls.append("effect")

    active = asyncio.create_task(bulkhead.execute(owner))
    waiting = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        waiting = asyncio.create_task(bulkhead.execute(queued))
        await wait_for_waiters(bulkhead, 1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert bulkhead.stats["waiting"] == 0
        assert bulkhead.stats["active"] == 1
        assert not calls
    finally:
        release.set()
        await asyncio.gather(
            active, *([waiting] if waiting else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_simultaneous_idle_admissions_do_not_consume_queue_capacity():
    bulkhead = Bulkhead("burst", BulkheadConfig(max_concurrent=3, max_waiting=1))
    release = asyncio.Event()
    calls = []

    async def operation():
        calls.append("effect")
        await release.wait()

    tasks = [asyncio.create_task(bulkhead.execute(operation)) for _ in range(3)]
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(calls) == 3
        assert bulkhead.stats["active"] == 3
        assert bulkhead.stats["waiting"] == 0
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_release_active_cleanup_early():
    bulkhead = Bulkhead("cleanup", BulkheadConfig(max_concurrent=1, max_waiting=1))
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def operation():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")

    async def queued():
        calls.append("next")

    owner = asyncio.create_task(bulkhead.execute(operation))
    waiter = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        owner.cancel()
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        waiter = asyncio.create_task(bulkhead.execute(queued))
        await wait_for_waiters(bulkhead, 1)
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        await asyncio.sleep(0)
        assert not owner.done()
        assert bulkhead.stats["active"] == 1
        assert not calls
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        await waiter
        assert calls == ["cleaned", "next"]
        assert bulkhead.stats["active"] == bulkhead.stats["waiting"] == 0
    finally:
        release.set()
        await asyncio.gather(
            owner, *([waiter] if waiter else []), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_cancellation_after_permit_handoff_returns_capacity_without_an_effect():
    bulkhead = Bulkhead("handoff", BulkheadConfig(max_concurrent=1, max_waiting=2))
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation(name):
        calls.append(name)
        if name == "owner":
            entered.set()
            await release.wait()
        return name

    owner = asyncio.create_task(bulkhead.execute(operation, "owner"))
    queued = []
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        head = asyncio.create_task(bulkhead.execute(operation, "cancelled head"))
        queued.append(head)
        await wait_for_waiters(bulkhead, 1)
        tail = asyncio.create_task(bulkhead.execute(operation, "tail"))
        queued.append(tail)
        await wait_for_waiters(bulkhead, 2)
        # The owner's return hands capacity to the head before waking its caller.
        owner.add_done_callback(lambda _: head.cancel())
        release.set()
        assert await owner == "owner"
        with pytest.raises(asyncio.CancelledError):
            await head
        assert await asyncio.wait_for(tail, timeout=2) == "tail"
        assert calls == ["owner", "tail"]
        assert bulkhead.stats["active"] == bulkhead.stats["waiting"] == 0
        assert bulkhead.stats["available"] == 1
    finally:
        release.set()
        for task in queued:
            task.cancel()
        await asyncio.gather(owner, *queued, return_exceptions=True)


@pytest.mark.asyncio
async def test_fifo_queue_skips_cancelled_waiters_and_rejects_excess_work():
    bulkhead = Bulkhead("fifo", BulkheadConfig(max_concurrent=1, max_waiting=3))
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation(name):
        calls.append(name)
        if name == "owner":
            entered.set()
            await release.wait()
        assert bulkhead.stats["active"] == 1
        assert bulkhead.stats["available"] == 0
        return name

    owner = asyncio.create_task(bulkhead.execute(operation, "owner"))
    queued = []
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        for index, name in enumerate(["first", "cancelled middle", "last"], 1):
            queued.append(asyncio.create_task(bulkhead.execute(operation, name)))
            await wait_for_waiters(bulkhead, index)
        with pytest.raises(BulkheadFullError, match="full"):
            await bulkhead.execute(operation, "rejected")
        queued[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued[1]
        await wait_for_waiters(bulkhead, 2)
        release.set()
        await asyncio.gather(owner, queued[0], queued[2])
        assert calls == ["owner", "first", "last"]
        assert bulkhead.stats["available"] == 1
        assert bulkhead.stats["active"] == bulkhead.stats["waiting"] == 0
    finally:
        release.set()
        await asyncio.gather(owner, *queued, return_exceptions=True)


@pytest.mark.asyncio
async def test_handoff_skips_expired_waiters_even_before_their_timer_runs(monkeypatch):
    now = 100.0
    monkeypatch.setattr(resilience, "time", SimpleNamespace(monotonic=lambda: now))
    bulkhead = Bulkhead(
        "expired", BulkheadConfig(max_concurrent=1, max_waiting=2, timeout_seconds=10)
    )
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation(name):
        calls.append(name)
        if name == "owner":
            entered.set()
            await release.wait()
        return name

    owner = asyncio.create_task(bulkhead.execute(operation, "owner"))
    queued = []
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        expired = asyncio.create_task(bulkhead.execute(operation, "expired"))
        queued.append(expired)
        await wait_for_waiters(bulkhead, 1)
        now += 5
        live = asyncio.create_task(bulkhead.execute(operation, "live"))
        queued.append(live)
        await wait_for_waiters(bulkhead, 2)
        now += 5
        release.set()
        await owner
        with pytest.raises(BulkheadFullError, match="Timeout waiting"):
            await expired
        assert await live == "live"
        assert calls == ["owner", "live"]
        assert bulkhead.stats["active"] == bulkhead.stats["waiting"] == 0
    finally:
        release.set()
        await asyncio.gather(owner, *queued, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [0, 0.01])
async def test_queue_timeout_retires_only_the_waiter(seconds):
    bulkhead = Bulkhead(
        "timeout",
        BulkheadConfig(max_concurrent=1, max_waiting=1, timeout_seconds=seconds),
    )
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation(name):
        calls.append(name)
        if name == "owner":
            entered.set()
            await release.wait()
        return name

    owner = asyncio.create_task(bulkhead.execute(operation, "owner"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(BulkheadFullError, match="Timeout waiting"):
            await bulkhead.execute(operation, "expired")
        assert bulkhead.stats["active"] == 1
        assert bulkhead.stats["waiting"] == 0
        assert calls == ["owner"]
        release.set()
        await owner
        assert await bulkhead.execute(operation, "later") == "later"
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_caller_cancellation_wins_after_cleanup_returns_or_raises(cleanup_fails):
    bulkhead = Bulkhead("cancellation", BulkheadConfig(max_concurrent=1))
    entered = asyncio.Event()
    calls = []

    async def operation():
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            calls.append("cleaned")
            if cleanup_fails:
                raise ValueError("cleanup failed") from None
            return "late result"

    owner = asyncio.create_task(bulkhead.execute(operation))
    await asyncio.wait_for(entered.wait(), timeout=2)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert calls == ["cleaned"]
    assert bulkhead.stats["available"] == 1


@pytest.mark.asyncio
async def test_configuration_changes_do_not_resize_an_existing_bulkhead():
    config = BulkheadConfig(max_concurrent=1, max_waiting=0)
    bulkhead = Bulkhead("fixed capacity", config)
    config.max_concurrent = 10
    config.max_waiting = 10
    bulkhead.config.max_concurrent = 20
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()

    owner = asyncio.create_task(bulkhead.execute(operation))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert bulkhead.stats["max_concurrent"] == 1
        assert bulkhead.stats["available"] == 0
        with pytest.raises(BulkheadFullError):
            await bulkhead.execute(operation)
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.asyncio
async def test_decorator_retains_metadata_and_inherits_request_context():
    context = contextvars.ContextVar("bulkhead_request", default="unset")
    bulkhead = Bulkhead("metadata")

    async def operation(value: str) -> str:
        """One request."""
        return f"{context.get()}:{value}"

    wrapped = bulkhead(operation)
    token = context.set("request-1")
    try:
        assert await wrapped("value") == "request-1:value"
        assert wrapped.__name__ == operation.__name__
        assert wrapped.__doc__ == operation.__doc__
        assert inspect.signature(wrapped) == inspect.signature(operation)
        assert inspect.iscoroutinefunction(wrapped)
    finally:
        context.reset(token)


@pytest.mark.asyncio
async def test_upstream_failure_retains_identity_and_releases_capacity():
    bulkhead = Bulkhead("failure", BulkheadConfig(max_concurrent=1, max_waiting=0))
    failure = ValueError("upstream failed")

    async def operation():
        raise failure

    with pytest.raises(ValueError) as caught:
        await bulkhead.execute(operation)
    assert caught.value is failure
    assert bulkhead.stats["available"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"max_concurrent": 0},
        {"max_concurrent": -1},
        {"max_concurrent": True},
        {"max_waiting": -1},
        {"max_waiting": 1.5},
        {"timeout_seconds": -1},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_limits_are_rejected_before_work(options):
    with pytest.raises(ValueError):
        BulkheadConfig(**options)
