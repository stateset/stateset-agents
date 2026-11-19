"""Circuit admission and completion races with explicit event barriers."""

import asyncio

import pytest

from stateset_agents.api import resilience
from stateset_agents.api.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)


class Clock:
    def __init__(self):
        self.wall = 1_000.0
        self.ticks = 100.0

    def time(self):
        return self.wall

    def monotonic(self):
        return self.ticks

    def advance(self, seconds):
        self.wall += seconds
        self.ticks += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    # Replace only this module's clock, not asyncio's scheduling clock.
    monkeypatch.setattr(resilience, "time", clock)
    return clock


def make_breaker(**kwargs):
    return CircuitBreaker(
        "lifecycle",
        CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2, timeout_seconds=5, **kwargs
        ),
    )


async def success():
    return "ok"


async def fail():
    raise OSError("upstream unavailable")


async def open_breaker(breaker):
    with pytest.raises(OSError, match="upstream unavailable"):
        await breaker.call(fail)
    assert breaker.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_half_open_admits_one_probe_and_requires_sequential_successes(clock):
    breaker = make_breaker()
    await open_breaker(breaker)
    clock.advance(5)
    entered, release = asyncio.Event(), asyncio.Event()

    async def probe():
        entered.set()
        await release.wait()
        return "probe result"

    task = asyncio.create_task(breaker.call(probe))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert breaker.state is CircuitState.HALF_OPEN
        assert breaker.probe_in_flight
        for _ in range(3):
            with pytest.raises(CircuitOpenError, match="HALF_OPEN"):
                await breaker.call(success)
        assert breaker.stats.rejected_requests == 3
        release.set()
        assert await task == "probe result"
        assert not breaker.probe_in_flight
        assert breaker.state is CircuitState.HALF_OPEN
        assert breaker.stats.success_count == 1
        assert await breaker.call(success) == "ok"
        assert breaker.state is CircuitState.CLOSED
        assert breaker.stats.total_requests == 6
    finally:
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("phase", ["open", "half_open", "recovered", "reset"])
async def test_late_closed_call_cannot_change_a_new_generation(clock, outcome, phase):
    breaker = make_breaker()
    entered, release = asyncio.Event(), asyncio.Event()

    async def late_call():
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise OSError("late failure")
        return "late result"

    task = asyncio.create_task(breaker.call(late_call))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await open_breaker(breaker)
        if phase in {"half_open", "recovered"}:
            clock.advance(5)
            await breaker.call(success)
        if phase == "recovered":
            await breaker.call(success)
        if phase == "reset":
            breaker.reset()
        snapshot = breaker.stats.to_dict()
        clock.advance(1)
        release.set()
        if outcome == "failure":
            with pytest.raises(OSError, match="late failure"):
                await task
        else:
            assert await task == "late result"
        assert breaker.stats.to_dict() == snapshot
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_fallback_does_not_block_recovery_admission(clock):
    entered, release = asyncio.Event(), asyncio.Event()

    async def fallback(value):
        entered.set()
        await release.wait()
        return value

    breaker = make_breaker(fallback=fallback)
    await open_breaker(breaker)
    task = asyncio.create_task(breaker.call(success, "cached result"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        clock.advance(5)
        assert await asyncio.wait_for(breaker.call(success), timeout=0.2) == "ok"
        assert breaker.state is CircuitState.HALF_OPEN
        release.set()
        assert await task == "cached result"
        assert breaker.stats.success_count == 1
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_half_open_rejection_uses_fallback_without_counting_recovery(clock):
    breaker = make_breaker(fallback=lambda value: f"cached {value}")
    await open_breaker(breaker)
    clock.advance(5)
    entered, release = asyncio.Event(), asyncio.Event()

    async def probe():
        entered.set()
        await release.wait()
        return "live result"

    task = asyncio.create_task(breaker.call(probe))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert await breaker.call(success, "value") == "cached value"
        assert breaker.stats.success_count == 0
        assert breaker.stats.rejected_requests == 1
        release.set()
        assert await task == "live result"
        assert breaker.stats.success_count == 1
    finally:
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("wall_jump", [-1_000, 1_000])
async def test_recovery_cooldown_uses_elapsed_time_not_wall_time(clock, wall_jump):
    breaker = make_breaker()
    await open_breaker(breaker)
    failure_time = breaker.stats.last_failure_time
    clock.wall += wall_jump
    clock.advance(4)
    with pytest.raises(CircuitOpenError):
        await breaker.call(success)
    clock.advance(1)
    assert await breaker.call(success) == "ok"
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.stats.last_failure_time == failure_time
    assert breaker.stats.last_state_change == clock.wall


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancelled", "excluded", "unexpected"])
async def test_inconclusive_probe_releases_admission_without_recovery_evidence(
    clock, outcome
):
    breaker = make_breaker(excluded_exceptions=(ValueError,))
    await open_breaker(breaker)
    clock.advance(5)
    await breaker.call(success)
    assert breaker.stats.success_count == 1
    entered, release = asyncio.Event(), asyncio.Event()
    error = ValueError("excluded") if outcome == "excluded" else KeyError("unexpected")

    async def probe():
        entered.set()
        await release.wait()
        raise error

    task = asyncio.create_task(breaker.call(probe))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert breaker.probe_in_flight
        if outcome == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        else:
            release.set()
            with pytest.raises(type(error)) as caught:
                await asyncio.wait_for(task, timeout=1)
            assert caught.value is error
        assert not breaker.probe_in_flight
        assert breaker.state is CircuitState.HALF_OPEN
        assert breaker.stats.success_count == 1
        assert breaker.stats.failure_count == 1
        await breaker.call(success)
        assert breaker.state is CircuitState.CLOSED
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
async def test_old_probe_completion_cannot_release_a_new_probe(clock, outcome):
    breaker = make_breaker()
    entered, release, new_entered, new_release = (asyncio.Event() for _ in range(4))

    async def old_probe():
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise OSError("old probe failed")
        return "old result"

    async def new_probe():
        new_entered.set()
        await new_release.wait()
        return "new result"

    await open_breaker(breaker)
    clock.advance(5)
    old_task = asyncio.create_task(breaker.call(old_probe))
    new_task = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        breaker.reset()
        assert not breaker.probe_in_flight
        await open_breaker(breaker)
        clock.advance(5)
        new_task = asyncio.create_task(breaker.call(new_probe))
        await asyncio.wait_for(new_entered.wait(), timeout=1)
        snapshot = breaker.stats.to_dict()
        if outcome == "cancelled":
            old_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await old_task
        else:
            release.set()
            if outcome == "failure":
                with pytest.raises(OSError, match="old probe failed"):
                    await old_task
            else:
                assert await old_task == "old result"
        assert breaker.stats.to_dict() == snapshot
        assert breaker.probe_in_flight
        with pytest.raises(CircuitOpenError, match="HALF_OPEN"):
            await breaker.call(success)
        new_release.set()
        assert await new_task == "new result"
        assert not breaker.probe_in_flight
        assert breaker.stats.success_count == 1
    finally:
        release.set()
        new_release.set()
        await asyncio.gather(
            old_task,
            *([new_task] if new_task is not None else []),
            return_exceptions=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
async def test_repeated_cancellation_cannot_detach_completion_bookkeeping(
    clock, monkeypatch, outcome
):
    breaker = make_breaker()
    await open_breaker(breaker)
    clock.advance(5)
    entered, release, completing = (asyncio.Event() for _ in range(3))
    complete = breaker._complete_attempt

    async def observed_complete(*args):
        completing.set()
        await complete(*args)

    monkeypatch.setattr(breaker, "_complete_attempt", observed_complete)

    async def probe():
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise OSError("probe failed")
        return "ok"

    task = asyncio.create_task(breaker.call(probe))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        async with breaker._lock:
            if outcome == "cancelled":
                task.cancel()
            else:
                release.set()
            await asyncio.wait_for(completing.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert breaker.probe_in_flight
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert not breaker.probe_in_flight
        if outcome == "failure":
            assert breaker.state is CircuitState.OPEN
            assert breaker.stats.failure_count == 2
        else:
            assert breaker.state is CircuitState.HALF_OPEN
            assert breaker.stats.success_count == (1 if outcome == "success" else 0)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_probe_starts_a_fresh_full_cooldown(clock):
    breaker = make_breaker()
    await open_breaker(breaker)
    clock.advance(5)
    await breaker.call(success)
    clock.advance(2)
    await open_breaker(breaker)
    assert breaker.stats.last_failure_time == clock.wall
    assert breaker.stats.success_count == 0
    assert not breaker.probe_in_flight
    clock.advance(4)
    with pytest.raises(CircuitOpenError):
        await breaker.call(success)
    clock.advance(1)
    await breaker.call(success)
    assert breaker.state is CircuitState.HALF_OPEN
    await breaker.call(success)
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_closed_calls_remain_concurrent_and_success_resets_failure_streak(clock):
    breaker = CircuitBreaker(
        "closed", CircuitBreakerConfig(failure_threshold=2, timeout_seconds=5)
    )
    entered, other_entered, release = (asyncio.Event() for _ in range(3))

    async def request(started):
        started.set()
        await release.wait()
        return "ok"

    first = asyncio.create_task(breaker.call(request, entered))
    second = asyncio.create_task(breaker.call(request, other_entered))
    try:
        await asyncio.wait_for(
            asyncio.gather(entered.wait(), other_entered.wait()), timeout=1
        )
        assert not breaker.probe_in_flight
        with pytest.raises(OSError):
            await breaker.call(fail)
        assert breaker.stats.failure_count == 1
        release.set()
        assert await asyncio.gather(first, second) == ["ok", "ok"]
        assert breaker.stats.failure_count == 0
        with pytest.raises(OSError):
            await breaker.call(fail)
        assert breaker.state is CircuitState.CLOSED
        await open_breaker(breaker)
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_rejected_fallback_failure_cannot_complete_the_active_probe(clock):
    error = OSError("fallback failed")

    async def fallback():
        raise error

    breaker = make_breaker(fallback=fallback)
    await open_breaker(breaker)
    clock.advance(5)
    entered, release = asyncio.Event(), asyncio.Event()

    async def probe():
        entered.set()
        await release.wait()

    task = asyncio.create_task(breaker.call(probe))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        with pytest.raises(OSError) as caught:
            await breaker.call(success)
        assert caught.value is error
        assert breaker.probe_in_flight
        assert breaker.state is CircuitState.HALF_OPEN
        assert breaker.stats.failure_count == 1
        assert breaker.stats.success_count == 0
        assert breaker.stats.rejected_requests == 1
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_immediate_failures_open_before_admitting_the_next_ready_call(clock):
    breaker = CircuitBreaker(
        "burst", CircuitBreakerConfig(failure_threshold=3, timeout_seconds=5)
    )
    calls = 0

    async def immediate_failure():
        nonlocal calls
        calls += 1
        raise OSError("immediate failure")

    outcomes = await asyncio.gather(
        *(breaker.call(immediate_failure) for _ in range(20)), return_exceptions=True
    )
    assert calls == 3
    assert sum(isinstance(outcome, OSError) for outcome in outcomes) == 3
    assert sum(isinstance(outcome, CircuitOpenError) for outcome in outcomes) == 17
    assert breaker.stats.failure_count == 3
    assert breaker.stats.total_requests == 20
    assert breaker.stats.rejected_requests == 17
