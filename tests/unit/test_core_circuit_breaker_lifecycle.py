"""Core circuit-breaker ownership across synchronous and asynchronous calls."""

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from stateset_agents.core import error_handling as errors
from stateset_agents.core.error_handling import (
    CircuitBreaker,
    CircuitBreakerState,
    ErrorHandler,
    GRPOException,
    circuit_breaker,
)


def fail():
    raise ValueError("upstream failed")


def opened_breaker(**kwargs):
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0, **kwargs)
    with pytest.raises(ValueError):
        breaker.call(fail)
    return breaker


@pytest.mark.asyncio
async def test_async_failures_are_counted_after_they_are_awaited():
    breaker = CircuitBreaker(failure_threshold=1)
    calls = []

    async def operation():
        calls.append("effect")
        raise ValueError("async failed")

    with pytest.raises(ValueError, match="async failed"):
        await breaker.call(operation)
    assert breaker.state is CircuitBreakerState.OPEN
    assert breaker.failure_count == 1
    with pytest.raises(GRPOException, match="OPEN"):
        await breaker.call(operation)
    assert calls == ["effect"]


@pytest.mark.asyncio
async def test_async_recovery_does_not_close_on_coroutine_creation():
    breaker = opened_breaker()
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()
        return "recovered"

    task = asyncio.create_task(breaker.call(operation))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert breaker.state is CircuitBreakerState.HALF_OPEN
        assert breaker.probe_in_flight
        release.set()
        assert await task == "recovered"
        assert breaker.state is CircuitBreakerState.CLOSED
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


def test_threads_admit_only_one_recovery_probe():
    breaker = opened_breaker()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def recovery():
        entered.set()
        assert release.wait(timeout=3)
        return "recovered"

    def competitor():
        calls.append("unexpected effect")

    with ThreadPoolExecutor(max_workers=2) as executor:
        recovery_task = executor.submit(breaker.call, recovery)
        try:
            assert entered.wait(timeout=2)
            with pytest.raises(GRPOException, match="probe|OPEN"):
                executor.submit(breaker.call, competitor).result(timeout=2)
            assert not calls
        finally:
            release.set()
        assert recovery_task.result(timeout=2) == "recovered"


def test_old_success_cannot_close_a_new_recovery_probe():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
    old_entered, old_release = threading.Event(), threading.Event()
    probe_entered, probe_release = threading.Event(), threading.Event()

    def old_success():
        old_entered.set()
        assert old_release.wait(timeout=3)
        return "old result"

    def probe():
        probe_entered.set()
        assert probe_release.wait(timeout=3)
        return "probe result"

    with ThreadPoolExecutor(max_workers=2) as executor:
        old = executor.submit(breaker.call, old_success)
        recovery = None
        try:
            assert old_entered.wait(timeout=2)
            with pytest.raises(ValueError):
                breaker.call(fail)
            recovery = executor.submit(breaker.call, probe)
            assert probe_entered.wait(timeout=2)
            old_release.set()
            assert old.result(timeout=2) == "old result"
            assert breaker.state is CircuitBreakerState.HALF_OPEN
            assert breaker.probe_in_flight
        finally:
            old_release.set()
            probe_release.set()
            if recovery is not None:
                assert recovery.result(timeout=2) == "probe result"


@pytest.mark.asyncio
async def test_async_decorator_counts_failures_in_the_registered_breaker():
    handler = ErrorHandler()

    @circuit_breaker(
        "service", errors.CircuitBreakerConfig(failure_threshold=1), handler
    )
    async def operation():
        raise ValueError("decorated failure")

    with pytest.raises(ValueError):
        await operation()
    assert handler.get_circuit_breaker("service").is_open


def test_wall_clock_jumps_do_not_shortcut_recovery_cooldown(monkeypatch):
    wall = 1000.0
    mono = 50.0
    monkeypatch.setattr(
        errors, "time", SimpleNamespace(time=lambda: wall, monotonic=lambda: mono)
    )
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
    with pytest.raises(ValueError):
        breaker.call(fail)
    assert breaker.last_failure_time == 1000
    wall += 100000
    with pytest.raises(GRPOException, match="OPEN"):
        breaker.call(lambda: "too early")
    wall = -1000
    mono += 10
    assert breaker.call(lambda: "recovered") == "recovered"


@pytest.mark.asyncio
async def test_cancelled_probe_keeps_ownership_until_upstream_cleanup_finishes():
    breaker = opened_breaker(expected_exception=BaseException)
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    task = asyncio.create_task(breaker.call(operation))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert breaker.probe_in_flight
        with pytest.raises(GRPOException, match="probe"):
            breaker.call(lambda: "too early")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert breaker.state is CircuitBreakerState.HALF_OPEN
        assert not breaker.probe_in_flight
        assert breaker.failure_count == 1
        assert breaker.call(lambda: "recovered") == "recovered"
        assert breaker.state is CircuitBreakerState.CLOSED
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_excluded_probe_error_releases_admission_without_claiming_recovery():
    breaker = opened_breaker(expected_exception=(ValueError, LookupError))
    failure_time = breaker.last_failure_time
    excluded = TypeError("caller input")

    async def operation():
        raise excluded

    with pytest.raises(TypeError) as caught:
        await breaker.call(operation)
    assert caught.value is excluded
    assert breaker.get_status() == {
        "state": "half_open",
        "failure_count": 1,
        "last_failure_time": failure_time,
        "probe_in_flight": False,
    }
    assert breaker.call(lambda: "recovered") == "recovered"


@pytest.mark.asyncio
async def test_async_and_threaded_callers_share_one_recovery_slot():
    breaker = opened_breaker()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def probe():
        entered.set()
        await release.wait()
        return "recovered"

    def competitor():
        calls.append("unexpected effect")

    task = asyncio.create_task(breaker.call(probe))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(GRPOException, match="probe"):
            await asyncio.to_thread(breaker.call, competitor)
        with pytest.raises(GRPOException, match="probe"):
            await breaker.call(probe)
        assert not calls
    finally:
        release.set()
        assert await task == "recovered"


@pytest.mark.asyncio
@pytest.mark.parametrize("old_fails", [False, True])
async def test_reset_fences_old_probe_without_releasing_new_probe(old_fails):
    breaker = opened_breaker()
    old_entered, old_release = asyncio.Event(), asyncio.Event()
    new_entered, new_release = asyncio.Event(), asyncio.Event()

    async def old_probe():
        old_entered.set()
        await old_release.wait()
        if old_fails:
            raise ValueError("old failure")
        return "old result"

    async def new_probe():
        new_entered.set()
        await new_release.wait()
        return "new result"

    old = asyncio.create_task(breaker.call(old_probe))
    new = None
    try:
        await asyncio.wait_for(old_entered.wait(), timeout=2)
        breaker.reset()
        with pytest.raises(ValueError):
            breaker.call(fail)
        new = asyncio.create_task(breaker.call(new_probe))
        await asyncio.wait_for(new_entered.wait(), timeout=2)
        status = breaker.get_status()
        old_release.set()
        if old_fails:
            with pytest.raises(ValueError, match="old failure"):
                await old
        else:
            assert await old == "old result"
        assert breaker.get_status() == status
        assert breaker.probe_in_flight
        with pytest.raises(GRPOException, match="probe"):
            breaker.call(lambda: "unexpected effect")
    finally:
        old_release.set()
        new_release.set()
        await asyncio.gather(old, *([new] if new else []), return_exceptions=True)
    assert breaker.state is CircuitBreakerState.CLOSED


@pytest.mark.asyncio
async def test_stale_failure_cannot_extend_new_cooldown(monkeypatch):
    now = 50.0
    monkeypatch.setattr(
        errors, "time", SimpleNamespace(time=lambda: now, monotonic=lambda: now)
    )
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
    entered, release = asyncio.Event(), asyncio.Event()

    async def old_call():
        entered.set()
        await release.wait()
        raise ValueError("late failure")

    old = asyncio.create_task(breaker.call(old_call))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(ValueError):
            breaker.call(fail)
        status = breaker.get_status()
        now += 9
        release.set()
        with pytest.raises(ValueError, match="late failure"):
            await old
        assert breaker.get_status() == status
        now += 1
        assert breaker.call(lambda: "recovered") == "recovered"
    finally:
        release.set()
        await asyncio.gather(old, return_exceptions=True)


def test_failed_probe_starts_a_full_new_cooldown(monkeypatch):
    now = 50.0
    monkeypatch.setattr(
        errors, "time", SimpleNamespace(time=lambda: now, monotonic=lambda: now)
    )
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
    with pytest.raises(ValueError):
        breaker.call(fail)
    now += 10
    with pytest.raises(ValueError):
        breaker.call(fail)
    assert breaker.failure_count == 2
    assert breaker.last_failure_time == 60
    assert not breaker.probe_in_flight
    now += 9
    with pytest.raises(GRPOException, match="OPEN"):
        breaker.call(lambda: "too early")
    now += 1
    assert breaker.call(lambda: "recovered") == "recovered"
    assert breaker.failure_count == 0


def test_closed_successes_preserve_legacy_cumulative_failure_policy():
    breaker = CircuitBreaker(failure_threshold=2)
    with pytest.raises(ValueError):
        breaker.call(fail)
    assert breaker.call(lambda: "success") == "success"
    assert breaker.failure_count == 1
    with pytest.raises(ValueError):
        breaker.call(fail)
    assert breaker.is_open


def test_closed_calls_run_concurrently_without_holding_the_state_lock():
    breaker = CircuitBreaker(failure_threshold=2)
    barrier = threading.Barrier(4)

    def operation():
        barrier.wait(timeout=3)
        raise ValueError("concurrent failure")

    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(breaker.call, operation) for _ in range(4)]
        for task in tasks:
            with pytest.raises(ValueError, match="concurrent failure"):
                task.result(timeout=4)
    assert breaker.is_open
    assert breaker.failure_count == 2


@pytest.mark.asyncio
async def test_async_callable_is_lazy_until_awaited():
    breaker = opened_breaker()
    calls = []

    class Operation:
        async def __call__(self, value):
            calls.append(value)
            return value

    pending = breaker.call(Operation(), "unused")
    try:
        assert breaker.is_open
        assert not breaker.probe_in_flight
        assert not calls
        assert await breaker.call(Operation(), "used") == "used"
        assert calls == ["used"]
    finally:
        pending.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("result_kind", ["coroutine", "future", "custom"])
async def test_sync_awaitable_factories_record_awaited_failures(result_kind):
    breaker = opened_breaker()
    failure = ValueError("awaited failure")

    async def operation():
        raise failure

    class CustomAwaitable:
        def __await__(self):
            return operation().__await__()

    def factory():
        if result_kind == "coroutine":
            return operation()
        if result_kind == "custom":
            return CustomAwaitable()
        result = asyncio.get_running_loop().create_future()
        result.set_exception(failure)
        return result

    pending = breaker.call(factory)
    assert breaker.probe_in_flight
    assert breaker.state is CircuitBreakerState.HALF_OPEN
    with pytest.raises(ValueError) as caught:
        await pending
    assert caught.value is failure
    assert breaker.is_open
    assert not breaker.probe_in_flight
    assert breaker.failure_count == 2


@pytest.mark.asyncio
async def test_decorators_preserve_metadata_and_async_introspection():
    handler = ErrorHandler()

    def sync_operation(value: str) -> str:
        """Synchronous operation."""
        return value

    async def async_operation(value: str) -> str:
        """Asynchronous operation."""
        return value

    sync_wrapped = circuit_breaker("sync", errors.CircuitBreakerConfig(), handler)(
        sync_operation
    )
    async_wrapped = circuit_breaker("async", errors.CircuitBreakerConfig(), handler)(
        async_operation
    )
    for original, wrapped in (
        (sync_operation, sync_wrapped),
        (async_operation, async_wrapped),
    ):
        assert wrapped.__name__ == original.__name__
        assert wrapped.__doc__ == original.__doc__
        assert inspect.signature(wrapped) == inspect.signature(original)
        assert inspect.iscoroutinefunction(wrapped) == inspect.iscoroutinefunction(
            original
        )
    assert sync_wrapped("sync result") == "sync result"
    assert await async_wrapped("async result") == "async result"


@pytest.mark.parametrize(
    "options",
    [
        {"failure_threshold": 0},
        {"failure_threshold": -1},
        {"failure_threshold": True},
        {"recovery_timeout": -1},
        {"recovery_timeout": float("nan")},
    ],
)
def test_invalid_breaker_policy_is_rejected_before_work(options):
    with pytest.raises(ValueError):
        CircuitBreaker(**options)
