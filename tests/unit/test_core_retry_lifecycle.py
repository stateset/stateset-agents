"""Retry intent, bounded attempts, and cancellation preserve caller control."""

import asyncio
import inspect

import pytest

from stateset_agents.core import error_handling as errors
from stateset_agents.core.error_handling import (
    CircuitBreaker,
    NetworkException,
    RetryConfig,
    retry_async,
)


@pytest.mark.asyncio
async def test_explicit_empty_retry_policy_does_not_repeat_effects():
    calls = []

    @retry_async(RetryConfig(max_attempts=3, retry_on=[], base_delay=0))
    async def operation():
        calls.append("effect")
        raise NetworkException("request outcome unknown")

    with pytest.raises(NetworkException):
        await operation()
    assert calls == ["effect"]


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
def test_invalid_attempt_budget_fails_before_any_operation(attempts):
    with pytest.raises(ValueError, match="max_attempts"):
        RetryConfig(max_attempts=attempts)


@pytest.mark.asyncio
async def test_retry_preserves_function_metadata():
    @retry_async(RetryConfig())
    async def lookup(order_id: str) -> str:
        """Fetch one order."""
        return order_id

    assert lookup.__name__ == "lookup"
    assert lookup.__doc__ == "Fetch one order."
    assert inspect.signature(lookup) == inspect.signature(lookup.__wrapped__)
    assert inspect.iscoroutinefunction(lookup)
    assert await lookup("order-1") == "order-1"


@pytest.mark.asyncio
async def test_cancellation_is_never_retried():
    calls = []

    @retry_async(RetryConfig(max_attempts=3, retry_on=[BaseException], base_delay=0))
    async def operation():
        calls.append("effect")
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await operation()
    assert calls == ["effect"]


@pytest.mark.asyncio
async def test_retry_snapshots_policy_for_each_invocation():
    entered, release = asyncio.Event(), asyncio.Event()
    config = RetryConfig(max_attempts=3, base_delay=0)
    calls = []

    @retry_async(config)
    async def operation():
        calls.append("effect")
        entered.set()
        await release.wait()
        raise NetworkException("request failed")

    task = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        config.max_attempts = 1
        config.base_delay = 3600
        config.retry_on.clear()
        release.set()
        with pytest.raises(NetworkException):
            await asyncio.wait_for(task, timeout=2)
        assert len(calls) == 3
        with pytest.raises(NetworkException):
            await operation()
        assert len(calls) == 4
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_during_backoff_prevents_next_effect(monkeypatch):
    entered = asyncio.Event()
    calls = []
    real_sleep = asyncio.sleep

    async def sleep(delay):
        entered.set()
        await real_sleep(delay)

    monkeypatch.setattr(errors.asyncio, "sleep", sleep)

    @retry_async(RetryConfig(max_attempts=3, base_delay=3600, max_delay=3600))
    async def operation():
        calls.append("effect")
        raise NetworkException("request failed")

    task = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == ["effect"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_delay", "exponential_base", "max_delay", "expected_delays"),
    [
        (2, 2, 5, [2, 4, 5, 5]),
        (1, 1e308, 5, [1, 5, 5, 5]),
        (0, 1e308, 5, [0, 0, 0, 0]),
        (1, 1e308, 0, [0, 0, 0, 0]),
    ],
)
async def test_backoff_is_bounded_and_preserves_last_error(
    monkeypatch, base_delay, exponential_base, max_delay, expected_delays
):
    delays, calls = [], []
    failure = NetworkException("request failed")

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(errors.asyncio, "sleep", sleep)

    @retry_async(
        RetryConfig(
            max_attempts=5,
            base_delay=base_delay,
            exponential_base=exponential_base,
            max_delay=max_delay,
            jitter=False,
        )
    )
    async def operation():
        calls.append("effect")
        raise failure

    with pytest.raises(NetworkException) as caught:
        await operation()
    assert caught.value is failure
    assert len(calls) == 5
    assert delays == expected_delays


@pytest.mark.asyncio
async def test_retry_configuration_owns_its_exception_list():
    retry_on = [ValueError]
    config = RetryConfig(max_attempts=2, base_delay=0, retry_on=retry_on)
    retry_on.clear()
    calls = []

    @retry_async(config)
    async def operation():
        calls.append("effect")
        if len(calls) == 1:
            raise ValueError("try again")
        return "recovered"

    assert await operation() == "recovered"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_scaled_overflow_keeps_the_correct_delay_below_the_cap(monkeypatch):
    delays = []
    failure = NetworkException("attempt failed")

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(errors.asyncio, "sleep", sleep)

    @retry_async(
        RetryConfig(
            max_attempts=4,
            base_delay=1e-300,
            exponential_base=1e200,
            max_delay=1e200,
            jitter=False,
        )
    )
    async def operation():
        raise failure

    with pytest.raises(NetworkException) as caught:
        await operation()
    assert caught.value is failure
    assert delays == pytest.approx([1e-300, 1e-100, 1e100], rel=1e-12, abs=0)


@pytest.mark.asyncio
async def test_simultaneous_calls_have_independent_attempt_budgets():
    both_started = asyncio.Event()
    calls = {"first": 0, "second": 0}

    @retry_async(RetryConfig(max_attempts=3, base_delay=0))
    async def operation(name):
        calls[name] += 1
        if all(calls.values()):
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        if calls[name] < 3:
            raise NetworkException("try again")
        return name

    assert await asyncio.gather(operation("first"), operation("second")) == [
        "first",
        "second",
    ]
    assert calls == {"first": 3, "second": 3}


@pytest.mark.asyncio
async def test_breaker_around_retry_observes_only_the_final_outcome():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
    failure = NetworkException("attempt failed")
    calls = []

    @retry_async(RetryConfig(max_attempts=3, base_delay=0))
    async def operation(recover=False):
        calls.append("effect")
        if recover and len(calls) == 5:
            return "recovered"
        raise failure

    with pytest.raises(NetworkException) as caught:
        await breaker.call(operation)
    assert caught.value is failure
    assert breaker.is_open
    assert breaker.failure_count == 1
    assert len(calls) == 3
    assert await breaker.call(operation, recover=True) == "recovered"
    assert breaker.failure_count == 0
    assert not breaker.probe_in_flight
    assert len(calls) == 5


@pytest.mark.parametrize(
    "options",
    [
        {"base_delay": -1},
        {"base_delay": float("inf")},
        {"max_delay": -1},
        {"max_delay": float("nan")},
        {"exponential_base": 0},
        {"exponential_base": -1},
        {"exponential_base": float("inf")},
    ],
)
def test_invalid_backoff_policy_is_rejected_before_work(options):
    with pytest.raises(ValueError):
        RetryConfig(**options)
