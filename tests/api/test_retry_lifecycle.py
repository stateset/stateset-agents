"""API retry budgets, classification, and cancellation remain stable per call."""

import asyncio
import inspect
import math
import sys

import pytest

from stateset_agents.api import resilience
from stateset_agents.api.resilience import (
    Bulkhead,
    BulkheadConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    RetryStrategy,
    retry,
    timeout,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_type",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit],
)
async def test_cancellation_and_process_control_are_never_retried(failure_type):
    strategy = RetryStrategy(
        RetryConfig(max_attempts=3, base_delay=0, retryable_exceptions=(BaseException,))
    )
    failure = failure_type("stop")
    calls = []

    async def operation():
        calls.append("effect")
        raise failure

    with pytest.raises(failure_type) as caught:
        await strategy.execute(operation)
    assert caught.value is failure
    assert calls == ["effect"]


@pytest.mark.asyncio
async def test_each_invocation_uses_one_policy_snapshot():
    config = RetryConfig(max_attempts=3, base_delay=0)
    strategy = RetryStrategy(config)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    failure = ValueError("attempt failed")

    async def operation():
        calls.append("effect")
        entered.set()
        await release.wait()
        raise failure

    task = asyncio.create_task(strategy.execute(operation))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        config.max_attempts = 1
        config.base_delay = 3600
        config.retryable_exceptions = ()
        config.non_retryable_exceptions = (ValueError,)
        release.set()
        with pytest.raises(ValueError) as caught:
            await asyncio.wait_for(task, timeout=2)
        assert caught.value is failure
        assert len(calls) == 3
        with pytest.raises(ValueError):
            await strategy.execute(operation)
        assert len(calls) == 4
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("base_delay", [0, 1])
async def test_exponential_overflow_cannot_mask_the_final_error(
    monkeypatch, base_delay
):
    delays, calls = [], []
    failure = ValueError("attempt failed")
    strategy = RetryStrategy(
        RetryConfig(
            max_attempts=4,
            base_delay=base_delay,
            exponential_base=1e308,
            max_delay=5,
            jitter=False,
        )
    )

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(resilience.asyncio, "sleep", sleep)

    async def operation():
        calls.append("effect")
        raise failure

    with pytest.raises(ValueError) as caught:
        await strategy.execute(operation)
    assert caught.value is failure
    assert len(calls) == 4
    assert delays == ([0, 0, 0] if base_delay == 0 else [1, 5, 5])


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
def test_invalid_attempt_budget_fails_before_any_operation(attempts):
    with pytest.raises(ValueError, match="max_attempts"):
        RetryConfig(max_attempts=attempts)


@pytest.mark.asyncio
async def test_scaled_overflow_keeps_the_correct_delay_below_the_cap(monkeypatch):
    strategy = RetryStrategy(
        RetryConfig(
            max_attempts=4,
            base_delay=1e-300,
            exponential_base=1e200,
            max_delay=1e200,
            jitter=False,
        )
    )
    delays = []
    failure = ValueError("request failed")

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(resilience.asyncio, "sleep", sleep)

    async def operation():
        raise failure

    with pytest.raises(ValueError) as caught:
        await strategy.execute(operation)
    assert caught.value is failure
    assert delays == pytest.approx([1e-300, 1e-100, 1e100], rel=1e-12, abs=0)


@pytest.mark.asyncio
async def test_cancellation_during_backoff_stops_before_the_next_attempt(monkeypatch):
    entered = asyncio.Event()
    calls = []
    real_sleep = asyncio.sleep

    async def sleep(delay):
        entered.set()
        await real_sleep(delay)

    monkeypatch.setattr(resilience.asyncio, "sleep", sleep)
    strategy = RetryStrategy(
        RetryConfig(
            max_attempts=3,
            base_delay=3600,
            max_delay=3600,
            retryable_exceptions=(BaseException,),
        )
    )

    async def operation():
        calls.append("effect")
        raise ValueError("attempt failed")

    task = asyncio.create_task(strategy.execute(operation))
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
async def test_concurrent_calls_have_independent_attempt_budgets():
    strategy = RetryStrategy(RetryConfig(max_attempts=3, base_delay=0))
    both_started = asyncio.Event()
    calls = {"first": 0, "second": 0}

    async def operation(name):
        calls[name] += 1
        if all(calls.values()):
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        if calls[name] < 3:
            raise ValueError("attempt failed")
        return name

    assert await asyncio.gather(
        strategy.execute(operation, "first"), strategy.execute(operation, "second")
    ) == ["first", "second"]
    assert calls == {"first": 3, "second": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filters",
    [
        {"retryable_exceptions": ()},
        {"retryable_exceptions": (TypeError,)},
        {
            "retryable_exceptions": (Exception,),
            "non_retryable_exceptions": (ValueError,),
        },
    ],
)
async def test_empty_unmatched_and_excluded_filters_never_repeat_effects(
    monkeypatch, filters
):
    calls, delays = [], []
    failure = ValueError("request outcome unknown")
    strategy = RetryStrategy(RetryConfig(max_attempts=3, base_delay=0, **filters))

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(resilience.asyncio, "sleep", sleep)

    async def operation():
        calls.append("effect")
        raise failure

    with pytest.raises(ValueError) as caught:
        await strategy.execute(operation)
    assert caught.value is failure
    assert calls == ["effect"]
    assert not delays


@pytest.mark.asyncio
async def test_config_owns_exception_filters_and_normalizes_single_classes():
    retryable = [ValueError]
    excluded = [TypeError]
    config = RetryConfig(
        max_attempts=2,
        base_delay=0,
        retryable_exceptions=retryable,
        non_retryable_exceptions=excluded,
    )
    retryable.clear()
    excluded.append(ValueError)
    calls = []

    async def operation():
        calls.append("effect")
        if len(calls) == 1:
            raise ValueError("attempt failed")
        return "result"

    assert await RetryStrategy(config).execute(operation) == "result"
    assert calls == ["effect", "effect"]
    assert RetryConfig(retryable_exceptions=ValueError).retryable_exceptions == (
        ValueError,
    )


@pytest.mark.asyncio
async def test_invalid_configuration_mutation_is_rejected_before_invocation():
    strategy = RetryStrategy(RetryConfig())
    strategy.config.max_attempts = 0
    calls = []

    async def operation():
        calls.append("effect")

    with pytest.raises(ValueError, match="max_attempts"):
        await strategy.execute(operation)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_delay", "exponential_base", "max_delay", "expected"),
    [
        (2, 2, 5, [2, 4, 5]),
        (4, 0.5, 5, [4, 2, 1]),
        (5, 1, 1, [1, 1, 1]),
        (1, 1e308, 0, [0, 0, 0]),
    ],
)
async def test_backoff_cap_and_no_sleep_after_final_attempt(
    monkeypatch, base_delay, exponential_base, max_delay, expected
):
    delays = []
    failures = []
    strategy = RetryStrategy(
        RetryConfig(
            max_attempts=4,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=False,
        )
    )

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(resilience.asyncio, "sleep", sleep)

    async def operation():
        failure = ValueError(f"attempt {len(failures) + 1}")
        failures.append(failure)
        raise failure

    with pytest.raises(ValueError) as caught:
        await strategy.execute(operation)
    assert len(failures) == 4
    assert caught.value is failures[-1]
    assert delays == expected


@pytest.mark.parametrize("random_value", [0, 0.5, 1])
def test_api_jitter_preserves_upward_adjustment_after_capping(
    monkeypatch, random_value
):
    monkeypatch.setattr(resilience.random, "random", lambda: random_value)
    strategy = RetryStrategy(RetryConfig(base_delay=4, max_delay=5))
    assert strategy._calculate_delay(2) == 5 * (1 + random_value * 0.5)


def test_jitter_overflow_remains_finite(monkeypatch):
    monkeypatch.setattr(resilience.random, "random", lambda: 1.0)
    strategy = RetryStrategy(
        RetryConfig(base_delay=sys.float_info.max, max_delay=sys.float_info.max)
    )
    assert math.isfinite(strategy._calculate_delay(1))


@pytest.mark.asyncio
async def test_breaker_observes_one_final_retry_outcome():
    breaker = CircuitBreaker(
        "retry",
        CircuitBreakerConfig(
            failure_threshold=1, success_threshold=1, timeout_seconds=0
        ),
    )
    strategy = RetryStrategy(RetryConfig(max_attempts=3, base_delay=0))
    calls = []

    @strategy
    async def operation(recover=False):
        calls.append("effect")
        if recover and len(calls) == 5:
            return "recovered"
        raise ValueError("attempt failed")

    with pytest.raises(ValueError):
        await breaker.call(operation)
    assert len(calls) == 3
    assert breaker.state is CircuitState.OPEN
    assert breaker.stats.failure_count == 1
    assert await breaker.call(operation, recover=True) == "recovered"
    assert len(calls) == 5
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
@pytest.mark.parametrize("outer_retry", [False, True])
async def test_retry_after_timeout_waits_for_cleanup_in_both_bulkhead_orders(
    outer_retry,
):
    bulkhead = Bulkhead("retry", BulkheadConfig(max_concurrent=1))
    strategy = RetryStrategy(RetryConfig(max_attempts=2, base_delay=0))
    cleaning, release = asyncio.Event(), asyncio.Event()
    calls = []

    @timeout(0.02)
    async def operation():
        if calls:
            calls.append("recovered")
            return "result"
        calls.append("first")
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            calls.append("cleaned")

    wrapped = (
        strategy(bulkhead(operation)) if outer_retry else bulkhead(strategy(operation))
    )
    task = asyncio.create_task(wrapped())
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert calls == ["first"]
        assert bulkhead.stats["active"] == 1
        assert not task.done()
        release.set()
        assert await task == "result"
        assert calls == ["first", "cleaned", "recovered"]
        assert bulkhead.stats["active"] == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_decorators_retain_metadata_arguments_and_result_identity():
    result = object()
    calls = []

    async def operation(order_id: str, *, version: int = 1) -> object:
        """One request."""
        calls.append((order_id, version))
        return result

    for wrap in (RetryStrategy(), retry()):
        wrapped = wrap(operation)
        assert wrapped.__name__ == operation.__name__
        assert wrapped.__doc__ == operation.__doc__
        assert inspect.signature(wrapped) == inspect.signature(operation)
        assert inspect.iscoroutinefunction(wrapped)
        assert await wrapped("order-1", version=2) is result
    assert calls == [("order-1", 2), ("order-1", 2)]


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
def test_invalid_backoff_is_rejected_before_work(options):
    with pytest.raises(ValueError):
        RetryConfig(**options)


@pytest.mark.parametrize("field", ["retryable_exceptions", "non_retryable_exceptions"])
@pytest.mark.parametrize(
    "value", [None, (ValueError("instance"),), (str,), "ValueError"]
)
def test_invalid_filters_are_rejected_before_work(field, value):
    with pytest.raises(TypeError, match=field):
        RetryConfig(**{field: value})
