"""
Resilience Patterns Module

Production-ready resilience patterns including circuit breaker, retry,
timeout, and bulkhead implementations for fault-tolerant APIs.
"""

import asyncio
import copy
import functools
import inspect
import logging
import math
import random
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ParamSpec, TypeVar, cast

from stateset_agents.exceptions import INFERENCE_EXCEPTIONS
from stateset_agents.utils.async_calls import drain_owned_operation
from stateset_agents.utils.backoff import capped_exponential_delay

logger = logging.getLogger(__name__)

CIRCUIT_EXCEPTIONS = INFERENCE_EXCEPTIONS
HEALTHCHECK_EXCEPTIONS = INFERENCE_EXCEPTIONS

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")


# ============================================================================
# Circuit Breaker
# ============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes before closing
    timeout_seconds: float = 30.0  # Time before half-open
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures
    fallback: Callable | None = None  # Fallback function when open


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: float | None = None
    last_state_change: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change,
        }


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Prevents cascading failures by temporarily stopping requests
    to a failing service.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests are rejected
    - HALF_OPEN: One recovery probe at a time, until enough probes succeed

    Each admitted call belongs to a state generation. Results from an older
    generation still reach their callers but cannot change current recovery
    counters. Use an instance on one event loop; reset is synchronous and must
    run on that same loop when calls are active.

    Usage:
        breaker = CircuitBreaker(name="external_api")

        @breaker
        async def call_external_api():
            ...
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._generation = 0
        self._probe_in_flight = False
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._stats.state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    @property
    def probe_in_flight(self) -> bool:
        """Return whether the current recovery generation owns an active probe."""
        return self._probe_in_flight

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with circuit breaker."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)

        return cast(F, wrapper)

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute an admitted call or use the configured rejection fallback.

        Half-open recovery admits one probe at a time. Cancellation and excluded
        exceptions release admission without supplying recovery evidence.
        Completion bookkeeping finishes before caller cancellation propagates.
        """
        async with self._lock:
            self._stats.total_requests += 1
            if self._stats.state == CircuitState.OPEN and self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
            rejected_state = None
            if self._stats.state == CircuitState.OPEN or (
                self._stats.state == CircuitState.HALF_OPEN and self._probe_in_flight
            ):
                self._stats.rejected_requests += 1
                rejected_state = self._stats.state
            elif self._stats.state == CircuitState.HALF_OPEN:
                self._probe_in_flight = True
            generation = self._generation

        # A fallback is user work, not a probe, and must not hold the state lock.
        if rejected_state is not None:
            if self.config.fallback:
                return await self._execute_fallback(*args, **kwargs)
            if rejected_state == CircuitState.HALF_OPEN:
                raise CircuitOpenError(
                    f"Circuit {self.name} is HALF_OPEN. "
                    "Recovery probe already in progress. Request rejected."
                )
            raise CircuitOpenError(f"Circuit {self.name} is OPEN. Request rejected.")

        succeeded = False
        failure: Exception | None = None
        try:
            result = await func(*args, **kwargs)
            succeeded = True
            return result
        except CIRCUIT_EXCEPTIONS as e:
            if isinstance(e, Exception) and not isinstance(
                e, self.config.excluded_exceptions
            ):
                failure = e
            raise
        finally:
            completion = self._complete_attempt(generation, succeeded, failure)
            if self._lock.locked():
                await drain_owned_operation(completion)
            else:
                # With no competing lock owner, completion has no suspension
                # point. Commit before another ready caller can be admitted.
                await completion

    async def _complete_attempt(
        self, generation: int, succeeded: bool, failure: Exception | None
    ) -> None:
        """Commit an outcome and release its probe, with no awaits inside the lock."""
        async with self._lock:
            if generation != self._generation:
                return
            self._probe_in_flight = False
            if succeeded:
                self._on_success()
            elif failure is not None:
                self._on_failure(failure)

    def _on_success(self) -> None:
        """Handle current-generation success while holding the state lock."""
        self._stats.success_count += 1

        if self._stats.state == CircuitState.HALF_OPEN:
            if self._stats.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self._stats.state == CircuitState.CLOSED:
            self._stats.failure_count = 0

    def _on_failure(self, exception: Exception) -> None:
        """Handle current-generation failure while holding the state lock."""
        self._stats.failure_count += 1
        self._stats.last_failure_time = time.time()
        self._stats.success_count = 0

        logger.warning(
            f"Circuit {self.name} recorded failure: {exception}",
            extra={"circuit": self.name, "failures": self._stats.failure_count},
        )

        if self._stats.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._stats.state == CircuitState.CLOSED:
            if self._stats.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._opened_at is None:
            return True
        elapsed = time.monotonic() - self._opened_at
        return elapsed >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._stats.state
        self._generation += 1
        self._probe_in_flight = False
        self._stats.state = new_state
        self._stats.last_state_change = time.time()
        self._opened_at = time.monotonic() if new_state == CircuitState.OPEN else None

        if new_state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            self._stats.success_count = 0
        if new_state == CircuitState.CLOSED:
            self._stats.failure_count = 0

        logger.info(
            f"Circuit {self.name} transitioned: {old_state.value} -> {new_state.value}",
            extra={
                "circuit": self.name,
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        )

    async def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function."""
        fallback = self.config.fallback
        if fallback is None:
            raise RuntimeError("Fallback function is not configured")
        if asyncio.iscoroutinefunction(fallback):
            return await fallback(*args, **kwargs)
        return fallback(*args, **kwargs)

    def reset(self) -> None:
        """Reset stats and admission, invalidating results from earlier calls."""
        self._generation += 1
        self._probe_in_flight = False
        self._opened_at = None
        self._stats = CircuitStats()
        logger.info(f"Circuit {self.name} manually reset")


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# ============================================================================
# Retry Pattern
# ============================================================================


@dataclass
class RetryConfig:
    """Validated retry policy; each invocation copies its current configuration."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: type[BaseException] | tuple[type[BaseException], ...] = (
        Exception,
    )
    non_retryable_exceptions: type[BaseException] | tuple[type[BaseException], ...] = ()

    def __post_init__(self) -> None:
        """Validate budgets and copy exception filters before any operation starts."""
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        for name in ("base_delay", "max_delay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.exponential_base) or self.exponential_base <= 0:
            raise ValueError("exponential_base must be finite and positive")
        for name in ("retryable_exceptions", "non_retryable_exceptions"):
            configured = getattr(self, name)
            try:
                exceptions = (
                    (configured,) if isinstance(configured, type) else tuple(configured)
                )
            except TypeError as exc:
                raise TypeError(f"{name} must contain exception classes") from exc
            if any(
                not isinstance(exc, type) or not issubclass(exc, BaseException)
                for exc in exceptions
            ):
                raise TypeError(f"{name} must contain exception classes")
            setattr(self, name, exceptions)


class RetryStrategy:
    """
    Retry strategy with exponential backoff and jitter.

    Usage:
        @retry(max_attempts=3, base_delay=1.0)
        async def unreliable_operation():
            ...
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        self.config = config or RetryConfig()

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with retry logic."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return cast(F, wrapper)

    async def execute(
        self, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Retry configured failures under one per-call policy, never interruption."""
        policy = replace(self.config)
        attempt = 0

        while True:
            attempt += 1
            try:
                return await func(*args, **kwargs)

            except (
                asyncio.CancelledError,
                KeyboardInterrupt,
                SystemExit,
                GeneratorExit,
            ):
                raise

            except policy.non_retryable_exceptions:
                # Don't retry these
                raise

            except policy.retryable_exceptions as e:
                if attempt == policy.max_attempts:
                    logger.error(
                        f"All {policy.max_attempts} retry attempts failed",
                        extra={"attempt": attempt, "error": str(e)},
                    )
                    raise

                delay = self._calculate_delay(attempt, policy)
                logger.warning(
                    f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}",
                    extra={"attempt": attempt, "delay": delay, "error": str(e)},
                )
                await asyncio.sleep(delay)

    def _calculate_delay(
        self, attempt: int, policy: RetryConfig | None = None
    ) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        config = policy if policy is not None else replace(self.config)
        delay = capped_exponential_delay(
            config.base_delay, config.exponential_base, attempt - 1, config.max_delay
        )

        if config.jitter:
            # Preserve the API's upward jitter after capping the base delay.
            delay = min(delay * (1 + random.random() * 0.5), sys.float_info.max)

        return delay


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> Callable[[F], F]:
    """Decorator factory for retry with custom config."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
    )
    strategy = RetryStrategy(config)
    return strategy


# ============================================================================
# Timeout Pattern
# ============================================================================


class TimeoutError(Exception):
    """Raised when operation times out."""

    pass


class _CallDeadlineExpired(Exception):
    """Distinguish our elapsed deadline from an upstream timeout exception."""


async def _cancel_and_drain_call(task: asyncio.Future[Any]) -> None:
    """Cancel owned work once, then retain it through repeated caller cancellation."""
    if not task.done():
        task.cancel()

    async def join() -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass

    await drain_owned_operation(join())


async def _await_owned_call(task: asyncio.Future[T], seconds: float | None = None) -> T:
    """Observe a call without forwarding repeated cancellation into its cleanup."""
    try:
        done, _ = await asyncio.wait({task}, timeout=seconds)
    except asyncio.CancelledError:
        try:
            await _cancel_and_drain_call(task)
        except Exception:
            # Retrieve cleanup failures; explicit caller cancellation takes precedence.
            pass
        raise
    if not done:
        await _cancel_and_drain_call(task)
        raise _CallDeadlineExpired
    return task.result()


def timeout(seconds: float | None) -> Callable[[F], F]:
    """
    Decorate async functions with a deadline and owned cancellation cleanup.

    Expiry cancels the operation once and drains it before raising TimeoutError,
    even if the operation suppresses cancellation and returns a late value.
    Cleanup may outlast the deadline. Upstream errors retain their original type.

    Usage:
        @timeout(5.0)
        async def slow_operation():
            ...
    """
    if seconds is not None and math.isnan(seconds):
        raise ValueError("timeout seconds must not be NaN")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            deadline = None if seconds is None else time.monotonic() + seconds

            async def invoke():
                if deadline is not None and time.monotonic() >= deadline:
                    raise _CallDeadlineExpired
                result = await func(*args, **kwargs)
                if deadline is not None and time.monotonic() >= deadline:
                    raise _CallDeadlineExpired
                return result

            try:
                if seconds is not None and seconds <= 0:
                    raise _CallDeadlineExpired
                return await _await_owned_call(asyncio.create_task(invoke()), seconds)
            except _CallDeadlineExpired as exc:
                raise TimeoutError(
                    f"Operation {func.__name__} timed out after {seconds}s"
                ) from exc

        return cast(F, wrapper)

    return decorator


# ============================================================================
# Bulkhead Pattern
# ============================================================================


@dataclass
class BulkheadConfig:
    """Bulkhead configuration."""

    max_concurrent: int = 10
    max_waiting: int = 100
    timeout_seconds: float | None = 30.0

    def __post_init__(self) -> None:
        """Reject unusable capacity and timeout settings before admitting work."""
        for name, minimum in (("max_concurrent", 1), ("max_waiting", 0)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.timeout_seconds is not None and (
            math.isnan(self.timeout_seconds) or self.timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be nonnegative and not NaN")


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""

    pass


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent access to a resource to prevent
    resource exhaustion. Limits are captured at construction. Use each instance
    on one event loop; admission and release have no suspension points. Active
    capacity includes reserved permits and work unwinding after cancellation.

    Usage:
        bulkhead = Bulkhead("db_pool", BulkheadConfig(max_concurrent=10))

        @bulkhead
        async def database_operation():
            ...
    """

    def __init__(
        self,
        name: str,
        config: BulkheadConfig | None = None,
    ) -> None:
        self.name = name
        self.config = replace(config) if config is not None else BulkheadConfig()
        self._max_concurrent = self.config.max_concurrent
        self._max_waiting = self.config.max_waiting
        self._timeout_seconds = self.config.timeout_seconds
        # True means capacity was reserved; False means the queue deadline expired.
        self._waiters: deque[tuple[asyncio.Future[bool], float | None]] = deque()
        self._active = 0

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with bulkhead."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return cast(F, wrapper)

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Admit work, retaining its capacity through owned cancellation cleanup."""
        await self._acquire()

        try:

            async def invoke():
                return await func(*args, **kwargs)

            return await _await_owned_call(asyncio.create_task(invoke()))
        finally:
            self._release()

    async def _acquire(self) -> None:
        """Reserve idle capacity or join the FIFO queue without invoking user work."""
        if self._active < self._max_concurrent and not self._waiters:
            self._active += 1
            return
        if len(self._waiters) >= self._max_waiting:
            raise BulkheadFullError(
                f"Bulkhead {self.name} is full. Max waiting: {self._max_waiting}"
            )
        loop = asyncio.get_running_loop()
        deadline = (
            None
            if self._timeout_seconds is None
            else time.monotonic() + self._timeout_seconds
        )
        future: asyncio.Future[bool] = loop.create_future()
        ticket = (future, deadline)
        self._waiters.append(ticket)

        try:
            await asyncio.wait({future}, timeout=self._timeout_seconds)
            if future.done() and future.result():
                return
            raise BulkheadFullError(f"Timeout waiting for bulkhead {self.name}")
        except BaseException:
            if future.done() and not future.cancelled() and future.result():
                # Cancellation won the race with a completed capacity handoff.
                self._release()
            else:
                if ticket in self._waiters:
                    self._waiters.remove(ticket)
                future.cancel()
            raise

    def _release(self) -> None:
        """Return one reservation, handing it directly to the oldest live waiter."""
        self._active -= 1
        while self._waiters:
            future, deadline = self._waiters.popleft()
            if deadline is not None and time.monotonic() >= deadline:
                future.set_result(False)
                continue
            self._active += 1
            future.set_result(True)
            break

    @property
    def stats(self) -> dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "max_concurrent": self._max_concurrent,
            "active": self._active,
            "waiting": len(self._waiters),
            "available": self._max_concurrent - self._active,
        }


# ============================================================================
# Health Check
# ============================================================================


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str | None = None
    latency_ms: float = 0.0
    last_check: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    details: dict[str, Any] = field(default_factory=dict)


class HealthCheckerState(str, Enum):
    """Lifecycle of health probe admission and cleanup ownership."""

    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(eq=False)
class _HealthProbe:
    name: str
    registration: object
    task: asyncio.Task[HealthCheckResult] | None = None
    waiters: int = 0
    cancel_requested: bool = False
    waiting_for_sync: bool = False


class HealthChecker:
    """
    Health observations with per-registration probe and cleanup ownership.

    Use an instance on one event loop. Simultaneous callers share a probe;
    cancellation of its last observer retires publication immediately while
    cleanup remains owned. Close the checker before closing probe dependencies.

    Usage:
        checker = HealthChecker()
        checker.add_check("database", check_database)
        checker.add_check("cache", check_cache)

        try:
            health = await checker.check_all()
        finally:
            await checker.aclose()
    """

    def __init__(self) -> None:
        self._checks: dict[str, Callable] = {}
        self._results: dict[str, HealthCheckResult] = {}
        self._registrations: dict[str, object] = {}
        self._current_probes: dict[str, _HealthProbe] = {}
        self._probes: set[_HealthProbe] = set()
        self._state = HealthCheckerState.READY
        self._close_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> HealthCheckerState:
        """Observe probe admission and cleanup without starting work."""
        return self._state

    def add_check(self, name: str, check_func: Callable) -> None:
        """Register a fresh check identity and invalidate its previous observation."""
        if self._state is not HealthCheckerState.READY:
            raise RuntimeError("Health checker is closing or closed")
        if not callable(check_func):
            raise TypeError("Health check must be callable")
        self._checks[name] = check_func
        self._registrations[name] = object()
        self._results.pop(name, None)
        self._current_probes.pop(name, None)

    def remove_check(self, name: str) -> None:
        """Remove a health check."""
        self._checks.pop(name, None)
        self._registrations.pop(name, None)
        self._results.pop(name, None)
        self._current_probes.pop(name, None)

    def clone_for_restart(self) -> "HealthChecker":
        """Create fresh probe ownership after confirmed close, retaining registrations."""
        if (
            self._state is not HealthCheckerState.CLOSED
            or self._close_task is None
            or not self._close_task.done()
            or self._close_task.cancelled()
            or self._close_task.exception() is not None
        ):
            raise RuntimeError("Finish health checker cleanup before replacing it")
        replacement = HealthChecker()
        for name, callback in self._checks.items():
            replacement.add_check(name, callback)
        return replacement

    async def check(self, name: str) -> HealthCheckResult:
        """Observe one current probe without letting waiter cancellation detach it."""
        if self._state is not HealthCheckerState.READY or name not in self._checks:
            return self._cached_result(name)
        probe = self._current_probes.get(name)
        if probe is None or (probe.task is not None and probe.task.done()):
            probe = self._start_probe(name)
        elif probe.cancel_requested:
            return self._cached_result(name)
        assert probe.task is not None
        if probe.task is asyncio.current_task():
            raise RuntimeError("A health probe cannot await its own result")
        probe.waiters += 1
        try:
            if not probe.task.done():
                await asyncio.wait({probe.task})
            if probe.task.cancelled():
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Health probe was cancelled",
                )
            return self._copy_result(name, probe.task.result())
        finally:
            probe.waiters -= 1
            if not probe.task.done() and probe.waiters == 0:
                self._cancel_probe(probe)

    def _start_probe(self, name: str) -> _HealthProbe:
        probe = _HealthProbe(name, self._registrations[name])
        self._current_probes[name] = probe
        self._probes.add(probe)
        self._results.pop(name, None)
        operation = self._run_probe(probe, self._checks[name])
        try:
            probe.task = asyncio.create_task(operation)
        except BaseException:
            operation.close()
            self._probes.discard(probe)
            if self._current_probes.get(name) is probe:
                self._current_probes.pop(name)
            raise
        probe.task.add_done_callback(lambda task: self._probe_done(probe, task))
        return probe

    def _cancel_probe(self, probe: _HealthProbe) -> None:
        if probe.cancel_requested:
            return
        probe.cancel_requested = True
        if self._registrations.get(probe.name) is probe.registration:
            self._results.pop(probe.name, None)
        # Cancelling to_thread's waiter cannot stop its thread. Keep that waiter
        # alive until the callable returns, including any awaitable it returns.
        if (
            probe.task is not None
            and not probe.task.done()
            and not probe.waiting_for_sync
        ):
            probe.task.cancel()

    async def _run_probe(
        self, probe: _HealthProbe, check_func: Callable
    ) -> HealthCheckResult:
        # Eager task factories may run this body before create_task returns.
        probe.task = asyncio.current_task()
        start = time.monotonic()
        try:
            if inspect.iscoroutinefunction(check_func) or inspect.iscoroutinefunction(
                getattr(check_func, "__call__", None)  # noqa: B004
            ):
                result = check_func()
            else:
                probe.waiting_for_sync = True
                try:
                    result = await asyncio.to_thread(check_func)
                finally:
                    probe.waiting_for_sync = False
            if inspect.isawaitable(result):
                if probe.cancel_requested:
                    if (
                        inspect.iscoroutine(result)
                        and inspect.getcoroutinestate(result) == inspect.CORO_CREATED
                    ):
                        result.close()
                    else:
                        await _cancel_and_drain_call(asyncio.ensure_future(result))
                    raise asyncio.CancelledError
                result = await result
            if probe.cancel_requested:
                raise asyncio.CancelledError
            if isinstance(result, HealthCheckResult):
                check_result = copy.deepcopy(result)
                check_result.status = HealthStatus(check_result.status)
                check_result.latency_ms = (time.monotonic() - start) * 1000
            else:
                check_result = HealthCheckResult(
                    name=probe.name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    latency_ms=(time.monotonic() - start) * 1000,
                )
        except HEALTHCHECK_EXCEPTIONS as e:
            check_result = HealthCheckResult(
                name=probe.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )
        if (
            self._state is HealthCheckerState.READY
            and not probe.cancel_requested
            and self._registrations.get(probe.name) is probe.registration
        ):
            self._results[probe.name] = check_result
        return check_result

    def _probe_done(
        self, probe: _HealthProbe, task: asyncio.Task[HealthCheckResult]
    ) -> None:
        self._probes.discard(probe)
        if self._current_probes.get(probe.name) is probe:
            self._current_probes.pop(probe.name)
        if not task.cancelled():
            error = task.exception()
            if error is not None and probe.waiters == 0:
                logger.debug(
                    "Unobserved health probe %s failed (%s)",
                    probe.name,
                    type(error).__name__,
                )

    def _cached_result(self, name: str) -> HealthCheckResult:
        if self._state is not HealthCheckerState.READY:
            message = f"Health checker is {self._state.value}"
        elif name not in self._checks:
            message = f"Check '{name}' not found"
        elif name in self._results:
            return self._results[name]
        else:
            probe = self._current_probes.get(name)
            message = (
                "Health check cleanup is in progress"
                if probe is not None and probe.cancel_requested
                else "Health check has not completed"
            )
        return HealthCheckResult(
            name=name, status=HealthStatus.UNHEALTHY, message=message
        )

    def _copy_result(self, name: str, result: HealthCheckResult) -> HealthCheckResult:
        try:
            return copy.deepcopy(result)
        except Exception:
            if self._results.get(name) is result:
                self._results.pop(name)
            raise

    def get_results(self) -> dict[str, HealthCheckResult]:
        """Return detached current observations, including unobserved checks as unhealthy."""
        return {
            name: self._copy_result(name, self._cached_result(name))
            for name in tuple(self._checks)
        }

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """Return detached batch measurements fenced by current registrations."""
        if any(probe.task is asyncio.current_task() for probe in self._probes):
            raise RuntimeError("A health probe cannot await a batch containing itself")
        registrations = dict(self._registrations)
        tasks = [asyncio.create_task(self.check(name)) for name in registrations]
        try:
            observations = dict(
                zip(registrations, await asyncio.gather(*tasks), strict=True)
            )
        finally:
            pending = [task for task in tasks if not task.done()]
            for task in pending:
                task.cancel()
            if pending:

                async def join() -> None:
                    await asyncio.gather(*pending, return_exceptions=True)

                await drain_owned_operation(join())
        # Another request may already have started a fresh probe. Preserve this
        # batch's measured results while fencing changes to registration identity.
        return {
            name: (
                observations[name]
                if self._state is HealthCheckerState.READY
                and name in observations
                and registrations[name] is self._registrations[name]
                else self._copy_result(name, self._cached_result(name))
            )
            for name in tuple(self._checks)
        }

    @staticmethod
    def aggregate_status(results: Mapping[str, HealthCheckResult]) -> HealthStatus:
        """Summarize exactly one result snapshot without reading live checker state."""
        statuses = [result.status for result in results.values()]
        if not statuses or any(
            status not in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            for status in statuses
        ):
            return HealthStatus.UNHEALTHY
        if all(status is HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.DEGRADED

    @property
    def overall_status(self) -> HealthStatus:
        """Summarize current registrations; missing observations cannot establish health."""
        return self.aggregate_status(
            {name: self._cached_result(name) for name in self._checks}
        )

    def get_status(self) -> dict[str, Any]:
        """Inspect admission and retained probe ownership without invoking checks."""
        return {
            "state": self._state.value,
            "overall_status": self.overall_status.value,
            "registered_checks": sorted(self._checks),
            "active_probes": len(self._probes),
            "cancelling_probes": sum(probe.cancel_requested for probe in self._probes),
            "pending_checks": sorted({probe.name for probe in self._probes}),
        }

    async def aclose(self) -> None:
        """Stop admission and join every owned probe before dependencies may close."""
        if any(probe.task is asyncio.current_task() for probe in self._probes):
            raise RuntimeError("A health probe cannot close its own checker")
        if self._close_task is None:
            self._state = HealthCheckerState.CLOSING
            self._close_task = asyncio.create_task(self._close())

        async def join() -> None:
            assert self._close_task is not None
            await self._close_task

        await drain_owned_operation(join())

    async def _close(self) -> None:
        probes = list(self._probes)
        for probe in probes:
            self._cancel_probe(probe)
        await asyncio.gather(
            *(probe.task for probe in probes if probe.task is not None),
            return_exceptions=True,
        )
        self._results.clear()
        self._current_probes.clear()
        self._probes.clear()
        self._state = HealthCheckerState.CLOSED


# ============================================================================
# Global Circuit Breaker Registry
# ============================================================================

_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_circuit_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    return {
        name: breaker.stats.to_dict() for name, breaker in _circuit_breakers.items()
    }
