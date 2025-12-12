"""
Resilience Patterns Module

Production-ready resilience patterns including circuit breaker, retry,
timeout, and bulkhead implementations for fault-tolerant APIs.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 3           # Successes before closing
    timeout_seconds: float = 30.0        # Time before half-open
    excluded_exceptions: tuple = ()      # Exceptions that don't count as failures
    fallback: Optional[Callable] = None  # Fallback function when open


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
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
    - HALF_OPEN: Testing if service recovered

    Usage:
        breaker = CircuitBreaker(name="external_api")

        @breaker
        async def call_external_api():
            ...
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._stats.state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        async with self._lock:
            self._stats.total_requests += 1

            # Check if we should transition from OPEN to HALF_OPEN
            if self._stats.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._stats.rejected_requests += 1
                    if self.config.fallback:
                        return await self._execute_fallback(*args, **kwargs)
                    raise CircuitOpenError(
                        f"Circuit {self.name} is OPEN. "
                        f"Request rejected."
                    )

        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result

        except Exception as e:
            # Check if this exception should be counted
            if isinstance(e, self.config.excluded_exceptions):
                raise

            await self._on_failure(e)
            raise

    async def _on_success(self) -> None:
        """Handle successful execution."""
        async with self._lock:
            self._stats.success_count += 1

            if self._stats.state == CircuitState.HALF_OPEN:
                if self._stats.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._stats.failure_count = 0

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed execution."""
        async with self._lock:
            self._stats.failure_count += 1
            self._stats.last_failure_time = time.time()
            self._stats.success_count = 0

            logger.warning(
                f"Circuit {self.name} recorded failure: {exception}",
                extra={"circuit": self.name, "failures": self._stats.failure_count}
            )

            if self._stats.state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._transition_to(CircuitState.OPEN)

            elif self._stats.state == CircuitState.CLOSED:
                if self._stats.failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._stats.last_failure_time is None:
            return True
        elapsed = time.time() - self._stats.last_failure_time
        return elapsed >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._stats.state
        self._stats.state = new_state
        self._stats.last_state_change = time.time()

        if new_state == CircuitState.CLOSED:
            self._stats.failure_count = 0
            self._stats.success_count = 0

        logger.info(
            f"Circuit {self.name} transitioned: {old_state.value} -> {new_state.value}",
            extra={"circuit": self.name, "old_state": old_state.value, "new_state": new_state.value}
        )

    async def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function."""
        if asyncio.iscoroutinefunction(self.config.fallback):
            return await self.config.fallback(*args, **kwargs)
        return self.config.fallback(*args, **kwargs)

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
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
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()


class RetryStrategy:
    """
    Retry strategy with exponential backoff and jitter.

    Usage:
        @retry(max_attempts=3, base_delay=1.0)
        async def unreliable_operation():
            ...
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with retry logic."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)

            except self.config.non_retryable_exceptions as e:
                # Don't retry these
                raise

            except self.config.retryable_exceptions as e:
                last_exception = e

                if attempt == self.config.max_attempts:
                    logger.error(
                        f"All {self.config.max_attempts} retry attempts failed",
                        extra={"attempt": attempt, "error": str(e)}
                    )
                    raise

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}",
                    extra={"attempt": attempt, "delay": delay, "error": str(e)}
                )
                await asyncio.sleep(delay)

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            # Add random jitter (0-50% of delay)
            delay = delay * (1 + random.random() * 0.5)

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


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Decorator to add timeout to async functions.

    Usage:
        @timeout(5.0)
        async def slow_operation():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Operation {func.__name__} timed out after {seconds}s"
                )
        return wrapper
    return decorator


# ============================================================================
# Bulkhead Pattern
# ============================================================================

@dataclass
class BulkheadConfig:
    """Bulkhead configuration."""
    max_concurrent: int = 10
    max_waiting: int = 100
    timeout_seconds: float = 30.0


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""
    pass


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent access to a resource to prevent
    resource exhaustion.

    Usage:
        bulkhead = Bulkhead(name="db_pool", max_concurrent=10)

        @bulkhead
        async def database_operation():
            ...
    """

    def __init__(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
    ):
        self.name = name
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._waiting = 0
        self._active = 0
        self._lock = asyncio.Lock()

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with bulkhead."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through bulkhead."""
        async with self._lock:
            if self._waiting >= self.config.max_waiting:
                raise BulkheadFullError(
                    f"Bulkhead {self.name} is full. "
                    f"Max waiting: {self.config.max_waiting}"
                )
            self._waiting += 1

        try:
            # Python 3.10 compatibility - use wait_for instead of asyncio.timeout
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._waiting -= 1
            raise BulkheadFullError(
                f"Timeout waiting for bulkhead {self.name}"
            )

        async with self._lock:
            self._waiting -= 1
            self._active += 1

        try:
            return await func(*args, **kwargs)
        finally:
            self._semaphore.release()
            async with self._lock:
                self._active -= 1

    @property
    def stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "max_concurrent": self.config.max_concurrent,
            "active": self._active,
            "waiting": self._waiting,
            "available": self.config.max_concurrent - self._active,
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
    message: Optional[str] = None
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Health check manager for service dependencies.

    Usage:
        checker = HealthChecker()
        checker.add_check("database", check_database)
        checker.add_check("cache", check_cache)

        health = await checker.check_all()
    """

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheckResult] = {}

    def add_check(self, name: str, check_func: Callable) -> None:
        """Add a health check function."""
        self._checks[name] = check_func

    def remove_check(self, name: str) -> None:
        """Remove a health check."""
        self._checks.pop(name, None)
        self._results.pop(name, None)

    async def check(self, name: str) -> HealthCheckResult:
        """Run a single health check."""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check '{name}' not found",
            )

        check_func = self._checks[name]
        start = time.monotonic()

        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            latency = (time.monotonic() - start) * 1000

            if isinstance(result, HealthCheckResult):
                result.latency_ms = latency
                self._results[name] = result
                return result

            # Simple boolean result
            status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            check_result = HealthCheckResult(
                name=name,
                status=status,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            check_result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
            )

        self._results[name] = check_result
        return check_result

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks concurrently."""
        tasks = [self.check(name) for name in self._checks]
        await asyncio.gather(*tasks)
        return self._results

    @property
    def overall_status(self) -> HealthStatus:
        """Get overall health status."""
        if not self._results:
            return HealthStatus.HEALTHY

        statuses = [r.status for r in self._results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED


# ============================================================================
# Global Circuit Breaker Registry
# ============================================================================

_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_circuit_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    return {
        name: breaker.stats.to_dict()
        for name, breaker in _circuit_breakers.items()
    }
