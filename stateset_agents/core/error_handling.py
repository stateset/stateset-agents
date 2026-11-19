"""
Enhanced Error Handling for GRPO Agent Framework

This module provides comprehensive error handling, retry mechanisms,
and recovery strategies for production-ready agent training and deployment.
"""

import asyncio
import inspect
import logging
import math
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast, overload

from stateset_agents.utils.backoff import capped_exponential_delay

from .errors import StateSetError

_P = ParamSpec("_P")
_T = TypeVar("_T")


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better handling"""

    TRAINING = "training"
    MODEL = "model"
    DATA = "data"
    NETWORK = "network"
    RESOURCE = "resource"
    VALIDATION = "validation"
    SYSTEM = "system"


class ErrorCode(Enum):
    """Structured error codes for consistent tracking."""

    UNKNOWN = "GRPO_E000"
    VALIDATION_FAILED = "GRPO_E001"
    TRAINING_FAILED = "GRPO_E100"
    MODEL_FAILED = "GRPO_E200"
    DATA_FAILED = "GRPO_E300"
    NETWORK_FAILED = "GRPO_E400"
    RESOURCE_FAILED = "GRPO_E500"
    SYSTEM_FAILED = "GRPO_E900"


DEFAULT_ERROR_CODES = {
    ErrorCategory.TRAINING: ErrorCode.TRAINING_FAILED,
    ErrorCategory.MODEL: ErrorCode.MODEL_FAILED,
    ErrorCategory.DATA: ErrorCode.DATA_FAILED,
    ErrorCategory.NETWORK: ErrorCode.NETWORK_FAILED,
    ErrorCategory.RESOURCE: ErrorCode.RESOURCE_FAILED,
    ErrorCategory.VALIDATION: ErrorCode.VALIDATION_FAILED,
    ErrorCategory.SYSTEM: ErrorCode.SYSTEM_FAILED,
}


@dataclass
class ErrorContext:
    """Rich error context for debugging"""

    error_id: str
    error_code: ErrorCode
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    component: str
    operation: str
    details: dict[str, Any]
    stack_trace: str
    recovery_actions: list[str]


class GRPOException(StateSetError):
    """Base exception for GRPO framework.

    Inherits from :class:`~stateset_agents.core.errors.StateSetError` so that
    ``except StateSetError`` catches GRPO errors too, unifying the two error
    hierarchies.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        recovery_actions: list[str] | None = None,
    ):
        # Call Exception.__init__ directly to avoid StateSetError's
        # required keyword args while preserving MRO compatibility.
        Exception.__init__(self, message)
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or DEFAULT_ERROR_CODES.get(
            category, ErrorCode.UNKNOWN
        )
        self.details = details or {}
        self.recovery_actions = recovery_actions or []
        self.timestamp = time.time()
        # StateSetError compatibility attributes
        self.cause: BaseException | None = None

    def __str__(self) -> str:
        parts = [f"[{self.error_code.value}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)

    def to_context(
        self, component: str = "unknown", operation: str = "unknown"
    ) -> ErrorContext:
        """Convert to rich error context"""
        return ErrorContext(
            error_id=f"{self.category.value}_{int(self.timestamp)}",
            error_code=self.error_code,
            category=self.category,
            severity=self.severity,
            timestamp=self.timestamp,
            component=component,
            operation=operation,
            details=self.details,
            stack_trace=traceback.format_exc(),
            recovery_actions=self.recovery_actions,
        )


class TrainingException(GRPOException):
    """Training-related errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.TRAINING
        super().__init__(message, **kwargs)


class ModelException(GRPOException):
    """Model-related errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.MODEL
        super().__init__(message, **kwargs)


class DataException(GRPOException):
    """Data processing errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.DATA
        super().__init__(message, **kwargs)


class NetworkException(GRPOException):
    """Network and API errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.NETWORK
        super().__init__(message, **kwargs)


class ResourceException(GRPOException):
    """Resource allocation errors (GPU, memory, etc.)"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.RESOURCE
        super().__init__(message, **kwargs)


class ValidationException(GRPOException):
    """Input validation errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.VALIDATION
        super().__init__(message, **kwargs)


class RetryConfig:
    """Configuration for retry mechanisms"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Sequence[type[BaseException]] | None = None,
    ) -> None:
        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        for name, value in (("base_delay", base_delay), ("max_delay", max_delay)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(exponential_base) or exponential_base <= 0:
            raise ValueError("exponential_base must be finite and positive")
        exceptions: list[type[BaseException]] = (
            [NetworkException, ResourceException]
            if retry_on is None
            else list(retry_on)
        )
        if any(
            not isinstance(exc, type) or not issubclass(exc, BaseException)
            for exc in exceptions
        ):
            raise TypeError("retry_on must contain exception classes")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = exceptions


class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: (
            type[BaseException] | tuple[type[BaseException], ...]
        ) = Exception,
    ) -> None:
        if (
            not isinstance(failure_threshold, int)
            or isinstance(failure_threshold, bool)
            or failure_threshold < 1
        ):
            raise ValueError("failure_threshold must be a positive integer")
        if math.isnan(recovery_timeout) or recovery_timeout < 0:
            raise ValueError("recovery_timeout must be nonnegative and not NaN")
        expected = (
            expected_exception
            if isinstance(expected_exception, tuple)
            else (expected_exception,)
        )
        if any(
            not isinstance(exc, type) or not issubclass(exc, BaseException)
            for exc in expected
        ):
            raise TypeError("expected_exception must contain exception classes")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Guard synchronous and awaited outcomes with one recovery probe at a time."""

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: (
            type[BaseException] | tuple[type[BaseException], ...]
        ) = Exception,
    ) -> None:
        # Backward/interop support: allow constructing directly from thresholds.
        self.config = config or CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
        )
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._last_failure_monotonic = 0.0
        self._lock = threading.RLock()
        self._generation = 0
        self._probe_in_flight = False

    @property
    def is_open(self) -> bool:
        """Return True when the breaker is OPEN."""
        with self._lock:
            return self.state == CircuitBreakerState.OPEN

    @property
    def probe_in_flight(self) -> bool:
        """Observe recovery ownership without reserving a probe."""
        with self._lock:
            return self._probe_in_flight

    def get_status(self) -> dict[str, Any]:
        """Read a consistent lifecycle snapshot without invoking upstream work."""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time,
                "probe_in_flight": self._probe_in_flight,
            }

    def reset(self) -> None:
        """Start a closed generation, fencing earlier results without cancelling work."""
        with self._lock:
            self._generation += 1
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0.0
            self._last_failure_monotonic = 0.0
            self._probe_in_flight = False

    def _admit(self) -> tuple[int, bool]:
        """Reserve admission and capture the generation under the state lock."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if (
                    time.monotonic() - self._last_failure_monotonic
                    < self.config.recovery_timeout
                ):
                    raise GRPOException(
                        "Circuit breaker is OPEN",
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.HIGH,
                    )
                self.state = CircuitBreakerState.HALF_OPEN
                self._generation += 1
            probe = self.state == CircuitBreakerState.HALF_OPEN
            if probe:
                if self._probe_in_flight:
                    raise GRPOException(
                        "Circuit breaker recovery probe is in progress",
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.HIGH,
                    )
                self._probe_in_flight = True
            return self._generation, probe

    def _finish(self, admission: tuple[int, bool], error: BaseException | None) -> None:
        """Commit only the outcome belonging to the admitting generation."""
        generation, probe = admission
        with self._lock:
            if generation != self._generation:
                return
            if probe:
                self._probe_in_flight = False
            if error is None:
                if probe:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self._generation += 1
                # Preserve the core breaker's cumulative closed-state failures.
                return
            if not isinstance(error, Exception) or not isinstance(
                error, self.config.expected_exception
            ):
                return
            self.failure_count += 1
            self.last_failure_time = time.time()
            self._last_failure_monotonic = time.monotonic()
            if probe or self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self._generation += 1

    @overload
    def call(
        self, func: Callable[_P, Awaitable[_T]], *args: _P.args, **kwargs: _P.kwargs
    ) -> Coroutine[Any, Any, _T]: ...

    @overload
    def call(
        self, func: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T: ...

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Protect a call, recording awaitable results only after they settle.

        Known async callables are admitted when awaited. A synchronous callable
        returning an awaitable starts under admission immediately; callers must
        await its returned wrapper to settle that ownership. Use call_async to
        defer invocation of an awaitable factory until execution is awaited.
        """
        if inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
            getattr(func, "__call__", None)  # noqa: B004 - Inspect async behavior.
        ):
            return self.call_async(func, *args, **kwargs)
        admission = self._admit()
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            self._finish(admission, exc)
            raise
        if inspect.isawaitable(result):
            return self._await_result(result, admission)
        self._finish(admission, None)
        return result

    async def call_async(
        self, func: Callable[_P, Awaitable[_T]], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        """Admit and invoke an async callable or awaitable factory when awaited."""
        admission = self._admit()
        try:
            result = await func(*args, **kwargs)
        except BaseException as exc:
            self._finish(admission, exc)
            raise
        self._finish(admission, None)
        return result

    async def _await_result(
        self, result: Awaitable[_T], admission: tuple[int, bool]
    ) -> _T:
        try:
            value = await result
        except BaseException as exc:
            self._finish(admission, exc)
            raise
        self._finish(admission, None)
        return value


class ErrorHandler:
    """Centralized error handling and recovery"""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        max_error_history: int = 10000,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: list[ErrorContext] = []
        self._max_error_history = max_error_history
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> ErrorContext:
        """Handle and log error with rich context"""

        if isinstance(error, GRPOException):
            error_context = error.to_context(component, operation)
        else:
            # Derive a readable error_id from the exception message, falling back to timestamp
            try:
                _msg = str(error).strip().lower().replace(" ", "_")
                _msg = "".join(ch for ch in _msg if ch.isalnum() or ch in ("_", "-"))
                _err_id = _msg if _msg else f"generic_{int(time.time())}"
            except (AttributeError, TypeError, ValueError):
                _err_id = f"generic_{int(time.time())}"
            error_context = ErrorContext(
                error_id=_err_id,
                error_code=DEFAULT_ERROR_CODES.get(
                    ErrorCategory.SYSTEM, ErrorCode.UNKNOWN
                ),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                timestamp=time.time(),
                component=component,
                operation=operation,
                details=context or {},
                stack_trace=traceback.format_exc(),
                recovery_actions=[],
            )

        self.error_history.append(error_context)

        # Enforce bounded history to prevent memory leaks
        if len(self.error_history) > self._max_error_history:
            # Remove oldest errors, keeping the most recent
            self.error_history = self.error_history[-self._max_error_history :]

        self._log_error(error_context)

        return error_context

    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level"""
        log_data = {
            "error_id": error_context.error_id,
            "error_code": error_context.error_code.value,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "component": error_context.component,
            "operation": error_context.operation,
            "details": error_context.details,
        }

        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {log_data}")
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR: {log_data}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR: {log_data}")
        else:
            self.logger.info(f"LOW SEVERITY ERROR: {log_data}")

    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Register a circuit breaker"""
        self.circuit_breakers[name] = CircuitBreaker(config)

    def get_circuit_breaker(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of recent errors"""
        recent_errors = [
            e
            for e in self.error_history
            if time.time() - e.timestamp < 3600  # Last hour
        ]

        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_error_code: dict[str, int] = {}

        for error in recent_errors:
            category = error.category.value
            severity = error.severity.value
            error_code = error.error_code.value

            by_category[category] = by_category.get(category, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_error_code[error_code] = by_error_code.get(error_code, 0) + 1

        return {
            "total_errors": len(recent_errors),
            "by_category": by_category,
            "by_severity": by_severity,
            "by_error_code": by_error_code,
            "circuit_breakers": {
                name: cb.state.value for name, cb in self.circuit_breakers.items()
            },
        }


def retry_async(
    config: RetryConfig,
) -> Callable[[Callable[_P, Awaitable[_T]]], Callable[_P, Coroutine[Any, Any, _T]]]:
    """Retry configured errors within a stable per-call policy, never cancellation."""

    def decorator(
        func: Callable[_P, Awaitable[_T]],
    ) -> Callable[_P, Coroutine[Any, Any, _T]]:
        @wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            policy = RetryConfig(
                max_attempts=config.max_attempts,
                base_delay=config.base_delay,
                max_delay=config.max_delay,
                exponential_base=config.exponential_base,
                jitter=config.jitter,
                retry_on=config.retry_on,
            )
            attempt = 0

            while True:
                try:
                    return await func(*args, **kwargs)
                except (
                    asyncio.CancelledError,
                    KeyboardInterrupt,
                    SystemExit,
                    GeneratorExit,
                ):
                    raise
                except tuple(policy.retry_on):
                    attempt += 1
                    if attempt == policy.max_attempts:
                        raise

                    delay = capped_exponential_delay(
                        policy.base_delay,
                        policy.exponential_base,
                        attempt - 1,
                        policy.max_delay,
                    )

                    # Add jitter to prevent thundering herd
                    if policy.jitter:
                        import random

                        delay *= 0.5 + random.random() * 0.5

                    await asyncio.sleep(delay)

        return wrapper

    return decorator


def circuit_breaker(
    name: str, config: CircuitBreakerConfig, error_handler: ErrorHandler
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Decorate sync or async work without hiding its metadata or awaited outcome."""

    error_handler.register_circuit_breaker(name, config)

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        def breaker() -> CircuitBreaker:
            cb = error_handler.get_circuit_breaker(name)
            if cb is None:
                raise RuntimeError(f"Circuit breaker {name!r} is not registered")
            return cb

        if inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
            getattr(func, "__call__", None)  # noqa: B004 - Inspect async behavior.
        ):

            @wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
                return await breaker().call_async(
                    cast(Callable[_P, Awaitable[Any]], func), *args, **kwargs
                )

            return cast(Callable[_P, _T], async_wrapper)

        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            return breaker().call(func, *args, **kwargs)

        return wrapper

    return decorator


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_global_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    return _global_error_handler


def set_global_error_handler(handler: ErrorHandler):
    """Set global error handler instance"""
    global _global_error_handler
    _global_error_handler = handler


# Convenience functions
def handle_error(
    error: Exception,
    component: str,
    operation: str,
    context: dict[str, Any] | None = None,
) -> ErrorContext:
    """Handle error using global handler"""
    return _global_error_handler.handle_error(error, component, operation, context)


def get_error_summary() -> dict[str, Any]:
    """Get error summary using global handler"""
    return _global_error_handler.get_error_summary()
