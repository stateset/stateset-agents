"""
Error handling helpers for the GRPO family of exceptions.

The exception hierarchy itself lives in :mod:`stateset_agents.core.errors`
(``StateSetError`` and its categories). This module keeps:

* ``GRPOException`` and its ``ValidationException`` / ``ResourceException``
  members, which the framework still raises. Each is also the matching
  :mod:`~stateset_agents.core.errors` category (``ValidationException`` is a
  ``ValidationError``) and its ``code`` is a canonical
  :class:`~stateset_agents.core.errors.ErrorCode`; the legacy ``GRPO_E###``
  code stays on ``error_code`` (a :class:`GRPOErrorCode`).
* ``ErrorHandler``, an in-memory error log returning :class:`ErrorRecord`.

Deprecated here (each access or call warns and names its replacement):

* ``ErrorCode`` (now :class:`GRPOErrorCode`; use ``core.errors.ErrorCode``)
  and ``TrainingException`` / ``ModelException`` / ``DataException`` /
  ``NetworkException`` (unused; use the ``core.errors`` categories) are
  removed in v0.72.0 with the stable root names that point at them;
* ``ErrorContext`` (now :class:`ErrorRecord`) is removed in v0.71.0;
* the retry and circuit-breaker helpers (``RetryConfig``, ``retry_async``,
  ``CircuitBreakerConfig``, ``CircuitBreaker``, ``circuit_breaker``) are
  removed in v0.72.0; :mod:`stateset_agents.api.resilience` is the one
  implementation.
"""

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._deprecation import warn_deprecated
from . import errors as _core_errors
from .errors import (
    DataError,
    ModelError,
    NetworkError,
    ResourceError,
    StateSetError,
    TrainingError,
    ValidationError,
)

#: Stable (root ``__all__``) names deprecated here are removed in this release.
_STABLE_REMOVAL = "0.72.0"
#: Provisional names deprecated here are removed in this release.
_PROVISIONAL_REMOVAL = "0.71.0"
_RESILIENCE = "stateset_agents.api.resilience"


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


class GRPOErrorCode(Enum):
    """Legacy ``GRPO_E###`` codes carried on ``GRPOException.error_code``.

    The canonical code of every framework error is
    :class:`stateset_agents.core.errors.ErrorCode` (``exc.code``).
    """

    UNKNOWN = "GRPO_E000"
    VALIDATION_FAILED = "GRPO_E001"
    TRAINING_FAILED = "GRPO_E100"
    MODEL_FAILED = "GRPO_E200"
    DATA_FAILED = "GRPO_E300"
    NETWORK_FAILED = "GRPO_E400"
    RESOURCE_FAILED = "GRPO_E500"
    SYSTEM_FAILED = "GRPO_E900"


DEFAULT_ERROR_CODES = {
    ErrorCategory.TRAINING: GRPOErrorCode.TRAINING_FAILED,
    ErrorCategory.MODEL: GRPOErrorCode.MODEL_FAILED,
    ErrorCategory.DATA: GRPOErrorCode.DATA_FAILED,
    ErrorCategory.NETWORK: GRPOErrorCode.NETWORK_FAILED,
    ErrorCategory.RESOURCE: GRPOErrorCode.RESOURCE_FAILED,
    ErrorCategory.VALIDATION: GRPOErrorCode.VALIDATION_FAILED,
    ErrorCategory.SYSTEM: GRPOErrorCode.SYSTEM_FAILED,
}

#: The canonical :class:`~stateset_agents.core.errors.ErrorCode` of each
#: category, used for ``GRPOException.code``.
CATEGORY_ERROR_CODES: dict[ErrorCategory, _core_errors.ErrorCode] = {
    ErrorCategory.TRAINING: TrainingError.default_code,
    ErrorCategory.MODEL: ModelError.default_code,
    ErrorCategory.DATA: DataError.default_code,
    ErrorCategory.NETWORK: NetworkError.default_code,
    ErrorCategory.RESOURCE: _core_errors.ErrorCode.RES_EXHAUSTED,
    ErrorCategory.VALIDATION: ValidationError.default_code,
    ErrorCategory.SYSTEM: _core_errors.ErrorCode.UNKNOWN,
}


@dataclass
class ErrorRecord:
    """One handled error as :class:`ErrorHandler` records it.

    Formerly ``ErrorContext``, a name it shared with the exception context in
    :mod:`stateset_agents.core.errors`.
    """

    error_id: str
    error_code: GRPOErrorCode
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

    A :class:`~stateset_agents.core.errors.StateSetError` whose ``code`` is the
    canonical :class:`~stateset_agents.core.errors.ErrorCode` of its category;
    the legacy ``GRPO_E###`` code is ``error_code`` and still heads
    ``str(exc)``. Each subclass also derives from the matching category error
    (``ValidationException`` is a ``ValidationError``, ...): one hierarchy.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: GRPOErrorCode | None = None,
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
            category, GRPOErrorCode.UNKNOWN
        )
        self.details = details or {}
        self.recovery_actions = recovery_actions or []
        self.timestamp = time.time()
        # StateSetError attributes, so ``except StateSetError as e`` reads the
        # same canonical ``e.code`` / ``e.context`` / ``e.to_dict()`` for every
        # framework error.
        self.code = CATEGORY_ERROR_CODES[category]
        self.context = _core_errors.ErrorContext(
            operation="unknown", component=category.value, details=self.details
        )
        self.cause: BaseException | None = None

    def __str__(self) -> str:
        parts = [f"[{self.error_code.value}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """``StateSetError.to_dict()`` plus the GRPO fields."""
        payload = super().to_dict()
        payload.update(
            {
                "grpo_error_code": self.error_code.value,
                "category": self.category.value,
                "severity": self.severity.value,
            }
        )
        return payload

    def to_context(
        self, component: str = "unknown", operation: str = "unknown"
    ) -> ErrorRecord:
        """Convert to the record :class:`ErrorHandler` keeps."""
        return ErrorRecord(
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


class ResourceException(GRPOException, ResourceError):
    """Resource allocation errors (GPU, memory, etc.)"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.RESOURCE
        super().__init__(message, **kwargs)


class ValidationException(GRPOException, ValidationError):
    """Input validation errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.VALIDATION
        super().__init__(message, **kwargs)


# The framework never raises these four; they are served (with a warning) by
# the module ``__getattr__`` below until v0.72.0.
class _TrainingException(GRPOException, TrainingError):
    """Training-related errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.TRAINING
        super().__init__(message, **kwargs)


class _ModelException(GRPOException, ModelError):
    """Model-related errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.MODEL
        super().__init__(message, **kwargs)


class _DataException(GRPOException, DataError):
    """Data processing errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.DATA
        super().__init__(message, **kwargs)


class _NetworkException(GRPOException, NetworkError):
    """Network and API errors"""

    def __init__(self, message: str, **kwargs):
        kwargs["category"] = ErrorCategory.NETWORK
        super().__init__(message, **kwargs)


# Keep the public names on the classes (pickling, reprs and tracebacks).
for _cls, _public in (
    (_TrainingException, "TrainingException"),
    (_ModelException, "ModelException"),
    (_DataException, "DataException"),
    (_NetworkException, "NetworkException"),
):
    _cls.__name__ = _cls.__qualname__ = _public
del _cls, _public

#: name -> (object, removal release, replacement)
_DEPRECATED_NAMES: dict[str, tuple[Any, str, str]] = {
    "ErrorCode": (
        GRPOErrorCode,
        _STABLE_REMOVAL,
        "stateset_agents.core.errors.ErrorCode (exc.code) or "
        "stateset_agents.core.error_handling.GRPOErrorCode (exc.error_code)",
    ),
    "ErrorContext": (
        ErrorRecord,
        _PROVISIONAL_REMOVAL,
        "stateset_agents.core.error_handling.ErrorRecord",
    ),
    "TrainingException": (
        _TrainingException,
        _STABLE_REMOVAL,
        "stateset_agents.core.errors.TrainingError",
    ),
    "ModelException": (
        _ModelException,
        _STABLE_REMOVAL,
        "stateset_agents.core.errors.ModelError",
    ),
    "DataException": (
        _DataException,
        _STABLE_REMOVAL,
        "stateset_agents.core.errors.DataError",
    ),
    "NetworkException": (
        _NetworkException,
        _STABLE_REMOVAL,
        "stateset_agents.core.errors.NetworkError",
    ),
}


def __getattr__(name: str) -> Any:
    """Serve the deprecated names with a dated warning."""
    entry = _DEPRECATED_NAMES.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value, removal, replacement = entry
    warn_deprecated(f"{__name__}.{name}", removal=removal, replacement=replacement)
    return value


class RetryConfig:
    """Configuration for :func:`retry_async`.

    Deprecated: :class:`stateset_agents.api.resilience.RetryConfig` with
    ``RetryStrategy`` / ``retry`` is the one retry implementation; this one
    will be removed in v0.72.0.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: list[type[Exception]] | None = None,
    ):
        warn_deprecated(
            f"{__name__}.RetryConfig",
            removal=_STABLE_REMOVAL,
            replacement=f"{_RESILIENCE}.RetryConfig",
        )
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on or [_NetworkException, ResourceException]


class CircuitBreakerConfig:
    """Configuration for :class:`CircuitBreaker`.

    Deprecated: :class:`stateset_agents.api.resilience.CircuitBreakerConfig`
    is the one circuit breaker; this one will be removed in v0.72.0.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        warn_deprecated(
            f"{__name__}.CircuitBreakerConfig",
            removal=_STABLE_REMOVAL,
            replacement=f"{_RESILIENCE}.CircuitBreakerConfig",
        )
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Synchronous circuit breaker.

    Deprecated: :class:`stateset_agents.api.resilience.CircuitBreaker` (async,
    with half-open success counting, fallbacks and the ``/metrics`` registry)
    is the one implementation; this one will be removed in v0.72.0.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        warn_deprecated(
            f"{__name__}.CircuitBreaker",
            removal=_STABLE_REMOVAL,
            replacement=f"{_RESILIENCE}.CircuitBreaker",
        )
        # Backward/interop support: allow constructing directly from thresholds.
        self.config = config or CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
        )
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    @property
    def is_open(self) -> bool:
        """Return True when the breaker is OPEN."""
        return self.state == CircuitBreakerState.OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise GRPOException(
                    "Circuit breaker is OPEN",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.HIGH,
                )

        try:
            result = func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
            return result

        except self.config.expected_exception:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN

            raise


def _readable_error_id(error: BaseException) -> str:
    """A readable id from the exception message, falling back to a timestamp."""
    try:
        text = str(error)
    # Process boundary: this runs while reporting an arbitrary (possibly
    # user-defined) exception; a broken ``__str__`` must not turn error
    # reporting into a second, unrelated crash.
    except Exception:  # noqa: BLE001
        text = ""
    slug = text.strip().lower().replace(" ", "_")
    slug = "".join(ch for ch in slug if ch.isalnum() or ch in ("_", "-"))
    return slug or f"generic_{int(time.time())}"


class ErrorHandler:
    """Centralized error handling and recovery"""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        max_error_history: int = 10000,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: list[ErrorRecord] = []
        self._max_error_history = max_error_history
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> ErrorRecord:
        """Handle and log error with rich context"""

        if isinstance(error, GRPOException):
            error_context = error.to_context(component, operation)
        else:
            error_context = ErrorRecord(
                error_id=_readable_error_id(error),
                error_code=DEFAULT_ERROR_CODES.get(
                    ErrorCategory.SYSTEM, GRPOErrorCode.UNKNOWN
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

    def _log_error(self, error_context: ErrorRecord):
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
        """Register a circuit breaker (deprecated with :class:`CircuitBreaker`)."""
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


def retry_async(config: RetryConfig):
    """Async retry decorator with exponential backoff.

    Deprecated: use :func:`stateset_agents.api.resilience.retry`; this will be
    removed in v0.72.0.
    """
    warn_deprecated(
        f"{__name__}.retry_async",
        removal=_STABLE_REMOVAL,
        replacement=f"{_RESILIENCE}.retry",
    )

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            last_exception: BaseException | None = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except tuple(config.retry_on) as e:
                    last_exception = e

                    # Don't wait after the last attempt
                    if attempt == config.max_attempts - 1:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base**attempt),
                        config.max_delay,
                    )

                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        import random

                        delay *= 0.5 + random.random() * 0.5

                    await asyncio.sleep(delay)

            # If we get here, all retries failed
            if last_exception is None:
                # Only reachable when max_attempts was mutated below 1 after
                # construction; never ``raise None`` (a confusing TypeError).
                raise ValueError(
                    f"max_attempts must be >= 1, got {config.max_attempts}"
                )
            raise last_exception

        return wrapper

    return decorator


def circuit_breaker(
    name: str, config: CircuitBreakerConfig, error_handler: ErrorHandler
):
    """Circuit breaker decorator.

    Deprecated: use :func:`stateset_agents.api.resilience.get_circuit_breaker`
    (the breaker is itself a decorator); this will be removed in v0.72.0.
    """
    warn_deprecated(
        f"{__name__}.circuit_breaker",
        removal=_STABLE_REMOVAL,
        replacement=f"{_RESILIENCE}.get_circuit_breaker",
    )

    error_handler.register_circuit_breaker(name, config)

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            cb = error_handler.get_circuit_breaker(name)
            if cb is None:
                # The registry was cleared after decoration; re-register so the
                # decorated function keeps its breaker instead of crashing.
                cb = CircuitBreaker(config)
                error_handler.circuit_breakers[name] = cb
            return cb.call(func, *args, **kwargs)

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
) -> ErrorRecord:
    """Handle error using global handler"""
    return _global_error_handler.handle_error(error, component, operation, context)


def get_error_summary() -> dict[str, Any]:
    """Get error summary using global handler"""
    return _global_error_handler.get_error_summary()
