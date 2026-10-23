"""
Standardized error handling for StateSet Agents: the one exception hierarchy.

Every framework exception derives from :class:`StateSetError`::

    StateSetError
    +-- ConfigurationError (ValueError)          bad configuration / arguments
    +-- ModelError                               model load, init, inference
    +-- TrainingError                            a training run failed
    |   +-- OutOfMemoryError
    |   +-- training.rollout_contract.RolloutContractError (ValueError)
    |   +-- training.diagnostics.TrainingSignalError (RuntimeError)
    |   +-- training.async_rollouts.AsyncRolloutError (RuntimeError)
    |   |   +-- AsyncRolloutClosed, AsyncRolloutTimeout, PolicyArtifactError,
    |   |       WorkerLeaseError, AsyncRolloutWorkerError, ...
    |   +-- training.backends.BackendError (RuntimeError)
    |       +-- BackendExecutionError
    +-- EnvironmentOperationError                an environment step failed
    +-- DataError                                data loading / processing
    +-- RewardError                              reward computation failed
    |   +-- core.reward_context.RewardUnavailableError (RuntimeError)
    |       +-- rewards.llm_judge.JudgeUnavailableError
    +-- ResourceError                            GPU, memory, pool slots
    +-- NetworkError                             remote calls
    +-- ValidationError                          invalid input
    +-- core.error_handling.GRPOException        (legacy family; members also
    |                                             derive from the category above)
    +-- api.errors.APIError                      HTTP errors (wire ErrorCode)
    +-- api.resilience.CircuitOpenError, OperationTimeoutError, BulkheadFullError
    +-- remote.executor.RemoteExecutionError

Errors raised where the framework used to raise a builtin keep that builtin
as a base class and print just their message (``message_only_str``), so
existing ``except ValueError`` / ``except RuntimeError`` handlers and error
output are unchanged (``ConfigurationError`` is a ``ValueError``,
``RewardUnavailableError`` a ``RuntimeError``).

Each class names its :class:`ErrorCode` in ``default_code``; ``code`` is
always an :class:`ErrorCode` member for the framework's own errors (the
gateway's ``APIError`` carries its wire code instead).
"""

from __future__ import annotations

import enum
import traceback
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .._deprecation import deprecated_alias_getattr


class ErrorCode(enum.Enum):
    """Structured error codes for all framework errors."""

    # Configuration errors (CFG_xxx)
    CFG_INVALID = "CFG_001"
    CFG_MISSING_KEY = "CFG_002"
    CFG_TYPE_MISMATCH = "CFG_003"
    CFG_VALIDATION_FAILED = "CFG_004"

    # Model errors (MDL_xxx)
    MDL_LOAD_FAILED = "MDL_001"
    MDL_INIT_FAILED = "MDL_002"
    MDL_INFERENCE_FAILED = "MDL_003"
    MDL_QUANTIZATION_FAILED = "MDL_004"
    MDL_NOT_SUPPORTED = "MDL_005"

    # Training errors (TRN_xxx)
    TRN_INIT_FAILED = "TRN_001"
    TRN_STEP_FAILED = "TRN_002"
    TRN_CHECKPOINT_FAILED = "TRN_003"
    TRN_OOM = "TRN_004"
    TRN_DIVERGED = "TRN_005"
    TRN_GRADIENT_EXPLODED = "TRN_006"
    TRN_ROLLOUT_CONTRACT = "TRN_007"
    TRN_NO_SIGNAL = "TRN_008"
    TRN_ASYNC_ROLLOUT = "TRN_009"
    TRN_BACKEND_FAILED = "TRN_010"

    # Environment errors (ENV_xxx)
    ENV_INIT_FAILED = "ENV_001"
    ENV_STEP_FAILED = "ENV_002"
    ENV_RESET_FAILED = "ENV_003"
    ENV_INVALID_ACTION = "ENV_004"

    # Data errors (DAT_xxx)
    DAT_LOAD_FAILED = "DAT_001"
    DAT_FORMAT_INVALID = "DAT_002"
    DAT_CORRUPTED = "DAT_003"

    # Reward errors (RWD_xxx)
    RWD_COMPUTE_FAILED = "RWD_001"
    RWD_INVALID_VALUE = "RWD_002"
    RWD_UNAVAILABLE = "RWD_003"

    # Network errors (NET_xxx)
    NET_TIMEOUT = "NET_001"
    NET_CONNECTION_FAILED = "NET_002"
    NET_API_ERROR = "NET_003"
    NET_CIRCUIT_OPEN = "NET_004"

    # Resource errors (RES_xxx)
    RES_OOM = "RES_001"
    RES_DISK_FULL = "RES_002"
    RES_GPU_UNAVAILABLE = "RES_003"
    RES_EXHAUSTED = "RES_004"

    # Validation errors (VAL_xxx)
    VAL_INVALID_INPUT = "VAL_001"
    VAL_TYPE_ERROR = "VAL_002"
    VAL_RANGE_ERROR = "VAL_003"

    # Unknown errors (UNK_xxx)
    UNKNOWN = "UNK_001"


@dataclass
class ErrorContext:
    """Context information for errors."""

    operation: str
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    traceback_str: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "component": self.component,
            "details": self.details,
            "traceback": self.traceback_str,
        }


class StateSetError(Exception):
    """Base exception for all StateSet Agents errors."""

    #: Structured code; subclass families may use their own enum.
    code: enum.Enum
    #: When true, ``str(exc)`` is the bare message, as for the builtin the class
    #: also derives from, so swapping ``raise ValueError(msg)`` for a typed error
    #: changes the type but not what callers print.
    message_only_str: ClassVar[bool] = False
    #: The code an instance gets when the caller does not pass one.
    default_code: ClassVar[ErrorCode] = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        context: ErrorContext | None = None,
        cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.message = message
        # ``UNKNOWN`` is the "not given" default, so a typed subclass raised as
        # ``SomeError(msg)`` reports its own code.
        self.code = self.default_code if code is ErrorCode.UNKNOWN else code
        self.context = context or ErrorContext(operation="unknown", component="unknown")
        self.cause = cause

    def __str__(self) -> str:
        if self.message_only_str:
            return self.message
        parts = [f"[{self.code.value}] {self.message}"]
        if self.context.details:
            parts.append(f"Context: {self.context.details}")
        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {self.cause}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code.value,
            "message": self.message,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(StateSetError, ValueError):
    """Raised when configuration or call arguments are invalid.

    Also a ``ValueError`` (what these checks raised before), and ``str(exc)``
    is the bare message.
    """

    message_only_str = True
    default_code = ErrorCode.CFG_INVALID

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=self.default_code,
            context=ErrorContext(
                operation="configuration", component="config", details=kwargs
            ),
        )


class ModelError(StateSetError):
    """Raised when model operations fail."""

    default_code = ErrorCode.MDL_LOAD_FAILED

    def __init__(
        self,
        message: str,
        code: ErrorCode | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"model_name": model_name, **kwargs}
        super().__init__(
            message,
            code=code or self.default_code,
            context=ErrorContext(
                operation="model_operation", component="model", details=details
            ),
        )


class TrainingError(StateSetError):
    """Raised when training fails."""

    default_code = ErrorCode.TRN_STEP_FAILED

    def __init__(
        self,
        message: str,
        code: ErrorCode | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"step": step, **kwargs}
        super().__init__(
            message,
            code=code or self.default_code,
            context=ErrorContext(
                operation="training", component="trainer", details=details
            ),
        )


class OutOfMemoryError(TrainingError):
    """Raised when GPU/CPU memory is exhausted."""

    def __init__(
        self,
        message: str = "Out of memory",
        allocated_gb: float | None = None,
        reserved_gb: float | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"allocated_gb": allocated_gb, "reserved_gb": reserved_gb, **kwargs}
        super().__init__(message, code=ErrorCode.TRN_OOM, **details)


class EnvironmentOperationError(StateSetError):
    """Raised when environment operations (init, reset, step) fail.

    Named so it does not shadow the builtin ``EnvironmentError`` (an alias of
    ``OSError``); ``core.errors.EnvironmentError`` is a deprecated alias.
    """

    default_code = ErrorCode.ENV_STEP_FAILED

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=self.default_code,
            context=ErrorContext(
                operation="environment", component="environment", details=kwargs
            ),
        )


class DataError(StateSetError):
    """Raised when data loading or processing fails."""

    default_code = ErrorCode.DAT_LOAD_FAILED

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=self.default_code,
            context=ErrorContext(
                operation="data_loading", component="data", details=kwargs
            ),
        )


class RewardError(StateSetError):
    """Raised when reward computation fails."""

    default_code = ErrorCode.RWD_COMPUTE_FAILED

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=self.default_code,
            context=ErrorContext(
                operation="reward_computation", component="reward", details=kwargs
            ),
        )


class ResourceError(StateSetError):
    """Raised when a resource (GPU, memory, disk, pool slot) is unavailable."""

    default_code = ErrorCode.RES_GPU_UNAVAILABLE

    def __init__(
        self,
        message: str,
        code: ErrorCode | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            code=code or self.default_code,
            context=ErrorContext(
                operation="resource", component="resource", details=kwargs
            ),
        )


class NetworkError(StateSetError):
    """Raised when network operations fail."""

    default_code = ErrorCode.NET_CONNECTION_FAILED

    def __init__(
        self,
        message: str,
        code: ErrorCode | None = None,
        url: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"url": url, **kwargs}
        super().__init__(
            message,
            code=code or self.default_code,
            context=ErrorContext(
                operation="network", component="network", details=details
            ),
        )


class ValidationError(StateSetError):
    """Raised when input validation fails."""

    default_code = ErrorCode.VAL_INVALID_INPUT

    def __init__(self, message: str, field: str | None = None, **kwargs: Any) -> None:
        details = {"field": field, **kwargs}
        super().__init__(
            message,
            code=self.default_code,
            context=ErrorContext(
                operation="validation", component="validator", details=details
            ),
        )


def wrap_exception(
    exc: BaseException,
    new_type: type[StateSetError],
    message: str | None = None,
    code: ErrorCode | None = None,
    **context_kwargs: Any,
) -> StateSetError:
    """Wrap an exception in a StateSetError with proper chaining.

    Args:
        exc: Original exception
        new_type: Type of StateSetError to wrap in
        message: Optional override message
        code: Optional error code
        **context_kwargs: Additional context

    Returns:
        Wrapped exception
    """
    msg = message or str(exc)
    context = ErrorContext(
        operation="exception_wrap",
        component="error_handler",
        details=context_kwargs,
        traceback_str=traceback.format_exc(),
    )

    result = new_type(msg, code=code or ErrorCode.UNKNOWN)
    result.cause = exc
    result.context = context
    return result


def raise_from(new_exc: StateSetError, cause: BaseException) -> None:
    """Raise a new exception from a cause with proper chaining.

    Args:
        new_exc: New exception to raise
        cause: Original cause
    """
    new_exc.cause = cause
    new_exc.context.traceback_str = traceback.format_exc()
    raise new_exc from cause


#: ``EnvironmentError`` shadowed the builtin (an alias of ``OSError``), so a
#: ``from stateset_agents.core.errors import *`` or a star-free import of it
#: changed what ``except EnvironmentError`` caught. Renamed in v0.70.0.
__getattr__ = deprecated_alias_getattr(
    __name__,
    {"EnvironmentError": (__name__, "EnvironmentOperationError", "0.71.0")},
)
