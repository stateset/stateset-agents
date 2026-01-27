"""
Standardized error handling for StateSet Agents.

Provides structured error codes, proper error chaining, and developer-friendly
error messages with context.
"""

from __future__ import annotations

import enum
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union


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

    # Network errors (NET_xxx)
    NET_TIMEOUT = "NET_001"
    NET_CONNECTION_FAILED = "NET_002"
    NET_API_ERROR = "NET_003"

    # Resource errors (RES_xxx)
    RES_OOM = "RES_001"
    RES_DISK_FULL = "RES_002"
    RES_GPU_UNAVAILABLE = "RES_003"

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
    details: Dict[str, Any] = field(default_factory=dict)
    traceback_str: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "component": self.component,
            "details": self.details,
            "traceback": self.traceback_str,
        }


class StateSetError(Exception):
    """Base exception for all StateSet Agents errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        context: Optional[ErrorContext] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or ErrorContext(
            operation="unknown", component="unknown"
        )
        self.cause = cause

    def __str__(self) -> str:
        parts = [f"[{self.code.value}] {self.message}"]
        if self.context.details:
            parts.append(f"Context: {self.context.details}")
        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {self.cause}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.code.value,
            "message": self.message,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(StateSetError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, code=ErrorCode.CFG_INVALID, context=ErrorContext(
                operation="configuration", component="config", details=kwargs
            )
        )


class ModelError(StateSetError):
    """Raised when model operations fail."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.MDL_LOAD_FAILED,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        details = {"model_name": model_name, **kwargs}
        super().__init__(
            message,
            code=code,
            context=ErrorContext(
                operation="model_operation", component="model", details=details
            ),
        )


class TrainingError(StateSetError):
    """Raised when training fails."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.TRN_STEP_FAILED,
        step: Optional[int] = None,
        **kwargs,
    ):
        details = {"step": step, **kwargs}
        super().__init__(
            message,
            code=code,
            context=ErrorContext(
                operation="training", component="trainer", details=details
            ),
        )


class OutOfMemoryError(TrainingError):
    """Raised when GPU/CPU memory is exhausted."""

    def __init__(
        self,
        message: str = "Out of memory",
        allocated_gb: Optional[float] = None,
        reserved_gb: Optional[float] = None,
        **kwargs,
    ):
        details = {"allocated_gb": allocated_gb, "reserved_gb": reserved_gb, **kwargs}
        super().__init__(
            message, code=ErrorCode.TRN_OOM, **details
        )


class EnvironmentError(StateSetError):
    """Raised when environment operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, code=ErrorCode.ENV_STEP_FAILED, context=ErrorContext(
                operation="environment", component="environment", details=kwargs
            )
        )


class DataError(StateSetError):
    """Raised when data loading or processing fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, code=ErrorCode.DAT_LOAD_FAILED, context=ErrorContext(
                operation="data_loading", component="data", details=kwargs
            )
        )


class RewardError(StateSetError):
    """Raised when reward computation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, code=ErrorCode.RWD_COMPUTE_FAILED, context=ErrorContext(
                operation="reward_computation", component="reward", details=kwargs
            )
        )


class NetworkError(StateSetError):
    """Raised when network operations fail."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.NET_CONNECTION_FAILED,
        url: Optional[str] = None,
        **kwargs,
    ):
        details = {"url": url, **kwargs}
        super().__init__(
            message,
            code=code,
            context=ErrorContext(
                operation="network", component="network", details=details
            ),
        )


class ValidationError(StateSetError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = {"field": field, **kwargs}
        super().__init__(
            message, code=ErrorCode.VAL_INVALID_INPUT, context=ErrorContext(
                operation="validation", component="validator", details=details
            )
        )


def wrap_exception(
    exc: BaseException,
    new_type: Type[StateSetError],
    message: Optional[str] = None,
    code: Optional[ErrorCode] = None,
    **context_kwargs,
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

    return new_type(
        msg,
        code=code or ErrorCode.UNKNOWN,
        context=context,
        cause=exc,
    )


def raise_from(
    new_exc: StateSetError, cause: BaseException
) -> None:
    """Raise a new exception from a cause with proper chaining.

    Args:
        new_exc: New exception to raise
        cause: Original cause
    """
    new_exc.cause = cause
    new_exc.context.traceback_str = traceback.format_exc()
    raise new_exc from cause
