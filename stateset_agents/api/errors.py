"""
API Error Handling Module

Unified error handling with consistent response format across all API services.
"""

import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# ============================================================================
# Error Codes
# ============================================================================

class ErrorCode(str, Enum):
    """Standardized error codes."""
    # Client errors (4xx)
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    CONFLICT = "CONFLICT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"

    # Business logic errors
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    INVALID_STATE = "INVALID_STATE"
    OPERATION_FAILED = "OPERATION_FAILED"
    TRAINING_FAILED = "TRAINING_FAILED"
    AGENT_ERROR = "AGENT_ERROR"


# ============================================================================
# Error Response Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Individual error detail."""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Human-readable error message")
    code: Optional[str] = Field(None, description="Machine-readable error code")


class ErrorResponse(BaseModel):
    """Standardized error response envelope."""
    error: Dict[str, Any] = Field(..., description="Error information")
    request_id: str = Field(..., description="Unique request identifier for tracing")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    path: str = Field(..., description="Request path that caused the error")
    documentation_url: Optional[str] = Field(None, description="Link to relevant documentation")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": [
                        {
                            "field": "prompts",
                            "message": "At least one prompt is required",
                            "code": "required"
                        }
                    ]
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-01-15T10:30:00.000Z",
                "path": "/api/v1/train"
            }
        }


# ============================================================================
# API Exceptions
# ============================================================================

class APIError(Exception):
    """Base API exception with consistent error information."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 500,
        details: Optional[List[ErrorDetail]] = None,
        documentation_url: Optional[str] = None,
        internal_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or []
        self.documentation_url = documentation_url
        self.internal_error = internal_error

    def to_response(self, request_id: str, path: str) -> Dict[str, Any]:
        """Convert to response dictionary."""
        response = {
            "error": {
                "code": self.code.value,
                "message": self.message,
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "path": path,
        }

        if self.details:
            response["error"]["details"] = [d.dict(exclude_none=True) for d in self.details]

        if self.documentation_url:
            response["documentation_url"] = self.documentation_url

        return response


class BadRequestError(APIError):
    """400 Bad Request."""
    def __init__(self, message: str, details: Optional[List[ErrorDetail]] = None):
        super().__init__(ErrorCode.BAD_REQUEST, message, 400, details)


class UnauthorizedError(APIError):
    """401 Unauthorized."""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(ErrorCode.UNAUTHORIZED, message, 401)


class ForbiddenError(APIError):
    """403 Forbidden."""
    def __init__(self, message: str = "Access denied"):
        super().__init__(ErrorCode.FORBIDDEN, message, 403)


class NotFoundError(APIError):
    """404 Not Found."""
    def __init__(self, resource: str, identifier: Optional[str] = None):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} '{identifier}' not found"
        super().__init__(ErrorCode.NOT_FOUND, message, 404)


class ConflictError(APIError):
    """409 Conflict."""
    def __init__(self, message: str):
        super().__init__(ErrorCode.CONFLICT, message, 409)


class ValidationError(APIError):
    """422 Validation Error."""
    def __init__(self, message: str, details: Optional[List[ErrorDetail]] = None):
        super().__init__(ErrorCode.VALIDATION_ERROR, message, 422, details)


class RateLimitError(APIError):
    """429 Rate Limit Exceeded."""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            ErrorCode.RATE_LIMIT_EXCEEDED,
            f"Rate limit exceeded. Retry after {retry_after} seconds.",
            429,
        )
        self.retry_after = retry_after


class InternalError(APIError):
    """500 Internal Server Error."""
    def __init__(self, message: str = "An internal error occurred", internal_error: Optional[Exception] = None):
        super().__init__(ErrorCode.INTERNAL_ERROR, message, 500, internal_error=internal_error)


class ServiceUnavailableError(APIError):
    """503 Service Unavailable."""
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(ErrorCode.SERVICE_UNAVAILABLE, message, 503)


class TrainingError(APIError):
    """Training-specific error."""
    def __init__(self, message: str, details: Optional[List[ErrorDetail]] = None):
        super().__init__(ErrorCode.TRAINING_FAILED, message, 500, details)


class AgentError(APIError):
    """Agent-specific error."""
    def __init__(self, message: str, details: Optional[List[ErrorDetail]] = None):
        super().__init__(ErrorCode.AGENT_ERROR, message, 500, details)


# ============================================================================
# Domain-Specific Exceptions
# ============================================================================

class AgentNotFoundError(NotFoundError):
    """Agent with given ID not found."""
    def __init__(self, agent_id: str):
        super().__init__("Agent", agent_id)
        self.agent_id = agent_id


class InvalidAgentConfigError(BadRequestError):
    """Agent configuration is invalid."""
    def __init__(self, details: str, validation_errors: Optional[List[ErrorDetail]] = None):
        super().__init__(f"Invalid agent configuration: {details}", validation_errors)


class ConversationNotFoundError(NotFoundError):
    """Conversation not found."""
    def __init__(self, conversation_id: str):
        super().__init__("Conversation", conversation_id)
        self.conversation_id = conversation_id


class TrainingJobNotFoundError(NotFoundError):
    """Training job not found."""
    def __init__(self, job_id: str):
        super().__init__("Training job", job_id)
        self.job_id = job_id


class TrainingConfigError(BadRequestError):
    """Training configuration is invalid."""
    def __init__(self, details: str, validation_errors: Optional[List[ErrorDetail]] = None):
        super().__init__(f"Invalid training configuration: {details}", validation_errors)


class TokenizationError(InternalError):
    """Failed to tokenize text."""
    def __init__(self, message: str = "Failed to calculate token count"):
        super().__init__(message)


class ModelLoadError(InternalError):
    """Failed to load model."""
    def __init__(self, model_name: str, reason: Optional[str] = None):
        message = f"Failed to load model '{model_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)
        self.model_name = model_name


class PromptInjectionError(BadRequestError):
    """Potential prompt injection detected."""
    def __init__(self, field: str = "input"):
        super().__init__(
            f"Request rejected: potentially harmful content detected in {field}",
            details=[ErrorDetail(
                field=field,
                message="Content appears to contain prompt injection attempt",
                code="prompt_injection"
            )]
        )


class ResourceExhaustedError(APIError):
    """Resource limit exceeded."""
    def __init__(self, resource: str, limit: int):
        super().__init__(
            ErrorCode.RESOURCE_EXHAUSTED,
            f"Resource limit exceeded for {resource}. Maximum allowed: {limit}",
            429
        )
        self.resource = resource
        self.limit = limit


class AuthLockoutError(APIError):
    """Account temporarily locked due to too many failed attempts."""
    def __init__(self, retry_after_seconds: int):
        super().__init__(
            ErrorCode.FORBIDDEN,
            f"Account temporarily locked due to too many failed authentication attempts. "
            f"Please try again in {retry_after_seconds} seconds.",
            403
        )
        self.retry_after_seconds = retry_after_seconds


# ============================================================================
# Error Response Builder
# ============================================================================

def build_error_response(
    request: Request,
    status_code: int,
    code: ErrorCode,
    message: str,
    details: Optional[List[Dict[str, Any]]] = None,
    documentation_url: Optional[str] = None,
) -> JSONResponse:
    """Build a standardized error response."""
    request_id = getattr(request.state, "request_id", "unknown")

    response_body = {
        "error": {
            "code": code.value,
            "message": message,
        },
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "path": str(request.url.path),
    }

    if details:
        response_body["error"]["details"] = details

    if documentation_url:
        response_body["documentation_url"] = documentation_url

    return JSONResponse(status_code=status_code, content=response_body)


# ============================================================================
# Exception Handlers
# ============================================================================

async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the error
    if exc.status_code >= 500:
        logger.error(
            f"API error: {exc.code.value} - {exc.message}",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "status_code": exc.status_code,
                "error_code": exc.code.value,
            },
            exc_info=exc.internal_error,
        )
    else:
        logger.warning(
            f"API error: {exc.code.value} - {exc.message}",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "status_code": exc.status_code,
                "error_code": exc.code.value,
            },
        )

    response = exc.to_response(request_id, str(request.url.path))

    headers = {}
    if isinstance(exc, RateLimitError):
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=exc.status_code,
        content=response,
        headers=headers,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Map status codes to error codes
    status_to_code = {
        400: ErrorCode.BAD_REQUEST,
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        405: ErrorCode.METHOD_NOT_ALLOWED,
        409: ErrorCode.CONFLICT,
        422: ErrorCode.VALIDATION_ERROR,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
    }

    code = status_to_code.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
    message = str(exc.detail) if exc.detail else "An error occurred"

    response = build_error_response(
        request=request,
        status_code=exc.status_code,
        code=code,
        message=message,
    )
    if exc.headers:
        response.headers.update(exc.headers)
    return response


async def starlette_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle Starlette HTTPException (e.g. 404/405 for missing routes)."""
    status_to_code = {
        400: ErrorCode.BAD_REQUEST,
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        405: ErrorCode.METHOD_NOT_ALLOWED,
        409: ErrorCode.CONFLICT,
        422: ErrorCode.VALIDATION_ERROR,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
    }

    code = status_to_code.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
    message = str(exc.detail) if exc.detail else "An error occurred"

    response = build_error_response(
        request=request,
        status_code=exc.status_code,
        code=code,
        message=message,
    )
    if exc.headers:
        response.headers.update(exc.headers)
    return response


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Convert Pydantic errors to our format
    details = []
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        details.append({
            "field": field_path,
            "message": error["msg"],
            "code": error["type"],
        })

    logger.warning(
        "Validation error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "errors": details,
        },
    )

    return build_error_response(
        request=request,
        status_code=422,
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details=details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the full traceback
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "exception_type": type(exc).__name__,
        },
        exc_info=True,
    )

    # Don't expose internal details in production
    return build_error_response(
        request=request,
        status_code=500,
        code=ErrorCode.INTERNAL_ERROR,
        message="An internal error occurred. Please try again later.",
    )


# ============================================================================
# Setup Function
# ============================================================================

def setup_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers."""
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    logger.info("Exception handlers configured successfully")
