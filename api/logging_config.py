"""
API Logging Configuration

Provides structured logging with request ID context.
"""

import contextvars
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Context variable for request ID
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestIDFilter(logging.Filter):
    """
    Logging filter that adds request_id to log records.

    This filter extracts the request ID from the context variable
    and adds it to the log record for structured logging.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to the log record."""
        record.request_id = request_id_ctx.get() or "-"
        return True


class StructuredFormatter(logging.Formatter):
    """
    Structured log formatter with consistent fields.

    Outputs logs in a consistent format suitable for log aggregation.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_request_id: bool = True,
        json_format: bool = False,
    ):
        """
        Initialize formatter.

        Args:
            include_timestamp: Include ISO timestamp in output.
            include_request_id: Include request ID in output.
            json_format: Output as JSON (for machine parsing).
        """
        self.include_timestamp = include_timestamp
        self.include_request_id = include_request_id
        self.json_format = json_format
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        if self.json_format:
            return self._format_json(record)
        return self._format_text(record)

    def _format_text(self, record: logging.LogRecord) -> str:
        """Format as human-readable text."""
        parts = []

        if self.include_timestamp:
            timestamp = datetime.utcnow().isoformat()
            parts.append(f"[{timestamp}]")

        parts.append(f"[{record.levelname}]")

        if self.include_request_id:
            request_id = getattr(record, "request_id", "-")
            parts.append(f"[{request_id}]")

        parts.append(f"{record.name}:")
        parts.append(record.getMessage())

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            parts.append(f"\n{exc_text}")

        return " ".join(parts)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Format as JSON."""
        import json

        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_request_id:
            log_data["request_id"] = getattr(record, "request_id", None)

        # Add extra fields
        for key in ("user_id", "path", "method", "status_code", "latency_ms"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_request_id: bool = True,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Output logs as JSON.
        include_request_id: Include request ID in logs.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Add request ID filter
    if include_request_id:
        console_handler.addFilter(RequestIDFilter())

    # Set formatter
    formatter = StructuredFormatter(
        include_timestamp=True,
        include_request_id=include_request_id,
        json_format=json_format,
    )
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def set_request_id(request_id: str) -> contextvars.Token:
    """
    Set the request ID for the current context.

    Args:
        request_id: The request ID to set.

    Returns:
        Token for resetting the context.
    """
    return request_id_ctx.set(request_id)


def get_request_id() -> Optional[str]:
    """
    Get the current request ID.

    Returns:
        The current request ID, or None if not set.
    """
    return request_id_ctx.get()


def clear_request_id(token: contextvars.Token) -> None:
    """
    Clear the request ID from context.

    Args:
        token: Token from set_request_id.
    """
    request_id_ctx.reset(token)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes request ID.

    Use this adapter when you need to add extra context to logs.
    """

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        """Initialize adapter with optional extra fields."""
        super().__init__(logger, extra or {})

    def process(
        self, msg: str, kwargs: Dict[str, Any]
    ) -> tuple:
        """Process log message and add context."""
        extra = kwargs.get("extra", {})
        extra["request_id"] = get_request_id()
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> LoggerAdapter:
    """
    Get a logger with request ID context.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger adapter with request ID support.
    """
    return LoggerAdapter(logging.getLogger(name))
