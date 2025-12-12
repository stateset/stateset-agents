"""
API Observability Module

OpenTelemetry tracing, structured logging, and metrics for production observability.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional, Callable
from functools import wraps

from .config import get_config

# Context variable for request tracing
_trace_context: ContextVar[Dict[str, Any]] = ContextVar("trace_context", default={})


# ============================================================================
# Structured Logging
# ============================================================================

class StructuredLogFormatter(logging.Formatter):
    """JSON structured log formatter for production logging."""

    def __init__(self, service_name: str = "stateset-agents-api"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get trace context
        trace_ctx = _trace_context.get()

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        # Add trace context if available
        if trace_ctx:
            log_entry["trace_id"] = trace_ctx.get("trace_id")
            log_entry["span_id"] = trace_ctx.get("span_id")
            log_entry["request_id"] = trace_ctx.get("request_id")

        # Add extra fields from record
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "path"):
            log_entry["path"] = record.path
        if hasattr(record, "method"):
            log_entry["method"] = record.method
        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code
        if hasattr(record, "elapsed_ms"):
            log_entry["elapsed_ms"] = record.elapsed_ms

        # Add any extra dict passed to logger
        extra = getattr(record, "__dict__", {})
        for key in ["client_ip", "user_agent", "error_code", "job_id", "conversation_id"]:
            if key in extra:
                log_entry[key] = extra[key]

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }

        # Add source location
        log_entry["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_entry, default=str)


class ConsoleLogFormatter(logging.Formatter):
    """Human-readable console formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Get trace context
        trace_ctx = _trace_context.get()
        request_id = trace_ctx.get("request_id", getattr(record, "request_id", ""))

        prefix = f"{timestamp} {color}{record.levelname:8}{self.RESET}"
        if request_id:
            prefix += f" [{request_id[:8]}]"

        message = record.getMessage()

        # Add extra context
        extras = []
        for key in ["path", "method", "status_code", "elapsed_ms", "user_id"]:
            if hasattr(record, key):
                extras.append(f"{key}={getattr(record, key)}")

        if extras:
            message += f" ({', '.join(extras)})"

        return f"{prefix} {record.name}: {message}"


def setup_logging(
    service_name: str = "stateset-agents-api",
    log_level: str = "INFO",
    log_format: str = "json",
) -> None:
    """
    Configure structured logging for the application.

    Args:
        service_name: Service name for log entries
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('json' for production, 'console' for development)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Set formatter based on format
    if log_format.lower() == "json":
        handler.setFormatter(StructuredLogFormatter(service_name))
    else:
        handler.setFormatter(ConsoleLogFormatter())

    root_logger.addHandler(handler)

    # Reduce noise from libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============================================================================
# Tracing
# ============================================================================

class Span:
    """Simple span for tracing operations."""

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ):
        self.name = name
        self.trace_id = trace_id or uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_span_id = parent_span_id
        self.start_time = time.monotonic()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.events: list = []
        self.status: str = "OK"
        self.status_message: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a span attribute."""
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.monotonic(),
            "attributes": attributes or {},
        })
        return self

    def set_status(self, status: str, message: Optional[str] = None) -> "Span":
        """Set span status."""
        self.status = status
        self.status_message = message
        return self

    def end(self) -> None:
        """End the span."""
        self.end_time = time.monotonic()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.monotonic()
        return (end - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for logging/export."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "status_message": self.status_message,
        }


class Tracer:
    """Simple tracer for creating and managing spans."""

    def __init__(self, service_name: str = "stateset-agents-api"):
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
        self._enabled = False

    def enable(self) -> None:
        """Enable tracing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable tracing."""
        self._enabled = False

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """Start a new span."""
        span = Span(name, trace_id, parent_span_id)

        if self._enabled:
            # Update context
            ctx = _trace_context.get().copy()
            ctx["trace_id"] = span.trace_id
            ctx["span_id"] = span.span_id
            _trace_context.set(ctx)

        return span

    def end_span(self, span: Span) -> None:
        """End a span and log it."""
        span.end()

        if self._enabled:
            self.logger.debug(
                f"Span completed: {span.name}",
                extra={
                    "span": span.to_dict(),
                }
            )


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
        config = get_config()
        if config.observability.enable_tracing:
            _tracer.enable()
    return _tracer


# ============================================================================
# Tracing Decorator
# ============================================================================

def trace(name: Optional[str] = None):
    """
    Decorator to trace a function.

    Usage:
        @trace("my_operation")
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            ctx = _trace_context.get()

            span = tracer.start_span(
                span_name,
                trace_id=ctx.get("trace_id"),
                parent_span_id=ctx.get("span_id"),
            )

            try:
                result = await func(*args, **kwargs)
                span.set_status("OK")
                return result
            except Exception as e:
                span.set_status("ERROR", str(e))
                raise
            finally:
                tracer.end_span(span)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            ctx = _trace_context.get()

            span = tracer.start_span(
                span_name,
                trace_id=ctx.get("trace_id"),
                parent_span_id=ctx.get("span_id"),
            )

            try:
                result = func(*args, **kwargs)
                span.set_status("OK")
                return result
            except Exception as e:
                span.set_status("ERROR", str(e))
                raise
            finally:
                tracer.end_span(span)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================================
# Context Management
# ============================================================================

def set_trace_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """Set trace context for the current request."""
    ctx = {
        "request_id": request_id or uuid.uuid4().hex,
        "trace_id": trace_id or uuid.uuid4().hex,
        "span_id": span_id or uuid.uuid4().hex[:16],
    }
    if user_id:
        ctx["user_id"] = user_id

    _trace_context.set(ctx)


def get_trace_context() -> Dict[str, Any]:
    """Get current trace context."""
    return _trace_context.get()


def clear_trace_context() -> None:
    """Clear trace context."""
    _trace_context.set({})


# ============================================================================
# OpenTelemetry Integration (Optional)
# ============================================================================

def setup_opentelemetry(
    service_name: str = "stateset-agents-api",
    endpoint: Optional[str] = None,
) -> bool:
    """
    Set up OpenTelemetry tracing (if available).

    Returns True if OpenTelemetry was configured successfully.
    """
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource

        # Set up resource
        resource = Resource(attributes={SERVICE_NAME: service_name})

        # Set up tracer provider
        provider = TracerProvider(resource=resource)

        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                logging.warning("OTLP exporter not available, using console exporter")
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        otel_trace.set_tracer_provider(provider)

        logging.info(f"OpenTelemetry configured for {service_name}")
        return True

    except ImportError:
        logging.debug("OpenTelemetry not available, using simple tracer")
        return False


# ============================================================================
# Initialization
# ============================================================================

def setup_observability() -> None:
    """Set up all observability components based on configuration."""
    config = get_config()

    # Set up logging
    setup_logging(
        service_name="stateset-agents-api",
        log_level=config.observability.log_level,
        log_format=config.observability.log_format,
    )

    # Set up tracing
    if config.observability.enable_tracing:
        if config.observability.tracing_endpoint:
            setup_opentelemetry(
                service_name="stateset-agents-api",
                endpoint=config.observability.tracing_endpoint,
            )
        get_tracer().enable()

    logging.info(
        "Observability configured",
        extra={
            "tracing_enabled": config.observability.enable_tracing,
            "prometheus_enabled": config.observability.enable_prometheus,
            "log_level": config.observability.log_level,
            "log_format": config.observability.log_format,
        }
    )
