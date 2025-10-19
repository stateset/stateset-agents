"""
Observability and Tracing System for GRPO Agent Framework

This module provides comprehensive observability including distributed tracing,
metrics collection, log correlation, and performance monitoring.
"""

import asyncio
import functools
import json
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Optional imports for enhanced functionality
try:
    from opentelemetry import baggage, metrics, trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.asyncio import AsyncIOInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class SpanKind(Enum):
    """Span kinds for different operation types"""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class TraceLevel(Enum):
    """Trace detail levels"""

    MINIMAL = "minimal"
    NORMAL = "normal"
    DETAILED = "detailed"
    VERBOSE = "verbose"


@dataclass
class SpanContext:
    """Span context information"""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
        }


@dataclass
class Span:
    """Span data structure"""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None

    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class TraceData:
    """Complete trace data"""

    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    def __post_init__(self):
        if self.spans:
            self.start_time = min(span.start_time for span in self.spans)
            end_times = [span.end_time for span in self.spans if span.end_time]
            if end_times:
                self.end_time = max(end_times)
                self.duration_ms = (
                    self.end_time - self.start_time
                ).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "trace_id": self.trace_id,
            "spans": [span.to_dict() for span in self.spans],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
        }


class ObservabilityTracer:
    """Custom tracer for observability"""

    def __init__(
        self,
        service_name: str = "stateset_agents",
        enable_jaeger: bool = False,
        enable_otlp: bool = False,
        jaeger_endpoint: str = "http://localhost:14268/api/traces",
        otlp_endpoint: str = "http://localhost:4317",
        trace_level: TraceLevel = TraceLevel.NORMAL,
    ):
        self.service_name = service_name
        self.trace_level = trace_level
        self.active_spans: Dict[str, Span] = {}
        self.completed_traces: deque = deque(maxlen=1000)
        self.current_context = threading.local()

        # Initialize OpenTelemetry if available
        if HAS_OPENTELEMETRY:
            self._setup_opentelemetry(
                enable_jaeger, enable_otlp, jaeger_endpoint, otlp_endpoint
            )

        # Performance metrics
        self.metrics = {
            "total_spans": 0,
            "active_spans": 0,
            "completed_traces": 0,
            "average_span_duration": 0.0,
            "spans_by_operation": defaultdict(int),
            "error_spans": 0,
        }

    def _setup_opentelemetry(
        self,
        enable_jaeger: bool,
        enable_otlp: bool,
        jaeger_endpoint: str,
        otlp_endpoint: str,
    ):
        """Setup OpenTelemetry tracing"""
        trace.set_tracer_provider(TracerProvider())

        # Setup exporters
        if enable_jaeger:
            jaeger_exporter = JaegerExporter(endpoint=jaeger_endpoint)
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)

        if enable_otlp:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)

        self.tracer = trace.get_tracer(self.service_name)

        # Setup metrics
        metrics.set_meter_provider(MeterProvider())
        self.meter = metrics.get_meter(self.service_name)

        # Initialize instruments
        self.span_counter = self.meter.create_counter(
            "spans_total", description="Total number of spans"
        )
        self.span_duration = self.meter.create_histogram(
            "span_duration_ms", description="Span duration in milliseconds"
        )

    def _generate_ids(self) -> tuple:
        """Generate trace and span IDs"""
        return str(uuid.uuid4()), str(uuid.uuid4())

    def _get_current_span(self) -> Optional[Span]:
        """Get current active span"""
        if not hasattr(self.current_context, "span_stack"):
            return None

        if not self.current_context.span_stack:
            return None

        return self.current_context.span_stack[-1]

    def _push_span(self, span: Span):
        """Push span to context stack"""
        if not hasattr(self.current_context, "span_stack"):
            self.current_context.span_stack = []

        self.current_context.span_stack.append(span)

    def _pop_span(self) -> Optional[Span]:
        """Pop span from context stack"""
        if not hasattr(self.current_context, "span_stack"):
            return None

        if not self.current_context.span_stack:
            return None

        return self.current_context.span_stack.pop()

    def start_span(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_context: Optional[SpanContext] = None,
        tags: Dict[str, Any] = None,
    ) -> Span:
        """Start a new span"""
        current_span = self._get_current_span()

        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        elif current_span:
            trace_id = current_span.trace_id
            parent_span_id = current_span.span_id
        else:
            trace_id, _ = self._generate_ids()
            parent_span_id = None

        _, span_id = self._generate_ids()

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            kind=kind,
            start_time=datetime.now(),
            tags=tags or {},
        )

        # Add to active spans
        self.active_spans[span_id] = span
        self._push_span(span)

        # Update metrics
        self.metrics["total_spans"] += 1
        self.metrics["active_spans"] += 1
        self.metrics["spans_by_operation"][operation_name] += 1

        # OpenTelemetry integration
        if HAS_OPENTELEMETRY and hasattr(self, "span_counter"):
            self.span_counter.add(1, {"operation": operation_name, "kind": kind.value})

        return span

    def finish_span(self, span: Span, error: Optional[Exception] = None):
        """Finish a span"""
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000

        if error:
            span.status = "error"
            span.error = str(error)
            span.tags["error"] = True
            span.tags["error.message"] = str(error)
            span.tags["error.type"] = type(error).__name__
            self.metrics["error_spans"] += 1

        # Remove from active spans
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]

        self._pop_span()

        # Update metrics
        self.metrics["active_spans"] -= 1

        # Update average duration
        total_duration = self.metrics["average_span_duration"] * (
            self.metrics["total_spans"] - 1
        )
        self.metrics["average_span_duration"] = (
            total_duration + span.duration_ms
        ) / self.metrics["total_spans"]

        # OpenTelemetry integration
        if HAS_OPENTELEMETRY and hasattr(self, "span_duration"):
            self.span_duration.record(
                span.duration_ms,
                {"operation": span.operation_name, "kind": span.kind.value},
            )

        # Check if trace is complete
        self._check_trace_completion(span.trace_id)

    def _check_trace_completion(self, trace_id: str):
        """Check if trace is complete and store it"""
        # Find all spans for this trace
        trace_spans = [
            span for span in self.active_spans.values() if span.trace_id == trace_id
        ]

        # If no active spans for this trace, it's complete
        if not trace_spans:
            # Find all completed spans for this trace (would need to store somewhere)
            # For now, we'll just track completion
            self.metrics["completed_traces"] += 1

    def add_span_tag(self, span: Span, key: str, value: Any):
        """Add tag to span"""
        span.tags[key] = value

    def add_span_log(self, span: Span, message: str, level: str = "info", **kwargs):
        """Add log to span"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs,
        }
        span.logs.append(log_entry)

    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        current_span = self._get_current_span()
        return current_span.trace_id if current_span else None

    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID"""
        current_span = self._get_current_span()
        return current_span.span_id if current_span else None

    def get_span_context(self) -> Optional[SpanContext]:
        """Get current span context"""
        current_span = self._get_current_span()
        if not current_span:
            return None

        return SpanContext(
            trace_id=current_span.trace_id,
            span_id=current_span.span_id,
            parent_span_id=current_span.parent_span_id,
        )

    @contextmanager
    def trace(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Dict[str, Any] = None,
    ):
        """Context manager for tracing"""
        span = self.start_span(operation_name, kind, tags=tags)

        try:
            yield span
        except Exception as e:
            self.finish_span(span, error=e)
            raise
        else:
            self.finish_span(span)

    @asynccontextmanager
    async def async_trace(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Dict[str, Any] = None,
    ):
        """Async context manager for tracing"""
        span = self.start_span(operation_name, kind, tags=tags)

        try:
            yield span
        except Exception as e:
            self.finish_span(span, error=e)
            raise
        else:
            self.finish_span(span)

    def get_metrics(self) -> Dict[str, Any]:
        """Get tracer metrics"""
        return {
            "total_spans": self.metrics["total_spans"],
            "active_spans": self.metrics["active_spans"],
            "completed_traces": self.metrics["completed_traces"],
            "average_span_duration_ms": self.metrics["average_span_duration"],
            "spans_by_operation": dict(self.metrics["spans_by_operation"]),
            "error_spans": self.metrics["error_spans"],
            "error_rate": self.metrics["error_spans"] / self.metrics["total_spans"]
            if self.metrics["total_spans"] > 0
            else 0,
        }


class ObservabilityManager:
    """Main observability manager"""

    def __init__(
        self,
        service_name: str = "stateset_agents",
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        enable_logging: bool = True,
        trace_level: TraceLevel = TraceLevel.NORMAL,
    ):
        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.enable_logging = enable_logging

        # Initialize tracer
        self.tracer = (
            ObservabilityTracer(service_name=service_name, trace_level=trace_level)
            if enable_tracing
            else None
        )

        # Initialize structured logger
        if enable_logging and HAS_STRUCTLOG:
            self._setup_structured_logging()

        # Performance tracking
        self.performance_metrics = {
            "api_requests": defaultdict(list),
            "training_iterations": [],
            "conversation_metrics": defaultdict(list),
            "system_metrics": defaultdict(list),
        }

    def _setup_structured_logging(self):
        """Setup structured logging with trace correlation"""

        def add_trace_context(_, __, event_dict):
            """Add trace context to log entries"""
            if self.tracer:
                trace_id = self.tracer.get_current_trace_id()
                span_id = self.tracer.get_current_span_id()

                if trace_id:
                    event_dict["trace_id"] = trace_id
                if span_id:
                    event_dict["span_id"] = span_id

            return event_dict

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                add_trace_context,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger(self.service_name)

    def trace_api_request(self, method: str, path: str, user_id: Optional[str] = None):
        """Trace API request"""
        if not self.tracer:
            return contextmanager(lambda: (yield))()

        tags = {"http.method": method, "http.url": path, "component": "api"}

        if user_id:
            tags["user.id"] = user_id

        return self.tracer.trace(f"API {method} {path}", SpanKind.SERVER, tags)

    def trace_training_iteration(self, iteration: int, batch_size: int):
        """Trace training iteration"""
        if not self.tracer:
            return contextmanager(lambda: (yield))()

        tags = {
            "training.iteration": iteration,
            "training.batch_size": batch_size,
            "component": "training",
        }

        return self.tracer.trace(
            f"Training Iteration {iteration}", SpanKind.INTERNAL, tags
        )

    def trace_conversation(self, conversation_id: str, user_id: Optional[str] = None):
        """Trace conversation"""
        if not self.tracer:
            return contextmanager(lambda: (yield))()

        tags = {"conversation.id": conversation_id, "component": "conversation"}

        if user_id:
            tags["user.id"] = user_id

        return self.tracer.trace(
            f"Conversation {conversation_id}", SpanKind.INTERNAL, tags
        )

    def trace_reward_computation(self, reward_type: str, conversation_id: str):
        """Trace reward computation"""
        if not self.tracer:
            return contextmanager(lambda: (yield))()

        tags = {
            "reward.type": reward_type,
            "conversation.id": conversation_id,
            "component": "reward",
        }

        return self.tracer.trace(
            f"Reward Computation {reward_type}", SpanKind.INTERNAL, tags
        )

    def record_performance_metric(
        self, category: str, operation: str, duration_ms: float, **metadata
    ):
        """Record performance metric"""
        metric_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        }

        self.performance_metrics[category].append(metric_data)

        # Keep only recent metrics
        if len(self.performance_metrics[category]) > 1000:
            self.performance_metrics[category] = self.performance_metrics[category][
                -1000:
            ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}

        for category, metrics in self.performance_metrics.items():
            if not metrics:
                continue

            durations = [m["duration_ms"] for m in metrics]

            summary[category] = {
                "total_operations": len(metrics),
                "average_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)]
                if durations
                else 0,
                "p99_duration_ms": sorted(durations)[int(len(durations) * 0.99)]
                if durations
                else 0,
            }

        return summary

    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get trace statistics"""
        if not self.tracer:
            return {"tracing": "disabled"}

        return self.tracer.get_metrics()

    def get_observability_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary"""
        return {
            "service_name": self.service_name,
            "timestamp": datetime.now().isoformat(),
            "tracing": self.get_trace_statistics(),
            "performance": self.get_performance_summary(),
            "features": {
                "tracing_enabled": self.enable_tracing,
                "metrics_enabled": self.enable_metrics,
                "logging_enabled": self.enable_logging,
                "opentelemetry_available": HAS_OPENTELEMETRY,
                "structlog_available": HAS_STRUCTLOG,
            },
        }


# Global observability instance
_global_observability: Optional[ObservabilityManager] = None


def get_observability() -> ObservabilityManager:
    """Get global observability manager"""
    global _global_observability

    if _global_observability is None:
        _global_observability = ObservabilityManager()

    return _global_observability


def setup_observability(
    service_name: str = "stateset_agents",
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_logging: bool = True,
    trace_level: TraceLevel = TraceLevel.NORMAL,
) -> ObservabilityManager:
    """Setup global observability"""
    global _global_observability

    _global_observability = ObservabilityManager(
        service_name=service_name,
        enable_tracing=enable_tracing,
        enable_metrics=enable_metrics,
        enable_logging=enable_logging,
        trace_level=trace_level,
    )

    return _global_observability


def trace_function(
    operation_name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    tags: Dict[str, Any] = None,
):
    """Decorator for tracing functions"""

    def decorator(func):
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                observability = get_observability()
                if not observability.tracer:
                    return await func(*args, **kwargs)

                async with observability.tracer.async_trace(op_name, kind, tags):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                observability = get_observability()
                if not observability.tracer:
                    return func(*args, **kwargs)

                with observability.tracer.trace(op_name, kind, tags):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
