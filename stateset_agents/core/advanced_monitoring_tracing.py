"""Distributed tracing helpers for advanced monitoring."""

from __future__ import annotations

import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any

from .advanced_monitoring_models import TraceSpan


class DistributedTracer:
    """Distributed tracing implementation."""

    def __init__(
        self,
        service_name: str = "grpo-service",
        *,
        opentelemetry_available: bool = False,
        trace_module: Any | None = None,
        jaeger_exporter_cls: Any | None = None,
        tracer_provider_cls: Any | None = None,
        batch_span_processor_cls: Any | None = None,
        logger: Any | None = None,
        handled_exceptions: tuple[type[BaseException], ...] = (Exception,),
    ):
        self.service_name = service_name
        self.active_spans: dict[str, TraceSpan] = {}
        self.completed_spans: list[TraceSpan] = []
        self.tracer = None
        self._opentelemetry_available = opentelemetry_available
        self._trace_module = trace_module
        self._jaeger_exporter_cls = jaeger_exporter_cls
        self._tracer_provider_cls = tracer_provider_cls
        self._batch_span_processor_cls = batch_span_processor_cls
        self._logger = logger
        self._handled_exceptions = handled_exceptions

        if self._opentelemetry_available and self._trace_module is not None:
            self._setup_opentelemetry()

    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing."""
        try:
            self._trace_module.set_tracer_provider(self._tracer_provider_cls())
            jaeger_exporter = self._jaeger_exporter_cls(
                agent_host_name="localhost",
                agent_port=6831,
            )
            span_processor = self._batch_span_processor_cls(jaeger_exporter)
            self._trace_module.get_tracer_provider().add_span_processor(span_processor)
            self.tracer = self._trace_module.get_tracer(self.service_name)
        except self._handled_exceptions as exc:
            if self._logger is not None:
                self._logger.warning("Failed to setup OpenTelemetry: %s", exc)

    @asynccontextmanager
    async def trace(
        self,
        operation_name: str,
        parent_span_id: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Create a trace span context manager."""
        span = self.start_span(operation_name, parent_span_id, tags)
        try:
            yield span
        except self._handled_exceptions as exc:
            span.tags["error"] = True
            span.tags["error.message"] = str(exc)
            span.logs.append(
                {
                    "timestamp": time.time(),
                    "level": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            raise
        finally:
            self.finish_span(span.span_id)

    def start_span(
        self,
        operation_name: str,
        parent_span_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> TraceSpan:
        """Start a new trace span."""
        parent_span = (
            self.active_spans.get(parent_span_id) if parent_span_id is not None else None
        )
        trace_id = parent_span.trace_id if parent_span is not None else str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags or {},
        )

        self.active_spans[span_id] = span

        if self.tracer:
            try:
                otel_span = self.tracer.start_span(operation_name)
                if tags:
                    for key, value in tags.items():
                        otel_span.set_attribute(key, value)
                span.tags["otel_span_id"] = str(otel_span.get_span_context().span_id)
            except self._handled_exceptions as exc:
                if self._logger is not None:
                    self._logger.error("OpenTelemetry span creation failed: %s", exc)

        return span

    def finish_span(self, span_id: str):
        """Finish a trace span."""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.finish()
            self.completed_spans.append(span)
            del self.active_spans[span_id]

            if len(self.completed_spans) > 1000:
                self.completed_spans = self.completed_spans[-1000:]

    def add_span_log(self, span_id: str, level: str, message: str, **kwargs):
        """Add a log event to an active span."""
        if span_id in self.active_spans:
            self.active_spans[span_id].logs.append(
                {"timestamp": time.time(), "level": level, "message": message, **kwargs}
            )

    def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """Get a summary for a completed trace id."""
        spans = [span for span in self.completed_spans if span.trace_id == trace_id]
        if not spans:
            return {}

        total_duration = max(span.end_time for span in spans) - min(
            span.start_time for span in spans
        )
        return {
            "trace_id": trace_id,
            "total_duration": total_duration,
            "span_count": len(spans),
            "operations": [span.operation_name for span in spans],
            "spans": [
                {
                    "span_id": span.span_id,
                    "operation": span.operation_name,
                    "duration": span.duration,
                    "tags": span.tags,
                }
                for span in spans
            ],
        }


__all__ = ["DistributedTracer"]
