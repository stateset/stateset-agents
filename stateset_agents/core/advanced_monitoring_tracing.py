"""Distributed tracing helpers for advanced monitoring."""

from __future__ import annotations

import asyncio
import atexit
import copy
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from stateset_agents.utils.async_calls import drain_owned_operation, run_sync_owned

from .advanced_monitoring_models import TraceSpan


class TracerState(str, Enum):
    """Local span admission and cleanup state."""

    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class DistributedTracer:
    """Local traces and an optional, privately owned OpenTelemetry pipeline.

    Use an instance on one event loop and await ``aclose()`` before that loop
    exits. Closing finishes unfinished observations as interrupted; it does not
    cancel the application operations they observe. SDK cleanup failures retain
    their handles and are not automatically retried.
    """

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
        self.tracer: Any | None = None
        self._opentelemetry_available = opentelemetry_available
        self._trace_module = trace_module
        self._jaeger_exporter_cls = jaeger_exporter_cls
        self._tracer_provider_cls = tracer_provider_cls
        self._batch_span_processor_cls = batch_span_processor_cls
        self._logger = logger
        self._handled_exceptions = handled_exceptions
        self._state = TracerState.READY
        self._provider: Any | None = None
        self._processor: Any | None = None
        self._exporter: Any | None = None
        self._processor_attached = False
        self._sdk_spans: dict[str, Any] = {}
        self._ending_spans: set[str] = set()
        self._end_errors: dict[str, BaseException] = {}
        self._setup_error: Exception | None = None
        self._sdk_shutdown_error: BaseException | None = None
        self._sdk_shutdown_attempted = False
        self._close_task: asyncio.Task[None] | None = None
        self._exit_hook: Callable[[], None] | None = None

        if self._opentelemetry_available and self._trace_module is not None:
            self._setup_opentelemetry()
            if any(
                value is not None
                for value in (self._provider, self._processor, self._exporter)
            ):
                self._exit_hook = self._close_at_exit
                atexit.register(self._exit_hook)

    def _setup_opentelemetry(self) -> None:
        """Create a private pipeline without mutating the global provider."""
        try:
            if (
                self._tracer_provider_cls is None
                or self._jaeger_exporter_cls is None
                or self._batch_span_processor_cls is None
            ):
                raise RuntimeError(
                    "OpenTelemetry provider, exporter, and processor are required"
                )
            self._provider = self._tracer_provider_cls(shutdown_on_exit=False)
            self._exporter = self._jaeger_exporter_cls(
                agent_host_name="localhost",
                agent_port=6831,
            )
            self._processor = self._batch_span_processor_cls(self._exporter)
            self._provider.add_span_processor(self._processor)
            self._processor_attached = True
            self.tracer = self._provider.get_tracer(self.service_name)
        except Exception as exc:
            # Local tracing remains available. Keep every returned SDK handle so
            # aclose can release partial setup without blocking this constructor.
            self._setup_error = exc
            if self._logger is not None:
                self._logger.warning("Failed to setup OpenTelemetry: %s", exc)

    @property
    def state(self) -> TracerState:
        """Observe local admission independently of SDK availability."""
        return self._state

    def get_status(self) -> dict[str, Any]:
        """Return lifecycle and retained ownership without exporting trace data."""
        return {
            "state": self._state.value,
            "active_spans": len(self.active_spans),
            "pending_sdk_spans": sorted(self._sdk_spans),
            "failed_span_ends": sorted(self._end_errors),
            "provider_owned": self._provider is not None,
            "processor_owned": self._processor is not None,
            "exporter_owned": self._exporter is not None,
            "sdk_available": self.tracer is not None
            and self._state is TracerState.READY,
            "setup_error_type": (
                type(self._setup_error).__name__ if self._setup_error else None
            ),
            "shutdown_error_type": (
                type(self._sdk_shutdown_error).__name__
                if self._sdk_shutdown_error
                else None
            ),
        }

    def _log_sdk_error(self, operation: str, error: BaseException) -> None:
        if self._logger is not None:
            self._logger.error(
                "OpenTelemetry %s failed (%s)", operation, type(error).__name__
            )

    @asynccontextmanager
    async def trace(
        self,
        operation_name: str,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ):
        """Create a trace span context manager."""
        span = self.start_span(operation_name, parent_span_id, tags)
        span_id = span.span_id
        try:
            yield span
        except asyncio.CancelledError:
            if span_id in self.active_spans:
                span.tags["cancelled"] = True
            raise
        except self._handled_exceptions as exc:
            if span_id in self.active_spans:
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
                sdk_span = self._sdk_spans.get(span_id)
                if sdk_span is not None:
                    try:
                        sdk_span.record_exception(exc)
                        if self._trace_module is not None:
                            sdk_span.set_status(
                                self._trace_module.Status(
                                    self._trace_module.StatusCode.ERROR, str(exc)
                                )
                            )
                    except Exception as error:
                        self._log_sdk_error("error annotation", error)
            raise
        finally:
            self.finish_span(span_id)

    def start_span(
        self,
        operation_name: str,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Start a new trace span."""
        if self._state is not TracerState.READY:
            raise RuntimeError(f"Tracer is {self._state.value}")
        parent_span = (
            self.active_spans.get(parent_span_id)
            if parent_span_id is not None
            else None
        )
        trace_id = (
            parent_span.trace_id if parent_span is not None else str(uuid.uuid4())
        )
        span_id = str(uuid.uuid4())

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=copy.deepcopy(tags) if tags is not None else {},
        )

        self.active_spans[span_id] = span

        if self.tracer is not None:
            try:
                sdk_parent = self._sdk_spans.get(parent_span_id or "")
                if sdk_parent is not None and self._trace_module is not None:
                    context = self._trace_module.set_span_in_context(sdk_parent)
                    otel_span = self.tracer.start_span(operation_name, context=context)
                else:
                    otel_span = self.tracer.start_span(operation_name)
                # Retain the span before annotations, which may fail independently.
                self._sdk_spans[span_id] = otel_span
                if span.tags:
                    for key, value in span.tags.items():
                        otel_span.set_attribute(key, value)
                sdk_context = otel_span.get_span_context()
                span.tags["otel_span_id"] = str(sdk_context.span_id)
                span.tags["otel_trace_id"] = str(sdk_context.trace_id)
            except Exception as exc:
                self._log_sdk_error("span creation", exc)

        return span

    def finish_span(self, span_id: str) -> None:
        """Finish local and SDK observations once, retaining failed SDK ends."""
        span = self.active_spans.pop(span_id, None)
        if span is not None:
            span.finish()
            self.completed_spans.append(span)
            if len(self.completed_spans) > 1000:
                self.completed_spans = self.completed_spans[-1000:]
        sdk_span = self._sdk_spans.get(span_id)
        if (
            sdk_span is None
            or span_id in self._ending_spans
            or span_id in self._end_errors
        ):
            return
        self._ending_spans.add(span_id)
        try:
            if span is not None:
                try:
                    for key in ("error", "error.message", "cancelled", "interrupted"):
                        if key in span.tags:
                            sdk_span.set_attribute(key, span.tags[key])
                except Exception as error:
                    self._log_sdk_error("completion annotation", error)
            try:
                sdk_span.end()
            except BaseException as error:
                self._end_errors[span_id] = error
                self._log_sdk_error("span end", error)
                if not isinstance(error, Exception):
                    raise
            else:
                self._sdk_spans.pop(span_id, None)
        finally:
            self._ending_spans.discard(span_id)

    def _shutdown_sdk(self) -> None:
        if self._sdk_shutdown_attempted:
            if self._sdk_shutdown_error is not None:
                raise self._sdk_shutdown_error
            return
        self._sdk_shutdown_attempted = True
        # An add_span_processor failure may occur before or after attachment.
        # In that uncertain case, close the processor directly and discard the
        # private provider; never ask both to repeat the same shutdown effect.
        owner = (
            self._provider
            if self._processor_attached
            else next(
                (
                    value
                    for value in (self._processor, self._exporter, self._provider)
                    if value is not None
                ),
                None,
            )
        )
        if owner is not None:
            try:
                owner.shutdown()
            except BaseException as error:
                self._sdk_shutdown_error = error
                raise
        self.tracer = None
        self._provider = self._processor = self._exporter = None

    def _unregister_exit_hook(self) -> None:
        if self._exit_hook is not None:
            atexit.unregister(self._exit_hook)
            self._exit_hook = None

    def _finish_unfinished_spans(self) -> BaseException | None:
        completion_error: BaseException | None = None
        for span_id, span in tuple(self.active_spans.items()):
            try:
                span.tags["interrupted"] = True
                self.finish_span(span_id)
            except BaseException as error:
                completion_error = completion_error or error
        # SDK ownership does not depend on callers retaining local history.
        for span_id in tuple(self._sdk_spans):
            try:
                self.finish_span(span_id)
            except BaseException as error:
                completion_error = completion_error or error
        return completion_error

    def _close_at_exit(self) -> None:
        """Preserve process-exit cleanup without replaying an explicit close."""
        if self._state is not TracerState.READY:
            return
        self._state = TracerState.CLOSING
        # Python leaves registry mutation during exit callbacks undefined.
        # Explicit aclose unregisters outside this callback instead.
        try:
            completion_error = self._finish_unfinished_spans()
            self._shutdown_sdk()
            if completion_error is not None:
                raise completion_error
            if self._end_errors:
                raise next(iter(self._end_errors.values()))
        except BaseException as error:
            self._state = TracerState.FAILED
            self._log_sdk_error("process-exit cleanup", error)
            if not isinstance(error, Exception):
                raise
        else:
            self._state = TracerState.CLOSED

    async def aclose(self) -> None:
        """End unfinished observations and join one owned SDK shutdown operation."""
        if self._close_task is asyncio.current_task():
            raise RuntimeError("Tracer cannot await its own shutdown")
        if self._close_task is None:
            self._state = TracerState.CLOSING
            self._unregister_exit_hook()
            operation = self._close()
            try:
                self._close_task = asyncio.create_task(operation)
            except BaseException:
                operation.close()
                raise

        async def join() -> None:
            assert self._close_task is not None
            await self._close_task

        await drain_owned_operation(join())

    async def _close(self) -> None:
        self._close_task = asyncio.current_task()
        try:
            completion_error = self._finish_unfinished_spans()
            await run_sync_owned(self._shutdown_sdk)
            if completion_error is not None:
                raise completion_error
            if self._end_errors:
                raise next(iter(self._end_errors.values()))
        except BaseException:
            self._state = TracerState.FAILED
            raise
        self._state = TracerState.CLOSED

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

        end_times = [span.end_time for span in spans if span.end_time is not None]
        total_duration = max(end_times) - min(span.start_time for span in spans)
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
                    "tags": copy.deepcopy(span.tags),
                }
                for span in spans
            ],
        }


__all__ = ["DistributedTracer", "TracerState"]
