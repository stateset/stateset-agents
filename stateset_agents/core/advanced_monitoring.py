"""
Advanced Monitoring and Observability for GRPO Agent Framework

This module provides comprehensive monitoring, metrics collection, distributed tracing,
and observability features for production-ready GRPO services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from functools import wraps
from typing import Any

import psutil

from stateset_agents.utils.async_calls import drain_owned_operation, run_sync_owned

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    CollectorRegistry = Counter = Gauge = Histogram = Summary = None
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)

from .advanced_monitoring_models import (
    Alert,
    AlertSeverity,
    MetricPoint,
    MetricType,
    TraceSpan,
)
from .advanced_monitoring_tracing import DistributedTracer

ADV_MONITORING_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    OSError,
    psutil.Error,
    asyncio.TimeoutError,
)

PROMETHEUS_METRIC_ALIASES = {
    "grpo.requests.total": "requests_total",
    "grpo.request.duration": "request_duration",
    "grpo.request.size": "request_size",
    "grpo.training.iterations": "training_iterations",
    "grpo.memory.usage": "memory_usage",
    "grpo.gpu.utilization": "gpu_utilization",
    "grpo.error_rate": "error_rate",
    "grpo.errors.total": "errors_total",
}


class MonitoringState(str, Enum):
    """Admission and background-work ownership for advanced monitoring."""

    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class _BackgroundLifecycle:
    """Local lifecycle shared by the three advanced monitoring task owners.

    Use one event loop. Passive metric/history access is independent of worker
    admission. Startup and shutdown each have one retained outcome; closing
    cannot discard workers, and failed workers are never silently restarted.
    """

    def __init__(self) -> None:
        self._state = MonitoringState.NEW
        self._start_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._observed_workers: set[str] = set()
        self._stopping_workers: set[str] = set()
        self._failure: BaseException | None = None

    def _background_children(self) -> dict[str, _BackgroundLifecycle]:
        return {}

    def _record_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
            logger.error(
                "%s background monitoring failed (%s)",
                type(self).__name__,
                type(error).__name__,
            )
        if self._state not in (MonitoringState.CLOSING, MonitoringState.CLOSED):
            self._state = MonitoringState.FAILED

    def _observe_workers(self) -> None:
        for name, task in self._workers.items():
            if not task.done() or name in self._observed_workers:
                continue
            self._observed_workers.add(name)
            error = None if task.cancelled() else task.exception()
            if error is not None:
                self._record_failure(error)
            elif (
                name not in self._stopping_workers
                and self._state is not MonitoringState.CLOSING
            ):
                self._record_failure(RuntimeError(f"Monitor worker {name} stopped"))
        if self._state is MonitoringState.RUNNING:
            for name, child in self._background_children().items():
                if child.state is not MonitoringState.RUNNING:
                    self._record_failure(
                        child._failure or RuntimeError(f"Monitor child {name} stopped")
                    )

    @property
    def state(self) -> MonitoringState:
        """Observe lifecycle state, including unexpected worker termination."""
        self._observe_workers()
        return self._state

    def get_status(self) -> dict[str, Any]:
        """Return detached lifecycle diagnostics without starting any work."""
        return {
            "state": self.state.value,
            "workers": {
                name: "done" if task.done() else "running"
                for name, task in self._workers.items()
            },
            "active_workers": sum(not task.done() for task in self._workers.values()),
            "error_type": type(self._failure).__name__ if self._failure else None,
            "children": {
                name: child.get_status()
                for name, child in self._background_children().items()
            },
        }

    def _owns_task(self, task: asyncio.Task[Any] | None) -> bool:
        return task is not None and (
            task in (self._start_task, self._close_task)
            or task in self._workers.values()
            or any(
                child._owns_task(task) for child in self._background_children().values()
            )
        )

    @staticmethod
    def _create_task(operation: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        try:
            return asyncio.create_task(operation)
        except BaseException:
            operation.close()
            raise

    def _spawn_worker(
        self, name: str, operation: Callable[[], Coroutine[Any, Any, None]]
    ) -> asyncio.Task[None]:
        async def run() -> None:
            # Eager factories run before create_task returns. Register ownership
            # before user callbacks can attempt to close their own monitor.
            current = asyncio.current_task()
            assert current is not None
            self._workers[name] = current
            await operation()

        task = self._create_task(run())
        self._workers[name] = task
        task.add_done_callback(lambda _: self._observe_workers())
        return task

    async def _start_background_tasks(self) -> None:
        raise NotImplementedError

    async def _close_background_resources(self) -> None:
        """Release dependencies after all background workers and children drain."""

    async def _start(self) -> None:
        self._start_task = asyncio.current_task()
        try:
            await self._start_background_tasks()
            self._observe_workers()
            if self._failure is not None:
                raise self._failure
            if self._state is MonitoringState.STARTING:
                self._state = MonitoringState.RUNNING
        except BaseException as error:
            self._record_failure(
                RuntimeError("Monitor startup was cancelled")
                if isinstance(error, asyncio.CancelledError)
                else error
            )
            raise

    async def start(self) -> None:
        """Start once; concurrent callers join the same admitted startup.

        A cancelled caller waits for admitted startup to settle. If it succeeds,
        the monitor remains owned and running until explicitly closed.
        """
        if self._owns_task(asyncio.current_task()):
            raise RuntimeError("Monitor workers cannot start their own owner")
        state = self.state
        if state is MonitoringState.RUNNING:
            return
        if state not in (MonitoringState.NEW, MonitoringState.STARTING):
            raise RuntimeError(f"Monitor is {state.value}; create a new monitor")
        if self._start_task is None:
            self._state = MonitoringState.STARTING
            try:
                self._start_task = self._create_task(self._start())
            except BaseException as error:
                self._record_failure(error)
                raise

        async def join() -> None:
            assert self._start_task is not None
            await self._start_task

        try:
            await drain_owned_operation(join())
        except BaseException:
            if self._start_task.done() and self._failure is not None:
                try:
                    await self.aclose()
                except BaseException:
                    pass  # Preserve the startup outcome after draining rollback.
            raise

    async def aclose(self) -> None:
        """Close admission and drain workers and children despite waiter cancellation."""
        if self._owns_task(asyncio.current_task()):
            raise RuntimeError("Monitor workers cannot close their own owner")
        if self._close_task is None:
            self._observe_workers()
            self._state = MonitoringState.CLOSING
            self._close_task = self._create_task(self._close())

        async def join() -> None:
            assert self._close_task is not None
            await self._close_task

        await drain_owned_operation(join())

    async def _close(self) -> None:
        self._close_task = asyncio.current_task()
        if self._start_task is not None:
            await asyncio.gather(self._start_task, return_exceptions=True)
        self._observe_workers()
        for name, task in self._workers.items():
            if not task.done():
                self._stopping_workers.add(name)
                task.cancel()
        await asyncio.gather(*self._workers.values(), return_exceptions=True)
        self._observe_workers()
        # Producers finish before their dependencies. Continue draining other
        # children even when one child reports a worker/cleanup failure.
        for child in self._background_children().values():
            try:
                await child.aclose()
            except BaseException as error:
                self._record_failure(error)
        try:
            await self._close_background_resources()
        except BaseException as error:
            self._record_failure(error)
        self._state = (
            MonitoringState.FAILED if self._failure else MonitoringState.CLOSED
        )
        if self._failure is not None:
            raise self._failure


class MetricsCollector(_BackgroundLifecycle):
    """Advanced metrics collection and aggregation"""

    def __init__(self, retention_period: int = 86400):  # 24 hours
        super().__init__()
        self.metrics: dict[str, deque] = defaultdict(lambda: deque())
        self.retention_period = retention_period
        self.aggregated_metrics: dict[str, dict[str, float]] = {}
        self.custom_collectors: list[Callable] = []

        # Prometheus metrics (if available)
        self.prometheus_registry = None
        self.prometheus_metrics: dict[str, Any] = {}

        if PROMETHEUS_AVAILABLE:
            self.prometheus_registry = CollectorRegistry()
            self._setup_prometheus_metrics()

        # Cleanup task (lazily initialized)
        self._cleanup_task: asyncio.Task | None = None

    async def _start_background_tasks(self) -> None:
        self._cleanup_task = self._spawn_worker("cleanup", self._cleanup_old_metrics)

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        requests_total = Counter(
            "grpo_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
            registry=self.prometheus_registry,
        )
        request_duration = Histogram(
            "grpo_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            registry=self.prometheus_registry,
        )
        request_size = Summary(
            "grpo_request_size_bytes",
            "Request size in bytes",
            ["method", "endpoint"],
            registry=self.prometheus_registry,
        )
        training_iterations = Counter(
            "grpo_training_iterations_total",
            "Total training iterations",
            ["agent_type", "strategy"],
            registry=self.prometheus_registry,
        )
        memory_usage = Gauge(
            "grpo_memory_usage_bytes",
            "Memory usage in bytes",
            ["component"],
            registry=self.prometheus_registry,
        )
        gpu_utilization = Gauge(
            "grpo_gpu_utilization_percent",
            "GPU utilization percentage",
            ["device"],
            registry=self.prometheus_registry,
        )
        error_rate = Gauge(
            "grpo_error_rate",
            "Error rate",
            ["component", "error_type"],
            registry=self.prometheus_registry,
        )
        errors_total = Counter(
            "grpo_errors_total",
            "Total number of errors",
            ["component", "error_type"],
            registry=self.prometheus_registry,
        )

        self.prometheus_metrics = {
            "requests_total": requests_total,
            "request_duration": request_duration,
            "request_size": request_size,
            "training_iterations": training_iterations,
            "memory_usage": memory_usage,
            "gpu_utilization": gpu_utilization,
            "error_rate": error_rate,
            "errors_total": errors_total,
        }
        for alias, target in PROMETHEUS_METRIC_ALIASES.items():
            if target in self.prometheus_metrics:
                self.prometheus_metrics[alias] = self.prometheus_metrics[target]

    def record_metric(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        metric_type: MetricType = MetricType.GAUGE,
    ):
        """Record a metric point"""
        current_time = time.time()
        labels = labels or {}

        # Store in internal metrics
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=current_time,
            labels=labels,
            metric_type=metric_type,
        )

        metric_key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.metrics[metric_key].append(metric_point)

        # Update Prometheus metrics
        prometheus_metric = self.prometheus_metrics.get(name)
        if PROMETHEUS_AVAILABLE and prometheus_metric is not None:
            label_values = [
                labels.get(label, "") for label in prometheus_metric._labelnames
            ]

            if metric_type == MetricType.COUNTER:
                prometheus_metric.labels(*label_values).inc(value)
            elif metric_type == MetricType.GAUGE:
                prometheus_metric.labels(*label_values).set(value)
            elif metric_type == MetricType.HISTOGRAM:
                prometheus_metric.labels(*label_values).observe(value)
            elif metric_type == MetricType.SUMMARY:
                prometheus_metric.labels(*label_values).observe(value)

    def add_custom_collector(self, collector: Callable[[], dict[str, float]]):
        """Add custom metric collector"""
        self.custom_collectors.append(collector)

    async def collect_system_metrics(self):
        """Collect system metrics"""
        # CPU metrics
        cpu_percent = await run_sync_owned(psutil.cpu_percent, interval=0.1)
        self.record_metric("system.cpu.utilization", cpu_percent, {"unit": "percent"})

        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric("system.memory.used", memory.used, {"unit": "bytes"})
        self.record_metric(
            "system.memory.available", memory.available, {"unit": "bytes"}
        )
        self.record_metric("system.memory.percent", memory.percent, {"unit": "percent"})

        # Disk metrics
        disk = psutil.disk_usage("/")
        self.record_metric("system.disk.used", disk.used, {"unit": "bytes"})
        self.record_metric("system.disk.free", disk.free, {"unit": "bytes"})
        self.record_metric("system.disk.percent", disk.percent, {"unit": "percent"})

        # Network metrics
        network = psutil.net_io_counters()
        self.record_metric(
            "system.network.bytes_sent", network.bytes_sent, {"unit": "bytes"}
        )
        self.record_metric(
            "system.network.bytes_recv", network.bytes_recv, {"unit": "bytes"}
        )

        # GPU metrics (if available)
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_cached = torch.cuda.memory_reserved(i)
                    utilization = (
                        torch.cuda.utilization(i)
                        if hasattr(torch.cuda, "utilization")
                        else 0
                    )

                    self.record_metric(
                        "gpu.memory.allocated",
                        memory_allocated,
                        {"device": f"cuda:{i}", "unit": "bytes"},
                    )
                    self.record_metric(
                        "gpu.memory.cached",
                        memory_cached,
                        {"device": f"cuda:{i}", "unit": "bytes"},
                    )
                    self.record_metric(
                        "gpu.utilization",
                        utilization,
                        {"device": f"cuda:{i}", "unit": "percent"},
                    )
        except ImportError:
            pass

        # Custom collectors
        for collector in self.custom_collectors:
            try:
                custom_metrics = collector()
                for name, value in custom_metrics.items():
                    self.record_metric(f"custom.{name}", value)
            except ADV_MONITORING_EXCEPTIONS as e:
                logger.error(f"Custom collector failed: {e}")

    def get_metrics(
        self,
        name_pattern: str | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> dict[str, list[MetricPoint]]:
        """Get metrics with optional filtering"""
        current_time = time.time()
        start_time = time_range[0] if time_range else current_time - 3600  # Last hour
        end_time = time_range[1] if time_range else current_time

        filtered_metrics = {}

        for metric_key, points in self.metrics.items():
            metric_name = metric_key.split(":")[0]

            # Name pattern filtering
            if name_pattern and name_pattern not in metric_name:
                continue

            # Time range filtering
            filtered_points = [
                point for point in points if start_time <= point.timestamp <= end_time
            ]

            if filtered_points:
                filtered_metrics[metric_key] = filtered_points

        return filtered_metrics

    def aggregate_metrics(
        self, name: str, aggregation: str = "avg", time_window: int = 300
    ) -> dict[str, float]:
        """Aggregate metrics over time window"""
        current_time = time.time()
        start_time = current_time - time_window

        aggregated = {}

        for metric_key, points in self.metrics.items():
            if not metric_key.startswith(name):
                continue

            # Filter by time window
            recent_points = [
                point.value for point in points if point.timestamp >= start_time
            ]

            if not recent_points:
                continue

            # Perform aggregation
            if aggregation == "avg":
                value = statistics.mean(recent_points)
            elif aggregation == "sum":
                value = sum(recent_points)
            elif aggregation == "min":
                value = min(recent_points)
            elif aggregation == "max":
                value = max(recent_points)
            elif aggregation == "count":
                value = len(recent_points)
            elif aggregation == "median":
                value = statistics.median(recent_points)
            elif aggregation == "p95":
                value = (
                    statistics.quantiles(recent_points, n=20)[18]
                    if len(recent_points) > 1
                    else recent_points[0]
                )
            else:
                value = statistics.mean(recent_points)  # Default to average

            aggregated[metric_key] = value

        return aggregated

    async def _cleanup_old_metrics(self):
        """Clean up old metrics periodically"""
        while self._state is not MonitoringState.CLOSING:
            try:
                current_time = time.time()
                cutoff_time = current_time - self.retention_period

                for _metric_key, points in self.metrics.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()

                await asyncio.sleep(3600)  # Clean up every hour
            except ADV_MONITORING_EXCEPTIONS as e:
                if self._state is MonitoringState.CLOSING:
                    raise
                logger.error(f"Metrics cleanup failed: {e}")
                await asyncio.sleep(60)


class AlertManager(_BackgroundLifecycle):
    """Advanced alerting system"""

    def __init__(self, metrics_collector: MetricsCollector):
        super().__init__()
        self.metrics_collector = metrics_collector
        self.alerts: dict[str, Alert] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[dict[str, Any]] = []
        self.notification_handlers: list[Callable] = []

        # Alert checking task (lazily initialized)
        self._alert_task: asyncio.Task | None = None

    async def _start_background_tasks(self) -> None:
        self._alert_task = self._spawn_worker("alerts", self._check_alerts)

    def add_alert(self, alert: Alert):
        """Add an alert definition"""
        self.alerts[alert.id] = alert

    def add_notification_handler(self, handler: Callable[[Alert, float], None]):
        """Add notification handler"""
        self.notification_handlers.append(handler)

    async def _check_alerts(self):
        """Continuously check alert conditions"""
        while self._state is not MonitoringState.CLOSING:
            try:
                current_time = time.time()

                for alert in self.alerts.values():
                    if not alert.enabled:
                        continue

                    try:
                        # Evaluate alert condition
                        should_trigger = await self._evaluate_alert_condition(alert)
                        if self._state is MonitoringState.CLOSING:
                            return

                        if should_trigger:
                            if alert.id not in self.active_alerts:
                                # New alert
                                self.active_alerts[alert.id] = alert
                                alert.last_triggered = current_time

                                # Record in history
                                self.alert_history.append(
                                    {
                                        "alert_id": alert.id,
                                        "name": alert.name,
                                        "severity": alert.severity.value,
                                        "triggered_at": current_time,
                                        "condition": alert.condition,
                                    }
                                )

                                # Send notifications
                                for handler in self.notification_handlers:
                                    try:
                                        handler(alert, current_time)
                                    except ADV_MONITORING_EXCEPTIONS as e:
                                        if self._state is MonitoringState.CLOSING:
                                            raise
                                        logger.error(
                                            f"Notification handler failed: {e}"
                                        )
                        else:
                            # Alert resolved
                            if alert.id in self.active_alerts:
                                del self.active_alerts[alert.id]

                    except ADV_MONITORING_EXCEPTIONS as e:
                        if self._state is MonitoringState.CLOSING:
                            raise
                        logger.error(f"Alert evaluation failed for {alert.id}: {e}")

                await asyncio.sleep(10)  # Check every 10 seconds
            except ADV_MONITORING_EXCEPTIONS as e:
                if self._state is MonitoringState.CLOSING:
                    raise
                logger.error(f"Alert checking loop failed: {e}")
                await asyncio.sleep(60)

    async def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate alert condition"""
        # Simple condition evaluation (can be extended)
        if ">" in alert.condition:
            metric_name, threshold_str = alert.condition.split(">")
            metric_name = metric_name.strip()
            threshold = float(threshold_str.strip())

            # Get recent metric values
            recent_metrics = self.metrics_collector.aggregate_metrics(
                metric_name, "avg", int(alert.duration)
            )

            for _metric_key, value in recent_metrics.items():
                if value > threshold:
                    return True

        elif "<" in alert.condition:
            metric_name, threshold_str = alert.condition.split("<")
            metric_name = metric_name.strip()
            threshold = float(threshold_str.strip())

            recent_metrics = self.metrics_collector.aggregate_metrics(
                metric_name, "avg", int(alert.duration)
            )

            for _metric_key, value in recent_metrics.items():
                if value < threshold:
                    return True

        return False

    def get_active_alerts(self) -> list[Alert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get alert history"""
        return self.alert_history[-limit:]


class AdvancedMonitoringService(_BackgroundLifecycle):
    """Monitoring facade with explicit ownership of its three async workers.

    Await ``start()`` to enable background collection and alerts, then await
    ``aclose()`` before closing the event loop. Passive metric recording remains
    available without startup and after shutdown. Tracing exporters are released
    after all workers finish; new traces are rejected once the tracer closes.
    """

    def __init__(
        self,
        enable_metrics: bool = True,
        enable_alerts: bool = True,
        enable_tracing: bool = True,
        retention_period: int = 86400,
    ):
        super().__init__()
        self.metrics_collector = (
            MetricsCollector(retention_period) if enable_metrics else None
        )
        self.alert_manager = (
            AlertManager(self.metrics_collector)
            if enable_alerts and self.metrics_collector
            else None
        )
        self.tracer = (
            DistributedTracer(
                opentelemetry_available=OPENTELEMETRY_AVAILABLE,
                trace_module=trace if OPENTELEMETRY_AVAILABLE else None,
                jaeger_exporter_cls=JaegerExporter if OPENTELEMETRY_AVAILABLE else None,
                tracer_provider_cls=TracerProvider if OPENTELEMETRY_AVAILABLE else None,
                batch_span_processor_cls=(
                    BatchSpanProcessor if OPENTELEMETRY_AVAILABLE else None
                ),
                logger=logger,
                handled_exceptions=ADV_MONITORING_EXCEPTIONS,
            )
            if enable_tracing
            else None
        )

        self._monitoring_task: asyncio.Task | None = None
        self._start_time = time.time()

        if self.metrics_collector:
            self._setup_default_alerts()
            # Note: monitoring loop is started lazily via start() method

    def _background_children(self) -> dict[str, _BackgroundLifecycle]:
        children: dict[str, _BackgroundLifecycle] = {}
        if self.alert_manager is not None:
            children["alerts"] = self.alert_manager
        if self.metrics_collector is not None:
            children["metrics"] = self.metrics_collector
        return children

    async def _start_background_tasks(self) -> None:
        if self.metrics_collector:
            await self.metrics_collector.start()
            self._monitoring_task = self._spawn_worker(
                "monitoring", self._monitoring_loop
            )
        if self.alert_manager:
            await self.alert_manager.start()

    async def _close_background_resources(self) -> None:
        if self.tracer is not None:
            await self.tracer.aclose()

    def get_status(self) -> dict[str, Any]:
        """Inspect worker ownership and the separate tracing pipeline lifecycle."""
        status = super().get_status()
        status["tracing"] = (
            self.tracer.get_status() if self.tracer is not None else None
        )
        return status

    def _setup_default_alerts(self):
        """Setup default alerts"""
        if not self.alert_manager:
            return

        # High CPU usage alert
        self.alert_manager.add_alert(
            Alert(
                id="high_cpu_usage",
                name="High CPU Usage",
                condition="system.cpu.utilization > 90",
                severity=AlertSeverity.WARNING,
                threshold=90.0,
                duration=300.0,  # 5 minutes
            )
        )

        # High memory usage alert
        self.alert_manager.add_alert(
            Alert(
                id="high_memory_usage",
                name="High Memory Usage",
                condition="system.memory.percent > 85",
                severity=AlertSeverity.WARNING,
                threshold=85.0,
                duration=300.0,
            )
        )

        # High error rate alert
        self.alert_manager.add_alert(
            Alert(
                id="high_error_rate",
                name="High Error Rate",
                condition="grpo.error_rate > 0.1",
                severity=AlertSeverity.ERROR,
                threshold=0.1,
                duration=60.0,  # 1 minute
            )
        )

    async def record_metric(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        metric_type: MetricType = MetricType.GAUGE,
    ):
        """Record a metric through the service facade."""
        if self.metrics_collector:
            self.metrics_collector.record_metric(name, value, labels, metric_type)

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._state is not MonitoringState.CLOSING:
            try:
                if self.metrics_collector:
                    await self.metrics_collector.collect_system_metrics()
                if self._state is MonitoringState.CLOSING:
                    return
                await asyncio.sleep(30)  # Collect every 30 seconds
            except ADV_MONITORING_EXCEPTIONS as e:
                if self._state is MonitoringState.CLOSING:
                    raise
                logger.error(f"Monitoring loop failed: {e}")
                await asyncio.sleep(60)

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "grpo.requests.total",
                1,
                {"method": method, "endpoint": endpoint, "status": str(status)},
                MetricType.COUNTER,
            )

            self.metrics_collector.record_metric(
                "grpo.request.duration",
                duration,
                {"method": method, "endpoint": endpoint},
                MetricType.HISTOGRAM,
            )

    def record_training_iteration(
        self, agent_type: str, strategy: str, metrics: dict[str, float]
    ):
        """Record training iteration metrics"""
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "grpo.training.iterations",
                1,
                {"agent_type": agent_type, "strategy": strategy},
                MetricType.COUNTER,
            )

            for metric_name, value in metrics.items():
                self.metrics_collector.record_metric(
                    f"grpo.training.{metric_name}",
                    value,
                    {"agent_type": agent_type, "strategy": strategy},
                )

    def record_error(self, component: str, error_type: str, error: Exception):
        """Record error metrics"""
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "grpo.errors.total",
                1,
                {"component": component, "error_type": error_type},
                MetricType.COUNTER,
            )

    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **tags):
        """Trace an operation"""
        if self.tracer:
            async with self.tracer.trace(operation_name, tags=tags) as span:
                yield span
        else:
            yield None

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary"""
        current_time = time.time()
        uptime = current_time - self._start_time

        summary = {
            "status": "healthy",
            "lifecycle": self.get_status(),
            "uptime": uptime,
            "timestamp": current_time,
            "components": {
                "metrics_collector": bool(self.metrics_collector),
                "alert_manager": bool(self.alert_manager),
                "tracer": bool(self.tracer),
            },
        }

        if self.metrics_collector:
            # Get recent system metrics
            recent_metrics = self.metrics_collector.aggregate_metrics(
                "system", "avg", 300
            )
            summary["system_metrics"] = recent_metrics

        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            summary["active_alerts"] = len(active_alerts)

            if active_alerts:
                summary["status"] = "degraded"
                critical_alerts = [
                    a for a in active_alerts if a.severity == AlertSeverity.CRITICAL
                ]
                if critical_alerts:
                    summary["status"] = "unhealthy"

        if self.state is not MonitoringState.RUNNING:
            summary["status"] = "unhealthy"
        return summary

    def get_metrics_dashboard(self) -> dict[str, Any]:
        """Get metrics for dashboard display"""
        if not self.metrics_collector:
            return {}

        current_time = time.time()
        dashboard: dict[str, Any] = {
            "timestamp": current_time,
            "system": {},
            "grpo": {},
            "alerts": [],
        }

        # System metrics
        system_metrics = self.metrics_collector.aggregate_metrics("system", "avg", 300)
        for metric_key, value in system_metrics.items():
            metric_name = metric_key.split(":")[0].replace("system.", "")
            dashboard["system"][metric_name] = value

        # GRPO metrics
        grpo_metrics = self.metrics_collector.aggregate_metrics("grpo", "avg", 300)
        for metric_key, value in grpo_metrics.items():
            metric_name = metric_key.split(":")[0].replace("grpo.", "")
            dashboard["grpo"][metric_name] = value

        # Active alerts
        if self.alert_manager:
            dashboard["alerts"] = [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "condition": alert.condition,
                }
                for alert in self.alert_manager.get_active_alerts()
            ]

        return dashboard


# Global monitoring instance
_monitoring_service: AdvancedMonitoringService | None = None


def get_monitoring_service() -> AdvancedMonitoringService:
    """Get or create global monitoring service"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = AdvancedMonitoringService()
    return _monitoring_service


@contextmanager
def monitor_operation(operation_name: str, **tags):
    """Context manager for monitoring operations"""
    monitoring = get_monitoring_service()
    start_time = time.time()

    try:
        yield
    except ADV_MONITORING_EXCEPTIONS as e:
        if monitoring.metrics_collector:
            monitoring.record_error("operation", type(e).__name__, e)
        raise
    finally:
        duration = time.time() - start_time
        if monitoring.metrics_collector:
            monitoring.metrics_collector.record_metric(
                f"operation.{operation_name}.duration",
                duration,
                tags,
                MetricType.HISTOGRAM,
            )


# Decorators for automatic monitoring
def monitor_async_function(operation_name: str | None = None):
    """Decorator for monitoring async functions"""

    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitoring = get_monitoring_service()
            start_time = time.time()

            if monitoring.tracer:
                async with monitoring.tracer.trace(name):
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except ADV_MONITORING_EXCEPTIONS as e:
                        monitoring.record_error("function", type(e).__name__, e)
                        raise
                    finally:
                        duration = time.time() - start_time
                        if monitoring.metrics_collector:
                            monitoring.metrics_collector.record_metric(
                                f"function.{name}.duration",
                                duration,
                                {"function": name},
                                MetricType.HISTOGRAM,
                            )
            else:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except ADV_MONITORING_EXCEPTIONS as e:
                    monitoring.record_error("function", type(e).__name__, e)
                    raise
                finally:
                    duration = time.time() - start_time
                    if monitoring.metrics_collector:
                        monitoring.metrics_collector.record_metric(
                            f"function.{name}.duration",
                            duration,
                            {"function": name},
                            MetricType.HISTOGRAM,
                        )

        return wrapper

    return decorator


def monitor_function(operation_name: str | None = None):
    """Decorator for monitoring sync functions"""

    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            monitoring = get_monitoring_service()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            except ADV_MONITORING_EXCEPTIONS as e:
                monitoring.record_error("function", type(e).__name__, e)
                raise
            finally:
                duration = time.time() - start_time
                if monitoring.metrics_collector:
                    monitoring.metrics_collector.record_metric(
                        f"function.{name}.duration",
                        duration,
                        {"function": name},
                        MetricType.HISTOGRAM,
                    )

        return wrapper

    return decorator


if __name__ == "__main__":
    # Example usage
    async def main():
        monitoring = AdvancedMonitoringService()

        # Record some metrics
        monitoring.record_request("POST", "/api/train", 200, 1.5)
        monitoring.record_training_iteration(
            "MultiTurnAgent", "grpo", {"reward": 0.8, "loss": 0.2, "episodes": 100}
        )

        # Use tracing
        async with monitoring.trace_operation(
            "example_operation", component="test"
        ) as span:
            await asyncio.sleep(0.1)
            if span:
                monitoring.tracer.add_span_log(
                    span.span_id, "info", "Operation completed"
                )

        # Get health summary
        health = monitoring.get_health_summary()
        print("Health Summary:", json.dumps(health, indent=2))

        # Get dashboard
        dashboard = monitoring.get_metrics_dashboard()
        print("Dashboard:", json.dumps(dashboard, indent=2))
        await monitoring.aclose()

    asyncio.run(main())


__all__ = [
    "ADV_MONITORING_EXCEPTIONS",
    "Alert",
    "AlertManager",
    "AlertSeverity",
    "AdvancedMonitoringService",
    "DistributedTracer",
    "MetricPoint",
    "MetricType",
    "MetricsCollector",
    "MonitoringState",
    "OPENTELEMETRY_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
    "TraceSpan",
    "get_monitoring_service",
    "monitor_async_function",
    "monitor_function",
    "monitor_operation",
]
