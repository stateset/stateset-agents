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
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from collections.abc import Callable
from typing import Any

import psutil

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


class MetricsCollector:
    """Advanced metrics collection and aggregation"""

    def __init__(self, retention_period: int = 86400):  # 24 hours
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

    def _ensure_cleanup_task(self):
        """Ensure cleanup task is running (lazy initialization)"""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_old_metrics())
            except RuntimeError:
                # No running event loop, will be started later
                pass

    async def start(self):
        """Start the metrics collector background tasks"""
        self._ensure_cleanup_task()

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
        cpu_percent = psutil.cpu_percent(interval=0.1)
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
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - self.retention_period

                for _metric_key, points in self.metrics.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()

                await asyncio.sleep(3600)  # Clean up every hour
            except ADV_MONITORING_EXCEPTIONS as e:
                logger.error(f"Metrics cleanup failed: {e}")
                await asyncio.sleep(60)


class AlertManager:
    """Advanced alerting system"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: dict[str, Alert] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[dict[str, Any]] = []
        self.notification_handlers: list[Callable] = []

        # Alert checking task (lazily initialized)
        self._alert_task: asyncio.Task | None = None

    def _ensure_alert_task(self):
        """Ensure alert checking task is running (lazy initialization)"""
        if self._alert_task is None or self._alert_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._alert_task = loop.create_task(self._check_alerts())
            except RuntimeError:
                # No running event loop, will be started later
                pass

    async def start(self):
        """Start the alert manager background tasks"""
        self._ensure_alert_task()

    def add_alert(self, alert: Alert):
        """Add an alert definition"""
        self.alerts[alert.id] = alert

    def add_notification_handler(self, handler: Callable[[Alert, float], None]):
        """Add notification handler"""
        self.notification_handlers.append(handler)

    async def _check_alerts(self):
        """Continuously check alert conditions"""
        while True:
            try:
                current_time = time.time()

                for alert in self.alerts.values():
                    if not alert.enabled:
                        continue

                    try:
                        # Evaluate alert condition
                        should_trigger = await self._evaluate_alert_condition(alert)

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
                                        logger.error(
                                            f"Notification handler failed: {e}"
                                        )
                        else:
                            # Alert resolved
                            if alert.id in self.active_alerts:
                                del self.active_alerts[alert.id]

                    except ADV_MONITORING_EXCEPTIONS as e:
                        logger.error(f"Alert evaluation failed for {alert.id}: {e}")

                await asyncio.sleep(10)  # Check every 10 seconds
            except ADV_MONITORING_EXCEPTIONS as e:
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


class AdvancedMonitoringService:
    """Comprehensive monitoring service"""

    def __init__(
        self,
        enable_metrics: bool = True,
        enable_alerts: bool = True,
        enable_tracing: bool = True,
        retention_period: int = 86400,
    ):
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
                tracer_provider_cls=TracerProvider
                if OPENTELEMETRY_AVAILABLE
                else None,
                batch_span_processor_cls=BatchSpanProcessor
                if OPENTELEMETRY_AVAILABLE
                else None,
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

    def _ensure_monitoring_task(self):
        """Ensure monitoring task is running (lazy initialization)"""
        if self._monitoring_task is None or self._monitoring_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._monitoring_task = loop.create_task(self._monitoring_loop())
            except RuntimeError:
                # No running event loop, will be started later
                pass

    async def start(self):
        """Start all monitoring background tasks"""
        if self.metrics_collector:
            await self.metrics_collector.start()
            self._ensure_monitoring_task()
        if self.alert_manager:
            await self.alert_manager.start()

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
        while True:
            try:
                if self.metrics_collector:
                    await self.metrics_collector.collect_system_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except ADV_MONITORING_EXCEPTIONS as e:
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
    "OPENTELEMETRY_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
    "TraceSpan",
    "get_monitoring_service",
    "monitor_async_function",
    "monitor_function",
    "monitor_operation",
]
