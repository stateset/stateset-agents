"""
Comprehensive Monitoring System for GRPO Agent Framework

This module provides advanced monitoring capabilities including metrics collection,
alerting, health checks, and integration with external monitoring systems.
"""

import asyncio
import json
import logging
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Optional imports for enhanced functionality
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class MetricType(Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Metric data structure"""

    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    help_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "help_text": self.help_text,
        }


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    name: str
    severity: AlertSeverity
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class HealthCheck:
    """Health check definition"""

    name: str
    check_func: Callable[[], bool]
    timeout: float = 5.0
    critical: bool = False

    async def run(self) -> Dict[str, Any]:
        """Run the health check"""
        start_time = time.time()

        try:
            loop = asyncio.get_running_loop()
            if asyncio.iscoroutinefunction(self.check_func):
                task = loop.create_task(self.check_func())
            else:
                task = loop.run_in_executor(None, self.check_func)

            result = await asyncio.wait_for(task, timeout=self.timeout)

            duration = time.time() - start_time

            return {
                "name": self.name,
                "status": "healthy" if result else "unhealthy",
                "duration_seconds": duration,
                "critical": self.critical,
                "timestamp": datetime.now().isoformat(),
            }

        except asyncio.TimeoutError:
            return {
                "name": self.name,
                "status": "timeout",
                "duration_seconds": self.timeout,
                "critical": self.critical,
                "timestamp": datetime.now().isoformat(),
                "error": "Health check timed out",
            }

        except Exception as e:
            return {
                "name": self.name,
                "status": "error",
                "duration_seconds": time.time() - start_time,
                "critical": self.critical,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }


class MetricsCollector:
    """Collects and manages metrics with thread-safe operations"""

    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and HAS_PROMETHEUS
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.prometheus_metrics: Dict[str, Any] = {}

        # Thread lock for protecting shared state
        self._lock = threading.RLock()

        # System metrics
        self.system_metrics = {
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "disk_usage": deque(maxlen=100),
            "network_io": deque(maxlen=100),
        }

        # Application metrics
        self.app_metrics = {
            "requests_total": 0,
            "requests_per_second": deque(maxlen=60),
            "response_times": deque(maxlen=1000),
            "active_conversations": 0,
            "training_iterations": 0,
            "reward_scores": deque(maxlen=100),
            "error_count": 0,
        }

        # Initialize Prometheus metrics
        if self.enable_prometheus:
            self._setup_prometheus_metrics()

        # Start system metrics collection
        self._start_system_monitoring()

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        if not HAS_PROMETHEUS:
            return

        # Request metrics
        self.prometheus_metrics["requests_total"] = Counter(
            "grpo_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
        )

        self.prometheus_metrics["request_duration"] = Histogram(
            "grpo_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
        )

        # System metrics
        self.prometheus_metrics["cpu_usage"] = Gauge(
            "grpo_cpu_usage_percent", "CPU usage percentage"
        )

        self.prometheus_metrics["memory_usage"] = Gauge(
            "grpo_memory_usage_bytes", "Memory usage in bytes"
        )

        # Application metrics
        self.prometheus_metrics["active_conversations"] = Gauge(
            "grpo_active_conversations", "Number of active conversations"
        )

        self.prometheus_metrics["training_iterations"] = Counter(
            "grpo_training_iterations_total", "Total training iterations"
        )

        self.prometheus_metrics["reward_score"] = Histogram(
            "grpo_reward_score", "Reward scores distribution"
        )

        self.prometheus_metrics["errors_total"] = Counter(
            "grpo_errors_total", "Total number of errors", ["error_type"]
        )

    def _start_system_monitoring(self):
        """Start system metrics collection in background"""
        if not HAS_PSUTIL:
            return

        # Create a logger for the monitoring thread
        monitoring_logger = logging.getLogger(__name__ + ".system_monitoring")

        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)

                    # Memory usage
                    memory = psutil.virtual_memory()

                    # Disk usage
                    disk = psutil.disk_usage("/")

                    # Network I/O
                    net_io = psutil.net_io_counters()

                    # Thread-safe update of all metrics
                    with self._lock:
                        self.system_metrics["cpu_usage"].append(cpu_percent)
                        self.system_metrics["memory_usage"].append(memory.percent)
                        self.system_metrics["disk_usage"].append(disk.percent)
                        self.system_metrics["network_io"].append(
                            {
                                "bytes_sent": net_io.bytes_sent,
                                "bytes_recv": net_io.bytes_recv,
                                "timestamp": time.time(),
                            }
                        )

                    # Update Prometheus metrics (these are thread-safe by design)
                    if self.enable_prometheus:
                        self.prometheus_metrics["cpu_usage"].set(cpu_percent)
                        self.prometheus_metrics["memory_usage"].set(memory.used)

                except Exception as e:
                    monitoring_logger.warning(f"Error collecting system metrics: {e}")

                time.sleep(30)  # Collect every 30 seconds

        # Start in background thread
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()

    def record_metric(self, metric: Metric):
        """Record a metric"""
        self.metrics[metric.name].append(metric)

        # Update Prometheus metrics
        if self.enable_prometheus and metric.name in self.prometheus_metrics:
            prom_metric = self.prometheus_metrics[metric.name]

            if metric.type == MetricType.COUNTER:
                prom_metric.inc(metric.value)
            elif metric.type == MetricType.GAUGE:
                prom_metric.set(metric.value)
            elif metric.type == MetricType.HISTOGRAM:
                prom_metric.observe(metric.value)

    def increment_counter(
        self, name: str, value: float = 1.0, labels: Dict[str, str] = None
    ):
        """Increment a counter metric"""
        metric = Metric(
            name=name, type=MetricType.COUNTER, value=value, labels=labels or {}
        )
        self.record_metric(metric)

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        metric = Metric(
            name=name, type=MetricType.GAUGE, value=value, labels=labels or {}
        )
        self.record_metric(metric)

    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram metric"""
        metric = Metric(
            name=name, type=MetricType.HISTOGRAM, value=value, labels=labels or {}
        )
        self.record_metric(metric)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {},
            "application_metrics": {},
            "custom_metrics": {},
        }

        # System metrics
        if HAS_PSUTIL:
            summary["system_metrics"] = {
                "cpu_usage": {
                    "current": list(self.system_metrics["cpu_usage"])[-1]
                    if self.system_metrics["cpu_usage"]
                    else 0,
                    "average": statistics.mean(self.system_metrics["cpu_usage"])
                    if self.system_metrics["cpu_usage"]
                    else 0,
                    "max": max(self.system_metrics["cpu_usage"])
                    if self.system_metrics["cpu_usage"]
                    else 0,
                },
                "memory_usage": {
                    "current": list(self.system_metrics["memory_usage"])[-1]
                    if self.system_metrics["memory_usage"]
                    else 0,
                    "average": statistics.mean(self.system_metrics["memory_usage"])
                    if self.system_metrics["memory_usage"]
                    else 0,
                    "max": max(self.system_metrics["memory_usage"])
                    if self.system_metrics["memory_usage"]
                    else 0,
                },
                "disk_usage": {
                    "current": list(self.system_metrics["disk_usage"])[-1]
                    if self.system_metrics["disk_usage"]
                    else 0,
                    "average": statistics.mean(self.system_metrics["disk_usage"])
                    if self.system_metrics["disk_usage"]
                    else 0,
                    "max": max(self.system_metrics["disk_usage"])
                    if self.system_metrics["disk_usage"]
                    else 0,
                },
            }

        # Application metrics
        summary["application_metrics"] = {
            "requests_total": self.app_metrics["requests_total"],
            "requests_per_second": list(self.app_metrics["requests_per_second"])[-1]
            if self.app_metrics["requests_per_second"]
            else 0,
            "average_response_time": statistics.mean(self.app_metrics["response_times"])
            if self.app_metrics["response_times"]
            else 0,
            "active_conversations": self.app_metrics["active_conversations"],
            "training_iterations": self.app_metrics["training_iterations"],
            "average_reward_score": statistics.mean(self.app_metrics["reward_scores"])
            if self.app_metrics["reward_scores"]
            else 0,
            "error_count": self.app_metrics["error_count"],
        }

        # Custom metrics
        for metric_name, metric_list in self.metrics.items():
            if metric_list:
                latest_metric = metric_list[-1]
                summary["custom_metrics"][metric_name] = {
                    "type": latest_metric.type.value,
                    "value": latest_metric.value,
                    "labels": latest_metric.labels,
                    "timestamp": latest_metric.timestamp.isoformat(),
                }

        return summary


class AlertManager:
    """Manages alerts and notifications with thread-safe operations"""

    def __init__(self):
        self.alerts: deque = deque(maxlen=10000)  # Bounded to prevent memory leak
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable] = []
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__ + ".AlertManager")

    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity,
        message: str,
        labels: Dict[str, str] = None,
    ):
        """Add an alert rule"""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "message": message,
            "labels": labels or {},
        }
        self.alert_rules.append(rule)

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)

    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics (thread-safe)"""
        # Take a snapshot of rules to avoid holding lock during condition evaluation
        with self._lock:
            rules_snapshot = list(self.alert_rules)

        for rule in rules_snapshot:
            try:
                condition_met = rule["condition"](metrics)

                with self._lock:
                    if condition_met:
                        # Check if alert already exists
                        existing_alert = next(
                            (
                                a
                                for a in self.alerts
                                if a.name == rule["name"] and not a.resolved
                            ),
                            None,
                        )

                        if not existing_alert:
                            # Create new alert
                            alert = Alert(
                                id=str(uuid.uuid4()),
                                name=rule["name"],
                                severity=rule["severity"],
                                message=rule["message"],
                                labels=rule["labels"],
                            )

                            self.alerts.append(alert)

                            # Send notifications (outside lock to avoid deadlock)
                            handlers_copy = list(self.notification_handlers)
                    else:
                        existing_alert = None
                        handlers_copy = []

                        # Check if we should resolve existing alert
                        for a in self.alerts:
                            if a.name == rule["name"] and not a.resolved:
                                a.resolved = True
                                a.resolved_at = datetime.now()
                                break

                # Send notifications outside the lock
                if condition_met and not existing_alert:
                    for handler in handlers_copy:
                        try:
                            handler(alert)
                        except Exception as e:
                            self._logger.warning(f"Notification handler error: {e}")

            except Exception as e:
                self._logger.warning(f"Error checking alert rule {rule['name']}: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [a for a in self.alerts if not a.resolved]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_alerts = self.get_active_alerts()

        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "alerts_by_severity": {
                severity.value: len(
                    [a for a in active_alerts if a.severity == severity]
                )
                for severity in AlertSeverity
            },
            "recent_alerts": [a.to_dict() for a in active_alerts[-10:]],
        }


class HealthChecker:
    """Manages health checks"""

    def __init__(self):
        self.health_checks: List[HealthCheck] = []

    def add_health_check(self, health_check: HealthCheck):
        """Add a health check"""
        self.health_checks.append(health_check)

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = []

        for check in self.health_checks:
            result = await check.run()
            results.append(result)

        # Overall health status
        all_healthy = all(r["status"] == "healthy" for r in results)

        critical_failing = any(
            r["status"] != "healthy" and r["critical"] for r in results
        )

        overall_status = (
            "healthy"
            if all_healthy
            else ("critical" if critical_failing else "degraded")
        )

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "summary": {
                "total_checks": len(results),
                "healthy": len([r for r in results if r["status"] == "healthy"]),
                "unhealthy": len([r for r in results if r["status"] != "healthy"]),
                "critical_failing": len(
                    [r for r in results if r["status"] != "healthy" and r["critical"]]
                ),
            },
        }


class MonitoringService:
    """Main monitoring service that orchestrates all monitoring functionality"""

    def __init__(
        self,
        enable_prometheus: bool = True,
        prometheus_port: int = 8000,
        enable_health_checks: bool = True,
        enable_alerts: bool = True,
    ):
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.enable_health_checks = enable_health_checks
        self.enable_alerts = enable_alerts

        # Initialize components
        self.metrics_collector = MetricsCollector(enable_prometheus)
        self.alert_manager = AlertManager() if enable_alerts else None
        self.health_checker = HealthChecker() if enable_health_checks else None

        # Logger for this service
        self._logger = logging.getLogger(__name__ + ".MonitoringService")

        # Start Prometheus server
        if enable_prometheus and HAS_PROMETHEUS:
            try:
                start_http_server(prometheus_port)
                self._logger.info(f"Prometheus metrics server started on port {prometheus_port}")
            except Exception as e:
                self._logger.error(f"Failed to start Prometheus server: {e}")

        # Setup default health checks
        self._setup_default_health_checks()

        # Setup default alert rules
        self._setup_default_alert_rules()

        # Start monitoring loop
        self._start_monitoring_loop()

    def _setup_default_health_checks(self):
        """Setup default health checks"""
        if not self.health_checker:
            return

        # Database health check
        def database_check():
            # This would check database connectivity
            return True

        self.health_checker.add_health_check(
            HealthCheck("database", database_check, critical=True)
        )

        # Cache health check
        def cache_check():
            # This would check cache connectivity
            return True

        self.health_checker.add_health_check(
            HealthCheck("cache", cache_check, critical=False)
        )

        # System resources check
        def system_resources_check():
            if not HAS_PSUTIL:
                return True

            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            return cpu_percent < 90 and memory_percent < 90

        self.health_checker.add_health_check(
            HealthCheck("system_resources", system_resources_check, critical=False)
        )

    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        if not self.alert_manager:
            return

        # High CPU usage
        def high_cpu_condition(metrics):
            cpu_usage = (
                metrics.get("system_metrics", {}).get("cpu_usage", {}).get("current", 0)
            )
            return cpu_usage > 80

        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            high_cpu_condition,
            AlertSeverity.MEDIUM,
            "CPU usage is above 80%",
        )

        # High memory usage
        def high_memory_condition(metrics):
            memory_usage = (
                metrics.get("system_metrics", {})
                .get("memory_usage", {})
                .get("current", 0)
            )
            return memory_usage > 85

        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            high_memory_condition,
            AlertSeverity.MEDIUM,
            "Memory usage is above 85%",
        )

        # High error rate
        def high_error_rate_condition(metrics):
            error_count = metrics.get("application_metrics", {}).get("error_count", 0)
            return error_count > 100

        self.alert_manager.add_alert_rule(
            "high_error_rate",
            high_error_rate_condition,
            AlertSeverity.HIGH,
            "Error count is above 100",
        )

    def _start_monitoring_loop(self):
        """Start the monitoring loop"""

        def monitoring_loop():
            while True:
                try:
                    # Get current metrics
                    metrics = self.metrics_collector.get_metrics_summary()

                    # Check alerts
                    if self.alert_manager:
                        self.alert_manager.check_alerts(metrics)

                except Exception as e:
                    self._logger.warning(f"Error in monitoring loop: {e}")

                time.sleep(60)  # Check every minute

        # Start in background thread
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()

    @contextmanager
    def track_request(self, method: str, endpoint: str):
        """Context manager for tracking API requests"""
        start_time = time.time()

        try:
            yield

            # Record successful request
            duration = time.time() - start_time
            self.metrics_collector.increment_counter(
                "requests_total",
                labels={"method": method, "endpoint": endpoint, "status": "success"},
            )
            self.metrics_collector.observe_histogram(
                "request_duration",
                duration,
                labels={"method": method, "endpoint": endpoint},
            )

            # Update app metrics
            self.metrics_collector.app_metrics["requests_total"] += 1
            self.metrics_collector.app_metrics["response_times"].append(duration)

        except Exception as e:
            # Record failed request
            duration = time.time() - start_time
            self.metrics_collector.increment_counter(
                "requests_total",
                labels={"method": method, "endpoint": endpoint, "status": "error"},
            )
            self.metrics_collector.increment_counter(
                "errors_total", labels={"error_type": type(e).__name__}
            )

            # Update app metrics
            self.metrics_collector.app_metrics["error_count"] += 1

            raise

    async def log_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Log a custom metric"""
        metric = Metric(
            name=name, type=MetricType.GAUGE, value=value, labels=labels or {}
        )
        self.metrics_collector.record_metric(metric)

    async def log_event(self, event_type: str, metadata: Dict[str, Any]):
        """Log an event"""
        self.metrics_collector.increment_counter(
            "events_total", labels={"event_type": event_type}
        )

        # Could also send to external systems like Elasticsearch
        print(f"Event logged: {event_type} - {metadata}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return self.metrics_collector.get_metrics_summary()

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        if not self.health_checker:
            return {"status": "health_checks_disabled"}

        return await self.health_checker.run_all_checks()

    async def get_alerts(self) -> Dict[str, Any]:
        """Get alert summary"""
        if not self.alert_manager:
            return {"alerts": "disabled"}

        return self.alert_manager.get_alert_summary()

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": await self.get_metrics(),
            "health": await self.get_health_status(),
            "alerts": await self.get_alerts(),
        }


# Global monitoring instance
_global_monitoring: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get global monitoring service"""
    global _global_monitoring

    if _global_monitoring is None:
        _global_monitoring = MonitoringService()

    return _global_monitoring


def setup_monitoring(
    enable_prometheus: bool = True,
    prometheus_port: int = 8000,
    enable_health_checks: bool = True,
    enable_alerts: bool = True,
) -> MonitoringService:
    """Setup global monitoring"""
    global _global_monitoring

    _global_monitoring = MonitoringService(
        enable_prometheus=enable_prometheus,
        prometheus_port=prometheus_port,
        enable_health_checks=enable_health_checks,
        enable_alerts=enable_alerts,
    )

    return _global_monitoring
