"""
Advanced Real-Time Monitoring Dashboard

Production-grade monitoring with live metrics, alerts, and visualization.
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import prometheus_client as prom
    from prometheus_client import Counter, Gauge, Histogram, Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time"""

    timestamp: datetime
    metrics: Dict[str, float]
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert triggered by metric threshold"""

    name: str
    severity: str  # info, warning, error, critical
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False


class MetricAggregator:
    """Aggregates metrics over time windows"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def add(self, metric_name: str, value: float):
        """Add a metric value"""
        self.metrics[metric_name].append(value)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = list(self.metrics.get(metric_name, []))

        if not values:
            return {}

        import numpy as np

        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}


class AlertManager:
    """Manages alerts and thresholds"""

    def __init__(self, alert_callback: Optional[Callable[[Alert], None]] = None):
        self.alert_callback = alert_callback
        self.thresholds: Dict[str, List[Tuple[str, float, Callable]]] = defaultdict(
            list
        )
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)

    def add_threshold(
        self,
        metric_name: str,
        severity: str,
        threshold: float,
        condition: Callable[[float, float], bool],
    ):
        """
        Add an alert threshold

        Args:
            metric_name: Name of metric to monitor
            severity: Alert severity level
            threshold: Threshold value
            condition: Function(value, threshold) -> bool to check if alert should fire
        """
        self.thresholds[metric_name].append((severity, threshold, condition))

    def check_metric(self, metric_name: str, value: float):
        """Check if metric triggers any alerts"""
        if metric_name not in self.thresholds:
            return

        for severity, threshold, condition in self.thresholds[metric_name]:
            alert_key = f"{metric_name}_{severity}"

            if condition(value, threshold):
                # Trigger alert
                if alert_key not in self.active_alerts:
                    alert = Alert(
                        name=alert_key,
                        severity=severity,
                        message=f"{metric_name} {condition.__name__} {threshold}: {value:.2f}",
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=threshold,
                        timestamp=datetime.now(),
                    )
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)

                    if self.alert_callback:
                        self.alert_callback(alert)

                    logger.warning(f"Alert triggered: {alert.message}")

            else:
                # Resolve alert if it exists
                if alert_key in self.active_alerts:
                    alert = self.active_alerts.pop(alert_key)
                    alert.resolved = True
                    logger.info(f"Alert resolved: {alert.message}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alert_history if a.timestamp >= cutoff]


class AdvancedDashboard:
    """
    Advanced monitoring dashboard with real-time metrics and alerts
    """

    def __init__(
        self,
        enable_prometheus: bool = True,
        metrics_window: int = 10000,
        alert_callback: Optional[Callable[[Alert], None]] = None,
    ):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.aggregator = MetricAggregator(window_size=metrics_window)
        self.alert_manager = AlertManager(alert_callback=alert_callback)

        # Prometheus metrics (if available)
        if self.enable_prometheus:
            self._init_prometheus_metrics()

        # Snapshot history
        self.snapshots: deque = deque(maxlen=1000)

        # Start background tasks
        self._tasks: List[asyncio.Task] = []

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not self.enable_prometheus:
            return

        # Training metrics
        self.prom_train_loss = Gauge(
            "grpo_train_loss", "Training loss", ["episode", "model"]
        )
        self.prom_train_reward = Gauge(
            "grpo_train_reward", "Training reward", ["episode", "model"]
        )
        self.prom_train_kl = Gauge("grpo_kl_divergence", "KL divergence", ["model"])

        # System metrics
        self.prom_gpu_memory = Gauge("gpu_memory_usage_bytes", "GPU memory usage")
        self.prom_cpu_percent = Gauge("cpu_usage_percent", "CPU usage percentage")

        # Request metrics
        self.prom_requests = Counter("api_requests_total", "Total API requests")
        self.prom_latency = Histogram(
            "api_latency_seconds", "API latency", buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )

        logger.info("✓ Prometheus metrics initialized")

    async def log_metric(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ):
        """Log a metric value"""
        # Add to aggregator
        self.aggregator.add(name, value)

        # Check alerts
        self.alert_manager.check_metric(name, value)

        # Update Prometheus
        if self.enable_prometheus:
            self._update_prometheus(name, value, tags or {})

    def _update_prometheus(self, name: str, value: float, tags: Dict[str, str]):
        """Update Prometheus metrics"""
        if not self.enable_prometheus:
            return

        # Map custom metrics to Prometheus
        if "loss" in name.lower():
            self.prom_train_loss.labels(**tags).set(value)
        elif "reward" in name.lower():
            self.prom_train_reward.labels(**tags).set(value)
        elif "kl" in name.lower():
            self.prom_train_kl.labels(**tags).set(value)

    def add_alert_threshold(
        self, metric_name: str, severity: str, threshold: float, condition_type: str
    ):
        """
        Add an alert threshold

        Args:
            metric_name: Name of metric
            severity: info, warning, error, critical
            threshold: Threshold value
            condition_type: 'above', 'below', 'equals'
        """
        conditions = {
            "above": lambda v, t: v > t,
            "below": lambda v, t: v < t,
            "equals": lambda v, t: abs(v - t) < 1e-6,
        }

        condition = conditions.get(condition_type, conditions["above"])
        condition.__name__ = condition_type

        self.alert_manager.add_threshold(metric_name, severity, threshold, condition)

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        return self.aggregator.get_stats(metric_name)

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all metric statistics"""
        return self.aggregator.get_all_stats()

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return self.alert_manager.get_active_alerts()

    def take_snapshot(self, metrics: Dict[str, float], tags: Optional[Dict[str, str]] = None):
        """Take a snapshot of current metrics"""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(), metrics=metrics, tags=tags or {}
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_snapshot_history(self, hours: int = 1) -> List[MetricSnapshot]:
        """Get snapshot history"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [s for s in self.snapshots if s.timestamp >= cutoff]

    async def start_monitoring(self):
        """Start background monitoring tasks"""
        self._tasks = [
            asyncio.create_task(self._monitor_system_metrics()),
            asyncio.create_task(self._periodic_snapshots()),
        ]

    async def stop_monitoring(self):
        """Stop background monitoring"""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _monitor_system_metrics(self):
        """Monitor system metrics in background"""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available for system monitoring")
            return

        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                await self.log_metric("system.cpu_percent", cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                await self.log_metric("system.memory_percent", memory.percent)

                # GPU metrics (if available)
                try:
                    import torch

                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            memory_allocated = torch.cuda.memory_allocated(i)
                            memory_reserved = torch.cuda.memory_reserved(i)
                            await self.log_metric(
                                f"gpu.{i}.memory_allocated", memory_allocated
                            )
                            await self.log_metric(
                                f"gpu.{i}.memory_reserved", memory_reserved
                            )
                except ImportError:
                    pass

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(10)

    async def _periodic_snapshots(self):
        """Take periodic metric snapshots"""
        while True:
            try:
                all_stats = self.get_all_stats()
                if all_stats:
                    # Flatten stats for snapshot
                    snapshot_metrics = {}
                    for metric_name, stats in all_stats.items():
                        for stat_name, value in stats.items():
                            snapshot_metrics[f"{metric_name}.{stat_name}"] = value

                    self.take_snapshot(snapshot_metrics)

                await asyncio.sleep(60)  # Snapshot every minute

            except Exception as e:
                logger.error(f"Error taking snapshot: {e}")
                await asyncio.sleep(60)

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get a summary for dashboard display"""
        stats = self.get_all_stats()
        alerts = self.get_active_alerts()
        recent_snapshots = self.get_snapshot_history(hours=1)

        return {
            "timestamp": datetime.now().isoformat(),
            "metrics_count": len(stats),
            "active_alerts": len(alerts),
            "alerts": [
                {
                    "name": a.name,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in alerts
            ],
            "metrics_summary": {
                name: {
                    "current": s.get("mean", 0),
                    "trend": "stable",  # Could compute from snapshots
                    "min": s.get("min", 0),
                    "max": s.get("max", 0),
                }
                for name, s in stats.items()
            },
            "snapshots_count": len(recent_snapshots),
        }

    def print_dashboard(self):
        """Print a text-based dashboard"""
        summary = self.get_dashboard_summary()

        print("\n" + "=" * 80)
        print("📊 GRPO Training Dashboard".center(80))
        print("=" * 80)
        print(f"Time: {summary['timestamp']}")
        print(f"Metrics Tracked: {summary['metrics_count']}")
        print(f"Active Alerts: {summary['active_alerts']}")
        print()

        # Alerts
        if summary["alerts"]:
            print("🚨 ACTIVE ALERTS:")
            for alert in summary["alerts"]:
                severity_icon = {
                    "info": "ℹ️",
                    "warning": "⚠️",
                    "error": "❌",
                    "critical": "🔥",
                }.get(alert["severity"], "⚠️")
                print(f"  {severity_icon} {alert['message']}")
            print()

        # Metrics
        print("📈 METRICS SUMMARY:")
        for name, metrics in summary["metrics_summary"].items():
            print(
                f"  {name:30s} Current: {metrics['current']:8.4f} "
                f"[{metrics['min']:8.4f}, {metrics['max']:8.4f}]"
            )

        print("=" * 80 + "\n")


# Convenience function
def create_production_dashboard(
    enable_alerts: bool = True,
) -> AdvancedDashboard:
    """Create a production-ready dashboard with standard thresholds"""

    def alert_handler(alert: Alert):
        """Handle alerts (could send to Slack, PagerDuty, etc.)"""
        logger.warning(f"Alert: {alert.message}")

    dashboard = AdvancedDashboard(
        enable_prometheus=True,
        metrics_window=10000,
        alert_callback=alert_handler if enable_alerts else None,
    )

    # Add standard alert thresholds
    if enable_alerts:
        # Training alerts
        dashboard.add_alert_threshold("train.loss", "warning", 10.0, "above")
        dashboard.add_alert_threshold("train.loss", "critical", 100.0, "above")
        dashboard.add_alert_threshold("train.kl_divergence", "warning", 0.5, "above")

        # System alerts
        dashboard.add_alert_threshold("system.cpu_percent", "warning", 80.0, "above")
        dashboard.add_alert_threshold("system.cpu_percent", "critical", 95.0, "above")
        dashboard.add_alert_threshold("system.memory_percent", "warning", 85.0, "above")

    return dashboard
