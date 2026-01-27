"""
Unit tests for the Advanced Monitoring module.

Tests cover metrics collection, alerting, distributed tracing,
and observability features.
"""

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateset_agents.core.advanced_monitoring import (
    Alert,
    AlertSeverity,
    MetricPoint,
    MetricsCollector,
    MetricType,
    TraceSpan,
)


class TestMetricType:
    """Test MetricType enum."""

    def test_metric_type_values(self):
        """Test that all metric types have expected values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_alert_severity_values(self):
        """Test that all alert severities have expected values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestMetricPoint:
    """Test MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test creating a MetricPoint."""
        point = MetricPoint(
            name="cpu_usage",
            value=0.75,
            timestamp=time.time(),
            labels={"host": "server1"},
            metric_type=MetricType.GAUGE,
        )

        assert point.name == "cpu_usage"
        assert point.value == 0.75
        assert point.labels["host"] == "server1"
        assert point.metric_type == MetricType.GAUGE

    def test_metric_point_defaults(self):
        """Test MetricPoint default values."""
        point = MetricPoint(
            name="test_metric",
            value=1.0,
            timestamp=time.time(),
        )

        assert point.labels == {}
        assert point.metric_type == MetricType.GAUGE


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an Alert."""
        alert = Alert(
            id="alert_001",
            name="High CPU Usage",
            condition="cpu_usage > 0.9",
            severity=AlertSeverity.WARNING,
            threshold=0.9,
            duration=60.0,
        )

        assert alert.id == "alert_001"
        assert alert.name == "High CPU Usage"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.threshold == 0.9
        assert alert.enabled is True

    def test_alert_defaults(self):
        """Test Alert default values."""
        alert = Alert(
            id="alert_001",
            name="Test Alert",
            condition="test > 0",
            severity=AlertSeverity.INFO,
            threshold=0.5,
            duration=30.0,
        )

        assert alert.enabled is True
        assert alert.last_triggered is None
        assert alert.metadata == {}


class TestTraceSpan:
    """Test TraceSpan dataclass."""

    def test_trace_span_creation(self):
        """Test creating a TraceSpan."""
        span = TraceSpan(
            trace_id="trace_001",
            span_id="span_001",
            parent_span_id=None,
            operation_name="process_request",
            start_time=time.time(),
        )

        assert span.trace_id == "trace_001"
        assert span.span_id == "span_001"
        assert span.parent_span_id is None
        assert span.operation_name == "process_request"
        assert span.end_time is None
        assert span.duration is None

    def test_trace_span_with_parent(self):
        """Test TraceSpan with parent span."""
        span = TraceSpan(
            trace_id="trace_001",
            span_id="span_002",
            parent_span_id="span_001",
            operation_name="child_operation",
            start_time=time.time(),
            tags={"component": "database"},
        )

        assert span.parent_span_id == "span_001"
        assert span.tags["component"] == "database"

    def test_trace_span_finish(self):
        """Test finishing a TraceSpan."""
        start_time = time.time()
        span = TraceSpan(
            trace_id="trace_001",
            span_id="span_001",
            parent_span_id=None,
            operation_name="test",
            start_time=start_time,
        )

        time.sleep(0.01)  # Small delay
        span.finish()

        assert span.end_time is not None
        assert span.end_time > span.start_time
        assert span.duration is not None
        assert span.duration >= 0.01


class TestMetricsCollector:
    """Test MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for testing."""
        with patch("core.advanced_monitoring.PROMETHEUS_AVAILABLE", False):
            collector = MetricsCollector(retention_period=3600)
            # Cleanup task is now lazily initialized, so no need to cancel
            return collector

    def test_collector_creation(self, collector):
        """Test collector creation."""
        assert collector.retention_period == 3600
        assert len(collector.metrics) == 0

    def test_record_metric_gauge(self, collector):
        """Test recording a gauge metric."""
        collector.record_metric(
            name="cpu_usage",
            value=0.75,
            labels={"host": "server1"},
            metric_type=MetricType.GAUGE,
        )

        assert len(collector.metrics) == 1
        metric_key = list(collector.metrics.keys())[0]
        assert "cpu_usage" in metric_key
        assert len(collector.metrics[metric_key]) == 1

    def test_record_metric_counter(self, collector):
        """Test recording a counter metric."""
        collector.record_metric(
            name="requests_total",
            value=100,
            labels={"endpoint": "/api/health"},
            metric_type=MetricType.COUNTER,
        )

        metric_key = list(collector.metrics.keys())[0]
        point = collector.metrics[metric_key][0]
        assert point.value == 100
        assert point.metric_type == MetricType.COUNTER

    def test_record_multiple_metrics(self, collector):
        """Test recording multiple metrics."""
        for i in range(5):
            collector.record_metric(
                name="test_metric",
                value=float(i),
                labels={"iteration": str(i)},
            )

        # Each has different labels, so different keys
        assert len(collector.metrics) == 5

    def test_record_same_metric_multiple_times(self, collector):
        """Test recording the same metric multiple times."""
        for i in range(3):
            collector.record_metric(
                name="cpu_usage",
                value=0.5 + i * 0.1,
                labels={"host": "server1"},
            )

        assert len(collector.metrics) == 1
        metric_key = list(collector.metrics.keys())[0]
        assert len(collector.metrics[metric_key]) == 3


class TestMetricsCollectorWithPrometheus:
    """Test MetricsCollector with Prometheus integration."""

    @pytest.fixture
    def collector_with_prometheus(self):
        """Create a MetricsCollector with mocked Prometheus."""
        with patch("core.advanced_monitoring.PROMETHEUS_AVAILABLE", True), \
             patch("core.advanced_monitoring.CollectorRegistry"), \
             patch("core.advanced_monitoring.Counter"), \
             patch("core.advanced_monitoring.Gauge"), \
             patch("core.advanced_monitoring.Histogram"), \
             patch("core.advanced_monitoring.Summary"):
            collector = MetricsCollector()
            # Cleanup task is now lazily initialized, so no need to cancel
            return collector

    def test_prometheus_metrics_setup(self, collector_with_prometheus):
        """Test Prometheus metrics are set up correctly."""
        assert collector_with_prometheus.prometheus_registry is not None
        assert len(collector_with_prometheus.prometheus_metrics) > 0


class TestAdvancedMonitoringService:
    """Test AdvancedMonitoringService class."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock monitoring service."""
        with patch("core.advanced_monitoring.PROMETHEUS_AVAILABLE", False), \
             patch("core.advanced_monitoring.OPENTELEMETRY_AVAILABLE", False):
            from stateset_agents.core.advanced_monitoring import AdvancedMonitoringService
            service = AdvancedMonitoringService()
            return service

    def test_service_creation(self, mock_service):
        """Test service creation."""
        assert mock_service is not None
        assert mock_service.metrics_collector is not None

    def test_record_request(self, mock_service):
        """Test recording API requests."""
        mock_service.record_request("GET", "/api/health", 200, 0.05)

        # Verify metrics were recorded
        assert len(mock_service.metrics_collector.metrics) > 0

    def test_record_training_iteration(self, mock_service):
        """Test recording training iterations."""
        mock_service.record_training_iteration(
            "MultiTurnAgent",
            "grpo",
            {"loss": 0.5, "reward": 0.8}
        )

        # Should record multiple metrics
        assert len(mock_service.metrics_collector.metrics) >= 1

    def test_record_error(self, mock_service):
        """Test recording errors."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            mock_service.record_error("api", "ValueError", e)

        # Should have error metrics
        assert len(mock_service.metrics_collector.metrics) >= 1

    def test_get_health_summary(self, mock_service):
        """Test getting health summary."""
        summary = mock_service.get_health_summary()

        assert "status" in summary
        assert summary["status"] in ["healthy", "degraded", "unhealthy"]

    def test_get_metrics_dashboard(self, mock_service):
        """Test getting metrics dashboard."""
        # Record some metrics first
        mock_service.record_request("GET", "/api/health", 200, 0.05)
        mock_service.record_training_iteration("TestAgent", "grpo", {"loss": 0.5})

        dashboard = mock_service.get_metrics_dashboard()

        assert isinstance(dashboard, dict)


class TestDistributedTracing:
    """Test distributed tracing functionality."""

    def test_create_trace(self):
        """Test creating a new trace."""
        span = TraceSpan(
            trace_id="trace_001",
            span_id="span_001",
            parent_span_id=None,
            operation_name="root_operation",
            start_time=time.time(),
        )

        assert span.trace_id == "trace_001"
        assert span.parent_span_id is None

    def test_create_child_span(self):
        """Test creating child spans."""
        parent = TraceSpan(
            trace_id="trace_001",
            span_id="span_001",
            parent_span_id=None,
            operation_name="parent",
            start_time=time.time(),
        )

        child = TraceSpan(
            trace_id=parent.trace_id,
            span_id="span_002",
            parent_span_id=parent.span_id,
            operation_name="child",
            start_time=time.time(),
        )

        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id

    def test_span_with_logs(self):
        """Test adding logs to spans."""
        span = TraceSpan(
            trace_id="trace_001",
            span_id="span_001",
            parent_span_id=None,
            operation_name="test",
            start_time=time.time(),
            logs=[
                {"timestamp": time.time(), "event": "start"},
                {"timestamp": time.time(), "event": "processing"},
            ],
        )

        assert len(span.logs) == 2


class TestAlertManager:
    """Test alert management functionality."""

    def test_create_cpu_alert(self):
        """Test creating a CPU usage alert."""
        alert = Alert(
            id="cpu_high",
            name="High CPU Usage",
            condition="cpu_usage > threshold",
            severity=AlertSeverity.WARNING,
            threshold=0.9,
            duration=60.0,
            metadata={"escalation": "ops-team"},
        )

        assert alert.threshold == 0.9
        assert alert.metadata["escalation"] == "ops-team"

    def test_create_memory_alert(self):
        """Test creating a memory usage alert."""
        alert = Alert(
            id="memory_critical",
            name="Critical Memory Usage",
            condition="memory_usage > threshold",
            severity=AlertSeverity.CRITICAL,
            threshold=0.95,
            duration=30.0,
        )

        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.duration == 30.0

    def test_alert_disabled(self):
        """Test disabled alert."""
        alert = Alert(
            id="test_alert",
            name="Test",
            condition="test",
            severity=AlertSeverity.INFO,
            threshold=0.5,
            duration=10.0,
            enabled=False,
        )

        assert alert.enabled is False


class TestMonitorAsyncFunction:
    """Test the monitor_async_function decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Test basic decorator functionality."""
        from stateset_agents.core.advanced_monitoring import monitor_async_function

        @monitor_async_function("test_operation")
        async def sample_function():
            await asyncio.sleep(0.01)
            return "result"

        result = await sample_function()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_with_exception(self):
        """Test decorator handles exceptions."""
        from stateset_agents.core.advanced_monitoring import monitor_async_function

        @monitor_async_function("failing_operation")
        async def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_function()


class TestSystemMetrics:
    """Test system metrics collection."""

    def test_collect_system_metrics(self):
        """Test collecting system metrics with psutil."""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        assert 0 <= cpu_percent <= 100
        assert memory.percent >= 0


class TestMetricAggregation:
    """Test metric aggregation functionality."""

    @pytest.fixture
    def collector_with_data(self):
        """Create a collector with sample data."""
        with patch("core.advanced_monitoring.PROMETHEUS_AVAILABLE", False):
            collector = MetricsCollector()
            # Cleanup task is now lazily initialized, so no need to cancel

            # Add sample metrics
            for i in range(10):
                collector.record_metric(
                    name="latency",
                    value=0.1 + i * 0.01,
                    labels={"endpoint": "/api/test"},
                )

            return collector

    def test_metrics_stored(self, collector_with_data):
        """Test that metrics are stored correctly."""
        assert len(collector_with_data.metrics) == 1
        metric_key = list(collector_with_data.metrics.keys())[0]
        assert len(collector_with_data.metrics[metric_key]) == 10

    def test_metrics_values(self, collector_with_data):
        """Test metric values are correct."""
        metric_key = list(collector_with_data.metrics.keys())[0]
        values = [p.value for p in collector_with_data.metrics[metric_key]]

        assert min(values) >= 0.1
        assert max(values) <= 0.2


class TestMetricRetention:
    """Test metric retention and cleanup."""

    @pytest.fixture
    def collector_short_retention(self):
        """Create a collector with short retention."""
        with patch("core.advanced_monitoring.PROMETHEUS_AVAILABLE", False):
            collector = MetricsCollector(retention_period=1)  # 1 second
            # Cleanup task is now lazily initialized, so no need to cancel
            return collector

    def test_retention_period_set(self, collector_short_retention):
        """Test retention period is set correctly."""
        assert collector_short_retention.retention_period == 1


class TestGetMonitoringService:
    """Test the get_monitoring_service function."""

    def test_get_monitoring_service(self):
        """Test getting the global monitoring service."""
        from stateset_agents.core.advanced_monitoring import get_monitoring_service

        service1 = get_monitoring_service()
        service2 = get_monitoring_service()

        # Should return the same instance
        assert service1 is service2
