"""Model types for advanced monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""

    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Alert definition."""

    id: str
    name: str
    condition: str
    severity: AlertSeverity
    threshold: float
    duration: float
    enabled: bool = True
    last_triggered: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Distributed tracing span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: float
    end_time: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    duration: float | None = None

    def finish(self):
        """Finish the span."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


__all__ = [
    "Alert",
    "AlertSeverity",
    "MetricPoint",
    "MetricType",
    "TraceSpan",
]
