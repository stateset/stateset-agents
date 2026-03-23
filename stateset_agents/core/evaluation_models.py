"""Model types for the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def _format_confidence_level(confidence_level: float) -> str:
    confidence_pct = float(confidence_level) * 100
    if confidence_pct.is_integer():
        return f"{int(confidence_pct)}%"
    return f"{confidence_pct:.1f}%"


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    QUALITY = "quality"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SAFETY = "safety"
    CONSISTENCY = "consistency"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    metrics: list[str] = field(
        default_factory=lambda: ["relevance", "coherence", "helpfulness", "latency"]
    )
    num_samples: int | None = None
    timeout_seconds: float = 30.0
    parallel_workers: int = 4
    cache_responses: bool = True
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95


@dataclass
class EvaluationSample:
    """A single evaluation sample."""

    id: str
    input: str | list[dict[str, str]]
    expected_output: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result of a single metric computation."""

    metric_name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""

    sample_id: str
    input: str | list[dict[str, str]]
    output: str
    expected_output: str | None
    metrics: list[MetricResult]
    latency_ms: float
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def get_metric(self, name: str) -> float | None:
        """Get a specific metric value."""
        for metric in self.metrics:
            if metric.metric_name == name:
                return metric.value
        return None


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    config: EvaluationConfig
    samples: list[SampleResult]
    aggregate_metrics: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    @property
    def total_samples(self) -> int:
        return len(self.samples)

    @property
    def successful_samples(self) -> int:
        return sum(1 for sample in self.samples if sample.error is None)

    @property
    def failed_samples(self) -> int:
        return sum(1 for sample in self.samples if sample.error is not None)

    @property
    def success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.successful_samples / self.total_samples

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"Total Samples: {self.total_samples}",
            f"Successful: {self.successful_samples}",
            f"Failed: {self.failed_samples}",
            f"Success Rate: {self.success_rate:.1%}",
            "",
            "METRICS:",
        ]
        confidence_label = _format_confidence_level(self.config.confidence_level)

        for metric_name, value in self.aggregate_metrics.items():
            interval = self.confidence_intervals.get(metric_name)
            if interval:
                lines.append(
                    f"  {metric_name}: {value:.4f} ({confidence_label} CI: "
                    f"[{interval[0]:.4f}, {interval[1]:.4f}])"
                )
            else:
                lines.append(f"  {metric_name}: {value:.4f}")

        duration = self.duration_seconds
        if duration is not None:
            lines.append(f"\nDuration: {duration:.2f}s")
            if duration > 0:
                lines.append(
                    f"Throughput: {self.total_samples / duration:.2f} samples/s"
                )
            else:
                lines.append("Throughput: n/a")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "success_rate": self.success_rate,
            "aggregate_metrics": self.aggregate_metrics,
            "confidence_intervals": self.confidence_intervals,
            "duration_seconds": self.duration_seconds,
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "metrics": {
                        metric.metric_name: metric.value for metric in sample.metrics
                    },
                    "latency_ms": sample.latency_ms,
                    "error": sample.error,
                }
                for sample in self.samples
            ],
        }


__all__ = [
    "EvaluationConfig",
    "EvaluationResults",
    "EvaluationSample",
    "MetricResult",
    "MetricType",
    "SampleResult",
]
