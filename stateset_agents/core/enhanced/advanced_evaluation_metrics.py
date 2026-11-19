"""Evaluation values with explicit, value-bound measurement provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

METRIC_GROUPS = {
    "performance": (
        "response_quality",
        "coherence_score",
        "relevance_score",
        "helpfulness_score",
        "safety_score",
    ),
    "efficiency": ("response_time", "memory_usage", "throughput"),
    "robustness": ("error_rate", "recovery_rate", "edge_case_handling"),
    "user_experience": ("user_satisfaction", "engagement_score", "conversation_flow"),
    "advanced": ("reasoning_quality", "memory_utilization", "adaptability_score"),
    "statistics": ("confidence_interval", "standard_deviation", "sample_size"),
}
METRIC_NAMES = frozenset(name for names in METRIC_GROUPS.values() for name in names)
_UNIT_INTERVAL_METRICS = frozenset(
    name
    for group in ("performance", "robustness", "user_experience", "advanced")
    for name in METRIC_GROUPS[group]
    if name != "memory_utilization"
)


class MetricEvidenceStatus(str, Enum):
    """Whether a value is observed, derived from a proxy, or unsupported."""

    MEASURED = "measured"
    INFERRED = "inferred"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MetricEvidence:
    """A declaration bound to one numeric value and measurement method.

    This records the producer's declaration, not an independent attestation.
    ``source`` identifies the method; comparisons also require matching cases.
    """

    status: MetricEvidenceStatus
    value: float | int | tuple[float, float] | None = None
    source: str | None = None
    sample_size: int | None = None
    reason: str | None = None
    scope: str | None = None

    @property
    def available(self) -> bool:
        """Whether this declaration supports using the captured value."""
        return self.status is not MetricEvidenceStatus.UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Return detached, JSON-compatible evidence metadata."""
        return {
            "status": self.status.value,
            "value": self.value,
            "source": self.source,
            "sample_size": self.sample_size,
            "reason": self.reason,
            "scope": self.scope,
        }


def _validate_value(name: str, value: Any) -> None:
    if name not in METRIC_NAMES:
        raise ValueError(f"Unknown evaluation metric: {name}")
    if name == "confidence_interval":
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
            or not 0 <= value[0] <= value[1] <= 1
            or not all(math.isfinite(item) for item in value)
        ):
            raise ValueError(
                "A quality confidence interval must be a finite ordered pair in [0, 1]"
            )
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if name in _UNIT_INTERVAL_METRICS and value > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    if name == "sample_size" and not isinstance(value, int):
        raise ValueError("sample_size must be an integer")


@dataclass
class EvaluationMetrics:
    """Legacy numeric values with separate, explicit evidence declarations.

    Constructor values and direct assignments do not establish measurement.
    Use ``record`` to bind a measured/inferred value to its method. A subsequent
    value change invalidates that declaration until the producer records it again.
    """

    response_quality: float = 0.0
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    helpfulness_score: float = 0.0
    safety_score: float = 1.0
    response_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    recovery_rate: float = 1.0
    edge_case_handling: float = 0.0
    user_satisfaction: float = 0.0
    engagement_score: float = 0.0
    conversation_flow: float = 0.0
    reasoning_quality: float = 0.0
    memory_utilization: float = 0.0
    adaptability_score: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    standard_deviation: float = 0.0
    sample_size: int = 0

    def __post_init__(self) -> None:
        self._evidence: dict[str, MetricEvidence] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        evidence: MetricEvidence | None = getattr(self, "_evidence", {}).get(name)
        if evidence is not None and evidence.available:
            try:
                _validate_value(name, value)
                unchanged = value == evidence.value
            except ValueError:
                unchanged = False
            if not unchanged:
                self._evidence[name] = MetricEvidence(
                    MetricEvidenceStatus.UNAVAILABLE,
                    reason="The value changed after evidence was recorded",
                )
        object.__setattr__(self, name, value)

    def record(
        self,
        name: str,
        value: float | int | tuple[float, float],
        *,
        source: str,
        status: MetricEvidenceStatus | str = MetricEvidenceStatus.MEASURED,
        sample_size: int | None = None,
        scope: str | None = None,
    ) -> None:
        """Validate and record a value and its measurement or inference method."""
        _validate_value(name, value)
        status = MetricEvidenceStatus(status)
        if status is MetricEvidenceStatus.UNAVAILABLE:
            raise ValueError("Use mark_unavailable() for an unsupported metric")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Metric evidence requires a nonempty source method")
        if sample_size is not None and (
            type(sample_size) is not int or sample_size <= 0
        ):
            raise ValueError("Evidence sample_size must be a positive integer or None")
        if scope is not None and (not isinstance(scope, str) or not scope.strip()):
            raise ValueError("Evidence scope must be a nonempty string or None")
        evidence = MetricEvidence(status, value, source, sample_size, scope=scope)
        setattr(self, name, value)
        if not hasattr(self, "_evidence"):
            self._evidence = {}
        self._evidence[name] = evidence

    def mark_unavailable(self, name: str, reason: str) -> None:
        """Retire evidence without rewriting the compatible numeric attribute."""
        if name not in METRIC_NAMES:
            raise ValueError(f"Unknown evaluation metric: {name}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Unavailable evidence requires a reason")
        if not hasattr(self, "_evidence"):
            self._evidence = {}
        self._evidence[name] = MetricEvidence(
            MetricEvidenceStatus.UNAVAILABLE, reason=reason
        )

    def evidence(self, name: str) -> MetricEvidence:
        """Observe evidence, treating changed or invalid numeric values as unavailable."""
        if name not in METRIC_NAMES:
            raise ValueError(f"Unknown evaluation metric: {name}")
        evidence: MetricEvidence | None = getattr(self, "_evidence", {}).get(name)
        if evidence is None:
            return MetricEvidence(
                MetricEvidenceStatus.UNAVAILABLE,
                reason="No measurement or inference was recorded",
            )
        if not evidence.available:
            return evidence
        value = getattr(self, name)
        try:
            _validate_value(name, value)
        except ValueError:
            return MetricEvidence(
                MetricEvidenceStatus.UNAVAILABLE, reason="The current value is invalid"
            )
        if value != evidence.value:
            return MetricEvidence(
                MetricEvidenceStatus.UNAVAILABLE,
                reason="The value changed after evidence was recorded",
            )
        return evidence

    def to_dict(self, *, include_unavailable_values: bool = True) -> dict[str, Any]:
        """Serialize groups and provenance; optionally replace unsupported values with None.

        The default retains existing numeric fields for compatibility. Reports
        use the evidence-only projection so defaults cannot look measured.
        """
        evidence = {
            name: self.evidence(name)
            for names in METRIC_GROUPS.values()
            for name in names
        }
        groups = {
            group: {
                name: (
                    getattr(self, name)
                    if include_unavailable_values
                    else evidence[name].value
                )
                for name in names
            }
            for group, names in METRIC_GROUPS.items()
        }
        return {
            **groups,
            "provenance": {name: item.to_dict() for name, item in evidence.items()},
        }
