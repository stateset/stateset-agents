"""Model types for the intelligent orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class OrchestrationMode(Enum):
    """Modes of orchestration."""

    MANUAL = "manual"
    SEMI_AUTOMATED = "semi_automated"
    FULLY_AUTOMATED = "fully_automated"
    ADAPTIVE = "adaptive"


class OptimizationObjective(Enum):
    """Optimization objectives."""

    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    SPEED = "speed"
    ROBUSTNESS = "robustness"
    BALANCED = "balanced"


class ComponentStatus(Enum):
    """Status of framework components."""

    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    OPTIMIZING = "optimizing"
    UPDATING = "updating"


@dataclass
class OrchestrationConfig:
    """Configuration for intelligent orchestration."""

    mode: OrchestrationMode = OrchestrationMode.ADAPTIVE
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED
    enable_adaptive_learning: bool = True
    enable_nas: bool = True
    enable_multimodal: bool = True
    enable_auto_optimization: bool = True
    performance_threshold: float = 0.8
    adaptation_interval: timedelta = timedelta(minutes=30)
    nas_interval: timedelta = timedelta(hours=4)
    optimization_interval: timedelta = timedelta(hours=1)
    max_gpu_memory_usage: float = 0.9
    max_cpu_usage: float = 0.8
    max_concurrent_processes: int = 4

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for status and debugging APIs."""
        return {
            "mode": self.mode.value,
            "optimization_objective": self.optimization_objective.value,
            "enable_adaptive_learning": self.enable_adaptive_learning,
            "enable_nas": self.enable_nas,
            "enable_multimodal": self.enable_multimodal,
            "enable_auto_optimization": self.enable_auto_optimization,
            "performance_threshold": self.performance_threshold,
            "adaptation_interval_minutes": self.adaptation_interval.total_seconds()
            / 60,
            "nas_interval_hours": self.nas_interval.total_seconds() / 3600,
            "optimization_interval_hours": self.optimization_interval.total_seconds()
            / 3600,
            "max_gpu_memory_usage": self.max_gpu_memory_usage,
            "max_cpu_usage": self.max_cpu_usage,
            "max_concurrent_processes": self.max_concurrent_processes,
        }


@dataclass
class ComponentState:
    """State of a framework component."""

    component_id: str
    status: ComponentStatus
    last_update: datetime
    performance_metrics: dict[str, float] = field(default_factory=dict)
    resource_usage: dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    success_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_performance(self, metrics: dict[str, float]):
        """Update performance metrics."""
        self.performance_metrics.update(metrics)
        self.last_update = datetime.now()
        self.success_count += 1

    def record_error(self, error_info: str):
        """Record an error."""
        self.error_count += 1
        self.metadata["last_error"] = error_info
        self.metadata["last_error_time"] = datetime.now().isoformat()
        self.status = ComponentStatus.ERROR


@dataclass
class OrchestrationDecision:
    """A decision made by the orchestrator."""

    decision_id: str
    timestamp: datetime
    decision_type: str
    component: str
    action: str
    reasoning: str
    expected_benefit: float
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision for logs and responses."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "decision_type": self.decision_type,
            "component": self.component,
            "action": self.action,
            "reasoning": self.reasoning,
            "expected_benefit": self.expected_benefit,
            "confidence": self.confidence,
            "parameters": self.parameters,
        }


__all__ = [
    "ComponentState",
    "ComponentStatus",
    "OptimizationObjective",
    "OrchestrationConfig",
    "OrchestrationDecision",
    "OrchestrationMode",
]
