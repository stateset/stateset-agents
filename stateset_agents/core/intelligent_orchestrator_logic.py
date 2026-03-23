"""Pure decision and scoring helpers for the intelligent orchestrator."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .intelligent_orchestrator_models import (
    OrchestrationConfig,
    OrchestrationDecision,
)


def calculate_recent_performance(history: list[float]) -> float:
    """Calculate recent performance summary, using median for robustness."""
    if not history:
        return 0.5

    recent_window = min(50, len(history))
    recent_scores = sorted(history[-recent_window:])
    mid = len(recent_scores) // 2
    if len(recent_scores) % 2 == 1:
        return float(recent_scores[mid])
    return float(recent_scores[mid - 1] + recent_scores[mid]) / 2.0


def calculate_performance_trend(history: list[float]) -> float:
    """Calculate performance trend where positive means improvement."""
    if len(history) < 20:
        return 0.0

    recent = history[-10:]
    earlier = history[-20:-10]
    return (sum(recent) / len(recent)) - (sum(earlier) / len(earlier))


def assess_resource_pressure(resource_usage_history: dict[str, list[float]]) -> float:
    """Assess current resource pressure from recent CPU and memory usage."""
    pressure_factors = []

    if "memory" in resource_usage_history:
        recent_memory = resource_usage_history["memory"][-10:] or [0.5]
        pressure_factors.append(sum(recent_memory) / len(recent_memory))

    if "cpu" in resource_usage_history:
        recent_cpu = resource_usage_history["cpu"][-10:] or [0.5]
        pressure_factors.append(sum(recent_cpu) / len(recent_cpu))

    if not pressure_factors:
        return 0.5

    return sum(pressure_factors) / len(pressure_factors)


def assess_adaptation_need(
    learning_insights: dict[str, Any], performance_trend: float
) -> bool:
    """Assess whether the orchestrator should adapt its strategy."""
    adaptation_signals = []

    if performance_trend < -0.05:
        adaptation_signals.append(True)

    if "curriculum_insights" in learning_insights:
        task_progress = learning_insights["curriculum_insights"].get(
            "task_progress", {}
        )
        for progress in task_progress.values():
            if progress.get("success_rate", 1.0) < 0.6:
                adaptation_signals.append(True)
                break

    if "exploration_insights" in learning_insights:
        epsilon = learning_insights["exploration_insights"].get("current_epsilon", 0.1)
        if epsilon < 0.05:
            adaptation_signals.append(True)

    return any(adaptation_signals)


def make_automated_decision(
    config: OrchestrationConfig,
    reward: float,
    recent_performance: float,
    resource_pressure: float,
) -> OrchestrationDecision:
    """Build the automated orchestration decision."""
    if recent_performance < config.performance_threshold:
        if resource_pressure < 0.7:
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="performance_optimization",
                component="orchestrator",
                action="aggressive_optimize",
                reasoning=(
                    f"Performance {recent_performance:.3f} below threshold "
                    f"{config.performance_threshold}"
                ),
                expected_benefit=0.3,
                confidence=0.8,
            )
        return OrchestrationDecision(
            decision_id="",
            timestamp=datetime.now(),
            decision_type="resource_optimization",
            component="orchestrator",
            action="gentle_optimize",
            reasoning=(
                "Performance low but resources constrained "
                f"({resource_pressure:.2f})"
            ),
            expected_benefit=0.1,
            confidence=0.6,
        )

    exploration_probability = max(0.1, 1.0 - recent_performance)
    if reward > recent_performance:
        return OrchestrationDecision(
            decision_id="",
            timestamp=datetime.now(),
            decision_type="exploration",
            component="orchestrator",
            action="explore_improvements",
            reasoning="Performance good, exploring for better solutions",
            expected_benefit=0.05,
            confidence=exploration_probability,
        )
    return OrchestrationDecision(
        decision_id="",
        timestamp=datetime.now(),
        decision_type="maintenance",
        component="orchestrator",
        action="maintain_status",
        reasoning="Performance stable, maintaining current approach",
        expected_benefit=0.0,
        confidence=0.9,
    )


def make_adaptive_decision(
    config: OrchestrationConfig,
    learning_insights: dict[str, Any],
    performance_trend: float,
    recent_performance: float,
    nas_optimization_due: bool,
) -> OrchestrationDecision:
    """Build the adaptive orchestration decision."""
    if assess_adaptation_need(learning_insights, performance_trend):
        curriculum_insights = learning_insights.get("curriculum_insights", {})
        difficulty = curriculum_insights.get("current_difficulty", 0.5)
        if difficulty < 0.3:
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="curriculum_adaptation",
                component="adaptive_learning",
                action="increase_difficulty",
                reasoning="Current difficulty too low for optimal learning",
                expected_benefit=0.2,
                confidence=0.7,
            )
        if difficulty > 0.9:
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="curriculum_adaptation",
                component="adaptive_learning",
                action="provide_support",
                reasoning="Current difficulty too high, agent struggling",
                expected_benefit=0.15,
                confidence=0.8,
            )

    if nas_optimization_due and recent_performance < 0.85:
        return OrchestrationDecision(
            decision_id="",
            timestamp=datetime.now(),
            decision_type="architecture_optimization",
            component="nas",
            action="schedule_nas_optimization",
            reasoning="Performance plateau detected, architecture optimization due",
            expected_benefit=0.25,
            confidence=0.6,
        )

    if performance_trend > 0.05:
        return OrchestrationDecision(
            decision_id="",
            timestamp=datetime.now(),
            decision_type="exploration",
            component="orchestrator",
            action="continue_current_strategy",
            reasoning="Positive performance trend, continuing current approach",
            expected_benefit=0.1,
            confidence=0.8,
        )

    return OrchestrationDecision(
        decision_id="",
        timestamp=datetime.now(),
        decision_type="optimization",
        component="orchestrator",
        action="explore_alternatives",
        reasoning="Performance stagnant, exploring alternative strategies",
        expected_benefit=0.15,
        confidence=0.5,
    )


def make_conservative_decision(
    config: OrchestrationConfig,
    success: bool,
    recent_performance: float,
) -> OrchestrationDecision:
    """Build the conservative orchestration decision."""
    if recent_performance < config.performance_threshold * 0.8:
        return OrchestrationDecision(
            decision_id="",
            timestamp=datetime.now(),
            decision_type="alert",
            component="orchestrator",
            action="request_human_intervention",
            reasoning=(
                "Performance "
                f"{recent_performance:.3f} significantly below threshold"
            ),
            expected_benefit=0.0,
            confidence=1.0,
        )
    if not success:
        return OrchestrationDecision(
            decision_id="",
            timestamp=datetime.now(),
            decision_type="recovery",
            component="orchestrator",
            action="gentle_recovery",
            reasoning="Recent failure detected, applying gentle recovery",
            expected_benefit=0.05,
            confidence=0.7,
        )
    return OrchestrationDecision(
        decision_id="",
        timestamp=datetime.now(),
        decision_type="maintenance",
        component="orchestrator",
        action="monitor_and_maintain",
        reasoning="Normal operation, monitoring for changes",
        expected_benefit=0.0,
        confidence=0.9,
    )


__all__ = [
    "assess_adaptation_need",
    "assess_resource_pressure",
    "calculate_performance_trend",
    "calculate_recent_performance",
    "make_adaptive_decision",
    "make_automated_decision",
    "make_conservative_decision",
]
