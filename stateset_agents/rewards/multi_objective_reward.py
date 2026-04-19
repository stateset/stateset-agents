"""Multi-objective reward composition and factory helpers."""

import logging
from typing import Any, cast

import numpy as np

from stateset_agents.core.reward_base import RewardFunction, RewardResult
from stateset_agents.utils.cache import CacheService
from stateset_agents.utils.monitoring import MonitoringService
from .multi_objective_components import (
    ActionOrientedRewardComponent,
    BaseRewardComponent,
    EmpathyRewardComponent,
    LengthRewardComponent,
    MULTI_REWARD_EXCEPTIONS,
    ModelBasedRewardComponent,
    ProfessionalismRewardComponent,
    ReasoningRewardComponent,
    RewardComponent,
    SimilarityRewardComponent,
)

logger = logging.getLogger(__name__)


class MultiObjectiveRewardFunction(RewardFunction):
    """
    Multi-objective reward function that combines multiple reward components
    with weighted scoring and sophisticated evaluation criteria.
    """

    def __init__(
        self,
        components: list[BaseRewardComponent] | None = None,
        weight: float = 1.0,
        normalization_method: str = "weighted_sum",
        cache_service: CacheService | None = None,
        monitoring_service: MonitoringService | None = None,
    ):
        """
        Initialize multi-objective reward function.

        Args:
            components: List of reward components
            weight: Overall weight for this reward function
            normalization_method: How to combine component scores
            cache_service: Optional cache service
            monitoring_service: Optional monitoring service
        """
        super().__init__(weight=weight)

        self.components = components or []
        self.normalization_method = normalization_method
        self.cache = cache_service
        self.monitoring = monitoring_service

        self._normalize_component_weights()
        self.evaluation_count = 0
        self.component_scores: dict[str, list[float]] = {}

    def _normalize_component_weights(self) -> None:
        """Normalize component weights to a stable sum of 1.0."""
        total_weight = sum(component.weight for component in self.components)
        if total_weight <= 0:
            return
        for component in self.components:
            component.weight = component.weight / total_weight

    def add_component(self, component: BaseRewardComponent) -> None:
        """Add a reward component."""
        self.components.append(component)
        self._normalize_component_weights()

    def remove_component(self, name: str) -> None:
        """Remove a reward component by name."""
        self.components = [c for c in self.components if c.name != name]
        self._normalize_component_weights()

    async def compute_reward(
        self,
        turns: list[Any] | None = None,
        context: dict[str, Any] | None = None,
        trajectory: Any | None = None,
        turn: Any | None = None,
        **_: Any,
    ) -> RewardResult:
        """
        Compute multi-objective reward.

        Args:
            turns: List of conversation turns (dicts or ConversationTurn objects).
            context: Optional context information

        Returns:
            RewardResult with combined score and component breakdown
        """
        # Support trainer-style calls: compute_reward(trajectory=?, turn=?, context=?)
        if turns is None and turn is not None:
            normalized: list[dict[str, Any]] = []
            if context and isinstance(context, dict):
                user_query = context.get("user_query") or context.get("prompt")
                if user_query:
                    normalized.append({"role": "user", "content": str(user_query)})
            normalized.append(self._normalize_turn(turn))
            turns = normalized

        turns = turns or []
        normalized_turns = [self._normalize_turn(item) for item in turns]

        if not self.components:
            return RewardResult(
                score=0.0,
                breakdown={"total_score": 0.0},
                metadata={"error": "No components configured"},
            )

        # Compute scores for each component
        component_scores: dict[str, float] = {}
        weighted_scores: list[float] = []

        for component in self.components:
            try:
                score = await component.compute_score(normalized_turns, context)
                component_scores[component.name] = score
                weighted_scores.append(score * component.weight)

                # Track component performance
                if component.name not in self.component_scores:
                    self.component_scores[component.name] = []
                self.component_scores[component.name].append(score)

            except MULTI_REWARD_EXCEPTIONS as e:
                logger.error(f"Component {component.name} failed: {e}")
                component_scores[component.name] = 0.0
                weighted_scores.append(0.0)

        # Combine scores
        if self.normalization_method == "weighted_sum":
            final_score = sum(weighted_scores)
        elif self.normalization_method == "weighted_average":
            total_weight = sum(component.weight for component in self.components)
            final_score = (
                sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
            )
        elif self.normalization_method == "geometric_mean":
            # Geometric mean of weighted scores
            if all(s > 0 for s in weighted_scores):
                final_score = np.prod(weighted_scores) ** (1.0 / len(weighted_scores))
            else:
                final_score = 0.0
        else:
            final_score = sum(weighted_scores)

        # Ensure score is in [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        self.evaluation_count += 1

        # Create breakdown
        weights = {c.name: c.weight for c in self.components}
        breakdown: dict[str, Any] = {
            "total_score": final_score,
            "evaluation_count": float(self.evaluation_count),
            "components": component_scores,
            "weights": weights,
            "normalization_method": self.normalization_method,
        }
        metadata = {
            "components": component_scores,
            "weights": weights,
            "normalization_method": self.normalization_method,
        }

        # Log metrics
        if self.monitoring:
            await self.monitoring.log_metric(
                "multi_objective_reward.evaluation",
                final_score,
                labels={"method": self.normalization_method},
            )

        result = RewardResult(
            score=final_score,
            components=component_scores,
            breakdown=cast(dict[str, float], breakdown),
            metadata=metadata,
        )
        # Some trainers expect `total_reward`; keep it as an alias for score.
        result.total_reward = float(final_score)
        return result

    def _normalize_turn(self, item: Any) -> dict[str, Any]:
        """Normalize turn inputs while preserving metadata-rich objects."""
        if isinstance(item, dict):
            return dict(item)

        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            normalized_dict = to_dict()
            if isinstance(normalized_dict, dict):
                return normalized_dict

        role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        metadata = getattr(item, "metadata", None)
        normalized: dict[str, Any] = {
            "role": str(role) if role is not None else "assistant",
            "content": str(content) if content is not None else str(item),
        }
        if isinstance(metadata, dict):
            normalized["metadata"] = dict(metadata)
        return normalized

    def get_component_statistics(self) -> dict[str, dict[str, float]]:
        """Get statistics for each component"""
        stats: dict[str, dict[str, float]] = {}

        for name, scores in self.component_scores.items():
            if scores:
                stats[name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "count": float(len(scores)),
                }

        return stats

    def get_info(self) -> dict[str, Any]:
        """Get reward function information"""
        return {
            "type": "multi_objective",
            "components": [c.get_info() for c in self.components],
            "normalization_method": self.normalization_method,
            "evaluation_count": self.evaluation_count,
            "component_statistics": self.get_component_statistics(),
        }


# Convenience functions for creating common multi-objective rewards
def create_customer_service_reward(
    expected_responses: list[str] | None = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for customer service"""
    components = [
        EmpathyRewardComponent(weight=0.25),
        ActionOrientedRewardComponent(weight=0.25),
        ProfessionalismRewardComponent(weight=0.25),
        LengthRewardComponent(weight=0.1, min_length=20, optimal_range=(50, 200)),
        SimilarityRewardComponent(weight=0.15, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


def create_domain_reward(
    domain: str,
    expected_responses: list[str] | None = None,
    weight: float = 1.0,
    **kwargs,
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward function for a given domain.

    This is a small convenience wrapper used by many examples. It maps common
    domain names (e.g., ``"technical_support"``) to the corresponding
    multi-objective reward template.
    """

    domain_lower = domain.strip().lower()

    if domain_lower in {"customer_service", "customer", "cs", "support"}:
        return create_customer_service_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    if domain_lower in {
        "technical_support",
        "tech_support",
        "technical",
        "it",
        "it_support",
        "helpdesk",
    }:
        return create_technical_support_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    if domain_lower in {"sales", "marketing", "product", "sales_assistant"}:
        return create_sales_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    if domain_lower in {"education", "educational", "tutoring", "tutor"}:
        return create_educational_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    if domain_lower in {"creative", "creative_writing", "writing"}:
        return create_creative_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    # Default to customer service heuristics for conversational alignment.
    return create_customer_service_reward(
        expected_responses=expected_responses, weight=weight, **kwargs
    )


def create_technical_support_reward(
    expected_responses: list[str] | None = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for technical support"""
    components = [
        ActionOrientedRewardComponent(weight=0.4),  # High weight for solutions
        ProfessionalismRewardComponent(weight=0.2),
        LengthRewardComponent(weight=0.1, min_length=30, optimal_range=(100, 300)),
        SimilarityRewardComponent(weight=0.3, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


def create_educational_reward(
    expected_responses: list[str] | None = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for educational content"""
    components = [
        ActionOrientedRewardComponent(weight=0.3),
        ProfessionalismRewardComponent(weight=0.2),
        LengthRewardComponent(weight=0.1, min_length=50, optimal_range=(150, 400)),
        SimilarityRewardComponent(weight=0.4, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


def create_sales_reward(
    expected_responses: list[str] | None = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for sales interactions"""
    # Add sales-specific components
    sales_action_keywords = [
        "buy",
        "purchase",
        "order",
        "discount",
        "offer",
        "deal",
        "save",
        "recommend",
        "suggest",
        "perfect for",
        "ideal",
        "best choice",
        "limited time",
        "special",
        "exclusive",
        "value",
        "benefit",
    ]

    components = [
        ActionOrientedRewardComponent(
            weight=0.35, action_keywords=sales_action_keywords
        ),
        ProfessionalismRewardComponent(weight=0.25),
        EmpathyRewardComponent(weight=0.15),
        LengthRewardComponent(weight=0.1, min_length=30, optimal_range=(80, 250)),
        SimilarityRewardComponent(weight=0.15, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


def create_creative_reward(
    expected_responses: list[str] | None = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for creative content"""
    creative_keywords = [
        "imagine",
        "creative",
        "unique",
        "original",
        "innovative",
        "artistic",
        "beautiful",
        "inspiring",
        "fascinating",
        "amazing",
        "wonderful",
        "story",
        "narrative",
        "character",
        "scene",
        "vivid",
        "descriptive",
    ]

    components = [
        ActionOrientedRewardComponent(weight=0.2, action_keywords=creative_keywords),
        LengthRewardComponent(weight=0.2, min_length=50, optimal_range=(200, 500)),
        SimilarityRewardComponent(weight=0.6, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


__all__ = [
    "ActionOrientedRewardComponent",
    "BaseRewardComponent",
    "EmpathyRewardComponent",
    "LengthRewardComponent",
    "MULTI_REWARD_EXCEPTIONS",
    "ModelBasedRewardComponent",
    "MultiObjectiveRewardFunction",
    "ProfessionalismRewardComponent",
    "ReasoningRewardComponent",
    "RewardComponent",
    "SimilarityRewardComponent",
    "create_creative_reward",
    "create_customer_service_reward",
    "create_domain_reward",
    "create_educational_reward",
    "create_sales_reward",
    "create_technical_support_reward",
]
