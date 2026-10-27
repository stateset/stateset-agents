"""Multi-objective reward composition and factory helpers."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from stateset_agents.core.reward_base import (
    OUTAGE_POLICIES,
    RewardFunction,
    RewardResult,
    RewardUnavailableError,
)
from stateset_agents.core.reward_context import (
    COMPONENT_FAILURES_KEY,
    REWARD_FALLBACK_KEY,
    declared_score_range,
    scores_latest_turn_only,
)
from stateset_agents.utils.cache import CacheService
from stateset_agents.utils.monitoring import MonitoringService

from .multi_objective_components import (
    MULTI_REWARD_EXCEPTIONS,
    ActionOrientedRewardComponent,
    BaseRewardComponent,
    EmpathyRewardComponent,
    GroundedComponent,
    LengthRewardComponent,
    ModelBasedRewardComponent,
    ProfessionalismRewardComponent,
    ReasoningRewardComponent,
    RewardComponent,
    SimilarityRewardComponent,
    ground_components,
)

logger = logging.getLogger(__name__)

#: Rounding slack at the edges of a score range (as in
#: ``reward_context.check_score_range``).
_RANGE_SLACK = 1e-9
#: The range of a component that declares none.
_DEFAULT_COMPONENT_RANGE = (0.0, 1.0)


def _component_range(component: Any) -> tuple[float, float]:
    """The ``score_range`` ``component`` declares, else ``[0, 1]``."""
    return declared_score_range(component) or _DEFAULT_COMPONENT_RANGE


def _checked_component_score(component: Any, score: float) -> float:
    """``score``, or ``ValueError`` when it is not finite or lies outside the
    component's score range (never clamped)."""
    name = getattr(component, "name", type(component).__name__)
    if not math.isfinite(score):
        raise ValueError(f"Reward component {name} returned a non-finite score {score}")
    low, high = _component_range(component)
    slack = _RANGE_SLACK * max(1.0, abs(low), abs(high))
    if not low - slack <= score <= high + slack:
        raise ValueError(
            f"Reward component {name} scored {score}, outside its score range "
            f"[{low}, {high}]"
        )
    return score


@dataclass
class _ComponentScores:
    """The components that scored in one evaluation, in order."""

    scores: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    by_name: dict[str, float] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)

    def add(self, name: str, score: float, weight: float) -> None:
        self.scores.append(score)
        self.weights.append(weight)
        self.by_name[name] = score


class MultiObjectiveRewardFunction(RewardFunction):
    """
    Multi-objective reward function that combines multiple reward components
    with weighted scoring and sophisticated evaluation criteria.

    A component score must be finite and inside the component's
    ``score_range`` (``[0, 1]`` when it declares none), or the evaluation
    raises ``ValueError``; scores are never clamped.

    Outages (a component raising ``RewardUnavailableError``) follow
    :attr:`outage_policy`, exactly as for ``CompositeReward``:

    * ``"raise"`` (the default): the reward raises ``RewardUnavailableError``
      too, so the episode is dropped rather than scored by the surviving
      components alone.
    * ``"renormalize"`` (opt-in: ``outage_policy="renormalize"`` or
      ``reward.outage_policy = "renormalize"``): the failed component is left
      out and its weight spread over the rest; the result lists it in
      ``metadata["failed_components"]``, counts it in
      ``breakdown["unavailable_components"]`` and is flagged
      ``metadata["reward_fallback"]``.

    A result is also flagged ``metadata["reward_fallback"]`` when a component
    that scored reports a ``fallback_reason`` (a model judge without a client
    scoring by length). Environments and ``score_episode`` copy the flag and
    the failed components to the trajectory's metadata
    (``reward_fallback``, ``reward_failed_components``).
    """

    #: How a component outage is handled (see the class docs).
    outage_policy: str = "raise"

    def __init__(
        self,
        components: list[BaseRewardComponent] | None = None,
        weight: float = 1.0,
        normalization_method: str = "weighted_sum",
        cache_service: CacheService | None = None,
        monitoring_service: MonitoringService | None = None,
        outage_policy: str | None = None,
    ):
        """
        Initialize multi-objective reward function.

        Args:
            components: List of reward components
            weight: Overall weight for this reward function
            normalization_method: How to combine component scores
            cache_service: Optional cache service
            monitoring_service: Optional monitoring service
            outage_policy: ``"raise"`` (the default) or ``"renormalize"``
                (see the class docs)
        """
        super().__init__(weight=weight)

        self.components = components or []
        self.normalization_method = normalization_method
        self.cache = cache_service
        self.monitoring = monitoring_service
        if outage_policy is not None:
            self.outage_policy = outage_policy
            self._checked_outage_policy()

        self._normalize_component_weights()
        self.evaluation_count = 0
        self.component_scores: dict[str, list[float]] = {}

    @property
    def score_range(self) -> tuple[float, float]:
        """The lowest and highest bounds of the components' score ranges
        (``[0, 1]`` for components that declare none): every combination
        method keeps the combined score inside them."""
        ranges = [_component_range(c) for c in self.components]
        if not ranges:
            return _DEFAULT_COMPONENT_RANGE
        return min(low for low, _ in ranges), max(high for _, high in ranges)

    def _normalized_weight_snapshot(
        self, components: tuple[BaseRewardComponent, ...]
    ) -> tuple[float, ...]:
        """Validated component weights scaled to sum to 1 (unchanged if 0)."""
        for component in components:
            self._validate_component_weight(component)
        weights = tuple(float(component.weight) for component in components)
        total_weight = math.fsum(weights)
        if not math.isfinite(total_weight):
            raise ValueError("total component weight must be finite")
        if total_weight <= 0:
            return weights
        return tuple(weight / total_weight for weight in weights)

    def _normalize_component_weights(self) -> None:
        """Normalize component weights to a stable sum of 1.0."""
        components = tuple(self.components)
        for component, weight in zip(
            components, self._normalized_weight_snapshot(components), strict=True
        ):
            component.weight = weight

    @staticmethod
    def _validate_component_weight(component: BaseRewardComponent) -> None:
        """Reject weights outside the nonnegative finite proof domain."""
        try:
            valid = math.isfinite(component.weight) and component.weight >= 0
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("component weight must be finite and nonnegative") from exc
        if not valid:
            raise ValueError("component weight must be finite and nonnegative")

    def add_component(self, component: BaseRewardComponent) -> None:
        """Add a reward component."""
        self._validate_component_weight(component)
        self.components.append(component)
        try:
            self._normalize_component_weights()
        except ValueError:
            self.components.pop()
            raise

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

        Raises:
            RewardUnavailableError: a component is unavailable (with the
                default ``outage_policy = "raise"``), or every component is.
            ValueError: no components, or a component score that is not
                finite or lies outside the component's score range.
        """
        scoped, whole = self._turns_in_scope(turns, turn, context)
        if not self.components:
            raise ValueError("MultiObjectiveRewardFunction has no components")
        policy = self._checked_outage_policy()

        # Components are mutable and their callbacks may suspend. Revalidate
        # and use one normalized weight snapshot for the whole evaluation,
        # without writing it back: renormalizing stored weights on every call
        # drifts them by an ulp per call and made scores differ between calls.
        components = tuple(self.components)
        component_weights = self._normalized_weight_snapshot(components)
        scored = await self._score_components(
            components, component_weights, (scoped, whole), context, policy
        )
        final_score = self._settled_score(
            self._combine(scored, math.fsum(component_weights))
        )
        self.evaluation_count += 1

        weights = {
            component.name: weight
            for component, weight in zip(components, component_weights, strict=True)
        }
        breakdown: dict[str, Any] = {
            "total_score": final_score,
            "evaluation_count": float(self.evaluation_count),
            "components": scored.by_name,
            "weights": weights,
            "normalization_method": self.normalization_method,
        }
        metadata: dict[str, Any] = {
            "components": scored.by_name,
            "weights": weights,
            "normalization_method": self.normalization_method,
            COMPONENT_FAILURES_KEY: scored.failed,
        }
        fallbacks = list(scored.fallbacks)
        if scored.failed:
            # An opted-in renormalization is a fallback score: flag it.
            fallbacks.append(f"renormalized without {', '.join(scored.failed)}")
            breakdown["unavailable_components"] = float(len(scored.failed))
        if fallbacks:
            metadata[REWARD_FALLBACK_KEY] = "; ".join(fallbacks)

        # Log metrics
        if self.monitoring:
            await self.monitoring.log_metric(
                "multi_objective_reward.evaluation",
                final_score,
                labels={"method": self.normalization_method},
            )

        result = RewardResult(
            score=final_score,
            components=scored.by_name,
            breakdown=cast(dict[str, float], breakdown),
            metadata=metadata,
        )
        # Some trainers expect `total_reward`; keep it as an alias for score.
        result.total_reward = float(final_score)
        return result

    def _turns_in_scope(
        self, turns: list[Any] | None, turn: Any, context: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The turns to score, as dicts: those in scope (honoring the
        latest-turn scope) and the whole conversation (for judges)."""
        # Support trainer-style calls: compute_reward(trajectory=?, turn=?, context=?)
        if turns is None and turn is not None:
            normalized: list[dict[str, Any]] = []
            if context and isinstance(context, dict):
                user_query = context.get("user_query") or context.get("prompt")
                if user_query:
                    normalized.append({"role": "user", "content": str(user_query)})
            normalized.append(self._normalize_turn(turn))
            turns = normalized

        normalized_turns = [self._normalize_turn(item) for item in turns or []]
        if not scores_latest_turn_only(context):
            return normalized_turns, normalized_turns
        # One step's reward: keep the conversation, but only the newest reply
        # among assistant turns, so components that aggregate over replies
        # score that one.
        assistant_indices = [
            index
            for index, item in enumerate(normalized_turns)
            if item.get("role") == "assistant"
        ]
        earlier = set(assistant_indices[:-1])
        scoped = [
            item for index, item in enumerate(normalized_turns) if index not in earlier
        ]
        return scoped, normalized_turns

    def _checked_outage_policy(self) -> str:
        policy = self.outage_policy
        if policy not in OUTAGE_POLICIES:
            from stateset_agents.core.errors import ConfigurationError

            raise ConfigurationError(
                "MultiObjectiveRewardFunction.outage_policy must be one of "
                f"{OUTAGE_POLICIES}, got {policy!r}"
            )
        return policy

    async def _score_components(
        self,
        components: tuple[BaseRewardComponent, ...],
        weights: tuple[float, ...],
        turns: tuple[list[dict[str, Any]], list[dict[str, Any]]],
        context: dict[str, Any] | None,
        policy: str,
    ) -> _ComponentScores:
        """Every component's checked score; outages follow ``policy``.

        ``turns`` is (the turns in scope, the whole conversation): a judge
        (``sees_whole_conversation``) scores the newest reply itself and
        reads every earlier turn, replies included, as its context.
        """
        scoped, whole = turns
        scored = _ComponentScores()
        for component, weight in zip(components, weights, strict=True):
            judge = getattr(component, "sees_whole_conversation", False) is True
            try:
                score = await component.compute_score(
                    whole if judge else scoped, context
                )
            except RewardUnavailableError as error:
                # Only an outage is absorbed, and only when renormalizing is
                # opted into; other exceptions are bugs and propagate.
                if (
                    getattr(component, "fail_on_error", False) is True
                    or policy != "renormalize"
                ):
                    raise RewardUnavailableError(
                        f"MultiObjectiveRewardFunction: component "
                        f"{component.name} is unavailable: {error}"
                    ) from error
                logger.warning(
                    "Reward component %s is unavailable; its weight is spread "
                    "over the other components (outage_policy='renormalize'): %s",
                    component.name,
                    error,
                )
                scored.failed.append(str(component.name))
                continue
            score = _checked_component_score(component, score)
            scored.add(component.name, score, weight)
            reason = getattr(component, "fallback_reason", None)
            if isinstance(reason, str) and reason:
                scored.fallbacks.append(reason)
            # Track component performance
            self.component_scores.setdefault(component.name, []).append(score)
        if not scored.scores:
            # No component scored: fail rather than return a valid-looking 0.0.
            raise RewardUnavailableError(
                "MultiObjectiveRewardFunction: every component failed "
                f"({', '.join(scored.failed)})"
            )
        return scored

    def _combine(self, scored: _ComponentScores, configured_weight: float) -> float:
        """Combine the component scores with ``normalization_method``."""
        weights = scored.weights
        surviving_weight = math.fsum(weights)
        if scored.failed and surviving_weight > 0:
            # Opted-in renormalization: a failed component's weight is spread
            # over the components that scored, as CompositeReward does.
            scale = configured_weight / surviving_weight
            weights = [weight * scale for weight in weights]
        weighted = [
            score * weight for score, weight in zip(scored.scores, weights, strict=True)
        ]
        total_weight = math.fsum(weights)
        if self.normalization_method == "weighted_average":
            return math.fsum(weighted) / total_weight if total_weight > 0 else 0.0
        if self.normalization_method == "geometric_mean":
            # Weighted geometric mean: prod(score_i ** w_i) with weights
            # summing to 1, so all-perfect components score 1.0.
            if total_weight <= 0 or any(score <= 0 for score in scored.scores):
                return 0.0
            return float(
                math.exp(
                    math.fsum(
                        w * math.log(score)
                        for score, w in zip(scored.scores, weights, strict=True)
                    )
                    / total_weight
                )
            )
        # "weighted_sum" (and any unknown method)
        return math.fsum(weighted)

    def _settled_score(self, score: float) -> float:
        """``score`` checked against :attr:`score_range`.

        In-range component scores combine to a score in that range, so only
        float rounding (at most ``1e-9``) can land outside it; that is
        snapped to the bound. Anything further raises ``ValueError``: a score
        is never clamped into range.
        """
        low, high = self.score_range
        slack = _RANGE_SLACK * max(1.0, abs(low), abs(high))
        if not (math.isfinite(score) and low - slack <= score <= high + slack):
            raise ValueError(
                f"MultiObjectiveRewardFunction combined score {score} is outside "
                f"its score range [{low}, {high}]"
            )
        return min(max(score, low), high)

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


# Convenience functions for creating common multi-objective rewards.
#
# These are heuristic shaping signals, not training objectives. Every
# component is gated on task grounding (GroundedComponent), so a canned
# template or keyword stuffing scores below a reply that answers the request.
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

    return MultiObjectiveRewardFunction(
        components=ground_components(components), weight=weight, **kwargs
    )


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

    return MultiObjectiveRewardFunction(
        components=ground_components(components), weight=weight, **kwargs
    )


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

    return MultiObjectiveRewardFunction(
        components=ground_components(components), weight=weight, **kwargs
    )


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

    return MultiObjectiveRewardFunction(
        components=ground_components(components), weight=weight, **kwargs
    )


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

    return MultiObjectiveRewardFunction(
        components=ground_components(components), weight=weight, **kwargs
    )


__all__ = [
    "ActionOrientedRewardComponent",
    "BaseRewardComponent",
    "EmpathyRewardComponent",
    "GroundedComponent",
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
