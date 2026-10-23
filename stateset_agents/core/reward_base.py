"""
Base reward modeling classes and types for multi-turn agent training.

This module provides the foundational abstractions for reward functions
including the base class, result types, and composite reward implementation.
"""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .reward_context import (
    COMPONENT_FAILURES_KEY,
    REWARD_FALLBACK_KEY,
    UNIT_SCORE_RANGE,
    RewardUnavailableError,
    build_reward_context,
    check_child_score_range,
    reward_fallback,
)
from .trajectory import ConversationTurn

logger = logging.getLogger(__name__)


class SyncRewardCallError(RuntimeError):
    """A reward was called synchronously (``reward(...)``) inside a running
    event loop; await ``reward.compute_reward(...)`` there instead."""


#: ``CompositeReward.outage_policy`` values.
OUTAGE_POLICIES = ("raise", "renormalize")


class RewardType(Enum):
    """Types of rewards"""

    IMMEDIATE = "immediate"  # Reward for single turn
    CUMULATIVE = "cumulative"  # Reward for entire conversation
    SPARSE = "sparse"  # Reward only at episode end
    DENSE = "dense"  # Reward at every step


@dataclass(init=False)
class RewardResult:
    """Result of reward computation"""

    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None

    def __init__(
        self,
        score: float,
        components: dict[str, float] | None = None,
        breakdown: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        explanation: str | None = None,
    ):
        self.score = float(score)
        if breakdown is None and components is not None:
            breakdown = components
        if components is None and breakdown is not None:
            components = breakdown

        self.breakdown = dict(breakdown or {})
        self.components = dict(components or {})
        self.metadata = dict(metadata or {})
        self.explanation = explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "total_reward": self.total_reward,
            "breakdown": self.breakdown,
            "components": self.components,
            "metadata": self.metadata,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardResult":
        return cls(
            score=float(data.get("score", data.get("total_reward", 0.0))),
            breakdown=data.get("breakdown") or data.get("components") or {},
            components=data.get("components") or data.get("breakdown") or {},
            metadata=data.get("metadata") or {},
            explanation=data.get("explanation"),
        )

    @property
    def total_reward(self) -> float:
        """Compatibility alias used by several training paths."""
        return self.score

    @total_reward.setter
    def total_reward(self, value: float) -> None:
        self.score = float(value)


class RewardFunction(ABC):
    """
    Abstract base class for reward functions
    """

    #: True for style heuristics (keyword, phrase or length scoring) that a
    #: canned reply can satisfy without answering the request. The shipped
    #: factories wrap these in ``core.grounding.GroundedReward``.
    shaping: bool = False

    def __init__(
        self,
        weight: float = 1.0,
        reward_type: RewardType = RewardType.IMMEDIATE,
        name: str | None = None,
    ):
        self.weight = weight
        self.reward_type = reward_type
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Compute reward for conversation turns"""
        pass

    async def compute_turn_reward(
        self,
        turn: ConversationTurn,
        context: dict[str, Any] | None = None,
        conversation_history: list[ConversationTurn] | None = None,
    ) -> RewardResult:
        """Compute reward for a single turn"""
        turns = conversation_history or []
        turns = turns + [turn]
        return await self.compute_reward(turns, context)

    def __call__(self, *args: Any, **kwargs: Any) -> "RewardResult":
        """Make reward function callable (sync wrapper for async compute_reward).

        Raises :class:`SyncRewardCallError` inside a running event loop; use
        ``await reward.compute_reward(...)`` there. Errors raised by the
        reward itself propagate unchanged.
        """
        if _event_loop_running():
            raise SyncRewardCallError(
                "Cannot call RewardFunction synchronously from within an async "
                "context. Use `await reward_function.compute_reward(...)` instead "
                "of `reward_function(...)`."
            )
        return asyncio.run(self.compute_reward(*args, **kwargs))


def _event_loop_running() -> bool:
    """Whether this thread is running an event loop (no message parsing)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class CompositeReward(RewardFunction):
    """Combines multiple reward functions with weights.

    Outages (a component raising ``RewardUnavailableError``) follow
    :attr:`outage_policy`:

    * ``"raise"`` (the default): the composite raises
      ``RewardUnavailableError`` too, so the episode is dropped rather than
      scored by the surviving components alone (a judge outage must not turn
      the conciseness term into the whole reward).
    * ``"renormalize"`` (opt-in, ``composite.outage_policy = "renormalize"``):
      the failed component is left out and its weight spread over the rest;
      the result lists it in ``metadata["failed_components"]``, counts it in
      ``breakdown["unavailable_components"]`` and is flagged
      ``metadata["reward_fallback"]``, which environments and
      ``score_episode`` copy to the trajectory (with
      ``metadata["reward_failed_components"]``).

    Every child score must be finite and inside the child's declared
    ``score_range``, or ``[0, 1]`` when it declares none (a nested
    ``CompositeReward`` spans its own weights and is checked through its
    children); otherwise the composite raises ``ValueError`` rather than
    weighting an out-of-range score into the total. Scores are never clamped.
    """

    #: How a component outage is handled (see the class docs).
    outage_policy: str = "raise"

    def __init__(
        self,
        reward_functions: list[RewardFunction],
        combination_method: str = "weighted_sum",
        normalize_weights: bool = False,
    ):
        super().__init__(name="CompositeReward")
        self.reward_functions = reward_functions
        self.combination_method = combination_method
        self.normalize_weights = normalize_weights

    @property
    def reward_type(self) -> RewardType:
        """Defer a composition to episode end when any child needs the full trace."""
        if any(
            reward.reward_type in (RewardType.CUMULATIVE, RewardType.SPARSE)
            for reward in getattr(self, "reward_functions", [])
        ):
            return RewardType.CUMULATIVE
        return self._reward_type

    @reward_type.setter
    def reward_type(self, value: RewardType) -> None:
        self._reward_type = value

    @property
    def fail_on_error(self) -> bool:
        """Preserve critical evidence requirements through nested composition."""
        return any(
            getattr(reward, "fail_on_error", False) for reward in self.reward_functions
        )

    @property
    def requires_environment_context(self) -> bool:
        """Request final environment state when any component needs it."""
        return any(
            getattr(reward, "requires_environment_context", False)
            for reward in self.reward_functions
        )

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Compute composite reward from all functions"""

        # An empty composite has nothing to score with; a 0.0 would look like
        # a real (worst) score.
        if not self.reward_functions:
            raise ValueError("CompositeReward has no reward functions configured")

        if self.outage_policy not in OUTAGE_POLICIES:
            from .errors import ConfigurationError

            raise ConfigurationError(
                f"CompositeReward.outage_policy must be one of {OUTAGE_POLICIES}, "
                f"got {self.outage_policy!r}"
            )
        results: list[tuple[RewardFunction, RewardResult]] = []
        failed: list[str] = []
        for reward_fn in self.reward_functions:
            try:
                child_context = context
                if (
                    self.requires_environment_context
                    and not getattr(reward_fn, "requires_environment_context", False)
                    and isinstance(context, dict)
                ):
                    # This composite was handed the evidence context (state
                    # plus "scenario", no lifted keys). Ordinary components get
                    # the standard context built from it, so they see both the
                    # scenario's keys and the environment state.
                    child_context = build_reward_context(context, reward_fn=reward_fn)
                result = await reward_fn.compute_reward(turns, child_context)
            except RewardUnavailableError as e:
                # An outage. By default the whole composite is unavailable,
                # so the episode is dropped; renormalizing is an opt-in.
                # Anything else (TypeError, AttributeError, bad data) is a bug
                # and propagates rather than silently reweighting the reward.
                if (
                    getattr(reward_fn, "fail_on_error", False)
                    or self.outage_policy != "renormalize"
                ):
                    raise RewardUnavailableError(
                        f"CompositeReward: component {reward_fn.name} is "
                        f"unavailable: {e}"
                    ) from e
                logger.warning(
                    "Reward function %s is unavailable; its weight is spread "
                    "over the other components (outage_policy='renormalize'): %s",
                    reward_fn.name,
                    e,
                )
                failed.append(str(reward_fn.name))
                continue
            if not math.isfinite(float(result.score)):
                # A NaN/inf component would silently poison the total.
                raise ValueError(
                    f"Reward function {reward_fn.name} returned a non-finite "
                    f"score {result.score}"
                )
            # A child is held to its declared score_range or, declaring
            # none, to [0, 1]: a 7.0 must not be weighted into the sum. A
            # nested composite spans its own weights and checks its children.
            check_child_score_range(
                reward_fn,
                float(result.score),
                default=(
                    None if isinstance(reward_fn, CompositeReward) else UNIT_SCORE_RANGE
                ),
            )
            results.append((reward_fn, result))

        # No component scored: there is no reward, so fail rather than
        # return a valid-looking 0.0 that environments and trainers would use.
        if not results:
            raise RewardUnavailableError(
                "CompositeReward: every reward function failed "
                f"({', '.join(failed)})"
            )

        # Combine rewards. math.fsum keeps the totals exactly rounded and
        # independent of summation order and Python version.
        component_weights = [float(rf.weight) for rf, _ in results]
        if self.normalize_weights and math.fsum(component_weights) > 0:
            weight_sum = math.fsum(component_weights)
            component_weights = [w / weight_sum for w in component_weights]
        elif failed and math.fsum(component_weights) > 0:
            # Spread a failed component's weight over the survivors so the
            # score keeps its scale; otherwise a failure scores as 0 and
            # biases advantages within the group.
            configured = math.fsum(float(rf.weight) for rf in self.reward_functions)
            scale = configured / math.fsum(component_weights)
            component_weights = [w * scale for w in component_weights]

        weighted_sum = math.fsum(
            rr.score * w for (_, rr), w in zip(results, component_weights, strict=True)
        )
        if self.combination_method == "weighted_sum":
            total_score = weighted_sum
        elif self.combination_method == "average":
            total_score = math.fsum(r.score for _, r in results) / len(results)
        elif self.combination_method == "min":
            total_score = min(r.score for _, r in results)
        elif self.combination_method == "max":
            total_score = max(r.score for _, r in results)
        else:
            logger.warning(
                f"Unknown combination method '{self.combination_method}', defaulting to weighted_sum"
            )
            total_score = weighted_sum

        components: dict[str, float] = {}
        for reward_fn, result in results:
            name = (reward_fn.name or reward_fn.__class__.__name__).lower()
            if name.endswith("reward"):
                name = name[: -len("reward")]
            components[name] = float(result.score)

        metadata: dict[str, Any] = {}
        fallbacks = [
            f"{rf.name}: {reason}"
            for rf, rr in results
            if (reason := reward_fallback(rr)) is not None
        ]
        if failed:
            # An opted-in renormalization is a fallback score: flag it.
            fallbacks.append(f"renormalized without {', '.join(failed)}")
            components["unavailable_components"] = float(len(failed))
        if fallbacks:
            # A child's opted-in fallback score is part of this score too.
            metadata[REWARD_FALLBACK_KEY] = "; ".join(fallbacks)
        return RewardResult(
            score=total_score,
            components=components,
            metadata={
                "component_scores": [r.score for _, r in results],
                "component_names": [rf.name for rf, _ in results],
                "component_breakdowns": [r.breakdown for _, r in results],
                "component_metadata": [r.metadata for _, r in results],
                "combination_method": self.combination_method,
                "normalize_weights": self.normalize_weights,
                COMPONENT_FAILURES_KEY: failed,
                **metadata,
            },
        )


# Custom reward function decorator
def reward_function(
    weight: float = 1.0, reward_type: RewardType = RewardType.IMMEDIATE
) -> Callable[[Callable[..., Any]], "RewardFunction"]:
    """Decorator for creating custom reward functions"""

    def decorator(func: Callable[..., Any]) -> "RewardFunction":
        class CustomReward(RewardFunction):
            def __init__(self) -> None:
                super().__init__(weight, reward_type, func.__name__)
                self.func = func

            async def compute_reward(
                self,
                turns: list["ConversationTurn"],
                context: dict[str, Any] | None = None,
            ) -> "RewardResult":
                score = await self.func(turns, context)
                return RewardResult(
                    score=score,
                    breakdown={self.name: score},
                    metadata={"custom_function": True},
                )

        return CustomReward()

    return decorator
