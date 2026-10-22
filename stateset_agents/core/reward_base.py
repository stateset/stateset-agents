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

from .reward_context import REWARD_FALLBACK_KEY, build_reward_context, reward_fallback
from .trajectory import ConversationTurn

logger = logging.getLogger(__name__)

REWARD_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
)


class RewardUnavailableError(RuntimeError):
    """A reward could not be computed for an external reason (judge or network
    outage, a verifier that cannot be reached).

    This is the only failure reward composers absorb: ``CompositeReward`` and
    ``MultiObjectiveRewardFunction`` leave the component out and spread its
    weight over the rest. Every other exception is a bug or bad data and
    propagates, so it is never mistaken for an outage.
    """


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

        Warning: Do not call this from within an async context.
        Use `await compute_reward()` directly instead.
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # We're in an async context - can't use asyncio.run()
            raise RuntimeError(
                "Cannot call RewardFunction synchronously from within an async context. "
                "Use `await reward_function.compute_reward(...)` instead of `reward_function(...)`."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No running loop, safe to use asyncio.run()
                return asyncio.run(self.compute_reward(*args, **kwargs))
            else:
                # Re-raise the "already in async context" error
                raise


class CompositeReward(RewardFunction):
    """
    Combines multiple reward functions with weights
    """

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
                # An outage: leave the component out and spread its weight.
                # Anything else (TypeError, AttributeError, bad data) is a bug
                # and propagates rather than silently reweighting the reward.
                if getattr(reward_fn, "fail_on_error", False):
                    raise
                logger.warning(
                    "Reward function %s is unavailable; its weight is spread "
                    "over the other components: %s",
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
                "failed_components": failed,
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
