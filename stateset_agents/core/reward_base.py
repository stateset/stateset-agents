"""
Base reward modeling classes and types for multi-turn agent training.

This module provides the foundational abstractions for reward functions
including the base class, result types, and composite reward implementation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)


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
    breakdown: Dict[str, float] = field(default_factory=dict)
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None

    def __init__(
        self,
        score: float,
        components: Optional[Dict[str, float]] = None,
        breakdown: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        explanation: Optional[str] = None,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "breakdown": self.breakdown,
            "components": self.components,
            "metadata": self.metadata,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardResult":
        return cls(
            score=float(data.get("score", 0.0)),
            breakdown=data.get("breakdown") or data.get("components") or {},
            components=data.get("components") or data.get("breakdown") or {},
            metadata=data.get("metadata") or {},
            explanation=data.get("explanation"),
        )


class RewardFunction(ABC):
    """
    Abstract base class for reward functions
    """

    def __init__(
        self,
        weight: float = 1.0,
        reward_type: RewardType = RewardType.IMMEDIATE,
        name: Optional[str] = None,
    ):
        self.weight = weight
        self.reward_type = reward_type
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward for conversation turns"""
        pass

    async def compute_turn_reward(
        self,
        turn: ConversationTurn,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> RewardResult:
        """Compute reward for a single turn"""
        turns = conversation_history or []
        turns = turns + [turn]
        return await self.compute_reward(turns, context)

    def __call__(self, *args, **kwargs):
        """Make reward function callable (sync wrapper for async compute_reward).

        Warning: Do not call this from within an async context.
        Use `await compute_reward()` directly instead.
        """
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
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
        reward_functions: List[RewardFunction],
        combination_method: str = "weighted_sum",
        normalize_weights: bool = False,
    ):
        super().__init__(name="CompositeReward")
        self.reward_functions = reward_functions
        self.combination_method = combination_method
        self.normalize_weights = normalize_weights

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute composite reward from all functions"""

        # Handle empty reward functions
        if not self.reward_functions:
            logger.warning("CompositeReward has no reward functions configured")
            return RewardResult(
                score=0.0,
                breakdown={},
                metadata={"error": "No reward functions configured"},
            )

        results: List[Tuple[RewardFunction, RewardResult]] = []
        for reward_fn in self.reward_functions:
            try:
                result = await reward_fn.compute_reward(turns, context)
                results.append((reward_fn, result))
            except Exception as e:
                logger.warning(f"Reward function {reward_fn.name} failed: {e}")
                # Continue with other reward functions
                continue

        # Handle case where all reward functions failed
        if not results:
            logger.error("All reward functions failed to compute")
            return RewardResult(
                score=0.0,
                components={},
                metadata={"error": "All reward functions failed"},
            )

        # Combine rewards with safe defaults for empty results
        component_weights = [float(rf.weight) for rf, _ in results]
        if self.normalize_weights and sum(component_weights) > 0:
            weight_sum = float(sum(component_weights))
            component_weights = [w / weight_sum for w in component_weights]

        if self.combination_method == "weighted_sum":
            total_score = sum(
                rr.score * w for (rf, rr), w in zip(results, component_weights)
            )
        elif self.combination_method == "average":
            total_score = float(np.mean([r.score for _, r in results])) if results else 0.0
        elif self.combination_method == "min":
            total_score = min((r.score for _, r in results), default=0.0)
        elif self.combination_method == "max":
            total_score = max((r.score for _, r in results), default=0.0)
        else:
            logger.warning(f"Unknown combination method '{self.combination_method}', defaulting to weighted_sum")
            total_score = sum(
                rr.score * w for (rf, rr), w in zip(results, component_weights)
            )

        components: Dict[str, float] = {}
        for reward_fn, result in results:
            name = (reward_fn.name or reward_fn.__class__.__name__).lower()
            if name.endswith("reward"):
                name = name[: -len("reward")]
            components[name] = float(result.score)

        return RewardResult(
            score=total_score,
            components=components,
            metadata={
                "component_scores": [r.score for _, r in results],
                "component_names": [rf.name for rf, _ in results],
                "combination_method": self.combination_method,
                "normalize_weights": self.normalize_weights,
            },
        )


# Custom reward function decorator
def reward_function(
    weight: float = 1.0, reward_type: RewardType = RewardType.IMMEDIATE
):
    """Decorator for creating custom reward functions"""

    def decorator(func):
        class CustomReward(RewardFunction):
            def __init__(self):
                super().__init__(weight, reward_type, func.__name__)
                self.func = func

            async def compute_reward(self, turns, context=None):
                score = await self.func(turns, context)
                return RewardResult(
                    score=score,
                    breakdown={self.name: score},
                    metadata={"custom_function": True},
                )

        return CustomReward()

    return decorator
