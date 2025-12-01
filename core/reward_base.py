"""
Base reward modeling classes and types for multi-turn agent training.

This module provides the foundational abstractions for reward functions
including the base class, result types, and composite reward implementation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of rewards"""

    IMMEDIATE = "immediate"  # Reward for single turn
    CUMULATIVE = "cumulative"  # Reward for entire conversation
    SPARSE = "sparse"  # Reward only at episode end
    DENSE = "dense"  # Reward at every step


@dataclass
class RewardResult:
    """Result of reward computation"""

    score: float
    breakdown: Dict[str, float]
    metadata: Dict[str, Any]
    explanation: Optional[str] = None


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
        """Make reward function callable"""
        return asyncio.run(self.compute_reward(*args, **kwargs))


class CompositeReward(RewardFunction):
    """
    Combines multiple reward functions with weights
    """

    def __init__(
        self,
        reward_functions: List[RewardFunction],
        combination_method: str = "weighted_sum",
    ):
        super().__init__(name="CompositeReward")
        self.reward_functions = reward_functions
        self.combination_method = combination_method

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute composite reward from all functions"""

        results = []
        for reward_fn in self.reward_functions:
            result = await reward_fn.compute_reward(turns, context)
            results.append(result)

        # Combine rewards
        if self.combination_method == "weighted_sum":
            total_score = sum(
                r.score * rf.weight for r, rf in zip(results, self.reward_functions)
            )
        elif self.combination_method == "average":
            total_score = np.mean([r.score for r in results])
        elif self.combination_method == "min":
            total_score = min(r.score for r in results)
        elif self.combination_method == "max":
            total_score = max(r.score for r in results)
        else:
            total_score = sum(
                r.score * rf.weight for r, rf in zip(results, self.reward_functions)
            )

        # Combine breakdowns
        combined_breakdown = {}
        for result, reward_fn in zip(results, self.reward_functions):
            for key, value in result.breakdown.items():
                combined_key = f"{reward_fn.name}_{key}"
                combined_breakdown[combined_key] = value

        return RewardResult(
            score=total_score,
            breakdown=combined_breakdown,
            metadata={
                "component_scores": [r.score for r in results],
                "component_names": [rf.name for rf in self.reward_functions],
                "combination_method": self.combination_method,
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
