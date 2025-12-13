"""
Factory functions and utility rewards for creating reward combinations.

This module provides:
- Factory functions for creating common reward combinations
- SimilarityAwareReward for response matching
- Adaptive reward creation utilities
"""

from typing import Any, Dict, List, Optional

from .basic_rewards import (
    ConcisenessReward,
    CorrectnessReward,
    EngagementReward,
    HelpfulnessReward,
    SafetyReward,
    TaskCompletionReward,
)
from .domain_rewards import (
    CustomerServiceReward,
    SalesAssistantReward,
    TechnicalSupportReward,
)
from .reward_base import CompositeReward, RewardFunction, RewardResult, RewardType
from .trajectory import ConversationTurn


class SimilarityAwareReward(RewardFunction):
    """
    Reward function that considers similarity to expected responses
    """

    def __init__(
        self,
        weight: float = 1.0,
        expected_responses_map: Optional[Dict[str, str]] = None,
        base_reward_fn: Optional[RewardFunction] = None,
    ):
        super().__init__(weight, RewardType.IMMEDIATE, "SimilarityAwareReward")
        self.expected_responses_map = expected_responses_map or {}
        self.base_reward_fn = base_reward_fn

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward with similarity consideration"""

        # Get base reward if provided
        base_score = 0.0
        base_breakdown = {}
        if self.base_reward_fn:
            base_result = await self.base_reward_fn.compute_reward(turns, context)
            base_score = base_result.score * 0.5  # Weight base reward
            base_breakdown = base_result.breakdown

        # Compute similarity scores
        similarity_scores = []
        similarity_breakdown = {}

        for i, turn in enumerate(turns):
            if turn.role == "assistant":
                # Get the prompt/query for this turn
                prompt = context.get("prompt") if context else None
                expected = self.expected_responses_map.get(prompt) if prompt else None

                if expected:
                    import difflib

                    similarity = difflib.SequenceMatcher(
                        None, turn.content.lower(), expected.lower()
                    ).ratio()
                    similarity_scores.append(similarity)
                    similarity_breakdown[f"turn_{i}_similarity"] = similarity

        # Combine scores
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            total_score = base_score + (avg_similarity * 0.5)
        else:
            total_score = base_score

        # Merge breakdowns
        breakdown = {**base_breakdown, **similarity_breakdown}

        return RewardResult(
            score=total_score,
            breakdown=breakdown,
            metadata={
                "has_expected_responses": bool(similarity_scores),
                "avg_similarity": sum(similarity_scores) / len(similarity_scores)
                if similarity_scores
                else 0.0,
            },
        )


# Utility functions for creating common reward combinations


def create_helpful_agent_reward() -> CompositeReward:
    """Create reward function for helpful assistant"""
    return CompositeReward(
        [
            HelpfulnessReward(weight=0.4),
            SafetyReward(weight=0.3),
            CorrectnessReward(weight=0.2),
            EngagementReward(weight=0.1),
        ]
    )


def create_customer_service_reward() -> CompositeReward:
    """Create reward function for customer service agent"""
    return CompositeReward(
        [
            HelpfulnessReward(weight=0.35),
            SafetyReward(weight=0.25),
            ConcisenessReward(weight=0.2),
            EngagementReward(weight=0.2),
        ]
    )


def create_task_agent_reward(task_criteria: Dict[str, Any]) -> CompositeReward:
    """Create reward function for task-oriented agent"""
    return CompositeReward(
        [
            TaskCompletionReward(weight=0.5, task_criteria=task_criteria),
            CorrectnessReward(weight=0.3),
            SafetyReward(weight=0.2),
        ]
    )


def create_tutoring_reward() -> CompositeReward:
    """Create reward function for tutoring agent"""
    return CompositeReward(
        [
            HelpfulnessReward(weight=0.3),
            CorrectnessReward(weight=0.3),
            EngagementReward(weight=0.25),
            SafetyReward(weight=0.15),
        ]
    )


def create_domain_reward(
    domain: str,
    weight: float = 1.0,
    expected_responses: Optional[Dict[str, str]] = None,
    **kwargs,
) -> RewardFunction:
    """Create a domain-specific reward function"""

    domain_lower = domain.lower()

    if domain_lower in ["customer_service", "support", "cs"]:
        return CustomerServiceReward(weight, expected_responses)
    elif domain_lower in ["technical", "tech_support", "technical_support", "it", "it_support"]:
        return TechnicalSupportReward(weight)
    elif domain_lower in ["sales", "marketing", "product", "sales_assistant"]:
        return SalesAssistantReward(weight)
    else:
        # Default to a general helpful agent reward
        return create_helpful_agent_reward()


def create_adaptive_reward(
    base_rewards: List[RewardFunction],
    expected_responses: Optional[Dict[str, str]] = None,
    similarity_weight: float = 0.3,
) -> CompositeReward:
    """Create an adaptive reward that combines base rewards with similarity matching"""

    reward_functions = base_rewards.copy()

    # Add similarity reward if expected responses provided
    if expected_responses:
        similarity_reward = SimilarityAwareReward(
            weight=similarity_weight,
            expected_responses_map=expected_responses,
            base_reward_fn=None,  # Don't double-count base rewards
        )
        reward_functions.append(similarity_reward)

    # Normalize weights
    total_weight = sum(rf.weight for rf in reward_functions)
    for rf in reward_functions:
        rf.weight = rf.weight / total_weight

    return CompositeReward(reward_functions)
