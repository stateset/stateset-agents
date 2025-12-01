"""
Reward modeling framework for multi-turn agent training.

This module provides flexible reward functions for evaluating agent performance
in multi-turn conversations and task-oriented interactions.

This is a facade module that re-exports all reward components for backwards
compatibility. For new code, consider importing directly from the specific
submodules:

- reward_base: RewardType, RewardResult, RewardFunction, CompositeReward
- basic_rewards: HelpfulnessReward, SafetyReward, CorrectnessReward, etc.
- domain_rewards: CustomerServiceReward, TechnicalSupportReward, etc.
- reward_factories: create_helpful_agent_reward, create_domain_reward, etc.
"""

# Re-export base classes and types
from .reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
    reward_function,
)

# Re-export basic reward functions
from .basic_rewards import (
    ConcisenessReward,
    CorrectnessReward,
    EngagementReward,
    HelpfulnessReward,
    SafetyReward,
    TaskCompletionReward,
)

# Re-export domain-specific rewards
from .domain_rewards import (
    CustomerServiceReward,
    DomainSpecificReward,
    SalesAssistantReward,
    TechnicalSupportReward,
)

# Re-export factories and utilities
from .reward_factories import (
    SimilarityAwareReward,
    create_adaptive_reward,
    create_customer_service_reward,
    create_domain_reward,
    create_helpful_agent_reward,
    create_task_agent_reward,
    create_tutoring_reward,
)

# Example usage of custom reward decorator
@reward_function(weight=0.5)
async def politeness_reward(turns, context=None) -> float:
    """Custom reward for politeness"""
    from .trajectory import ConversationTurn

    polite_phrases = [
        "please",
        "thank you",
        "you're welcome",
        "excuse me",
        "i apologize",
    ]

    assistant_turns = [t for t in turns if t.role == "assistant"]
    if not assistant_turns:
        return 0.0

    total_score = 0.0
    for turn in assistant_turns:
        content_lower = turn.content.lower()
        phrase_count = sum(1 for phrase in polite_phrases if phrase in content_lower)
        turn_score = min(phrase_count * 0.2, 1.0)
        total_score += turn_score

    return total_score / len(assistant_turns)


__all__ = [
    # Base types
    "RewardType",
    "RewardResult",
    "RewardFunction",
    "CompositeReward",
    "reward_function",
    # Basic rewards
    "HelpfulnessReward",
    "SafetyReward",
    "CorrectnessReward",
    "ConcisenessReward",
    "EngagementReward",
    "TaskCompletionReward",
    # Domain rewards
    "DomainSpecificReward",
    "CustomerServiceReward",
    "TechnicalSupportReward",
    "SalesAssistantReward",
    # Utility rewards
    "SimilarityAwareReward",
    # Factory functions
    "create_helpful_agent_reward",
    "create_customer_service_reward",
    "create_task_agent_reward",
    "create_tutoring_reward",
    "create_domain_reward",
    "create_adaptive_reward",
    # Example custom reward
    "politeness_reward",
]
