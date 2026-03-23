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

# Re-export base classes and types
from .reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
    reward_function,
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
]
