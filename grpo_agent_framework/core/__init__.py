"""Compatibility bridge for legacy ``grpo_agent_framework.core`` imports."""

import warnings

from stateset_agents.core.environment import Environment
from stateset_agents.core.reward import CompositeReward, RewardFunction
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory, Trajectory

warnings.warn(
    "Importing from 'grpo_agent_framework.core' is deprecated; use "
    "'stateset_agents.core' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Environment",
    "CompositeReward",
    "RewardFunction",
    "Trajectory",
    "MultiTurnTrajectory",
    "ConversationTurn",
]
