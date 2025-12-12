"""
GRPO Training Implementation with HuggingFace and W&B Integration

This module provides the core training infrastructure for multi-turn agents
using Group Relative Policy Optimization (GRPO) with seamless integration
to HuggingFace transformers and Weights & Biases.

This is a facade module that re-exports all trainer components for backwards
compatibility. For new code, consider importing directly from the specific
submodules:

- trainer_utils: require_torch, require_transformers, get_torch, etc.
- single_turn_trainer: SingleTurnGRPOTrainer
- multi_turn_trainer: MultiTurnGRPOTrainer, GRPOTrainer
- loss_computation: compute_grpo_loss, compute_enhanced_grpo_loss
"""

from __future__ import annotations

# Backwards-compatible globals.
# Some unit tests and legacy integrations patch `training.trainer.torch/np`.
try:  # pragma: no cover - optional dependency
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

# Re-export utility functions
from .trainer_utils import (
    get_amp,
    get_functional,
    get_torch,
    require_torch,
    require_transformers,
)

# Re-export trainers
from .single_turn_trainer import SingleTurnGRPOTrainer
from .multi_turn_trainer import GRPOTrainer, MultiTurnGRPOTrainer

# Re-export loss computation functions
from .loss_computation import compute_enhanced_grpo_loss, compute_grpo_loss

# Re-export core dependencies for type hints
from stateset_agents.core import agent as core_agent
from stateset_agents.core import environment as core_environment
from stateset_agents.core import reward as core_reward
from stateset_agents.core import trajectory as core_trajectory

Agent = core_agent.Agent
MultiTurnAgent = core_agent.MultiTurnAgent
AgentConfig = core_agent.AgentConfig

Environment = core_environment.Environment

MultiTurnTrajectory = core_trajectory.MultiTurnTrajectory
TrajectoryGroup = core_trajectory.TrajectoryGroup
ConversationTurn = core_trajectory.ConversationTurn

RewardFunction = core_reward.RewardFunction
CompositeReward = core_reward.CompositeReward

try:
    from utils.wandb_integration import WandBLogger as _WandBLogger
except ImportError:  # pragma: no cover - optional dependency
    _WandBLogger = None  # type: ignore[assignment]

WandBLogger = _WandBLogger  # type: ignore[assignment]

__all__ = [
    # Trainers
    "SingleTurnGRPOTrainer",
    "MultiTurnGRPOTrainer",
    "GRPOTrainer",
    # Loss computation
    "compute_grpo_loss",
    "compute_enhanced_grpo_loss",
    # Utilities
    "require_torch",
    "require_transformers",
    "get_torch",
    "get_functional",
    "get_amp",
    # Core types
    "Agent",
    "MultiTurnAgent",
    "AgentConfig",
    "Environment",
    "MultiTurnTrajectory",
    "TrajectoryGroup",
    "ConversationTurn",
    "RewardFunction",
    "CompositeReward",
    "WandBLogger",
]
