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

import warnings

# Re-export loss computation functions
from .loss_computation import compute_enhanced_grpo_loss, compute_grpo_loss
from .multi_turn_trainer import GRPOTrainer, MultiTurnGRPOTrainer

# Re-export trainers
from .single_turn_trainer import SingleTurnGRPOTrainer

# Re-export utility functions
from .trainer_utils import (
    get_amp,
    get_functional,
    get_torch,
    require_torch,
    require_transformers,
)

# ---- Deprecated re-exports ------------------------------------------------
# These core types should be imported from their canonical modules instead of
# from training.trainer.  We keep them accessible via __getattr__ so existing
# code doesn't break, but emit a DeprecationWarning on first access.

_DEPRECATED_ALIASES = {
    "Agent": "stateset_agents.core.agent",
    "MultiTurnAgent": "stateset_agents.core.agent",
    "AgentConfig": "stateset_agents.core.agent",
    "Environment": "stateset_agents.core.environment",
    "MultiTurnTrajectory": "stateset_agents.core.trajectory",
    "TrajectoryGroup": "stateset_agents.core.trajectory",
    "ConversationTurn": "stateset_agents.core.trajectory",
    "RewardFunction": "stateset_agents.core.reward",
    "CompositeReward": "stateset_agents.core.reward",
    "WandBLogger": "stateset_agents.utils.wandb_integration",
}


def __getattr__(name: str):
    if name in _DEPRECATED_ALIASES:
        canonical = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"Importing {name} from stateset_agents.training.trainer is deprecated. "
            f"Use 'from {canonical} import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        mod = importlib.import_module(canonical)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
]
