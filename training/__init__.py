"""
Training infrastructure for GRPO Agent Framework
"""

from .config import TrainingConfig, TrainingProfile, get_config_for_task
from .trainer import GRPOTrainer, MultiTurnGRPOTrainer

# TRL-based GRPO training
try:
    from .trl_grpo_trainer import (
        ModelManager,
        TRLGRPOConfig,
        TRLGRPODatasetBuilder,
        TRLGRPORewardFunction,
        TRLGRPOTrainerWrapper,
        train_customer_service_with_trl,
        train_with_trl_grpo,
    )

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

__all__ = [
    "GRPOTrainer",
    "MultiTurnGRPOTrainer",
    "TrainingConfig",
    "TrainingProfile",
    "get_config_for_task",
]

# Add TRL exports if available
if TRL_AVAILABLE:
    __all__.extend(
        [
            "TRLGRPOConfig",
            "ModelManager",
            "TRLGRPODatasetBuilder",
            "TRLGRPORewardFunction",
            "TRLGRPOTrainerWrapper",
            "train_with_trl_grpo",
            "train_customer_service_with_trl",
        ]
    )
