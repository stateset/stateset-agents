"""
Training infrastructure for GRPO Agent Framework
"""

from .trainer import GRPOTrainer, MultiTurnGRPOTrainer
from .config import TrainingConfig, TrainingProfile

# TRL-based GRPO training
try:
    from .trl_grpo_trainer import (
        TRLGRPOConfig,
        ModelManager,
        TRLGRPODatasetBuilder,
        TRLGRPORewardFunction,
        TRLGRPOTrainerWrapper,
        train_with_trl_grpo,
        train_customer_service_with_trl
    )
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

__all__ = [
    "GRPOTrainer",
    "MultiTurnGRPOTrainer", 
    "TrainingConfig",
    "TrainingProfile",
]

# Add TRL exports if available
if TRL_AVAILABLE:
    __all__.extend([
        "TRLGRPOConfig",
        "ModelManager",
        "TRLGRPODatasetBuilder",
        "TRLGRPORewardFunction",
        "TRLGRPOTrainerWrapper",
        "train_with_trl_grpo",
        "train_customer_service_with_trl"
    ])