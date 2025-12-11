"""
Training infrastructure for GRPO Agent Framework

Includes state-of-the-art RL algorithms:
- GRPO: Group Relative Policy Optimization
- GSPO: Group Sequence Policy Optimization
- GEPO: Group Expectation Policy Optimization (best for heterogeneous/distributed)
- DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization (best for reasoning)
- VAPO: Value-Augmented Policy Optimization (SOTA: 60.4 on AIME 2024)

Generation backends:
- vLLM: 5-20x faster generation with automatic log probability extraction
- HuggingFace: Standard generation fallback
"""

from .config import TrainingConfig, TrainingProfile, get_config_for_task
from .trainer import GRPOTrainer, MultiTurnGRPOTrainer, SingleTurnGRPOTrainer
from .train import train, TrainingMode

# vLLM backend for fast generation - use lazy import to avoid torchvision issues
# Don't import at module level, let users import when needed
VLLM_BACKEND_AVAILABLE = False
VLLM_AVAILABLE = False

try:
    # Just check if vllm_backend module exists without importing it
    import importlib.util
    spec = importlib.util.find_spec(".vllm_backend", package="training")
    if spec is not None:
        VLLM_BACKEND_AVAILABLE = True
except (ImportError, ValueError):
    pass

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

# GSPO trainer
try:
    from .gspo_trainer import (
        GSPOConfig,
        GSPOTrainer,
        train_with_gspo,
    )

    GSPO_AVAILABLE = True
except ImportError:
    GSPO_AVAILABLE = False

# GEPO trainer (Group Expectation Policy Optimization)
try:
    from .gepo_trainer import (
        GEPOConfig,
        GEPOTrainer,
        train_with_gepo,
    )

    GEPO_AVAILABLE = True
except ImportError:
    GEPO_AVAILABLE = False

# DAPO trainer (Decoupled Clip and Dynamic Sampling)
try:
    from .dapo_trainer import (
        DAPOConfig,
        DAPOTrainer,
        DAPORewardShaper,
        DynamicSamplingBuffer,
        train_with_dapo,
        train_reasoning_with_dapo,
    )

    DAPO_AVAILABLE = True
except ImportError:
    DAPO_AVAILABLE = False

# VAPO trainer (Value-Augmented Policy Optimization)
try:
    from .vapo_trainer import (
        VAPOConfig,
        VAPOTrainer,
        ValueHead,
        LengthAdaptiveGAE,
        train_with_vapo,
    )

    VAPO_AVAILABLE = True
except ImportError:
    VAPO_AVAILABLE = False

__all__ = [
    # Core trainers
    "GRPOTrainer",
    "MultiTurnGRPOTrainer",
    "SingleTurnGRPOTrainer",
    # High-level training interface
    "train",
    "TrainingMode",
    # Config
    "TrainingConfig",
    "TrainingProfile",
    "get_config_for_task",
    # vLLM backend availability flag
    "VLLM_BACKEND_AVAILABLE",
]

# vLLM exports are not added to __all__ to avoid module-level import
# Users should import directly from training.vllm_backend when needed:
# from training.vllm_backend import VLLMConfig, VLLMGenerator, etc.

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

# Add GSPO exports if available
if GSPO_AVAILABLE:
    __all__.extend(
        [
            "GSPOConfig",
            "GSPOTrainer",
            "train_with_gspo",
        ]
    )

# Add GEPO exports if available
if GEPO_AVAILABLE:
    __all__.extend(
        [
            "GEPOConfig",
            "GEPOTrainer",
            "train_with_gepo",
        ]
    )

# Add DAPO exports if available
if DAPO_AVAILABLE:
    __all__.extend(
        [
            "DAPOConfig",
            "DAPOTrainer",
            "DAPORewardShaper",
            "DynamicSamplingBuffer",
            "train_with_dapo",
            "train_reasoning_with_dapo",
        ]
    )

# Add VAPO exports if available
if VAPO_AVAILABLE:
    __all__.extend(
        [
            "VAPOConfig",
            "VAPOTrainer",
            "ValueHead",
            "LengthAdaptiveGAE",
            "train_with_vapo",
        ]
    )

# PPO trainer (baseline algorithm)
try:
    from .ppo_trainer import (
        PPOConfig,
        PPOTrainer,
        PPOValueHead,
        AdaptiveKLController,
        compute_gae,
        train_ppo,
    )

    PPO_AVAILABLE = True
    __all__.extend(
        [
            "PPOConfig",
            "PPOTrainer",
            "PPOValueHead",
            "AdaptiveKLController",
            "compute_gae",
            "train_ppo",
        ]
    )
except ImportError:
    PPO_AVAILABLE = False

# KL Controllers
try:
    from .kl_controllers import (
        KLController,
        FixedKLController,
        AdaptiveKLController as KLAdaptiveController,
        LinearKLScheduler,
        CosineKLScheduler,
        WarmupKLScheduler,
        HybridKLController,
        NoKLController,
        create_kl_controller,
    )

    KL_CONTROLLERS_AVAILABLE = True
    __all__.extend(
        [
            "KLController",
            "FixedKLController",
            "LinearKLScheduler",
            "CosineKLScheduler",
            "WarmupKLScheduler",
            "HybridKLController",
            "NoKLController",
            "create_kl_controller",
        ]
    )
except ImportError:
    KL_CONTROLLERS_AVAILABLE = False

# EMA Model Support
try:
    from .ema import (
        EMAModel,
        EMACallback,
        MultiEMA,
        create_ema_model,
    )

    EMA_AVAILABLE = True
    __all__.extend(
        [
            "EMAModel",
            "EMACallback",
            "MultiEMA",
            "create_ema_model",
        ]
    )
except ImportError:
    EMA_AVAILABLE = False

# RLAIF Trainer
try:
    from .rlaif_trainer import (
        RLAIFConfig,
        RLAIFTrainer,
        ConstitutionalAI,
        train_rlaif,
    )

    RLAIF_AVAILABLE = True
    __all__.extend(
        [
            "RLAIFConfig",
            "RLAIFTrainer",
            "ConstitutionalAI",
            "train_rlaif",
        ]
    )
except ImportError:
    RLAIF_AVAILABLE = False
