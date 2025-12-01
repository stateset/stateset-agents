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

# vLLM backend for fast generation
try:
    from .vllm_backend import (
        VLLM_AVAILABLE,
        VLLMConfig,
        VLLMGenerator,
        HuggingFaceGeneratorFallback,
        GenerationResult,
        BatchGenerationResult,
        create_generator,
        quick_generate,
    )

    VLLM_BACKEND_AVAILABLE = True
except ImportError:
    VLLM_BACKEND_AVAILABLE = False
    VLLM_AVAILABLE = False

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
    # Config
    "TrainingConfig",
    "TrainingProfile",
    "get_config_for_task",
    # vLLM backend availability flag
    "VLLM_BACKEND_AVAILABLE",
]

# Add vLLM exports if available
if VLLM_BACKEND_AVAILABLE:
    __all__.extend(
        [
            "VLLM_AVAILABLE",
            "VLLMConfig",
            "VLLMGenerator",
            "HuggingFaceGeneratorFallback",
            "GenerationResult",
            "BatchGenerationResult",
            "create_generator",
            "quick_generate",
        ]
    )

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
