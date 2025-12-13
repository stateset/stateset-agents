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

import logging

logger = logging.getLogger(__name__)

from .config import TrainingConfig, TrainingProfile, get_config_for_task
from .trainer import GRPOTrainer, MultiTurnGRPOTrainer, SingleTurnGRPOTrainer
from .train import train, TrainingMode
from .evaluation import EvaluationConfig, evaluate_agent

# vLLM backend for fast generation - use lazy import to avoid torchvision issues
# Don't import at module level, let users import when needed
VLLM_BACKEND_AVAILABLE = False
VLLM_AVAILABLE = False

try:
    # Just check if vllm_backend module exists without importing it
    import importlib.util
    spec = importlib.util.find_spec(".vllm_backend", package=__name__)
    if spec is not None:
        VLLM_BACKEND_AVAILABLE = True
except (ImportError, ValueError):
    pass

# Optional trainer/algorithm availability is computed without importing heavy deps.
import importlib
import importlib.util
from typing import Any, Dict, Tuple


def _has_spec(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# Lightweight dependency checks
try:  # pragma: no cover
    import trl  # type: ignore[import-not-found]

    TRL_AVAILABLE = hasattr(trl, "GRPOConfig")
except Exception:  # pragma: no cover
    TRL_AVAILABLE = False

_TORCH_AVAILABLE = _has_spec("torch")
GSPO_AVAILABLE = _TORCH_AVAILABLE
GEPO_AVAILABLE = _TORCH_AVAILABLE
DAPO_AVAILABLE = _TORCH_AVAILABLE
VAPO_AVAILABLE = _TORCH_AVAILABLE
PPO_AVAILABLE = _TORCH_AVAILABLE
KL_CONTROLLERS_AVAILABLE = _TORCH_AVAILABLE
EMA_AVAILABLE = _TORCH_AVAILABLE
RLAIF_AVAILABLE = _TORCH_AVAILABLE


_OPTIONAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    # TRL-based GRPO
    "ModelManager": (f"{__name__}.trl_grpo_trainer", "ModelManager"),
    "TRLGRPOConfig": (f"{__name__}.trl_grpo_trainer", "TRLGRPOConfig"),
    "TRLGRPODatasetBuilder": (f"{__name__}.trl_grpo_trainer", "TRLGRPODatasetBuilder"),
    "TRLGRPORewardFunction": (f"{__name__}.trl_grpo_trainer", "TRLGRPORewardFunction"),
    "TRLGRPOTrainerWrapper": (f"{__name__}.trl_grpo_trainer", "TRLGRPOTrainerWrapper"),
    "train_with_trl_grpo": (f"{__name__}.trl_grpo_trainer", "train_with_trl_grpo"),
    "train_customer_service_with_trl": (
        f"{__name__}.trl_grpo_trainer",
        "train_customer_service_with_trl",
    ),
    # GSPO / GEPO / DAPO / VAPO
    "GSPOConfig": (f"{__name__}.gspo_trainer", "GSPOConfig"),
    "GSPOTrainer": (f"{__name__}.gspo_trainer", "GSPOTrainer"),
    "train_with_gspo": (f"{__name__}.gspo_trainer", "train_with_gspo"),
    "GEPOConfig": (f"{__name__}.gepo_trainer", "GEPOConfig"),
    "GEPOTrainer": (f"{__name__}.gepo_trainer", "GEPOTrainer"),
    "train_with_gepo": (f"{__name__}.gepo_trainer", "train_with_gepo"),
    "DAPOConfig": (f"{__name__}.dapo_trainer", "DAPOConfig"),
    "DAPOTrainer": (f"{__name__}.dapo_trainer", "DAPOTrainer"),
    "DAPORewardShaper": (f"{__name__}.dapo_trainer", "DAPORewardShaper"),
    "DynamicSamplingBuffer": (f"{__name__}.dapo_trainer", "DynamicSamplingBuffer"),
    "train_with_dapo": (f"{__name__}.dapo_trainer", "train_with_dapo"),
    "train_reasoning_with_dapo": (
        f"{__name__}.dapo_trainer",
        "train_reasoning_with_dapo",
    ),
    "VAPOConfig": (f"{__name__}.vapo_trainer", "VAPOConfig"),
    "VAPOTrainer": (f"{__name__}.vapo_trainer", "VAPOTrainer"),
    "ValueHead": (f"{__name__}.vapo_trainer", "ValueHead"),
    "LengthAdaptiveGAE": (f"{__name__}.vapo_trainer", "LengthAdaptiveGAE"),
    "train_with_vapo": (f"{__name__}.vapo_trainer", "train_with_vapo"),
    # PPO
    "PPOConfig": (f"{__name__}.ppo_trainer", "PPOConfig"),
    "PPOTrainer": (f"{__name__}.ppo_trainer", "PPOTrainer"),
    "PPOValueHead": (f"{__name__}.ppo_trainer", "PPOValueHead"),
    "AdaptiveKLController": (f"{__name__}.ppo_trainer", "AdaptiveKLController"),
    "compute_gae": (f"{__name__}.ppo_trainer", "compute_gae"),
    "train_ppo": (f"{__name__}.ppo_trainer", "train_ppo"),
    # KL controllers
    "KLController": (f"{__name__}.kl_controllers", "KLController"),
    "FixedKLController": (f"{__name__}.kl_controllers", "FixedKLController"),
    "LinearKLScheduler": (f"{__name__}.kl_controllers", "LinearKLScheduler"),
    "CosineKLScheduler": (f"{__name__}.kl_controllers", "CosineKLScheduler"),
    "WarmupKLScheduler": (f"{__name__}.kl_controllers", "WarmupKLScheduler"),
    "HybridKLController": (f"{__name__}.kl_controllers", "HybridKLController"),
    "NoKLController": (f"{__name__}.kl_controllers", "NoKLController"),
    "create_kl_controller": (f"{__name__}.kl_controllers", "create_kl_controller"),
    # EMA
    "EMAModel": (f"{__name__}.ema", "EMAModel"),
    "EMACallback": (f"{__name__}.ema", "EMACallback"),
    "MultiEMA": (f"{__name__}.ema", "MultiEMA"),
    "create_ema_model": (f"{__name__}.ema", "create_ema_model"),
    # RLAIF
    "RLAIFConfig": (f"{__name__}.rlaif_trainer", "RLAIFConfig"),
    "RLAIFTrainer": (f"{__name__}.rlaif_trainer", "RLAIFTrainer"),
    "ConstitutionalAI": (f"{__name__}.rlaif_trainer", "ConstitutionalAI"),
    "train_rlaif": (f"{__name__}.rlaif_trainer", "train_rlaif"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GRPOTrainer",
    "MultiTurnGRPOTrainer",
    "SingleTurnGRPOTrainer",
    "train",
    "TrainingMode",
    "TrainingConfig",
    "TrainingProfile",
    "get_config_for_task",
    "EvaluationConfig",
    "evaluate_agent",
    "VLLM_BACKEND_AVAILABLE",
    "TRL_AVAILABLE",
    "GSPO_AVAILABLE",
    "GEPO_AVAILABLE",
    "DAPO_AVAILABLE",
    "VAPO_AVAILABLE",
    "PPO_AVAILABLE",
    "KL_CONTROLLERS_AVAILABLE",
    "EMA_AVAILABLE",
    "RLAIF_AVAILABLE",
]
