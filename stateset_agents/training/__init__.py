"""
Training infrastructure for GRPO Agent Framework

Includes state-of-the-art RL algorithms:
- GRPO: Group Relative Policy Optimization
- GSPO: Group Sequence Policy Optimization
- GEPO: Group Expectation Policy Optimization (best for heterogeneous/distributed)
- DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization (best for reasoning)
- VAPO: Value-Augmented Policy Optimization (SOTA: 60.4 on AIME 2024)

Offline RL algorithms:
- CQL: Conservative Q-Learning
- IQL: Implicit Q-Learning
- BCQ: Batch-Constrained Q-Learning
- BEAR: Bootstrapping Error Accumulation Reduction
- Decision Transformer: Sequence modeling approach

Sim-to-Real Transfer:
- Domain Randomization: Persona, topic, and style randomization
- System Identification: Learn user behavior models
- Progressive Transfer: Gradual sim-to-real transition

Generation backends:
- vLLM: 5-20x faster generation with automatic log probability extraction
- HuggingFace: Standard generation fallback
"""

import importlib
import importlib.util
import logging
from typing import Any

from .config import TrainingConfig, TrainingProfile, get_config_for_task
from .continual_learning import (
    ContinualLearningConfig,
    ContinualLearningManager,
    TrajectoryReplayBuffer,
)
from .evaluation import EvaluationConfig, evaluate_agent
from .train import TrainingMode, train
from .trainer import GRPOTrainer, MultiTurnGRPOTrainer, SingleTurnGRPOTrainer

logger = logging.getLogger(__name__)

# vLLM backend for fast generation - use lazy import to avoid torchvision issues
# Don't import at module level, let users import when needed
VLLM_BACKEND_AVAILABLE = False
VLLM_AVAILABLE = False

try:
    # Just check if vllm_backend module exists without importing it
    spec = importlib.util.find_spec(".vllm_backend", package=__name__)
    if spec is not None:
        VLLM_BACKEND_AVAILABLE = True
except (ImportError, ValueError):
    pass

# Optional trainer/algorithm availability is computed without importing heavy deps.


def _has_spec(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# Lightweight dependency checks
try:  # pragma: no cover
    import trl

    TRL_AVAILABLE = hasattr(trl, "GRPOConfig")
except ImportError:  # pragma: no cover
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

# Offline RL and Sim-to-Real availability
OFFLINE_RL_AVAILABLE = _TORCH_AVAILABLE
BCQ_AVAILABLE = _TORCH_AVAILABLE
BEAR_AVAILABLE = _TORCH_AVAILABLE
DECISION_TRANSFORMER_AVAILABLE = _TORCH_AVAILABLE
SIM_TO_REAL_AVAILABLE = _TORCH_AVAILABLE


AUTO_RESEARCH_AVAILABLE = True

_OPTIONAL_EXPORTS: dict[str, tuple[str, str]] = {
    # Auto-Research Loop
    "AutoResearchConfig": (f"{__name__}.auto_research.config", "AutoResearchConfig"),
    "AutoResearchLoop": (f"{__name__}.auto_research.experiment_loop", "AutoResearchLoop"),
    "run_auto_research": (f"{__name__}.auto_research.experiment_loop", "run_auto_research"),
    "ExperimentTracker": (f"{__name__}.auto_research.experiment_tracker", "ExperimentTracker"),
    "ExperimentRecord": (f"{__name__}.auto_research.experiment_tracker", "ExperimentRecord"),
    "CheckpointManager": (f"{__name__}.auto_research.checkpoint_manager", "CheckpointManager"),
    "ExperimentProposer": (f"{__name__}.auto_research.proposer", "ExperimentProposer"),
    "RandomProposer": (f"{__name__}.auto_research.proposer", "RandomProposer"),
    "PerturbationProposer": (f"{__name__}.auto_research.proposer", "PerturbationProposer"),
    "GridProposer": (f"{__name__}.auto_research.proposer", "GridProposer"),
    "BayesianProposer": (f"{__name__}.auto_research.proposer", "BayesianProposer"),
    "LLMProposer": (f"{__name__}.auto_research.llm_proposer", "LLMProposer"),
    "create_proposer": (f"{__name__}.auto_research.proposer", "create_proposer"),
    "create_auto_research_search_space": (f"{__name__}.auto_research.search_spaces", "create_auto_research_search_space"),
    "create_quick_search_space": (f"{__name__}.auto_research.search_spaces", "create_quick_search_space"),
    "get_auto_research_search_space": (f"{__name__}.auto_research.search_spaces", "get_auto_research_search_space"),
    # TRL-based GRPO
    "ModelManager": (f"{__name__}.trl_grpo_trainer", "ModelManager"),
    "TRLGRPOConfig": (f"{__name__}.trl_grpo_config", "TRLGRPOConfig"),
    "TRLGRPODatasetBuilder": (f"{__name__}.trl_grpo_trainer", "TRLGRPODatasetBuilder"),
    "TRLGRPORewardFunction": (f"{__name__}.trl_grpo_trainer", "TRLGRPORewardFunction"),
    "TRLGRPOTrainerWrapper": (f"{__name__}.trl_grpo_trainer", "TRLGRPOTrainerWrapper"),
    "train_with_trl_grpo": (f"{__name__}.trl_grpo_entrypoints", "train_with_trl_grpo"),
    "train_customer_service_with_trl": (
        f"{__name__}.trl_grpo_entrypoints",
        "train_customer_service_with_trl",
    ),
    # GSPO / GEPO / DAPO / VAPO
    "GSPOConfig": (f"{__name__}.gspo_trainer", "GSPOConfig"),
    "GSPOTrainer": (f"{__name__}.gspo_trainer", "GSPOTrainer"),
    "GSPO_Trainer": (f"{__name__}.gspo_trainer", "GSPO_Trainer"),
    "train_with_gspo": (f"{__name__}.gspo_trainer", "train_with_gspo"),
    # Qwen3.5 starter path
    "QWEN35_08B_BASE_MODEL": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_BASE_MODEL"),
    "QWEN35_08B_CONFIG_SUFFIXES": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_CONFIG_SUFFIXES"),
    "QWEN35_08B_DEFAULT_OUTPUT_DIR": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_DEFAULT_OUTPUT_DIR"),
    "QWEN35_08B_LORA_TARGET_MODULES": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_LORA_TARGET_MODULES"),
    "QWEN35_08B_POST_TRAINED_MODEL": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_POST_TRAINED_MODEL"),
    "QWEN35_08B_STARTER_PROFILE_CHOICES": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_STARTER_PROFILE_CHOICES"),
    "QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS"),
    "QWEN35_08B_SUPPORTED_VARIANTS": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_SUPPORTED_VARIANTS"),
    "QWEN35_08B_TASK_CHOICES": (f"{__name__}.qwen3_5_starter", "QWEN35_08B_TASK_CHOICES"),
    "Qwen35Config": (f"{__name__}.qwen3_5_starter", "Qwen35Config"),
    "create_qwen3_5_agent_config": (f"{__name__}.qwen3_5_starter", "create_qwen3_5_agent_config"),
    "create_qwen3_5_preview": (f"{__name__}.qwen3_5_starter", "create_qwen3_5_preview"),
    "describe_qwen3_5_starter_profiles": (f"{__name__}.qwen3_5_starter", "describe_qwen3_5_starter_profiles"),
    "finetune_qwen3_5_0_8b": (f"{__name__}.qwen3_5_starter", "finetune_qwen3_5_0_8b"),
    "get_qwen3_5_config": (f"{__name__}.qwen3_5_starter", "get_qwen3_5_config"),
    "get_qwen3_5_gspo_config": (f"{__name__}.qwen3_5_starter", "get_qwen3_5_gspo_config"),
    "get_qwen3_5_gspo_overrides": (f"{__name__}.qwen3_5_starter", "get_qwen3_5_gspo_overrides"),
    "get_qwen3_5_profile_description": (f"{__name__}.qwen3_5_starter", "get_qwen3_5_profile_description"),
    "get_qwen3_5_profile_overrides": (f"{__name__}.qwen3_5_starter", "get_qwen3_5_profile_overrides"),
    "get_qwen3_5_system_prompt": (f"{__name__}.qwen3_5_starter", "get_qwen3_5_system_prompt"),
    "load_qwen3_5_config_file": (f"{__name__}.qwen3_5_starter", "load_qwen3_5_config_file"),
    "run_qwen3_5_0_8b_config": (f"{__name__}.qwen3_5_starter", "run_qwen3_5_0_8b_config"),
    "summarize_qwen3_5_config": (f"{__name__}.qwen3_5_starter", "summarize_qwen3_5_config"),
    "validate_qwen3_5_config": (f"{__name__}.qwen3_5_starter", "validate_qwen3_5_config"),
    "write_qwen3_5_config_file": (f"{__name__}.qwen3_5_starter", "write_qwen3_5_config_file"),
    # Serving artifacts
    "build_serving_manifest": (f"{__name__}.serving_artifacts", "build_serving_manifest"),
    "export_merged_model_for_serving": (f"{__name__}.serving_artifacts", "export_merged_model_for_serving"),
    "write_serving_manifest": (f"{__name__}.serving_artifacts", "write_serving_manifest"),
    # Gemma 4 31B starter path
    "GEMMA4_31B_BASE_MODEL": (f"{__name__}.gemma4_starter", "GEMMA4_31B_BASE_MODEL"),
    "GEMMA4_31B_CONFIG_SUFFIXES": (f"{__name__}.gemma4_starter", "GEMMA4_31B_CONFIG_SUFFIXES"),
    "GEMMA4_31B_DEFAULT_OUTPUT_DIR": (f"{__name__}.gemma4_starter", "GEMMA4_31B_DEFAULT_OUTPUT_DIR"),
    "GEMMA4_31B_LORA_TARGET_MODULES": (f"{__name__}.gemma4_starter", "GEMMA4_31B_LORA_TARGET_MODULES"),
    "GEMMA4_31B_STARTER_PROFILE_CHOICES": (f"{__name__}.gemma4_starter", "GEMMA4_31B_STARTER_PROFILE_CHOICES"),
    "GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS": (f"{__name__}.gemma4_starter", "GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS"),
    "GEMMA4_31B_SUPPORTED_VARIANTS": (f"{__name__}.gemma4_starter", "GEMMA4_31B_SUPPORTED_VARIANTS"),
    "GEMMA4_31B_TASK_CHOICES": (f"{__name__}.gemma4_starter", "GEMMA4_31B_TASK_CHOICES"),
    "Gemma4Config": (f"{__name__}.gemma4_starter", "Gemma4Config"),
    "create_gemma4_31b_agent_config": (f"{__name__}.gemma4_starter", "create_gemma4_31b_agent_config"),
    "create_gemma4_31b_preview": (f"{__name__}.gemma4_starter", "create_gemma4_31b_preview"),
    "describe_gemma4_31b_starter_profiles": (f"{__name__}.gemma4_starter", "describe_gemma4_31b_starter_profiles"),
    "finetune_gemma4_31b": (f"{__name__}.gemma4_starter", "finetune_gemma4_31b"),
    "get_gemma4_31b_config": (f"{__name__}.gemma4_starter", "get_gemma4_31b_config"),
    "get_gemma4_31b_gspo_config": (f"{__name__}.gemma4_starter", "get_gemma4_31b_gspo_config"),
    "get_gemma4_31b_gspo_overrides": (f"{__name__}.gemma4_starter", "get_gemma4_31b_gspo_overrides"),
    "get_gemma4_31b_profile_description": (f"{__name__}.gemma4_starter", "get_gemma4_31b_profile_description"),
    "get_gemma4_31b_profile_overrides": (f"{__name__}.gemma4_starter", "get_gemma4_31b_profile_overrides"),
    "get_gemma4_31b_system_prompt": (f"{__name__}.gemma4_starter", "get_gemma4_31b_system_prompt"),
    "load_gemma4_31b_config_file": (f"{__name__}.gemma4_starter", "load_gemma4_31b_config_file"),
    "run_gemma4_31b_config": (f"{__name__}.gemma4_starter", "run_gemma4_31b_config"),
    "summarize_gemma4_31b_config": (f"{__name__}.gemma4_starter", "summarize_gemma4_31b_config"),
    "validate_gemma4_31b_config": (f"{__name__}.gemma4_starter", "validate_gemma4_31b_config"),
    "write_gemma4_31b_config_file": (f"{__name__}.gemma4_starter", "write_gemma4_31b_config_file"),
    # GLM 5.1 starter path
    "GLM5_1_BASE_MODEL": (f"{__name__}.glm5_1_starter", "GLM5_1_BASE_MODEL"),
    "GLM5_1_CONFIG_SUFFIXES": (f"{__name__}.glm5_1_starter", "GLM5_1_CONFIG_SUFFIXES"),
    "GLM5_1_DEFAULT_OUTPUT_DIR": (f"{__name__}.glm5_1_starter", "GLM5_1_DEFAULT_OUTPUT_DIR"),
    "GLM5_1_FP8_MODEL": (f"{__name__}.glm5_1_starter", "GLM5_1_FP8_MODEL"),
    "GLM5_1_LORA_TARGET_MODULES": (f"{__name__}.glm5_1_starter", "GLM5_1_LORA_TARGET_MODULES"),
    "GLM5_1_STARTER_PROFILE_CHOICES": (f"{__name__}.glm5_1_starter", "GLM5_1_STARTER_PROFILE_CHOICES"),
    "GLM5_1_STARTER_PROFILE_DESCRIPTIONS": (f"{__name__}.glm5_1_starter", "GLM5_1_STARTER_PROFILE_DESCRIPTIONS"),
    "GLM5_1_SUPPORTED_VARIANTS": (f"{__name__}.glm5_1_starter", "GLM5_1_SUPPORTED_VARIANTS"),
    "GLM5_1_TASK_CHOICES": (f"{__name__}.glm5_1_starter", "GLM5_1_TASK_CHOICES"),
    "Glm51Config": (f"{__name__}.glm5_1_starter", "Glm51Config"),
    "create_glm5_1_agent_config": (f"{__name__}.glm5_1_starter", "create_glm5_1_agent_config"),
    "create_glm5_1_preview": (f"{__name__}.glm5_1_starter", "create_glm5_1_preview"),
    "describe_glm5_1_starter_profiles": (f"{__name__}.glm5_1_starter", "describe_glm5_1_starter_profiles"),
    "finetune_glm5_1": (f"{__name__}.glm5_1_starter", "finetune_glm5_1"),
    "get_glm5_1_config": (f"{__name__}.glm5_1_starter", "get_glm5_1_config"),
    "get_glm5_1_gspo_config": (f"{__name__}.glm5_1_starter", "get_glm5_1_gspo_config"),
    "get_glm5_1_gspo_overrides": (f"{__name__}.glm5_1_starter", "get_glm5_1_gspo_overrides"),
    "get_glm5_1_profile_description": (f"{__name__}.glm5_1_starter", "get_glm5_1_profile_description"),
    "get_glm5_1_profile_overrides": (f"{__name__}.glm5_1_starter", "get_glm5_1_profile_overrides"),
    "get_glm5_1_serving_recommendations": (f"{__name__}.glm5_1_starter", "get_glm5_1_serving_recommendations"),
    "get_glm5_1_system_prompt": (f"{__name__}.glm5_1_starter", "get_glm5_1_system_prompt"),
    "load_glm5_1_config_file": (f"{__name__}.glm5_1_starter", "load_glm5_1_config_file"),
    "run_glm5_1_config": (f"{__name__}.glm5_1_starter", "run_glm5_1_config"),
    "summarize_glm5_1_config": (f"{__name__}.glm5_1_starter", "summarize_glm5_1_config"),
    "validate_glm5_1_config": (f"{__name__}.glm5_1_starter", "validate_glm5_1_config"),
    "write_glm5_1_config_file": (f"{__name__}.glm5_1_starter", "write_glm5_1_config_file"),
    "GEPOConfig": (f"{__name__}.gepo_trainer", "GEPOConfig"),
    "GEPOTrainer": (f"{__name__}.gepo_trainer", "GEPOTrainer"),
    "train_with_gepo": (f"{__name__}.gepo_trainer", "train_with_gepo"),
    "DAPOConfig": (f"{__name__}.dapo_config", "DAPOConfig"),
    "DAPOTrainer": (f"{__name__}.dapo_trainer", "DAPOTrainer"),
    "DAPORewardShaper": (f"{__name__}.dapo_trainer", "DAPORewardShaper"),
    "DynamicSamplingBuffer": (f"{__name__}.dapo_trainer", "DynamicSamplingBuffer"),
    "train_with_dapo": (f"{__name__}.dapo_entrypoints", "train_with_dapo"),
    "train_reasoning_with_dapo": (
        f"{__name__}.dapo_entrypoints",
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
    # Offline RL - CQL/IQL
    "CQLConfig": (f"{__name__}.offline_rl_algorithms", "CQLConfig"),
    "IQLConfig": (f"{__name__}.offline_rl_algorithms", "IQLConfig"),
    "ConservativeQLearning": (
        f"{__name__}.offline_rl_algorithms",
        "ConservativeQLearning",
    ),
    "ImplicitQLearning": (f"{__name__}.offline_rl_algorithms", "ImplicitQLearning"),
    "OfflineRLTrainer": (f"{__name__}.offline_rl_algorithms", "OfflineRLTrainer"),
    # Offline RL - BCQ
    "BCQConfig": (f"{__name__}.offline_rl_bcq", "BCQConfig"),
    "BatchConstrainedQLearning": (
        f"{__name__}.offline_rl_bcq",
        "BatchConstrainedQLearning",
    ),
    "ConversationalVAE": (f"{__name__}.offline_rl_bcq", "ConversationalVAE"),
    "BCQTrainer": (f"{__name__}.offline_rl_bcq", "BCQTrainer"),
    # Offline RL - BEAR
    "BEARConfig": (f"{__name__}.offline_rl_bear", "BEARConfig"),
    "ConversationalBEAR": (f"{__name__}.offline_rl_bear", "ConversationalBEAR"),
    "MMDKernel": (f"{__name__}.offline_rl_bear", "MMDKernel"),
    "BEARTrainer": (f"{__name__}.offline_rl_bear", "BEARTrainer"),
    # Decision Transformer
    "DecisionTransformerConfig": (
        f"{__name__}.decision_transformer",
        "DecisionTransformerConfig",
    ),
    "DecisionTransformer": (f"{__name__}.decision_transformer", "DecisionTransformer"),
    "DecisionTransformerTrainer": (
        f"{__name__}.decision_transformer",
        "DecisionTransformerTrainer",
    ),
    "ConversationEmbedder": (
        f"{__name__}.decision_transformer",
        "ConversationEmbedder",
    ),
    # Offline GRPO
    "OfflineGRPOConfig": (f"{__name__}.offline_grpo_trainer", "OfflineGRPOConfig"),
    "OfflineGRPOTrainer": (f"{__name__}.offline_grpo_trainer", "OfflineGRPOTrainer"),
    "OfflineRLAlgorithm": (f"{__name__}.offline_grpo_trainer", "OfflineRLAlgorithm"),
    # Domain Randomization
    "DomainRandomizationConfig": (
        f"{__name__}.domain_randomization",
        "DomainRandomizationConfig",
    ),
    "DomainRandomizer": (f"{__name__}.domain_randomization", "DomainRandomizer"),
    "PersonaGenerator": (f"{__name__}.domain_randomization", "PersonaGenerator"),
    "ScenarioGenerator": (f"{__name__}.domain_randomization", "ScenarioGenerator"),
    "UserPersona": (f"{__name__}.domain_randomization", "UserPersona"),
    # Sim-to-Real Transfer
    "SimToRealConfig": (f"{__name__}.sim_to_real", "SimToRealConfig"),
    "SimToRealTransfer": (f"{__name__}.sim_to_real", "SimToRealTransfer"),
    "UserBehaviorModel": (f"{__name__}.sim_to_real", "UserBehaviorModel"),
    "DomainAdaptationModule": (f"{__name__}.sim_to_real", "DomainAdaptationModule"),
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
    "ContinualLearningConfig",
    "ContinualLearningManager",
    "TrajectoryReplayBuffer",
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
    # Offline RL and Sim-to-Real
    "OFFLINE_RL_AVAILABLE",
    "BCQ_AVAILABLE",
    "BEAR_AVAILABLE",
    "DECISION_TRANSFORMER_AVAILABLE",
    "SIM_TO_REAL_AVAILABLE",
    "AUTO_RESEARCH_AVAILABLE",
    "QWEN35_08B_BASE_MODEL",
    "QWEN35_08B_CONFIG_SUFFIXES",
    "QWEN35_08B_DEFAULT_OUTPUT_DIR",
    "QWEN35_08B_LORA_TARGET_MODULES",
    "QWEN35_08B_POST_TRAINED_MODEL",
    "QWEN35_08B_STARTER_PROFILE_CHOICES",
    "QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS",
    "QWEN35_08B_SUPPORTED_VARIANTS",
    "QWEN35_08B_TASK_CHOICES",
    "Qwen35Config",
    "create_qwen3_5_agent_config",
    "create_qwen3_5_preview",
    "describe_qwen3_5_starter_profiles",
    "finetune_qwen3_5_0_8b",
    "get_qwen3_5_config",
    "get_qwen3_5_gspo_config",
    "get_qwen3_5_gspo_overrides",
    "get_qwen3_5_profile_description",
    "get_qwen3_5_profile_overrides",
    "get_qwen3_5_system_prompt",
    "load_qwen3_5_config_file",
    "run_qwen3_5_0_8b_config",
    "summarize_qwen3_5_config",
    "validate_qwen3_5_config",
    "write_qwen3_5_config_file",
    "build_serving_manifest",
    "export_merged_model_for_serving",
    "write_serving_manifest",
    "GEMMA4_31B_BASE_MODEL",
    "GEMMA4_31B_CONFIG_SUFFIXES",
    "GEMMA4_31B_DEFAULT_OUTPUT_DIR",
    "GEMMA4_31B_LORA_TARGET_MODULES",
    "GEMMA4_31B_STARTER_PROFILE_CHOICES",
    "GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS",
    "GEMMA4_31B_SUPPORTED_VARIANTS",
    "GEMMA4_31B_TASK_CHOICES",
    "Gemma4Config",
    "create_gemma4_31b_agent_config",
    "create_gemma4_31b_preview",
    "describe_gemma4_31b_starter_profiles",
    "finetune_gemma4_31b",
    "get_gemma4_31b_config",
    "get_gemma4_31b_gspo_config",
    "get_gemma4_31b_gspo_overrides",
    "get_gemma4_31b_profile_description",
    "get_gemma4_31b_profile_overrides",
    "get_gemma4_31b_system_prompt",
    "load_gemma4_31b_config_file",
    "run_gemma4_31b_config",
    "summarize_gemma4_31b_config",
    "validate_gemma4_31b_config",
    "write_gemma4_31b_config_file",
    "GLM5_1_BASE_MODEL",
    "GLM5_1_CONFIG_SUFFIXES",
    "GLM5_1_DEFAULT_OUTPUT_DIR",
    "GLM5_1_FP8_MODEL",
    "GLM5_1_LORA_TARGET_MODULES",
    "GLM5_1_STARTER_PROFILE_CHOICES",
    "GLM5_1_STARTER_PROFILE_DESCRIPTIONS",
    "GLM5_1_SUPPORTED_VARIANTS",
    "GLM5_1_TASK_CHOICES",
    "Glm51Config",
    "create_glm5_1_agent_config",
    "create_glm5_1_preview",
    "describe_glm5_1_starter_profiles",
    "finetune_glm5_1",
    "get_glm5_1_config",
    "get_glm5_1_gspo_config",
    "get_glm5_1_gspo_overrides",
    "get_glm5_1_profile_description",
    "get_glm5_1_profile_overrides",
    "get_glm5_1_serving_recommendations",
    "get_glm5_1_system_prompt",
    "load_glm5_1_config_file",
    "run_glm5_1_config",
    "summarize_glm5_1_config",
    "validate_glm5_1_config",
    "write_glm5_1_config_file",
]
