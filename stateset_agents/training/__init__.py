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
from types import ModuleType
from typing import Any

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


# Lightweight dependency checks.
#
# ``TRL_AVAILABLE`` is resolved lazily (see ``__getattr__``) because importing
# ``trl`` pulls in ``torch``, and importing this package must stay torch-free.
def _detect_trl() -> bool:  # pragma: no cover - depends on the environment
    try:
        import trl
    except ImportError:
        return False

    # `hasattr` only swallows AttributeError. trl's own lazy-module loader
    # (trl._lazy_module) wraps *any* failure while resolving an attribute
    # -- including a transitively optional, environment-dependent one like
    # an unusable/absent vllm integration -- in a bare RuntimeError, not
    # AttributeError. A plain `hasattr(trl, "GRPOConfig")` would let that
    # RuntimeError escape and crash this call instead of falling back to
    # False as intended.
    try:
        return hasattr(trl, "GRPOConfig")
    except Exception:  # noqa: BLE001 - trl's lazy-attr errors are not typed
        return False


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
    # Core training surface (lazy so that importing this package stays
    # torch-free; every one of these modules imports torch transitively).
    "TrainingConfig": (f"{__name__}.config", "TrainingConfig"),
    "TrainingProfile": (f"{__name__}.config", "TrainingProfile"),
    "get_config_for_task": (f"{__name__}.config", "get_config_for_task"),
    "ContinualLearningConfig": (
        f"{__name__}.continual_learning",
        "ContinualLearningConfig",
    ),
    "ContinualLearningManager": (
        f"{__name__}.continual_learning",
        "ContinualLearningManager",
    ),
    "TrajectoryReplayBuffer": (
        f"{__name__}.continual_learning",
        "TrajectoryReplayBuffer",
    ),
    "EvaluationConfig": (f"{__name__}.evaluation", "EvaluationConfig"),
    "evaluate_agent": (f"{__name__}.evaluation", "evaluate_agent"),
    "TrainingMode": (f"{__name__}.train", "TrainingMode"),
    "train": (f"{__name__}.train", "train"),
    "GRPOTrainer": (f"{__name__}.trainer", "GRPOTrainer"),
    "MultiTurnGRPOTrainer": (f"{__name__}.trainer", "MultiTurnGRPOTrainer"),
    "SingleTurnGRPOTrainer": (f"{__name__}.trainer", "SingleTurnGRPOTrainer"),
    # Auto-Research Loop
    "AutoResearchConfig": (f"{__name__}.auto_research.config", "AutoResearchConfig"),
    "AutoResearchLoop": (
        f"{__name__}.auto_research.experiment_loop",
        "AutoResearchLoop",
    ),
    "run_auto_research": (
        f"{__name__}.auto_research.experiment_loop",
        "run_auto_research",
    ),
    "ExperimentTracker": (
        f"{__name__}.auto_research.experiment_tracker",
        "ExperimentTracker",
    ),
    "ExperimentRecord": (
        f"{__name__}.auto_research.experiment_tracker",
        "ExperimentRecord",
    ),
    "CheckpointManager": (
        f"{__name__}.auto_research.checkpoint_manager",
        "CheckpointManager",
    ),
    "ExperimentProposer": (f"{__name__}.auto_research.proposer", "ExperimentProposer"),
    "RandomProposer": (f"{__name__}.auto_research.proposer", "RandomProposer"),
    "PerturbationProposer": (
        f"{__name__}.auto_research.proposer",
        "PerturbationProposer",
    ),
    "GridProposer": (f"{__name__}.auto_research.proposer", "GridProposer"),
    "BayesianProposer": (f"{__name__}.auto_research.proposer", "BayesianProposer"),
    "LLMProposer": (f"{__name__}.auto_research.llm_proposer", "LLMProposer"),
    "create_proposer": (f"{__name__}.auto_research.proposer", "create_proposer"),
    "create_auto_research_search_space": (
        f"{__name__}.auto_research.search_spaces",
        "create_auto_research_search_space",
    ),
    "create_quick_search_space": (
        f"{__name__}.auto_research.search_spaces",
        "create_quick_search_space",
    ),
    "get_auto_research_search_space": (
        f"{__name__}.auto_research.search_spaces",
        "get_auto_research_search_space",
    ),
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
    "QWEN35_08B_CONFIG_SUFFIXES": (
        f"{__name__}.qwen3_5_starter",
        "QWEN35_08B_CONFIG_SUFFIXES",
    ),
    "QWEN35_08B_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.qwen3_5_starter",
        "QWEN35_08B_DEFAULT_OUTPUT_DIR",
    ),
    "QWEN35_08B_LORA_TARGET_MODULES": (
        f"{__name__}.qwen3_5_starter",
        "QWEN35_08B_LORA_TARGET_MODULES",
    ),
    "QWEN35_08B_POST_TRAINED_MODEL": (
        f"{__name__}.qwen3_5_starter",
        "QWEN35_08B_POST_TRAINED_MODEL",
    ),
    "QWEN35_08B_STARTER_PROFILE_CHOICES": (
        f"{__name__}.qwen3_5_starter",
        "QWEN35_08B_STARTER_PROFILE_CHOICES",
    ),
    "QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.qwen3_5_starter",
        "QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "QWEN35_08B_SUPPORTED_VARIANTS": (
        f"{__name__}.qwen3_5_starter",
        "QWEN35_08B_SUPPORTED_VARIANTS",
    ),
    "QWEN35_08B_TASK_CHOICES": (
        f"{__name__}.qwen3_5_starter",
        "QWEN35_08B_TASK_CHOICES",
    ),
    "Qwen35Config": (f"{__name__}.qwen3_5_starter", "Qwen35Config"),
    "create_qwen3_5_agent_config": (
        f"{__name__}.qwen3_5_starter",
        "create_qwen3_5_agent_config",
    ),
    "create_qwen3_5_preview": (f"{__name__}.qwen3_5_starter", "create_qwen3_5_preview"),
    "describe_qwen3_5_starter_profiles": (
        f"{__name__}.qwen3_5_starter",
        "describe_qwen3_5_starter_profiles",
    ),
    "finetune_qwen3_5_0_8b": (f"{__name__}.qwen3_5_starter", "finetune_qwen3_5_0_8b"),
    "get_qwen3_5_config": (f"{__name__}.qwen3_5_starter", "get_qwen3_5_config"),
    "get_qwen3_5_gspo_config": (
        f"{__name__}.qwen3_5_starter",
        "get_qwen3_5_gspo_config",
    ),
    "get_qwen3_5_gspo_overrides": (
        f"{__name__}.qwen3_5_starter",
        "get_qwen3_5_gspo_overrides",
    ),
    "get_qwen3_5_profile_description": (
        f"{__name__}.qwen3_5_starter",
        "get_qwen3_5_profile_description",
    ),
    "get_qwen3_5_profile_overrides": (
        f"{__name__}.qwen3_5_starter",
        "get_qwen3_5_profile_overrides",
    ),
    "get_qwen3_5_system_prompt": (
        f"{__name__}.qwen3_5_starter",
        "get_qwen3_5_system_prompt",
    ),
    "load_qwen3_5_config_file": (
        f"{__name__}.qwen3_5_starter",
        "load_qwen3_5_config_file",
    ),
    "run_qwen3_5_0_8b_config": (
        f"{__name__}.qwen3_5_starter",
        "run_qwen3_5_0_8b_config",
    ),
    "summarize_qwen3_5_config": (
        f"{__name__}.qwen3_5_starter",
        "summarize_qwen3_5_config",
    ),
    "validate_qwen3_5_config": (
        f"{__name__}.qwen3_5_starter",
        "validate_qwen3_5_config",
    ),
    "write_qwen3_5_config_file": (
        f"{__name__}.qwen3_5_starter",
        "write_qwen3_5_config_file",
    ),
    # Kimi-K2.6 starter path
    "KIMI_K26_BASE_MODEL": (f"{__name__}.kimi_k2_6_starter", "KIMI_K26_BASE_MODEL"),
    "KIMI_K26_CONFIG_SUFFIXES": (
        f"{__name__}.kimi_k2_6_starter",
        "KIMI_K26_CONFIG_SUFFIXES",
    ),
    "KIMI_K26_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.kimi_k2_6_starter",
        "KIMI_K26_DEFAULT_OUTPUT_DIR",
    ),
    "KIMI_K26_LORA_TARGET_MODULES": (
        f"{__name__}.kimi_k2_6_starter",
        "KIMI_K26_LORA_TARGET_MODULES",
    ),
    "KIMI_K26_STARTER_PROFILE_CHOICES": (
        f"{__name__}.kimi_k2_6_starter",
        "KIMI_K26_STARTER_PROFILE_CHOICES",
    ),
    "KIMI_K26_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.kimi_k2_6_starter",
        "KIMI_K26_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "KIMI_K26_SUPPORTED_VARIANTS": (
        f"{__name__}.kimi_k2_6_starter",
        "KIMI_K26_SUPPORTED_VARIANTS",
    ),
    "KIMI_K26_TASK_CHOICES": (f"{__name__}.kimi_k2_6_starter", "KIMI_K26_TASK_CHOICES"),
    "KimiK26Config": (f"{__name__}.kimi_k2_6_starter", "KimiK26Config"),
    "create_kimi_k2_6_agent_config": (
        f"{__name__}.kimi_k2_6_starter",
        "create_kimi_k2_6_agent_config",
    ),
    "create_kimi_k2_6_preview": (
        f"{__name__}.kimi_k2_6_starter",
        "create_kimi_k2_6_preview",
    ),
    "describe_kimi_k2_6_starter_profiles": (
        f"{__name__}.kimi_k2_6_starter",
        "describe_kimi_k2_6_starter_profiles",
    ),
    "finetune_kimi_k2_6": (f"{__name__}.kimi_k2_6_starter", "finetune_kimi_k2_6"),
    "get_kimi_k2_6_config": (f"{__name__}.kimi_k2_6_starter", "get_kimi_k2_6_config"),
    "get_kimi_k2_6_gspo_config": (
        f"{__name__}.kimi_k2_6_starter",
        "get_kimi_k2_6_gspo_config",
    ),
    "get_kimi_k2_6_gspo_overrides": (
        f"{__name__}.kimi_k2_6_starter",
        "get_kimi_k2_6_gspo_overrides",
    ),
    "get_kimi_k2_6_profile_description": (
        f"{__name__}.kimi_k2_6_starter",
        "get_kimi_k2_6_profile_description",
    ),
    "get_kimi_k2_6_profile_overrides": (
        f"{__name__}.kimi_k2_6_starter",
        "get_kimi_k2_6_profile_overrides",
    ),
    "get_kimi_k2_6_system_prompt": (
        f"{__name__}.kimi_k2_6_starter",
        "get_kimi_k2_6_system_prompt",
    ),
    "load_kimi_k2_6_config_file": (
        f"{__name__}.kimi_k2_6_starter",
        "load_kimi_k2_6_config_file",
    ),
    "run_kimi_k2_6_config": (f"{__name__}.kimi_k2_6_starter", "run_kimi_k2_6_config"),
    "summarize_kimi_k2_6_config": (
        f"{__name__}.kimi_k2_6_starter",
        "summarize_kimi_k2_6_config",
    ),
    "validate_kimi_k2_6_config": (
        f"{__name__}.kimi_k2_6_starter",
        "validate_kimi_k2_6_config",
    ),
    "write_kimi_k2_6_config_file": (
        f"{__name__}.kimi_k2_6_starter",
        "write_kimi_k2_6_config_file",
    ),
    # Kimi-K3 starter path
    "KIMI_K3_BASE_MODEL": (f"{__name__}.kimi_k3_starter", "KIMI_K3_BASE_MODEL"),
    "KIMI_K3_CONFIG_SUFFIXES": (
        f"{__name__}.kimi_k3_starter",
        "KIMI_K3_CONFIG_SUFFIXES",
    ),
    "KIMI_K3_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.kimi_k3_starter",
        "KIMI_K3_DEFAULT_OUTPUT_DIR",
    ),
    "KIMI_K3_LORA_TARGET_MODULES": (
        f"{__name__}.kimi_k3_starter",
        "KIMI_K3_LORA_TARGET_MODULES",
    ),
    "KIMI_K3_STARTER_PROFILE_CHOICES": (
        f"{__name__}.kimi_k3_starter",
        "KIMI_K3_STARTER_PROFILE_CHOICES",
    ),
    "KIMI_K3_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.kimi_k3_starter",
        "KIMI_K3_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "KIMI_K3_SUPPORTED_VARIANTS": (
        f"{__name__}.kimi_k3_starter",
        "KIMI_K3_SUPPORTED_VARIANTS",
    ),
    "KIMI_K3_TASK_CHOICES": (f"{__name__}.kimi_k3_starter", "KIMI_K3_TASK_CHOICES"),
    "KimiK3Config": (f"{__name__}.kimi_k3_starter", "KimiK3Config"),
    "create_kimi_k3_agent_config": (
        f"{__name__}.kimi_k3_starter",
        "create_kimi_k3_agent_config",
    ),
    "create_kimi_k3_preview": (f"{__name__}.kimi_k3_starter", "create_kimi_k3_preview"),
    "describe_kimi_k3_starter_profiles": (
        f"{__name__}.kimi_k3_starter",
        "describe_kimi_k3_starter_profiles",
    ),
    "finetune_kimi_k3": (f"{__name__}.kimi_k3_starter", "finetune_kimi_k3"),
    "get_kimi_k3_config": (f"{__name__}.kimi_k3_starter", "get_kimi_k3_config"),
    "get_kimi_k3_gspo_config": (
        f"{__name__}.kimi_k3_starter",
        "get_kimi_k3_gspo_config",
    ),
    "get_kimi_k3_gspo_overrides": (
        f"{__name__}.kimi_k3_starter",
        "get_kimi_k3_gspo_overrides",
    ),
    "get_kimi_k3_profile_description": (
        f"{__name__}.kimi_k3_starter",
        "get_kimi_k3_profile_description",
    ),
    "get_kimi_k3_profile_overrides": (
        f"{__name__}.kimi_k3_starter",
        "get_kimi_k3_profile_overrides",
    ),
    "get_kimi_k3_system_prompt": (
        f"{__name__}.kimi_k3_starter",
        "get_kimi_k3_system_prompt",
    ),
    "load_kimi_k3_config_file": (
        f"{__name__}.kimi_k3_starter",
        "load_kimi_k3_config_file",
    ),
    "run_kimi_k3_config": (f"{__name__}.kimi_k3_starter", "run_kimi_k3_config"),
    "summarize_kimi_k3_config": (
        f"{__name__}.kimi_k3_starter",
        "summarize_kimi_k3_config",
    ),
    "validate_kimi_k3_config": (
        f"{__name__}.kimi_k3_starter",
        "validate_kimi_k3_config",
    ),
    "write_kimi_k3_config_file": (
        f"{__name__}.kimi_k3_starter",
        "write_kimi_k3_config_file",
    ),
    # Muse Glimmer starter path
    "MUSE_GLIMMER_BASE_MODEL": (
        f"{__name__}.muse_glimmer_starter",
        "MUSE_GLIMMER_BASE_MODEL",
    ),
    "MUSE_GLIMMER_CONFIG_SUFFIXES": (
        f"{__name__}.muse_glimmer_starter",
        "MUSE_GLIMMER_CONFIG_SUFFIXES",
    ),
    "MUSE_GLIMMER_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.muse_glimmer_starter",
        "MUSE_GLIMMER_DEFAULT_OUTPUT_DIR",
    ),
    "MUSE_GLIMMER_LORA_TARGET_MODULES": (
        f"{__name__}.muse_glimmer_starter",
        "MUSE_GLIMMER_LORA_TARGET_MODULES",
    ),
    "MUSE_GLIMMER_STARTER_PROFILE_CHOICES": (
        f"{__name__}.muse_glimmer_starter",
        "MUSE_GLIMMER_STARTER_PROFILE_CHOICES",
    ),
    "MUSE_GLIMMER_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.muse_glimmer_starter",
        "MUSE_GLIMMER_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "MUSE_GLIMMER_SUPPORTED_VARIANTS": (
        f"{__name__}.muse_glimmer_starter",
        "MUSE_GLIMMER_SUPPORTED_VARIANTS",
    ),
    "MUSE_GLIMMER_TASK_CHOICES": (
        f"{__name__}.muse_glimmer_starter",
        "MUSE_GLIMMER_TASK_CHOICES",
    ),
    "MuseGlimmerConfig": (f"{__name__}.muse_glimmer_starter", "MuseGlimmerConfig"),
    "create_muse_glimmer_agent_config": (
        f"{__name__}.muse_glimmer_starter",
        "create_muse_glimmer_agent_config",
    ),
    "create_muse_glimmer_preview": (
        f"{__name__}.muse_glimmer_starter",
        "create_muse_glimmer_preview",
    ),
    "describe_muse_glimmer_starter_profiles": (
        f"{__name__}.muse_glimmer_starter",
        "describe_muse_glimmer_starter_profiles",
    ),
    "finetune_muse_glimmer": (
        f"{__name__}.muse_glimmer_starter",
        "finetune_muse_glimmer",
    ),
    "get_muse_glimmer_config": (
        f"{__name__}.muse_glimmer_starter",
        "get_muse_glimmer_config",
    ),
    "get_muse_glimmer_gspo_config": (
        f"{__name__}.muse_glimmer_starter",
        "get_muse_glimmer_gspo_config",
    ),
    "get_muse_glimmer_gspo_overrides": (
        f"{__name__}.muse_glimmer_starter",
        "get_muse_glimmer_gspo_overrides",
    ),
    "get_muse_glimmer_profile_description": (
        f"{__name__}.muse_glimmer_starter",
        "get_muse_glimmer_profile_description",
    ),
    "get_muse_glimmer_profile_overrides": (
        f"{__name__}.muse_glimmer_starter",
        "get_muse_glimmer_profile_overrides",
    ),
    "get_muse_glimmer_system_prompt": (
        f"{__name__}.muse_glimmer_starter",
        "get_muse_glimmer_system_prompt",
    ),
    "load_muse_glimmer_config_file": (
        f"{__name__}.muse_glimmer_starter",
        "load_muse_glimmer_config_file",
    ),
    "run_muse_glimmer_config": (
        f"{__name__}.muse_glimmer_starter",
        "run_muse_glimmer_config",
    ),
    "summarize_muse_glimmer_config": (
        f"{__name__}.muse_glimmer_starter",
        "summarize_muse_glimmer_config",
    ),
    "validate_muse_glimmer_config": (
        f"{__name__}.muse_glimmer_starter",
        "validate_muse_glimmer_config",
    ),
    "write_muse_glimmer_config_file": (
        f"{__name__}.muse_glimmer_starter",
        "write_muse_glimmer_config_file",
    ),
    # Nemotron 3.5 starter path
    "NEMOTRON_3_5_BASE_MODEL": (
        f"{__name__}.nemotron_3_5_starter",
        "NEMOTRON_3_5_BASE_MODEL",
    ),
    "NEMOTRON_3_5_CONFIG_SUFFIXES": (
        f"{__name__}.nemotron_3_5_starter",
        "NEMOTRON_3_5_CONFIG_SUFFIXES",
    ),
    "NEMOTRON_3_5_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.nemotron_3_5_starter",
        "NEMOTRON_3_5_DEFAULT_OUTPUT_DIR",
    ),
    "NEMOTRON_3_5_LORA_TARGET_MODULES": (
        f"{__name__}.nemotron_3_5_starter",
        "NEMOTRON_3_5_LORA_TARGET_MODULES",
    ),
    "NEMOTRON_3_5_STARTER_PROFILE_CHOICES": (
        f"{__name__}.nemotron_3_5_starter",
        "NEMOTRON_3_5_STARTER_PROFILE_CHOICES",
    ),
    "NEMOTRON_3_5_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.nemotron_3_5_starter",
        "NEMOTRON_3_5_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "NEMOTRON_3_5_SUPPORTED_VARIANTS": (
        f"{__name__}.nemotron_3_5_starter",
        "NEMOTRON_3_5_SUPPORTED_VARIANTS",
    ),
    "NEMOTRON_3_5_TASK_CHOICES": (
        f"{__name__}.nemotron_3_5_starter",
        "NEMOTRON_3_5_TASK_CHOICES",
    ),
    "Nemotron35Config": (f"{__name__}.nemotron_3_5_starter", "Nemotron35Config"),
    "create_nemotron_3_5_agent_config": (
        f"{__name__}.nemotron_3_5_starter",
        "create_nemotron_3_5_agent_config",
    ),
    "create_nemotron_3_5_preview": (
        f"{__name__}.nemotron_3_5_starter",
        "create_nemotron_3_5_preview",
    ),
    "describe_nemotron_3_5_starter_profiles": (
        f"{__name__}.nemotron_3_5_starter",
        "describe_nemotron_3_5_starter_profiles",
    ),
    "finetune_nemotron_3_5": (
        f"{__name__}.nemotron_3_5_starter",
        "finetune_nemotron_3_5",
    ),
    "get_nemotron_3_5_config": (
        f"{__name__}.nemotron_3_5_starter",
        "get_nemotron_3_5_config",
    ),
    "get_nemotron_3_5_gspo_config": (
        f"{__name__}.nemotron_3_5_starter",
        "get_nemotron_3_5_gspo_config",
    ),
    "get_nemotron_3_5_gspo_overrides": (
        f"{__name__}.nemotron_3_5_starter",
        "get_nemotron_3_5_gspo_overrides",
    ),
    "get_nemotron_3_5_profile_description": (
        f"{__name__}.nemotron_3_5_starter",
        "get_nemotron_3_5_profile_description",
    ),
    "get_nemotron_3_5_profile_overrides": (
        f"{__name__}.nemotron_3_5_starter",
        "get_nemotron_3_5_profile_overrides",
    ),
    "get_nemotron_3_5_system_prompt": (
        f"{__name__}.nemotron_3_5_starter",
        "get_nemotron_3_5_system_prompt",
    ),
    "load_nemotron_3_5_config_file": (
        f"{__name__}.nemotron_3_5_starter",
        "load_nemotron_3_5_config_file",
    ),
    "run_nemotron_3_5_config": (
        f"{__name__}.nemotron_3_5_starter",
        "run_nemotron_3_5_config",
    ),
    "summarize_nemotron_3_5_config": (
        f"{__name__}.nemotron_3_5_starter",
        "summarize_nemotron_3_5_config",
    ),
    "validate_nemotron_3_5_config": (
        f"{__name__}.nemotron_3_5_starter",
        "validate_nemotron_3_5_config",
    ),
    "write_nemotron_3_5_config_file": (
        f"{__name__}.nemotron_3_5_starter",
        "write_nemotron_3_5_config_file",
    ),
    # Qwen3.8 27B starter path
    "QWEN38_27B_BASE_MODEL": (
        f"{__name__}.qwen3_8_starter",
        "QWEN38_27B_BASE_MODEL",
    ),
    "QWEN38_27B_CONFIG_SUFFIXES": (
        f"{__name__}.qwen3_8_starter",
        "QWEN38_27B_CONFIG_SUFFIXES",
    ),
    "QWEN38_27B_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.qwen3_8_starter",
        "QWEN38_27B_DEFAULT_OUTPUT_DIR",
    ),
    "QWEN38_27B_LORA_TARGET_MODULES": (
        f"{__name__}.qwen3_8_starter",
        "QWEN38_27B_LORA_TARGET_MODULES",
    ),
    "QWEN38_27B_STARTER_PROFILE_CHOICES": (
        f"{__name__}.qwen3_8_starter",
        "QWEN38_27B_STARTER_PROFILE_CHOICES",
    ),
    "QWEN38_27B_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.qwen3_8_starter",
        "QWEN38_27B_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "QWEN38_27B_SUPPORTED_VARIANTS": (
        f"{__name__}.qwen3_8_starter",
        "QWEN38_27B_SUPPORTED_VARIANTS",
    ),
    "QWEN38_27B_TASK_CHOICES": (
        f"{__name__}.qwen3_8_starter",
        "QWEN38_27B_TASK_CHOICES",
    ),
    "Qwen38Config": (f"{__name__}.qwen3_8_starter", "Qwen38Config"),
    "create_qwen3_8_agent_config": (
        f"{__name__}.qwen3_8_starter",
        "create_qwen3_8_agent_config",
    ),
    "create_qwen3_8_preview": (
        f"{__name__}.qwen3_8_starter",
        "create_qwen3_8_preview",
    ),
    "describe_qwen3_8_starter_profiles": (
        f"{__name__}.qwen3_8_starter",
        "describe_qwen3_8_starter_profiles",
    ),
    "finetune_qwen3_8": (
        f"{__name__}.qwen3_8_starter",
        "finetune_qwen3_8",
    ),
    "get_qwen3_8_config": (
        f"{__name__}.qwen3_8_starter",
        "get_qwen3_8_config",
    ),
    "get_qwen3_8_gspo_config": (
        f"{__name__}.qwen3_8_starter",
        "get_qwen3_8_gspo_config",
    ),
    "get_qwen3_8_gspo_overrides": (
        f"{__name__}.qwen3_8_starter",
        "get_qwen3_8_gspo_overrides",
    ),
    "get_qwen3_8_profile_description": (
        f"{__name__}.qwen3_8_starter",
        "get_qwen3_8_profile_description",
    ),
    "get_qwen3_8_profile_overrides": (
        f"{__name__}.qwen3_8_starter",
        "get_qwen3_8_profile_overrides",
    ),
    "get_qwen3_8_system_prompt": (
        f"{__name__}.qwen3_8_starter",
        "get_qwen3_8_system_prompt",
    ),
    "load_qwen3_8_config_file": (
        f"{__name__}.qwen3_8_starter",
        "load_qwen3_8_config_file",
    ),
    "run_qwen3_8_config": (
        f"{__name__}.qwen3_8_starter",
        "run_qwen3_8_config",
    ),
    "summarize_qwen3_8_config": (
        f"{__name__}.qwen3_8_starter",
        "summarize_qwen3_8_config",
    ),
    "validate_qwen3_8_config": (
        f"{__name__}.qwen3_8_starter",
        "validate_qwen3_8_config",
    ),
    "write_qwen3_8_config_file": (
        f"{__name__}.qwen3_8_starter",
        "write_qwen3_8_config_file",
    ),
    # Qwen3 Coder starter path
    "QWEN3_CODER_BASE_MODEL": (
        f"{__name__}.qwen3_coder_starter",
        "QWEN3_CODER_BASE_MODEL",
    ),
    "QWEN3_CODER_CONFIG_SUFFIXES": (
        f"{__name__}.qwen3_coder_starter",
        "QWEN3_CODER_CONFIG_SUFFIXES",
    ),
    "QWEN3_CODER_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.qwen3_coder_starter",
        "QWEN3_CODER_DEFAULT_OUTPUT_DIR",
    ),
    "QWEN3_CODER_LORA_TARGET_MODULES": (
        f"{__name__}.qwen3_coder_starter",
        "QWEN3_CODER_LORA_TARGET_MODULES",
    ),
    "QWEN3_CODER_STARTER_PROFILE_CHOICES": (
        f"{__name__}.qwen3_coder_starter",
        "QWEN3_CODER_STARTER_PROFILE_CHOICES",
    ),
    "QWEN3_CODER_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.qwen3_coder_starter",
        "QWEN3_CODER_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "QWEN3_CODER_SUPPORTED_VARIANTS": (
        f"{__name__}.qwen3_coder_starter",
        "QWEN3_CODER_SUPPORTED_VARIANTS",
    ),
    "QWEN3_CODER_TASK_CHOICES": (
        f"{__name__}.qwen3_coder_starter",
        "QWEN3_CODER_TASK_CHOICES",
    ),
    "Qwen3CoderConfig": (f"{__name__}.qwen3_coder_starter", "Qwen3CoderConfig"),
    "create_qwen3_coder_agent_config": (
        f"{__name__}.qwen3_coder_starter",
        "create_qwen3_coder_agent_config",
    ),
    "create_qwen3_coder_preview": (
        f"{__name__}.qwen3_coder_starter",
        "create_qwen3_coder_preview",
    ),
    "describe_qwen3_coder_starter_profiles": (
        f"{__name__}.qwen3_coder_starter",
        "describe_qwen3_coder_starter_profiles",
    ),
    "finetune_qwen3_coder": (f"{__name__}.qwen3_coder_starter", "finetune_qwen3_coder"),
    "get_qwen3_coder_config": (
        f"{__name__}.qwen3_coder_starter",
        "get_qwen3_coder_config",
    ),
    "get_qwen3_coder_gspo_config": (
        f"{__name__}.qwen3_coder_starter",
        "get_qwen3_coder_gspo_config",
    ),
    "get_qwen3_coder_gspo_overrides": (
        f"{__name__}.qwen3_coder_starter",
        "get_qwen3_coder_gspo_overrides",
    ),
    "get_qwen3_coder_profile_description": (
        f"{__name__}.qwen3_coder_starter",
        "get_qwen3_coder_profile_description",
    ),
    "get_qwen3_coder_profile_overrides": (
        f"{__name__}.qwen3_coder_starter",
        "get_qwen3_coder_profile_overrides",
    ),
    "get_qwen3_coder_system_prompt": (
        f"{__name__}.qwen3_coder_starter",
        "get_qwen3_coder_system_prompt",
    ),
    "load_qwen3_coder_config_file": (
        f"{__name__}.qwen3_coder_starter",
        "load_qwen3_coder_config_file",
    ),
    "run_qwen3_coder_config": (
        f"{__name__}.qwen3_coder_starter",
        "run_qwen3_coder_config",
    ),
    "summarize_qwen3_coder_config": (
        f"{__name__}.qwen3_coder_starter",
        "summarize_qwen3_coder_config",
    ),
    "validate_qwen3_coder_config": (
        f"{__name__}.qwen3_coder_starter",
        "validate_qwen3_coder_config",
    ),
    "write_qwen3_coder_config_file": (
        f"{__name__}.qwen3_coder_starter",
        "write_qwen3_coder_config_file",
    ),
    # gpt-oss starter path
    "GPT_OSS_120B_MODEL": (f"{__name__}.gpt_oss_starter", "GPT_OSS_120B_MODEL"),
    "GPT_OSS_BASE_MODEL": (f"{__name__}.gpt_oss_starter", "GPT_OSS_BASE_MODEL"),
    "GPT_OSS_CONFIG_SUFFIXES": (
        f"{__name__}.gpt_oss_starter",
        "GPT_OSS_CONFIG_SUFFIXES",
    ),
    "GPT_OSS_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.gpt_oss_starter",
        "GPT_OSS_DEFAULT_OUTPUT_DIR",
    ),
    "GPT_OSS_LORA_TARGET_MODULES": (
        f"{__name__}.gpt_oss_starter",
        "GPT_OSS_LORA_TARGET_MODULES",
    ),
    "GPT_OSS_STARTER_PROFILE_CHOICES": (
        f"{__name__}.gpt_oss_starter",
        "GPT_OSS_STARTER_PROFILE_CHOICES",
    ),
    "GPT_OSS_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.gpt_oss_starter",
        "GPT_OSS_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "GPT_OSS_SUPPORTED_VARIANTS": (
        f"{__name__}.gpt_oss_starter",
        "GPT_OSS_SUPPORTED_VARIANTS",
    ),
    "GPT_OSS_TASK_CHOICES": (f"{__name__}.gpt_oss_starter", "GPT_OSS_TASK_CHOICES"),
    "GptOssConfig": (f"{__name__}.gpt_oss_starter", "GptOssConfig"),
    "create_gpt_oss_agent_config": (
        f"{__name__}.gpt_oss_starter",
        "create_gpt_oss_agent_config",
    ),
    "create_gpt_oss_preview": (f"{__name__}.gpt_oss_starter", "create_gpt_oss_preview"),
    "describe_gpt_oss_starter_profiles": (
        f"{__name__}.gpt_oss_starter",
        "describe_gpt_oss_starter_profiles",
    ),
    "finetune_gpt_oss": (f"{__name__}.gpt_oss_starter", "finetune_gpt_oss"),
    "get_gpt_oss_config": (f"{__name__}.gpt_oss_starter", "get_gpt_oss_config"),
    "get_gpt_oss_gspo_config": (
        f"{__name__}.gpt_oss_starter",
        "get_gpt_oss_gspo_config",
    ),
    "get_gpt_oss_gspo_overrides": (
        f"{__name__}.gpt_oss_starter",
        "get_gpt_oss_gspo_overrides",
    ),
    "get_gpt_oss_profile_description": (
        f"{__name__}.gpt_oss_starter",
        "get_gpt_oss_profile_description",
    ),
    "get_gpt_oss_profile_overrides": (
        f"{__name__}.gpt_oss_starter",
        "get_gpt_oss_profile_overrides",
    ),
    "get_gpt_oss_system_prompt": (
        f"{__name__}.gpt_oss_starter",
        "get_gpt_oss_system_prompt",
    ),
    "load_gpt_oss_config_file": (
        f"{__name__}.gpt_oss_starter",
        "load_gpt_oss_config_file",
    ),
    "run_gpt_oss_config": (f"{__name__}.gpt_oss_starter", "run_gpt_oss_config"),
    "summarize_gpt_oss_config": (
        f"{__name__}.gpt_oss_starter",
        "summarize_gpt_oss_config",
    ),
    "validate_gpt_oss_config": (
        f"{__name__}.gpt_oss_starter",
        "validate_gpt_oss_config",
    ),
    "write_gpt_oss_config_file": (
        f"{__name__}.gpt_oss_starter",
        "write_gpt_oss_config_file",
    ),
    # DeepSeek V4 Flash starter path
    "DEEPSEEK_V4_FLASH_BASE_MODEL": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_FLASH_BASE_MODEL",
    ),
    "DEEPSEEK_V4_BASE_MODEL": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_BASE_MODEL",
    ),
    "DEEPSEEK_V4_CONFIG_SUFFIXES": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_CONFIG_SUFFIXES",
    ),
    "DEEPSEEK_V4_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_DEFAULT_OUTPUT_DIR",
    ),
    "DEEPSEEK_V4_LORA_TARGET_MODULES": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_LORA_TARGET_MODULES",
    ),
    "DEEPSEEK_V4_STARTER_PROFILE_CHOICES": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_STARTER_PROFILE_CHOICES",
    ),
    "DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "DEEPSEEK_V4_SUPPORTED_VARIANTS": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_SUPPORTED_VARIANTS",
    ),
    "DEEPSEEK_V4_TASK_CHOICES": (
        f"{__name__}.deepseek_v4_starter",
        "DEEPSEEK_V4_TASK_CHOICES",
    ),
    "DeepseekV4Config": (f"{__name__}.deepseek_v4_starter", "DeepseekV4Config"),
    "create_deepseek_v4_agent_config": (
        f"{__name__}.deepseek_v4_starter",
        "create_deepseek_v4_agent_config",
    ),
    "create_deepseek_v4_preview": (
        f"{__name__}.deepseek_v4_starter",
        "create_deepseek_v4_preview",
    ),
    "describe_deepseek_v4_starter_profiles": (
        f"{__name__}.deepseek_v4_starter",
        "describe_deepseek_v4_starter_profiles",
    ),
    "finetune_deepseek_v4": (f"{__name__}.deepseek_v4_starter", "finetune_deepseek_v4"),
    "get_deepseek_v4_config": (
        f"{__name__}.deepseek_v4_starter",
        "get_deepseek_v4_config",
    ),
    "get_deepseek_v4_gspo_config": (
        f"{__name__}.deepseek_v4_starter",
        "get_deepseek_v4_gspo_config",
    ),
    "get_deepseek_v4_gspo_overrides": (
        f"{__name__}.deepseek_v4_starter",
        "get_deepseek_v4_gspo_overrides",
    ),
    "get_deepseek_v4_profile_description": (
        f"{__name__}.deepseek_v4_starter",
        "get_deepseek_v4_profile_description",
    ),
    "get_deepseek_v4_profile_overrides": (
        f"{__name__}.deepseek_v4_starter",
        "get_deepseek_v4_profile_overrides",
    ),
    "get_deepseek_v4_system_prompt": (
        f"{__name__}.deepseek_v4_starter",
        "get_deepseek_v4_system_prompt",
    ),
    "load_deepseek_v4_config_file": (
        f"{__name__}.deepseek_v4_starter",
        "load_deepseek_v4_config_file",
    ),
    "run_deepseek_v4_config": (
        f"{__name__}.deepseek_v4_starter",
        "run_deepseek_v4_config",
    ),
    "summarize_deepseek_v4_config": (
        f"{__name__}.deepseek_v4_starter",
        "summarize_deepseek_v4_config",
    ),
    "validate_deepseek_v4_config": (
        f"{__name__}.deepseek_v4_starter",
        "validate_deepseek_v4_config",
    ),
    "write_deepseek_v4_config_file": (
        f"{__name__}.deepseek_v4_starter",
        "write_deepseek_v4_config_file",
    ),
    # Serving artifacts
    "build_serving_manifest": (
        f"{__name__}.serving_artifacts",
        "build_serving_manifest",
    ),
    "export_merged_model_for_serving": (
        f"{__name__}.serving_artifacts",
        "export_merged_model_for_serving",
    ),
    "write_serving_manifest": (
        f"{__name__}.serving_artifacts",
        "write_serving_manifest",
    ),
    # Gemma 4 31B starter path
    "GEMMA4_31B_BASE_MODEL": (f"{__name__}.gemma4_starter", "GEMMA4_31B_BASE_MODEL"),
    "GEMMA4_31B_CONFIG_SUFFIXES": (
        f"{__name__}.gemma4_starter",
        "GEMMA4_31B_CONFIG_SUFFIXES",
    ),
    "GEMMA4_31B_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.gemma4_starter",
        "GEMMA4_31B_DEFAULT_OUTPUT_DIR",
    ),
    "GEMMA4_31B_LORA_TARGET_MODULES": (
        f"{__name__}.gemma4_starter",
        "GEMMA4_31B_LORA_TARGET_MODULES",
    ),
    "GEMMA4_31B_STARTER_PROFILE_CHOICES": (
        f"{__name__}.gemma4_starter",
        "GEMMA4_31B_STARTER_PROFILE_CHOICES",
    ),
    "GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.gemma4_starter",
        "GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "GEMMA4_31B_SUPPORTED_VARIANTS": (
        f"{__name__}.gemma4_starter",
        "GEMMA4_31B_SUPPORTED_VARIANTS",
    ),
    "GEMMA4_31B_TASK_CHOICES": (
        f"{__name__}.gemma4_starter",
        "GEMMA4_31B_TASK_CHOICES",
    ),
    "Gemma4Config": (f"{__name__}.gemma4_starter", "Gemma4Config"),
    "create_gemma4_31b_agent_config": (
        f"{__name__}.gemma4_starter",
        "create_gemma4_31b_agent_config",
    ),
    "create_gemma4_31b_preview": (
        f"{__name__}.gemma4_starter",
        "create_gemma4_31b_preview",
    ),
    "describe_gemma4_31b_starter_profiles": (
        f"{__name__}.gemma4_starter",
        "describe_gemma4_31b_starter_profiles",
    ),
    "finetune_gemma4_31b": (f"{__name__}.gemma4_starter", "finetune_gemma4_31b"),
    "get_gemma4_31b_config": (f"{__name__}.gemma4_starter", "get_gemma4_31b_config"),
    "get_gemma4_31b_gspo_config": (
        f"{__name__}.gemma4_starter",
        "get_gemma4_31b_gspo_config",
    ),
    "get_gemma4_31b_gspo_overrides": (
        f"{__name__}.gemma4_starter",
        "get_gemma4_31b_gspo_overrides",
    ),
    "get_gemma4_31b_profile_description": (
        f"{__name__}.gemma4_starter",
        "get_gemma4_31b_profile_description",
    ),
    "get_gemma4_31b_profile_overrides": (
        f"{__name__}.gemma4_starter",
        "get_gemma4_31b_profile_overrides",
    ),
    "get_gemma4_31b_system_prompt": (
        f"{__name__}.gemma4_starter",
        "get_gemma4_31b_system_prompt",
    ),
    "load_gemma4_31b_config_file": (
        f"{__name__}.gemma4_starter",
        "load_gemma4_31b_config_file",
    ),
    "run_gemma4_31b_config": (f"{__name__}.gemma4_starter", "run_gemma4_31b_config"),
    "summarize_gemma4_31b_config": (
        f"{__name__}.gemma4_starter",
        "summarize_gemma4_31b_config",
    ),
    "validate_gemma4_31b_config": (
        f"{__name__}.gemma4_starter",
        "validate_gemma4_31b_config",
    ),
    "write_gemma4_31b_config_file": (
        f"{__name__}.gemma4_starter",
        "write_gemma4_31b_config_file",
    ),
    # GLM 5.1 starter path
    "GLM5_1_BASE_MODEL": (f"{__name__}.glm5_1_starter", "GLM5_1_BASE_MODEL"),
    "GLM5_1_CONFIG_SUFFIXES": (f"{__name__}.glm5_1_starter", "GLM5_1_CONFIG_SUFFIXES"),
    "GLM5_1_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.glm5_1_starter",
        "GLM5_1_DEFAULT_OUTPUT_DIR",
    ),
    "GLM5_1_FP8_MODEL": (f"{__name__}.glm5_1_starter", "GLM5_1_FP8_MODEL"),
    "GLM5_1_LORA_TARGET_MODULES": (
        f"{__name__}.glm5_1_starter",
        "GLM5_1_LORA_TARGET_MODULES",
    ),
    "GLM5_1_STARTER_PROFILE_CHOICES": (
        f"{__name__}.glm5_1_starter",
        "GLM5_1_STARTER_PROFILE_CHOICES",
    ),
    "GLM5_1_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.glm5_1_starter",
        "GLM5_1_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "GLM5_1_SUPPORTED_VARIANTS": (
        f"{__name__}.glm5_1_starter",
        "GLM5_1_SUPPORTED_VARIANTS",
    ),
    "GLM5_1_TASK_CHOICES": (f"{__name__}.glm5_1_starter", "GLM5_1_TASK_CHOICES"),
    "Glm51Config": (f"{__name__}.glm5_1_starter", "Glm51Config"),
    "create_glm5_1_agent_config": (
        f"{__name__}.glm5_1_starter",
        "create_glm5_1_agent_config",
    ),
    "create_glm5_1_preview": (f"{__name__}.glm5_1_starter", "create_glm5_1_preview"),
    "describe_glm5_1_starter_profiles": (
        f"{__name__}.glm5_1_starter",
        "describe_glm5_1_starter_profiles",
    ),
    "finetune_glm5_1": (f"{__name__}.glm5_1_starter", "finetune_glm5_1"),
    "get_glm5_1_config": (f"{__name__}.glm5_1_starter", "get_glm5_1_config"),
    "get_glm5_1_gspo_config": (f"{__name__}.glm5_1_starter", "get_glm5_1_gspo_config"),
    "get_glm5_1_gspo_overrides": (
        f"{__name__}.glm5_1_starter",
        "get_glm5_1_gspo_overrides",
    ),
    "get_glm5_1_profile_description": (
        f"{__name__}.glm5_1_starter",
        "get_glm5_1_profile_description",
    ),
    "get_glm5_1_profile_overrides": (
        f"{__name__}.glm5_1_starter",
        "get_glm5_1_profile_overrides",
    ),
    "get_glm5_1_serving_recommendations": (
        f"{__name__}.glm5_1_starter",
        "get_glm5_1_serving_recommendations",
    ),
    "get_glm5_1_system_prompt": (
        f"{__name__}.glm5_1_starter",
        "get_glm5_1_system_prompt",
    ),
    "load_glm5_1_config_file": (
        f"{__name__}.glm5_1_starter",
        "load_glm5_1_config_file",
    ),
    "run_glm5_1_config": (f"{__name__}.glm5_1_starter", "run_glm5_1_config"),
    "summarize_glm5_1_config": (
        f"{__name__}.glm5_1_starter",
        "summarize_glm5_1_config",
    ),
    "validate_glm5_1_config": (f"{__name__}.glm5_1_starter", "validate_glm5_1_config"),
    "write_glm5_1_config_file": (
        f"{__name__}.glm5_1_starter",
        "write_glm5_1_config_file",
    ),
    # GLM 5.2 starter path
    "GLM5_2_BASE_MODEL": (f"{__name__}.glm5_2_starter", "GLM5_2_BASE_MODEL"),
    "GLM5_2_CONFIG_SUFFIXES": (f"{__name__}.glm5_2_starter", "GLM5_2_CONFIG_SUFFIXES"),
    "GLM5_2_DEFAULT_OUTPUT_DIR": (
        f"{__name__}.glm5_2_starter",
        "GLM5_2_DEFAULT_OUTPUT_DIR",
    ),
    "GLM5_2_FP8_MODEL": (f"{__name__}.glm5_2_starter", "GLM5_2_FP8_MODEL"),
    "GLM5_2_LORA_TARGET_MODULES": (
        f"{__name__}.glm5_2_starter",
        "GLM5_2_LORA_TARGET_MODULES",
    ),
    "GLM5_2_STARTER_PROFILE_CHOICES": (
        f"{__name__}.glm5_2_starter",
        "GLM5_2_STARTER_PROFILE_CHOICES",
    ),
    "GLM5_2_STARTER_PROFILE_DESCRIPTIONS": (
        f"{__name__}.glm5_2_starter",
        "GLM5_2_STARTER_PROFILE_DESCRIPTIONS",
    ),
    "GLM5_2_SUPPORTED_VARIANTS": (
        f"{__name__}.glm5_2_starter",
        "GLM5_2_SUPPORTED_VARIANTS",
    ),
    "GLM5_2_TASK_CHOICES": (f"{__name__}.glm5_2_starter", "GLM5_2_TASK_CHOICES"),
    "Glm52Config": (f"{__name__}.glm5_2_starter", "Glm52Config"),
    "create_glm5_2_agent_config": (
        f"{__name__}.glm5_2_starter",
        "create_glm5_2_agent_config",
    ),
    "create_glm5_2_preview": (f"{__name__}.glm5_2_starter", "create_glm5_2_preview"),
    "describe_glm5_2_starter_profiles": (
        f"{__name__}.glm5_2_starter",
        "describe_glm5_2_starter_profiles",
    ),
    "finetune_glm5_2": (f"{__name__}.glm5_2_starter", "finetune_glm5_2"),
    "get_glm5_2_config": (f"{__name__}.glm5_2_starter", "get_glm5_2_config"),
    "get_glm5_2_gspo_config": (f"{__name__}.glm5_2_starter", "get_glm5_2_gspo_config"),
    "get_glm5_2_gspo_overrides": (
        f"{__name__}.glm5_2_starter",
        "get_glm5_2_gspo_overrides",
    ),
    "get_glm5_2_profile_description": (
        f"{__name__}.glm5_2_starter",
        "get_glm5_2_profile_description",
    ),
    "get_glm5_2_profile_overrides": (
        f"{__name__}.glm5_2_starter",
        "get_glm5_2_profile_overrides",
    ),
    "get_glm5_2_serving_recommendations": (
        f"{__name__}.glm5_2_starter",
        "get_glm5_2_serving_recommendations",
    ),
    "get_glm5_2_system_prompt": (
        f"{__name__}.glm5_2_starter",
        "get_glm5_2_system_prompt",
    ),
    "load_glm5_2_config_file": (
        f"{__name__}.glm5_2_starter",
        "load_glm5_2_config_file",
    ),
    "run_glm5_2_config": (f"{__name__}.glm5_2_starter", "run_glm5_2_config"),
    "summarize_glm5_2_config": (
        f"{__name__}.glm5_2_starter",
        "summarize_glm5_2_config",
    ),
    "validate_glm5_2_config": (f"{__name__}.glm5_2_starter", "validate_glm5_2_config"),
    "write_glm5_2_config_file": (
        f"{__name__}.glm5_2_starter",
        "write_glm5_2_config_file",
    ),
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


def _maybe_import_submodule(name: str) -> ModuleType | None:
    """Import real submodules like ``stateset_agents.training.trainer`` on demand."""
    module_name = f"{__name__}.{name}"
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        spec = None

    if spec is None:
        return None

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        # The module exists but its dependencies (usually torch) do not.
        # Report it as absent so ``hasattr`` stays False instead of raising.
        logger.debug(
            "optional training submodule %s is unimportable: %s", module_name, exc
        )
        return None
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    if name == "TRL_AVAILABLE":
        value = _detect_trl()
        globals()[name] = value
        return value
    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    submodule = _maybe_import_submodule(name)
    if submodule is not None:
        return submodule
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
    "KIMI_K26_BASE_MODEL",
    "KIMI_K26_CONFIG_SUFFIXES",
    "KIMI_K26_DEFAULT_OUTPUT_DIR",
    "KIMI_K26_LORA_TARGET_MODULES",
    "KIMI_K26_STARTER_PROFILE_CHOICES",
    "KIMI_K26_STARTER_PROFILE_DESCRIPTIONS",
    "KIMI_K26_SUPPORTED_VARIANTS",
    "KIMI_K26_TASK_CHOICES",
    "KimiK26Config",
    "create_kimi_k2_6_agent_config",
    "create_kimi_k2_6_preview",
    "describe_kimi_k2_6_starter_profiles",
    "finetune_kimi_k2_6",
    "get_kimi_k2_6_config",
    "get_kimi_k2_6_gspo_config",
    "get_kimi_k2_6_gspo_overrides",
    "get_kimi_k2_6_profile_description",
    "get_kimi_k2_6_profile_overrides",
    "get_kimi_k2_6_system_prompt",
    "load_kimi_k2_6_config_file",
    "run_kimi_k2_6_config",
    "summarize_kimi_k2_6_config",
    "validate_kimi_k2_6_config",
    "write_kimi_k2_6_config_file",
    "KIMI_K3_BASE_MODEL",
    "KIMI_K3_CONFIG_SUFFIXES",
    "KIMI_K3_DEFAULT_OUTPUT_DIR",
    "KIMI_K3_LORA_TARGET_MODULES",
    "KIMI_K3_STARTER_PROFILE_CHOICES",
    "KIMI_K3_STARTER_PROFILE_DESCRIPTIONS",
    "KIMI_K3_SUPPORTED_VARIANTS",
    "KIMI_K3_TASK_CHOICES",
    "KimiK3Config",
    "create_kimi_k3_agent_config",
    "create_kimi_k3_preview",
    "describe_kimi_k3_starter_profiles",
    "finetune_kimi_k3",
    "get_kimi_k3_config",
    "get_kimi_k3_gspo_config",
    "get_kimi_k3_gspo_overrides",
    "get_kimi_k3_profile_description",
    "get_kimi_k3_profile_overrides",
    "get_kimi_k3_system_prompt",
    "load_kimi_k3_config_file",
    "run_kimi_k3_config",
    "summarize_kimi_k3_config",
    "validate_kimi_k3_config",
    "write_kimi_k3_config_file",
    "MUSE_GLIMMER_BASE_MODEL",
    "MUSE_GLIMMER_CONFIG_SUFFIXES",
    "MUSE_GLIMMER_DEFAULT_OUTPUT_DIR",
    "MUSE_GLIMMER_LORA_TARGET_MODULES",
    "MUSE_GLIMMER_STARTER_PROFILE_CHOICES",
    "MUSE_GLIMMER_STARTER_PROFILE_DESCRIPTIONS",
    "MUSE_GLIMMER_SUPPORTED_VARIANTS",
    "MUSE_GLIMMER_TASK_CHOICES",
    "MuseGlimmerConfig",
    "create_muse_glimmer_agent_config",
    "create_muse_glimmer_preview",
    "describe_muse_glimmer_starter_profiles",
    "finetune_muse_glimmer",
    "get_muse_glimmer_config",
    "get_muse_glimmer_gspo_config",
    "get_muse_glimmer_gspo_overrides",
    "get_muse_glimmer_profile_description",
    "get_muse_glimmer_profile_overrides",
    "get_muse_glimmer_system_prompt",
    "load_muse_glimmer_config_file",
    "run_muse_glimmer_config",
    "summarize_muse_glimmer_config",
    "validate_muse_glimmer_config",
    "write_muse_glimmer_config_file",
    "NEMOTRON_3_5_BASE_MODEL",
    "NEMOTRON_3_5_CONFIG_SUFFIXES",
    "NEMOTRON_3_5_DEFAULT_OUTPUT_DIR",
    "NEMOTRON_3_5_LORA_TARGET_MODULES",
    "NEMOTRON_3_5_STARTER_PROFILE_CHOICES",
    "NEMOTRON_3_5_STARTER_PROFILE_DESCRIPTIONS",
    "NEMOTRON_3_5_SUPPORTED_VARIANTS",
    "NEMOTRON_3_5_TASK_CHOICES",
    "Nemotron35Config",
    "create_nemotron_3_5_agent_config",
    "create_nemotron_3_5_preview",
    "describe_nemotron_3_5_starter_profiles",
    "finetune_nemotron_3_5",
    "get_nemotron_3_5_config",
    "get_nemotron_3_5_gspo_config",
    "get_nemotron_3_5_gspo_overrides",
    "get_nemotron_3_5_profile_description",
    "get_nemotron_3_5_profile_overrides",
    "get_nemotron_3_5_system_prompt",
    "load_nemotron_3_5_config_file",
    "run_nemotron_3_5_config",
    "summarize_nemotron_3_5_config",
    "validate_nemotron_3_5_config",
    "write_nemotron_3_5_config_file",
    "QWEN38_27B_BASE_MODEL",
    "QWEN38_27B_CONFIG_SUFFIXES",
    "QWEN38_27B_DEFAULT_OUTPUT_DIR",
    "QWEN38_27B_LORA_TARGET_MODULES",
    "QWEN38_27B_STARTER_PROFILE_CHOICES",
    "QWEN38_27B_STARTER_PROFILE_DESCRIPTIONS",
    "QWEN38_27B_SUPPORTED_VARIANTS",
    "QWEN38_27B_TASK_CHOICES",
    "Qwen38Config",
    "create_qwen3_8_agent_config",
    "create_qwen3_8_preview",
    "describe_qwen3_8_starter_profiles",
    "finetune_qwen3_8",
    "get_qwen3_8_config",
    "get_qwen3_8_gspo_config",
    "get_qwen3_8_gspo_overrides",
    "get_qwen3_8_profile_description",
    "get_qwen3_8_profile_overrides",
    "get_qwen3_8_system_prompt",
    "load_qwen3_8_config_file",
    "run_qwen3_8_config",
    "summarize_qwen3_8_config",
    "validate_qwen3_8_config",
    "write_qwen3_8_config_file",
    "QWEN3_CODER_BASE_MODEL",
    "QWEN3_CODER_CONFIG_SUFFIXES",
    "QWEN3_CODER_DEFAULT_OUTPUT_DIR",
    "QWEN3_CODER_LORA_TARGET_MODULES",
    "QWEN3_CODER_STARTER_PROFILE_CHOICES",
    "QWEN3_CODER_STARTER_PROFILE_DESCRIPTIONS",
    "QWEN3_CODER_SUPPORTED_VARIANTS",
    "QWEN3_CODER_TASK_CHOICES",
    "Qwen3CoderConfig",
    "create_qwen3_coder_agent_config",
    "create_qwen3_coder_preview",
    "describe_qwen3_coder_starter_profiles",
    "finetune_qwen3_coder",
    "get_qwen3_coder_config",
    "get_qwen3_coder_gspo_config",
    "get_qwen3_coder_gspo_overrides",
    "get_qwen3_coder_profile_description",
    "get_qwen3_coder_profile_overrides",
    "get_qwen3_coder_system_prompt",
    "load_qwen3_coder_config_file",
    "run_qwen3_coder_config",
    "summarize_qwen3_coder_config",
    "validate_qwen3_coder_config",
    "write_qwen3_coder_config_file",
    "GPT_OSS_120B_MODEL",
    "GPT_OSS_BASE_MODEL",
    "GPT_OSS_CONFIG_SUFFIXES",
    "GPT_OSS_DEFAULT_OUTPUT_DIR",
    "GPT_OSS_LORA_TARGET_MODULES",
    "GPT_OSS_STARTER_PROFILE_CHOICES",
    "GPT_OSS_STARTER_PROFILE_DESCRIPTIONS",
    "GPT_OSS_SUPPORTED_VARIANTS",
    "GPT_OSS_TASK_CHOICES",
    "GptOssConfig",
    "create_gpt_oss_agent_config",
    "create_gpt_oss_preview",
    "describe_gpt_oss_starter_profiles",
    "finetune_gpt_oss",
    "get_gpt_oss_config",
    "get_gpt_oss_gspo_config",
    "get_gpt_oss_gspo_overrides",
    "get_gpt_oss_profile_description",
    "get_gpt_oss_profile_overrides",
    "get_gpt_oss_system_prompt",
    "load_gpt_oss_config_file",
    "run_gpt_oss_config",
    "summarize_gpt_oss_config",
    "validate_gpt_oss_config",
    "write_gpt_oss_config_file",
    "DEEPSEEK_V4_FLASH_BASE_MODEL",
    "DEEPSEEK_V4_BASE_MODEL",
    "DEEPSEEK_V4_CONFIG_SUFFIXES",
    "DEEPSEEK_V4_DEFAULT_OUTPUT_DIR",
    "DEEPSEEK_V4_LORA_TARGET_MODULES",
    "DEEPSEEK_V4_STARTER_PROFILE_CHOICES",
    "DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS",
    "DEEPSEEK_V4_SUPPORTED_VARIANTS",
    "DEEPSEEK_V4_TASK_CHOICES",
    "DeepseekV4Config",
    "create_deepseek_v4_agent_config",
    "create_deepseek_v4_preview",
    "describe_deepseek_v4_starter_profiles",
    "finetune_deepseek_v4",
    "get_deepseek_v4_config",
    "get_deepseek_v4_gspo_config",
    "get_deepseek_v4_gspo_overrides",
    "get_deepseek_v4_profile_description",
    "get_deepseek_v4_profile_overrides",
    "get_deepseek_v4_system_prompt",
    "load_deepseek_v4_config_file",
    "run_deepseek_v4_config",
    "summarize_deepseek_v4_config",
    "validate_deepseek_v4_config",
    "write_deepseek_v4_config_file",
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
    "GLM5_2_BASE_MODEL",
    "GLM5_2_CONFIG_SUFFIXES",
    "GLM5_2_DEFAULT_OUTPUT_DIR",
    "GLM5_2_FP8_MODEL",
    "GLM5_2_LORA_TARGET_MODULES",
    "GLM5_2_STARTER_PROFILE_CHOICES",
    "GLM5_2_STARTER_PROFILE_DESCRIPTIONS",
    "GLM5_2_SUPPORTED_VARIANTS",
    "GLM5_2_TASK_CHOICES",
    "Glm52Config",
    "create_glm5_2_agent_config",
    "create_glm5_2_preview",
    "describe_glm5_2_starter_profiles",
    "finetune_glm5_2",
    "get_glm5_2_config",
    "get_glm5_2_gspo_config",
    "get_glm5_2_gspo_overrides",
    "get_glm5_2_profile_description",
    "get_glm5_2_profile_overrides",
    "get_glm5_2_serving_recommendations",
    "get_glm5_2_system_prompt",
    "load_glm5_2_config_file",
    "run_glm5_2_config",
    "summarize_glm5_2_config",
    "validate_glm5_2_config",
    "write_glm5_2_config_file",
]
