"""Packaged DeepSeek V4 Flash GSPO starter helpers.

DeepSeek V4 Flash (``deepseek-ai/DeepSeek-V4-Flash``) is a large
Mixture-of-Experts model (``model_type: deepseek_v4``,
``DeepseekV4ForCausalLM``): 43 layers, hidden size 4096, Multi-head Latent
Attention (64 heads, 1 KV latent head), 256 routed experts with 6 active per
token, a 129280-token vocabulary, and up to 1M max position embeddings.
Weights are published on HuggingFace under MIT. The NVFP4/FP8 repos are
inference-only. This is a large MoE, so this starter assumes:

* QLoRA-only fine-tuning on the MLA projection matrices
* vLLM-backed generation during training
* Multi-node (or very-high-memory single-node) topology

The MLA attention does not use llama-style ``q_proj``/``k_proj``/``v_proj``
modules; the LoRA targets below use the checkpoint's actual MLA projection
names (``wq_a``/``wq_b``/``wkv``/``wo_a``/``wo_b``), verified against the
model's safetensors weight map.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.core.agent import AgentConfig
from stateset_agents.training import starter_common as _common
from stateset_agents.training.config import TrainingConfig, get_config_for_task

logger = logging.getLogger(__name__)

_FAMILY_LABEL = "DeepSeek V4"
_DISPLAY_NAME = "DeepSeek V4 Flash"

DEEPSEEK_V4_BASE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_V4_FLASH_BASE_MODEL = "deepseek-ai/DeepSeek-V4-Flash-Base"
DEEPSEEK_V4_SUPPORTED_VARIANTS = [
    DEEPSEEK_V4_BASE_MODEL,
    DEEPSEEK_V4_FLASH_BASE_MODEL,
]
DEEPSEEK_V4_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
DEEPSEEK_V4_STARTER_PROFILE_CHOICES = [
    "balanced",
    "memory",
    "quality",
]
DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS = {
    "balanced": "First DeepSeek V4 Flash run with QLoRA defaults, 4-bit quantization, and a moderate context budget.",
    "memory": "Lower-memory DeepSeek V4 Flash run with smaller groups and shorter context for tighter multi-node clusters.",
    "quality": "Heavier DeepSeek V4 Flash run with larger context and rollout sizes when you have B200/H200 headroom.",
}
DEEPSEEK_V4_DEFAULT_OUTPUT_DIR = "./outputs/deepseek_v4_gspo"
# DeepSeek V4 Flash uses MLA attention with low-rank q/kv/o projections.
# These module names are taken from the checkpoint's safetensors weight map
# (layers.<n>.attn.{wq_a,wq_b,wkv,wo_a,wo_b}); llama-style q_proj/k_proj/
# v_proj modules do NOT exist in this architecture. The 256-expert routed
# MLPs are impractical LoRA targets and are deliberately excluded.
DEEPSEEK_V4_LORA_TARGET_MODULES = [
    "wq_a",
    "wq_b",
    "wkv",
    "wo_a",
    "wo_b",
]
DEEPSEEK_V4_CONFIG_SUFFIXES = {".json", ".js", ".yaml", ".yml"}

_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "balanced": {},
    "memory": {
        "use_4bit": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "num_generations": 2,
        "num_outer_iterations": 10,
        "generations_per_iteration": 8,
        "max_new_tokens": 1024,
        "max_prompt_length": 4096,
        "max_completion_length": 1024,
    },
    "quality": {
        "use_4bit": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "num_generations": 8,
        "num_outer_iterations": 30,
        "generations_per_iteration": 16,
        "max_new_tokens": 2048,
        "max_prompt_length": 16384,
        "max_completion_length": 2048,
        "learning_rate": 1.5e-6,
    },
}


def get_deepseek_v4_system_prompt(task: str = "customer_service") -> str:
    """Return a task-specific system prompt for DeepSeek V4 Flash."""
    return _common.select_system_prompt(
        task,
        base_intro="You are DeepSeek, an AI assistant created by DeepSeek.",
        conversational=_common.CONVERSATIONAL_GROUNDED,
    )


def get_deepseek_v4_profile_overrides(
    starter_profile: str = "balanced",
) -> dict[str, Any]:
    """Return preset overrides for a DeepSeek V4 Flash starter profile."""
    return _common.select_profile_overrides(
        starter_profile,
        profiles=_PROFILE_OVERRIDES,
        choices=DEEPSEEK_V4_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def get_deepseek_v4_profile_description(starter_profile: str = "balanced") -> str:
    """Return the human-readable description for a starter profile."""
    return _common.select_profile_description(
        starter_profile,
        descriptions=DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS,
        choices=DEEPSEEK_V4_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def summarize_deepseek_v4_config(config: DeepseekV4Config) -> dict[str, Any]:
    """Summarize the most relevant first-run properties for a resolved config."""
    return _common.summarize_config(config)


def describe_deepseek_v4_starter_profiles(
    task: str = "customer_service",
    model_name: str = DEEPSEEK_V4_BASE_MODEL,
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    return _common.describe_starter_profiles(
        task=task,
        model_name=model_name,
        choices=DEEPSEEK_V4_STARTER_PROFILE_CHOICES,
        get_config=get_deepseek_v4_config,
        get_description=get_deepseek_v4_profile_description,
        summarize=summarize_deepseek_v4_config,
    )


@dataclass
class DeepseekV4Config(_common.StarterConfigMixin):
    """Lightweight configuration container for DeepSeek V4 Flash post-training."""

    model_name: str = DEEPSEEK_V4_BASE_MODEL
    task: str = "customer_service"
    starter_profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 64
    lora_alpha: int | None = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEEPSEEK_V4_LORA_TARGET_MODULES)
    )

    use_4bit: bool = True
    use_8bit: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    max_new_tokens: int = 1536
    max_prompt_length: int = 8192
    max_completion_length: int = 1536
    temperature: float = 1.0
    top_p: float = 0.95

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_generations: int = 4
    learning_rate: float = 2e-6
    num_iterations: int = 1
    num_outer_iterations: int = 20
    generations_per_iteration: int = 12
    clip_range_left: float = 1.5e-4
    clip_range_right: float = 2.5e-4

    use_vllm: bool = True
    use_reference_model: bool = True
    output_dir: str = DEEPSEEK_V4_DEFAULT_OUTPUT_DIR
    save_steps_every: int = 5

    use_wandb: bool = False
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"

    _system_prompt = staticmethod(get_deepseek_v4_system_prompt)
    _wandb_base_tags = ("deepseek-v4", "moe", "gspo")
    _wandb_project_default = "deepseek_v4-gspo"

    def validate(self) -> list[str]:
        return validate_deepseek_v4_config(self)


def get_deepseek_v4_config(
    model_name: str = DEEPSEEK_V4_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> DeepseekV4Config:
    """Create a tuned first-run DeepSeek V4 Flash configuration."""
    return _common.resolve_starter_config(
        DeepseekV4Config,
        get_deepseek_v4_profile_overrides,
        _DISPLAY_NAME,
        logger,
        model_name=model_name,
        task=task,
        starter_profile=starter_profile,
        use_lora=use_lora,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        output_dir=output_dir,
        **overrides,
    )


def create_deepseek_v4_agent_config(config: DeepseekV4Config) -> AgentConfig:
    """Create the matching AgentConfig for DeepSeek V4 Flash."""
    return _common.create_agent_config(config)


def get_deepseek_v4_gspo_overrides(config: DeepseekV4Config) -> dict[str, Any]:
    """Return the GSPO override payload for DeepSeek V4 Flash."""
    return _common.build_gspo_overrides(config)


def get_deepseek_v4_gspo_config(
    config: DeepseekV4Config,
    base_config: TrainingConfig | None = None,
):
    """Create the GSPOConfig used for DeepSeek V4 Flash post-training."""
    return _common.build_gspo_config(
        config, base_config, get_config_for_task, get_deepseek_v4_gspo_overrides
    )


def validate_deepseek_v4_config(config: DeepseekV4Config) -> list[str]:
    """Validate a DeepSeek V4 Flash first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in DEEPSEEK_V4_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in DEEPSEEK_V4_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    model_name_lower = config.model_name.lower()
    if "deepseek" not in model_name_lower:
        warnings.append("model_name does not look like a DeepSeek checkpoint")
    if "v4-flash" not in model_name_lower:
        warnings.append(
            "this helper is tuned for deepseek-ai/DeepSeek-V4-Flash; "
            "verify overrides carefully"
        )
    if "nvfp4" in model_name_lower or "fp8" in model_name_lower:
        warnings.append(
            "the NVFP4/FP8 repos are inference-only; post-train the BF16 "
            "deepseek-ai/DeepSeek-V4-Flash checkpoint instead"
        )
    if config.learning_rate > 5e-6:
        warnings.append("learning rate is high for a first DeepSeek V4 Flash GSPO run")
    if config.learning_rate < 5e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 1:
        warnings.append(
            "per-device batch size above 1 is almost certainly going to OOM on DeepSeek V4 Flash"
        )
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append(
            "LoRA is mandatory for first DeepSeek V4 Flash runs (large MoE, full FT not feasible)"
        )
    if not config.use_4bit and not config.use_8bit:
        warnings.append(
            "DeepSeek V4 Flash needs 4-bit quantization for any single-node fine-tuning attempt"
        )
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if not config.use_vllm:
        warnings.append(
            "vLLM-backed generation is strongly recommended for DeepSeek V4 Flash to keep rollouts tractable"
        )
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def create_deepseek_v4_preview(
    config: DeepseekV4Config,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    return _common.create_preview(
        config,
        warnings,
        agent_config_fn=create_deepseek_v4_agent_config,
        summarize_fn=summarize_deepseek_v4_config,
        gspo_overrides_fn=get_deepseek_v4_gspo_overrides,
    )


def load_deepseek_v4_config_file(path: str | Path) -> DeepseekV4Config:
    """Load a DeepSeek V4 Flash starter config from JSON or YAML."""
    return _common.load_config_file(
        path,
        config_cls=DeepseekV4Config,
        suffixes=DEEPSEEK_V4_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


def write_deepseek_v4_config_file(
    config: DeepseekV4Config,
    path: str | Path,
    include_preview: bool = False,
) -> Path:
    """Write a DeepSeek V4 Flash starter config to JSON or YAML."""
    return _common.write_config_file(
        config,
        path,
        include_preview,
        preview_fn=create_deepseek_v4_preview,
        suffixes=DEEPSEEK_V4_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def run_deepseek_v4_config(
    config: DeepseekV4Config,
    dry_run: bool = False,
) -> Any:
    """Run or preview a DeepSeek V4 Flash GSPO job from a resolved config object."""
    return await _common.run_starter_config(
        config,
        dry_run,
        preview_fn=create_deepseek_v4_preview,
        gspo_config_fn=get_deepseek_v4_gspo_config,
        agent_config_fn=create_deepseek_v4_agent_config,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def finetune_deepseek_v4(
    model_name: str = DEEPSEEK_V4_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    output_dir: str | None = None,
    num_outer_iterations: int | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    dry_run: bool = False,
) -> Any:
    """Run or preview a first GSPO post-training job for DeepSeek V4 Flash."""
    return await _common.finetune_starter(
        get_config_fn=get_deepseek_v4_config,
        run_fn=run_deepseek_v4_config,
        model_name=model_name,
        task=task,
        starter_profile=starter_profile,
        use_lora=use_lora,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        output_dir=output_dir,
        num_outer_iterations=num_outer_iterations,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        dry_run=dry_run,
    )


__all__ = [
    "DEEPSEEK_V4_BASE_MODEL",
    "DEEPSEEK_V4_CONFIG_SUFFIXES",
    "DEEPSEEK_V4_DEFAULT_OUTPUT_DIR",
    "DEEPSEEK_V4_FLASH_BASE_MODEL",
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
]
