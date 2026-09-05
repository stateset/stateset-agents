"""Packaged Qwen3.5-0.8B GSPO starter helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.core.agent import AgentConfig
from stateset_agents.training import starter_common as _common
from stateset_agents.training.config import TrainingConfig, get_config_for_task

logger = logging.getLogger(__name__)

_FAMILY_LABEL = "Qwen"
_DISPLAY_NAME = "Qwen3.5-0.8B"

QWEN35_08B_BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
QWEN35_08B_POST_TRAINED_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_08B_SUPPORTED_VARIANTS = [
    QWEN35_08B_BASE_MODEL,
    QWEN35_08B_POST_TRAINED_MODEL,
]
QWEN35_08B_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
QWEN35_08B_STARTER_PROFILE_CHOICES = [
    "balanced",
    "memory",
    "quality",
]
QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS = {
    "balanced": "Default first run with the standard Qwen 0.8B starter settings.",
    "memory": "Low-memory first run with 4-bit quantization and shorter context/group sizes.",
    "quality": "Heavier first run with larger context/group sizes when you have more headroom.",
}
QWEN35_08B_DEFAULT_OUTPUT_DIR = "./outputs/qwen3_5_0_8b_gspo"
QWEN35_08B_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
QWEN35_08B_CONFIG_SUFFIXES = {".json", ".js", ".yaml", ".yml"}

_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "balanced": {},
    "memory": {
        "use_4bit": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_generations": 2,
        "num_outer_iterations": 15,
        "generations_per_iteration": 16,
        "max_new_tokens": 512,
        "max_prompt_length": 768,
        "max_completion_length": 512,
    },
    "quality": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_generations": 6,
        "num_outer_iterations": 40,
        "generations_per_iteration": 48,
        "max_new_tokens": 1024,
        "max_prompt_length": 1536,
        "max_completion_length": 1024,
        "learning_rate": 6e-6,
    },
}


def get_qwen3_5_system_prompt(task: str = "customer_service") -> str:
    """Return a task-specific system prompt for Qwen3.5-0.8B."""
    return _common.select_system_prompt(
        task,
        base_intro="You are Qwen, an AI assistant created by Alibaba Cloud.",
    )


def get_qwen3_5_profile_overrides(starter_profile: str = "balanced") -> dict[str, Any]:
    """Return preset overrides for a starter profile."""
    return _common.select_profile_overrides(
        starter_profile,
        profiles=_PROFILE_OVERRIDES,
        choices=QWEN35_08B_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def get_qwen3_5_profile_description(starter_profile: str = "balanced") -> str:
    """Return the human-readable description for a starter profile."""
    return _common.select_profile_description(
        starter_profile,
        descriptions=QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS,
        choices=QWEN35_08B_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def summarize_qwen3_5_config(config: Qwen35Config) -> dict[str, Any]:
    """Summarize the most relevant first-run properties for a resolved config."""
    return _common.summarize_config(config)


def describe_qwen3_5_starter_profiles(
    task: str = "customer_service",
    model_name: str = QWEN35_08B_BASE_MODEL,
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    return _common.describe_starter_profiles(
        task=task,
        model_name=model_name,
        choices=QWEN35_08B_STARTER_PROFILE_CHOICES,
        get_config=get_qwen3_5_config,
        get_description=get_qwen3_5_profile_description,
        summarize=summarize_qwen3_5_config,
    )


@dataclass
class Qwen35Config(_common.StarterConfigMixin):
    """Lightweight configuration container for Qwen3.5-0.8B post-training."""

    model_name: str = QWEN35_08B_BASE_MODEL
    task: str = "customer_service"
    starter_profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 32
    lora_alpha: int | None = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(QWEN35_08B_LORA_TARGET_MODULES)
    )

    use_4bit: bool = False
    use_8bit: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    max_new_tokens: int = 768
    max_prompt_length: int = 1024
    max_completion_length: int = 768
    temperature: float = 0.7
    top_p: float = 0.9

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    learning_rate: float = 8e-6
    num_iterations: int = 1
    num_outer_iterations: int = 25
    generations_per_iteration: int = 32
    clip_range_left: float = 3e-4
    clip_range_right: float = 4e-4
    # Policy objective preset + field overrides (docs/OBJECTIVES.md); None
    # keeps the native GSPO objective.
    objective: str | None = None
    objective_overrides: dict[str, Any] | None = None

    output_dir: str = QWEN35_08B_DEFAULT_OUTPUT_DIR
    save_steps_every: int = 5

    use_wandb: bool = False
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"

    _system_prompt = staticmethod(get_qwen3_5_system_prompt)
    _wandb_base_tags = ("qwen3.5", "0.8b", "gspo")
    _wandb_project_default = "qwen3_5_0_8b-gspo"

    def validate(self) -> list[str]:
        return validate_qwen3_5_config(self)


def get_qwen3_5_config(
    model_name: str = QWEN35_08B_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> Qwen35Config:
    """Create a tuned first-run Qwen3.5-0.8B configuration."""
    return _common.resolve_starter_config(
        Qwen35Config,
        get_qwen3_5_profile_overrides,
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


def create_qwen3_5_agent_config(config: Qwen35Config) -> AgentConfig:
    """Create the matching AgentConfig for Qwen3.5-0.8B."""
    return _common.create_agent_config(config)


def get_qwen3_5_gspo_overrides(config: Qwen35Config) -> dict[str, Any]:
    """Return the GSPO override payload for Qwen3.5-0.8B."""
    return _common.build_gspo_overrides(config)


def get_qwen3_5_gspo_config(
    config: Qwen35Config,
    base_config: TrainingConfig | None = None,
):
    """Create the GSPOConfig used for Qwen3.5-0.8B post-training."""
    return _common.build_gspo_config(
        config, base_config, get_config_for_task, get_qwen3_5_gspo_overrides
    )


def validate_qwen3_5_config(config: Qwen35Config) -> list[str]:
    """Validate a Qwen3.5-0.8B first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in QWEN35_08B_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in QWEN35_08B_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    if "qwen" not in config.model_name.lower():
        warnings.append("model_name does not look like a Qwen checkpoint")
    if "qwen3.5-0.8b" not in config.model_name.lower():
        warnings.append(
            "this helper is tuned for Qwen/Qwen3.5-0.8B; verify overrides carefully"
        )
    if config.learning_rate > 2e-5:
        warnings.append("learning rate is high for a first Qwen3.5-0.8B GSPO run")
    if config.learning_rate < 1e-6:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 4:
        warnings.append("per-device batch size above 4 may increase OOM risk")
    if config.get_effective_batch_size() < 4:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Qwen3.5-0.8B run")
    if config.max_prompt_length > 4096:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 2048:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def create_qwen3_5_preview(
    config: Qwen35Config,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    return _common.create_preview(
        config,
        warnings,
        agent_config_fn=create_qwen3_5_agent_config,
        summarize_fn=summarize_qwen3_5_config,
        gspo_overrides_fn=get_qwen3_5_gspo_overrides,
    )


def load_qwen3_5_config_file(path: str | Path) -> Qwen35Config:
    """Load a Qwen3.5-0.8B starter config from JSON or YAML."""
    return _common.load_config_file(
        path,
        config_cls=Qwen35Config,
        suffixes=QWEN35_08B_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


def write_qwen3_5_config_file(
    config: Qwen35Config,
    path: str | Path,
    include_preview: bool = False,
) -> Path:
    """Write a Qwen3.5-0.8B starter config to JSON or YAML."""
    return _common.write_config_file(
        config,
        path,
        include_preview,
        preview_fn=create_qwen3_5_preview,
        suffixes=QWEN35_08B_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def run_qwen3_5_0_8b_config(
    config: Qwen35Config,
    dry_run: bool = False,
) -> Any:
    """Run or preview a Qwen3.5-0.8B GSPO job from a resolved config object."""
    return await _common.run_starter_config(
        config,
        dry_run,
        preview_fn=create_qwen3_5_preview,
        gspo_config_fn=get_qwen3_5_gspo_config,
        agent_config_fn=create_qwen3_5_agent_config,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def finetune_qwen3_5_0_8b(
    model_name: str = QWEN35_08B_BASE_MODEL,
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
    """Run or preview a first GSPO post-training job for Qwen3.5-0.8B."""
    return await _common.finetune_starter(
        get_config_fn=get_qwen3_5_config,
        run_fn=run_qwen3_5_0_8b_config,
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
    "get_qwen3_5_gspo_overrides",
    "get_qwen3_5_config",
    "get_qwen3_5_gspo_config",
    "get_qwen3_5_profile_description",
    "get_qwen3_5_profile_overrides",
    "get_qwen3_5_system_prompt",
    "load_qwen3_5_config_file",
    "run_qwen3_5_0_8b_config",
    "summarize_qwen3_5_config",
    "validate_qwen3_5_config",
    "write_qwen3_5_config_file",
]
