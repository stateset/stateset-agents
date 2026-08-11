"""Packaged Kimi-K3 GSPO starter helpers.

Kimi K3 launched on Moonshot's product surface on 2026-07-16 (~2.5T-param MoE,
1M+ token native context per press coverage), but HuggingFace weights, model
card, and license are not yet published. ``KIMI_K3_BASE_MODEL`` and the profile
presets below are provisional mirrors of the Kimi-K2.6 starter pending the
official release.
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

_FAMILY_LABEL = "Kimi"
_DISPLAY_NAME = "Kimi-K3"

KIMI_K3_BASE_MODEL = "moonshotai/Kimi-K3"
KIMI_K3_SUPPORTED_VARIANTS = [
    KIMI_K3_BASE_MODEL,
]
KIMI_K3_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
KIMI_K3_STARTER_PROFILE_CHOICES = [
    "balanced",
    "memory",
    "quality",
]
KIMI_K3_STARTER_PROFILE_DESCRIPTIONS = {
    "balanced": "Default Kimi-K3 first run with QLoRA-friendly settings and a moderate context budget.",
    "memory": "Lower-memory Kimi-K3 first run with smaller rollout groups and shorter context.",
    "quality": "Heavier Kimi-K3 first run with larger context and rollout sizes when you have more headroom.",
}
KIMI_K3_DEFAULT_OUTPUT_DIR = "./outputs/kimi_k3_gspo"
KIMI_K3_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
KIMI_K3_CONFIG_SUFFIXES = {".json", ".js", ".yaml", ".yml"}

_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "balanced": {
        "use_4bit": True,
    },
    "memory": {
        "use_4bit": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 24,
        "num_generations": 2,
        "num_outer_iterations": 10,
        "generations_per_iteration": 8,
        "max_new_tokens": 768,
        "max_prompt_length": 2048,
        "max_completion_length": 768,
        "learning_rate": 2e-6,
    },
    "quality": {
        "use_4bit": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "num_generations": 6,
        "num_outer_iterations": 24,
        "generations_per_iteration": 16,
        "max_new_tokens": 2048,
        "max_prompt_length": 8192,
        "max_completion_length": 2048,
        "learning_rate": 2e-6,
    },
}


def get_kimi_k3_system_prompt(task: str = "customer_service") -> str:
    """Return a task-specific system prompt for Kimi-K3."""
    return _common.select_system_prompt(
        task,
        base_intro="You are Kimi, an AI assistant created by Moonshot AI.",
    )


def get_kimi_k3_profile_overrides(starter_profile: str = "balanced") -> dict[str, Any]:
    """Return preset overrides for a starter profile."""
    return _common.select_profile_overrides(
        starter_profile,
        profiles=_PROFILE_OVERRIDES,
        choices=KIMI_K3_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def get_kimi_k3_profile_description(starter_profile: str = "balanced") -> str:
    """Return the human-readable description for a starter profile."""
    return _common.select_profile_description(
        starter_profile,
        descriptions=KIMI_K3_STARTER_PROFILE_DESCRIPTIONS,
        choices=KIMI_K3_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def summarize_kimi_k3_config(config: KimiK3Config) -> dict[str, Any]:
    """Summarize the most relevant first-run properties for a resolved config."""
    return _common.summarize_config(config)


def describe_kimi_k3_starter_profiles(
    task: str = "customer_service",
    model_name: str = KIMI_K3_BASE_MODEL,
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    return _common.describe_starter_profiles(
        task=task,
        model_name=model_name,
        choices=KIMI_K3_STARTER_PROFILE_CHOICES,
        get_config=get_kimi_k3_config,
        get_description=get_kimi_k3_profile_description,
        summarize=summarize_kimi_k3_config,
    )


@dataclass
class KimiK3Config(_common.StarterConfigMixin):
    """Lightweight configuration container for Kimi-K3 post-training."""

    model_name: str = KIMI_K3_BASE_MODEL
    task: str = "customer_service"
    starter_profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 64
    lora_alpha: int | None = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(KIMI_K3_LORA_TARGET_MODULES)
    )

    use_4bit: bool = False
    use_8bit: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    max_new_tokens: int = 1024
    max_prompt_length: int = 4096
    max_completion_length: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_generations: int = 4
    learning_rate: float = 3e-6
    num_iterations: int = 1
    num_outer_iterations: int = 16
    generations_per_iteration: int = 12
    clip_range_left: float = 2e-4
    clip_range_right: float = 3e-4

    output_dir: str = KIMI_K3_DEFAULT_OUTPUT_DIR
    save_steps_every: int = 10

    use_wandb: bool = False
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"

    _system_prompt = staticmethod(get_kimi_k3_system_prompt)
    _wandb_base_tags = ("kimi-k3", "gspo")
    _wandb_project_default = "kimi_k3-gspo"

    def validate(self) -> list[str]:
        return validate_kimi_k3_config(self)


def get_kimi_k3_config(
    model_name: str = KIMI_K3_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> KimiK3Config:
    """Create a tuned first-run Kimi-K3 configuration."""
    return _common.resolve_starter_config(
        KimiK3Config,
        get_kimi_k3_profile_overrides,
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


def create_kimi_k3_agent_config(config: KimiK3Config) -> AgentConfig:
    """Create the matching AgentConfig for Kimi-K3."""
    return _common.create_agent_config(config)


def get_kimi_k3_gspo_overrides(config: KimiK3Config) -> dict[str, Any]:
    """Return the GSPO override payload for Kimi-K3."""
    return _common.build_gspo_overrides(config)


def get_kimi_k3_gspo_config(
    config: KimiK3Config,
    base_config: TrainingConfig | None = None,
):
    """Create the GSPOConfig used for Kimi-K3 post-training."""
    return _common.build_gspo_config(
        config, base_config, get_config_for_task, get_kimi_k3_gspo_overrides
    )


def validate_kimi_k3_config(config: KimiK3Config) -> list[str]:
    """Validate a Kimi-K3 first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in KIMI_K3_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in KIMI_K3_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    if "kimi" not in config.model_name.lower():
        warnings.append("model_name does not look like a Kimi checkpoint")
    if "kimi-k3" not in config.model_name.lower():
        warnings.append(
            "this helper is tuned for moonshotai/Kimi-K3; verify overrides carefully"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Kimi-K3 GSPO run")
    if config.learning_rate < 1e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append("per-device batch size above 2 may increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Kimi-K3 run")
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def create_kimi_k3_preview(
    config: KimiK3Config,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    return _common.create_preview(
        config,
        warnings,
        agent_config_fn=create_kimi_k3_agent_config,
        summarize_fn=summarize_kimi_k3_config,
        gspo_overrides_fn=get_kimi_k3_gspo_overrides,
    )


def load_kimi_k3_config_file(path: str | Path) -> KimiK3Config:
    """Load a Kimi-K3 starter config from JSON or YAML."""
    return _common.load_config_file(
        path,
        config_cls=KimiK3Config,
        suffixes=KIMI_K3_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


def write_kimi_k3_config_file(
    config: KimiK3Config,
    path: str | Path,
    include_preview: bool = False,
) -> Path:
    """Write a Kimi-K3 starter config to JSON or YAML."""
    return _common.write_config_file(
        config,
        path,
        include_preview,
        preview_fn=create_kimi_k3_preview,
        suffixes=KIMI_K3_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def run_kimi_k3_config(
    config: KimiK3Config,
    dry_run: bool = False,
) -> Any:
    """Run or preview a Kimi-K3 GSPO job from a resolved config object."""
    return await _common.run_starter_config(
        config,
        dry_run,
        preview_fn=create_kimi_k3_preview,
        gspo_config_fn=get_kimi_k3_gspo_config,
        agent_config_fn=create_kimi_k3_agent_config,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def finetune_kimi_k3(
    model_name: str = KIMI_K3_BASE_MODEL,
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
    """Run or preview a first GSPO post-training job for Kimi-K3."""
    return await _common.finetune_starter(
        get_config_fn=get_kimi_k3_config,
        run_fn=run_kimi_k3_config,
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
]
