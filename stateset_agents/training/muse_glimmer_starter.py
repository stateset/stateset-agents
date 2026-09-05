"""Packaged Muse Glimmer GSPO starter helpers.

Muse Glimmer (``meta-models/Muse-Glimmer-30B``) is Meta's open agentic model
released August 2026: a ~30B-parameter dense causal transformer (52 layers,
GQA 16:1, 131K+ context) with a dedicated perception encoder, distilled from
Muse Spark and tuned for on-device agentic workloads. Weights are published on
HuggingFace under Apache-2.0. The presets below target QLoRA post-training of
the text stack on a single high-memory GPU.
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

_FAMILY_LABEL = "Muse Glimmer"
_DISPLAY_NAME = "Muse Glimmer"

MUSE_GLIMMER_BASE_MODEL = "meta-models/Muse-Glimmer-30B"
MUSE_GLIMMER_SUPPORTED_VARIANTS = [
    MUSE_GLIMMER_BASE_MODEL,
    "meta-models/Muse-Glimmer-30B-assistant",
]
MUSE_GLIMMER_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
MUSE_GLIMMER_STARTER_PROFILE_CHOICES = [
    "balanced",
    "memory",
    "quality",
]
MUSE_GLIMMER_STARTER_PROFILE_DESCRIPTIONS = {
    "balanced": "Default Muse Glimmer first run with QLoRA-friendly settings and a moderate context budget.",
    "memory": "Lower-memory Muse Glimmer first run with smaller rollout groups and shorter context.",
    "quality": "Heavier Muse Glimmer first run with larger context and rollout sizes when you have more headroom.",
}
MUSE_GLIMMER_DEFAULT_OUTPUT_DIR = "./outputs/muse_glimmer_gspo"
MUSE_GLIMMER_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
MUSE_GLIMMER_CONFIG_SUFFIXES = {".json", ".js", ".yaml", ".yml"}

_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "balanced": {
        "use_4bit": True,
    },
    "memory": {
        "use_4bit": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 24,
        "num_generations": 2,
        "num_outer_iterations": 12,
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


def get_muse_glimmer_system_prompt(task: str = "customer_service") -> str:
    """Return a task-specific system prompt for Muse Glimmer."""
    return _common.select_system_prompt(
        task,
        base_intro="You are Muse Glimmer, an AI assistant created by Meta.",
    )


def get_muse_glimmer_profile_overrides(
    starter_profile: str = "balanced",
) -> dict[str, Any]:
    """Return preset overrides for a starter profile."""
    return _common.select_profile_overrides(
        starter_profile,
        profiles=_PROFILE_OVERRIDES,
        choices=MUSE_GLIMMER_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def get_muse_glimmer_profile_description(starter_profile: str = "balanced") -> str:
    """Return the human-readable description for a starter profile."""
    return _common.select_profile_description(
        starter_profile,
        descriptions=MUSE_GLIMMER_STARTER_PROFILE_DESCRIPTIONS,
        choices=MUSE_GLIMMER_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def summarize_muse_glimmer_config(config: MuseGlimmerConfig) -> dict[str, Any]:
    """Summarize the most relevant first-run properties for a resolved config."""
    return _common.summarize_config(config)


def describe_muse_glimmer_starter_profiles(
    task: str = "customer_service",
    model_name: str = MUSE_GLIMMER_BASE_MODEL,
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    return _common.describe_starter_profiles(
        task=task,
        model_name=model_name,
        choices=MUSE_GLIMMER_STARTER_PROFILE_CHOICES,
        get_config=get_muse_glimmer_config,
        get_description=get_muse_glimmer_profile_description,
        summarize=summarize_muse_glimmer_config,
    )


@dataclass
class MuseGlimmerConfig(_common.StarterConfigMixin):
    """Lightweight configuration container for Muse Glimmer post-training."""

    model_name: str = MUSE_GLIMMER_BASE_MODEL
    task: str = "customer_service"
    starter_profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 64
    lora_alpha: int | None = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(MUSE_GLIMMER_LORA_TARGET_MODULES)
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
    # Policy objective preset + field overrides (docs/OBJECTIVES.md); None
    # keeps the native GSPO objective.
    objective: str | None = None
    objective_overrides: dict[str, Any] | None = None

    output_dir: str = MUSE_GLIMMER_DEFAULT_OUTPUT_DIR
    save_steps_every: int = 10

    use_wandb: bool = False
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"

    _system_prompt = staticmethod(get_muse_glimmer_system_prompt)
    _wandb_base_tags = ("muse-glimmer", "gspo")
    _wandb_project_default = "muse_glimmer-gspo"

    def validate(self) -> list[str]:
        return validate_muse_glimmer_config(self)


def get_muse_glimmer_config(
    model_name: str = MUSE_GLIMMER_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> MuseGlimmerConfig:
    """Create a tuned first-run Muse Glimmer configuration."""
    return _common.resolve_starter_config(
        MuseGlimmerConfig,
        get_muse_glimmer_profile_overrides,
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


def create_muse_glimmer_agent_config(config: MuseGlimmerConfig) -> AgentConfig:
    """Create the matching AgentConfig for Muse Glimmer."""
    return _common.create_agent_config(config)


def get_muse_glimmer_gspo_overrides(config: MuseGlimmerConfig) -> dict[str, Any]:
    """Return the GSPO override payload for Muse Glimmer."""
    return _common.build_gspo_overrides(config)


def get_muse_glimmer_gspo_config(
    config: MuseGlimmerConfig,
    base_config: TrainingConfig | None = None,
):
    """Create the GSPOConfig used for Muse Glimmer post-training."""
    return _common.build_gspo_config(
        config, base_config, get_config_for_task, get_muse_glimmer_gspo_overrides
    )


def validate_muse_glimmer_config(config: MuseGlimmerConfig) -> list[str]:
    """Validate a Muse Glimmer first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in MUSE_GLIMMER_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in MUSE_GLIMMER_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    if "glimmer" not in config.model_name.lower():
        warnings.append("model_name does not look like a Muse Glimmer checkpoint")
    if "muse-glimmer" not in config.model_name.lower():
        warnings.append(
            "this helper is tuned for meta-models/Muse-Glimmer-30B; verify overrides carefully"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Muse Glimmer GSPO run")
    if config.learning_rate < 1e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append("per-device batch size above 2 may increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Muse Glimmer run")
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def create_muse_glimmer_preview(
    config: MuseGlimmerConfig,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    return _common.create_preview(
        config,
        warnings,
        agent_config_fn=create_muse_glimmer_agent_config,
        summarize_fn=summarize_muse_glimmer_config,
        gspo_overrides_fn=get_muse_glimmer_gspo_overrides,
    )


def load_muse_glimmer_config_file(path: str | Path) -> MuseGlimmerConfig:
    """Load a Muse Glimmer starter config from JSON or YAML."""
    return _common.load_config_file(
        path,
        config_cls=MuseGlimmerConfig,
        suffixes=MUSE_GLIMMER_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


def write_muse_glimmer_config_file(
    config: MuseGlimmerConfig,
    path: str | Path,
    include_preview: bool = False,
) -> Path:
    """Write a Muse Glimmer starter config to JSON or YAML."""
    return _common.write_config_file(
        config,
        path,
        include_preview,
        preview_fn=create_muse_glimmer_preview,
        suffixes=MUSE_GLIMMER_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def run_muse_glimmer_config(
    config: MuseGlimmerConfig,
    dry_run: bool = False,
) -> Any:
    """Run or preview a Muse Glimmer GSPO job from a resolved config object."""
    return await _common.run_starter_config(
        config,
        dry_run,
        preview_fn=create_muse_glimmer_preview,
        gspo_config_fn=get_muse_glimmer_gspo_config,
        agent_config_fn=create_muse_glimmer_agent_config,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def finetune_muse_glimmer(
    model_name: str = MUSE_GLIMMER_BASE_MODEL,
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
    """Run or preview a first GSPO post-training job for Muse Glimmer."""
    return await _common.finetune_starter(
        get_config_fn=get_muse_glimmer_config,
        run_fn=run_muse_glimmer_config,
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
]
