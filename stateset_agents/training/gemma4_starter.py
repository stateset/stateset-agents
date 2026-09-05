"""Packaged Gemma 4 31B GSPO starter helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.core.agent import AgentConfig
from stateset_agents.training import starter_common as _common
from stateset_agents.training.config import TrainingConfig, get_config_for_task

logger = logging.getLogger(__name__)

_FAMILY_LABEL = "Gemma"
_DISPLAY_NAME = "Gemma 4 31B"

GEMMA4_31B_BASE_MODEL = "google/gemma-4-31B-it"
GEMMA4_31B_SUPPORTED_VARIANTS = [
    GEMMA4_31B_BASE_MODEL,
]
GEMMA4_31B_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
GEMMA4_31B_STARTER_PROFILE_CHOICES = [
    "balanced",
    "memory",
    "quality",
]
GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS = {
    "balanced": "First Gemma 4 31B run with QLoRA defaults, 4-bit quantization, and a moderate context budget.",
    "memory": "Lower-memory Gemma 4 31B run with smaller groups and shorter context for tighter GPUs.",
    "quality": "Heavier Gemma 4 31B run with larger context and rollout sizes when you have more headroom.",
}
GEMMA4_31B_DEFAULT_OUTPUT_DIR = "./outputs/gemma4_31b_gspo"
GEMMA4_31B_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
GEMMA4_31B_CONFIG_SUFFIXES = {".json", ".js", ".yaml", ".yml"}

_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "balanced": {},
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
    },
    "quality": {
        "use_4bit": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "num_generations": 6,
        "num_outer_iterations": 24,
        "generations_per_iteration": 16,
        "max_new_tokens": 1536,
        "max_prompt_length": 8192,
        "max_completion_length": 1536,
        "learning_rate": 2e-6,
    },
}


def get_gemma4_31b_system_prompt(task: str = "customer_service") -> str:
    """Return a task-specific system prompt for Gemma 4 31B."""
    return _common.select_system_prompt(
        task,
        base_intro=(
            "You are Gemma, a helpful AI assistant built from the Gemma 4 31B "
            "instruction-tuned checkpoint by Google DeepMind."
        ),
        conversational=_common.CONVERSATIONAL_GROUNDED,
    )


def get_gemma4_31b_profile_overrides(
    starter_profile: str = "balanced",
) -> dict[str, Any]:
    """Return preset overrides for a Gemma 4 31B starter profile."""
    return _common.select_profile_overrides(
        starter_profile,
        profiles=_PROFILE_OVERRIDES,
        choices=GEMMA4_31B_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def get_gemma4_31b_profile_description(starter_profile: str = "balanced") -> str:
    """Return the human-readable description for a starter profile."""
    return _common.select_profile_description(
        starter_profile,
        descriptions=GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS,
        choices=GEMMA4_31B_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def summarize_gemma4_31b_config(config: Gemma4Config) -> dict[str, Any]:
    """Summarize the most relevant first-run properties for a resolved config."""
    return _common.summarize_config(config)


def describe_gemma4_31b_starter_profiles(
    task: str = "customer_service",
    model_name: str = GEMMA4_31B_BASE_MODEL,
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    return _common.describe_starter_profiles(
        task=task,
        model_name=model_name,
        choices=GEMMA4_31B_STARTER_PROFILE_CHOICES,
        get_config=get_gemma4_31b_config,
        get_description=get_gemma4_31b_profile_description,
        summarize=summarize_gemma4_31b_config,
    )


@dataclass
class Gemma4Config(_common.StarterConfigMixin):
    """Lightweight configuration container for Gemma 4 31B post-training."""

    model_name: str = GEMMA4_31B_BASE_MODEL
    task: str = "customer_service"
    starter_profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 64
    lora_alpha: int | None = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(GEMMA4_31B_LORA_TARGET_MODULES)
    )

    use_4bit: bool = True
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
    num_outer_iterations: int = 20
    generations_per_iteration: int = 12
    clip_range_left: float = 2e-4
    clip_range_right: float = 3e-4
    # Policy objective preset + field overrides (docs/OBJECTIVES.md); None
    # keeps the native GSPO objective.
    objective: str | None = None
    objective_overrides: dict[str, Any] | None = None

    output_dir: str = GEMMA4_31B_DEFAULT_OUTPUT_DIR
    save_steps_every: int = 5

    use_wandb: bool = False
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"

    _system_prompt = staticmethod(get_gemma4_31b_system_prompt)
    _wandb_base_tags = ("gemma4", "31b", "gspo")
    _wandb_project_default = "gemma4_31b-gspo"

    def validate(self) -> list[str]:
        return validate_gemma4_31b_config(self)


def get_gemma4_31b_config(
    model_name: str = GEMMA4_31B_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> Gemma4Config:
    """Create a tuned first-run Gemma 4 31B configuration."""
    return _common.resolve_starter_config(
        Gemma4Config,
        get_gemma4_31b_profile_overrides,
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


def create_gemma4_31b_agent_config(config: Gemma4Config) -> AgentConfig:
    """Create the matching AgentConfig for Gemma 4 31B."""
    return _common.create_agent_config(
        config,
        tokenizer_kwargs={"padding_side": "left"},
    )


def get_gemma4_31b_gspo_overrides(config: Gemma4Config) -> dict[str, Any]:
    """Return the GSPO override payload for Gemma 4 31B."""
    return _common.build_gspo_overrides(config)


def get_gemma4_31b_gspo_config(
    config: Gemma4Config,
    base_config: TrainingConfig | None = None,
):
    """Create the GSPOConfig used for Gemma 4 31B post-training."""
    return _common.build_gspo_config(
        config, base_config, get_config_for_task, get_gemma4_31b_gspo_overrides
    )


def validate_gemma4_31b_config(config: Gemma4Config) -> list[str]:
    """Validate a Gemma 4 31B first-run configuration."""
    warnings: list[str] = []

    installed_transformers = _common.get_transformers_version()
    if installed_transformers is not None and installed_transformers < (4, 57, 1):
        warnings.append(
            "Gemma 4 31B is validated with transformers>=4.57.1; upgrade if model loading fails"
        )

    if config.starter_profile not in GEMMA4_31B_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in GEMMA4_31B_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    model_name_lower = config.model_name.lower()
    if "gemma" not in model_name_lower:
        warnings.append("model_name does not look like a Gemma checkpoint")
    if "gemma-4" not in model_name_lower or "31b" not in model_name_lower:
        warnings.append(
            "this helper is tuned for google/gemma-4-31B-it; verify overrides carefully"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Gemma 4 31B GSPO run")
    if config.learning_rate < 1e-6:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 1:
        warnings.append("per-device batch size above 1 is likely to increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is strongly recommended for the first Gemma 4 31B run")
    if not config.use_4bit and not config.use_8bit:
        warnings.append(
            "Gemma 4 31B usually needs quantization for starter runs; consider use_4bit=True"
        )
    if config.max_prompt_length > 8192:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 2048:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def create_gemma4_31b_preview(
    config: Gemma4Config,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    return _common.create_preview(
        config,
        warnings,
        agent_config_fn=create_gemma4_31b_agent_config,
        summarize_fn=summarize_gemma4_31b_config,
        gspo_overrides_fn=get_gemma4_31b_gspo_overrides,
    )


def load_gemma4_31b_config_file(path: str | Path) -> Gemma4Config:
    """Load a Gemma 4 31B starter config from JSON or YAML."""
    return _common.load_config_file(
        path,
        config_cls=Gemma4Config,
        suffixes=GEMMA4_31B_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


def write_gemma4_31b_config_file(
    config: Gemma4Config,
    path: str | Path,
    include_preview: bool = False,
) -> Path:
    """Write a Gemma 4 31B starter config to JSON or YAML."""
    return _common.write_config_file(
        config,
        path,
        include_preview,
        preview_fn=create_gemma4_31b_preview,
        suffixes=GEMMA4_31B_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def run_gemma4_31b_config(
    config: Gemma4Config,
    dry_run: bool = False,
) -> Any:
    """Run or preview a Gemma 4 31B GSPO job from a resolved config object."""
    return await _common.run_starter_config(
        config,
        dry_run,
        preview_fn=create_gemma4_31b_preview,
        gspo_config_fn=get_gemma4_31b_gspo_config,
        agent_config_fn=create_gemma4_31b_agent_config,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def finetune_gemma4_31b(
    model_name: str = GEMMA4_31B_BASE_MODEL,
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
    """Run or preview a first GSPO post-training job for Gemma 4 31B."""
    return await _common.finetune_starter(
        get_config_fn=get_gemma4_31b_config,
        run_fn=run_gemma4_31b_config,
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
]
