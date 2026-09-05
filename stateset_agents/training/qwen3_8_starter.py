"""Packaged Qwen3.8 27B GSPO starter helpers.

Qwen3.8 27B (``Qwen/Qwen3.8-27B``) is Alibaba's open model released
2026-08-05 under Apache-2.0: a 27.8B-parameter multimodal LM
(``model_type: qwen3_5``, architecture ``Qwen3_5ForConditionalGeneration``)
pairing a vision tower with a 64-layer text stack (hidden 5120, 24 heads /
4 KV heads, 248320-token vocabulary, 262144 max positions = 256K context).
BF16 weights are roughly 56GB, so budget ~160GB of disk and either an 80GB
card or ``--gpu-count 2``. The custom architecture requires
``trust_remote_code=True``; the presets below target QLoRA post-training of
the text stack.

**Why these LoRA targets.** The text stack uses *hybrid* attention,
confirmed against the published weight map:

- a minority of layers use standard attention (``self_attn``: ``q_proj``,
  ``k_proj``, ``v_proj``, ``o_proj`` — 96 tensors);
- most layers use Mamba-style linear attention (``linear_attn``:
  ``in_proj_qkv``, ``in_proj_a``, ``in_proj_b``, ``in_proj_z``,
  ``out_proj``, ``conv1d`` — 432 tensors);
- every layer has an MLP (``gate_proj``, ``up_proj``, ``down_proj`` — 192
  tensors).

Listing only the llama-style names would silently adapt just the minority
standard-attention layers, so ``QWEN38_27B_LORA_TARGET_MODULES`` covers all
three groups. ``conv1d`` is left out because LoRA's low-rank decomposition
targets ``nn.Linear``. The vision tower (``model.visual.*``) is excluded:
text-only SFT sends it no gradient. Note that ``out_proj`` appears in both
stacks; peft matches by leaf name, so adapting the text copies necessarily
reaches the vision ones too.
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

_FAMILY_LABEL = "Qwen3.8 27B"
_DISPLAY_NAME = "Qwen3.8 27B"

QWEN38_27B_BASE_MODEL = "Qwen/Qwen3.8-27B"
QWEN38_27B_SUPPORTED_VARIANTS = [
    QWEN38_27B_BASE_MODEL,
    "Qwen/Qwen3.8-27B-FP8",
]
QWEN38_27B_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
QWEN38_27B_STARTER_PROFILE_CHOICES = [
    "balanced",
    "memory",
    "quality",
]
QWEN38_27B_STARTER_PROFILE_DESCRIPTIONS = {
    "balanced": "Default Qwen3.8 27B first run with QLoRA-friendly settings and a moderate context budget.",
    "memory": "Lower-memory Qwen3.8 27B first run with smaller rollout groups and shorter context.",
    "quality": "Heavier Qwen3.8 27B first run with larger context and rollout sizes when you have more headroom.",
}
QWEN38_27B_DEFAULT_OUTPUT_DIR = "./outputs/qwen3_8_27b_gspo"
QWEN38_27B_LORA_TARGET_MODULES = [
    # standard attention (minority of layers)
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # Mamba-style linear attention (majority of layers)
    "in_proj_qkv",
    "out_proj",
    # MLP (every layer)
    "gate_proj",
    "up_proj",
    "down_proj",
]
QWEN38_27B_CONFIG_SUFFIXES = {".json", ".js", ".yaml", ".yml"}

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


def get_qwen3_8_system_prompt(task: str = "customer_service") -> str:
    """Return a task-specific system prompt for Qwen3.8 27B."""
    return _common.select_system_prompt(
        task,
        base_intro="You are Qwen, an AI assistant created by Alibaba Cloud.",
    )


def get_qwen3_8_profile_overrides(
    starter_profile: str = "balanced",
) -> dict[str, Any]:
    """Return preset overrides for a starter profile."""
    return _common.select_profile_overrides(
        starter_profile,
        profiles=_PROFILE_OVERRIDES,
        choices=QWEN38_27B_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def get_qwen3_8_profile_description(starter_profile: str = "balanced") -> str:
    """Return the human-readable description for a starter profile."""
    return _common.select_profile_description(
        starter_profile,
        descriptions=QWEN38_27B_STARTER_PROFILE_DESCRIPTIONS,
        choices=QWEN38_27B_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def summarize_qwen3_8_config(config: Qwen38Config) -> dict[str, Any]:
    """Summarize the most relevant first-run properties for a resolved config."""
    return _common.summarize_config(config)


def describe_qwen3_8_starter_profiles(
    task: str = "customer_service",
    model_name: str = QWEN38_27B_BASE_MODEL,
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    return _common.describe_starter_profiles(
        task=task,
        model_name=model_name,
        choices=QWEN38_27B_STARTER_PROFILE_CHOICES,
        get_config=get_qwen3_8_config,
        get_description=get_qwen3_8_profile_description,
        summarize=summarize_qwen3_8_config,
    )


@dataclass
class Qwen38Config(_common.StarterConfigMixin):
    """Lightweight configuration container for Qwen3.8 27B post-training."""

    model_name: str = QWEN38_27B_BASE_MODEL
    task: str = "customer_service"
    starter_profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 64
    lora_alpha: int | None = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(QWEN38_27B_LORA_TARGET_MODULES)
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

    output_dir: str = QWEN38_27B_DEFAULT_OUTPUT_DIR
    save_steps_every: int = 10

    use_wandb: bool = False
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"

    _system_prompt = staticmethod(get_qwen3_8_system_prompt)
    _wandb_base_tags = ("qwen3-8-27b", "gspo")
    _wandb_project_default = "qwen3_8_27b-gspo"

    def validate(self) -> list[str]:
        return validate_qwen3_8_config(self)


def get_qwen3_8_config(
    model_name: str = QWEN38_27B_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> Qwen38Config:
    """Create a tuned first-run Qwen3.8 27B configuration."""
    return _common.resolve_starter_config(
        Qwen38Config,
        get_qwen3_8_profile_overrides,
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


def create_qwen3_8_agent_config(config: Qwen38Config) -> AgentConfig:
    """Create the matching AgentConfig for Qwen3.8 27B."""
    return _common.create_agent_config(config)


def get_qwen3_8_gspo_overrides(config: Qwen38Config) -> dict[str, Any]:
    """Return the GSPO override payload for Qwen3.8 27B."""
    return _common.build_gspo_overrides(config)


def get_qwen3_8_gspo_config(
    config: Qwen38Config,
    base_config: TrainingConfig | None = None,
):
    """Create the GSPOConfig used for Qwen3.8 27B post-training."""
    return _common.build_gspo_config(
        config, base_config, get_config_for_task, get_qwen3_8_gspo_overrides
    )


def validate_qwen3_8_config(config: Qwen38Config) -> list[str]:
    """Validate a Qwen3.8 27B first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in QWEN38_27B_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in QWEN38_27B_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    if "qwen" not in config.model_name.lower():
        warnings.append("model_name does not look like a Qwen checkpoint")
    if "qwen3.8-27b" not in config.model_name.lower():
        warnings.append(
            "this helper is tuned for Qwen/Qwen3.8-27B; verify overrides carefully"
        )
    if "fp8" in config.model_name.lower():
        warnings.append(
            "the FP8 variant is inference-oriented; post-train the BF16 checkpoint instead"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Qwen3.8 27B GSPO run")
    if config.learning_rate < 1e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append("per-device batch size above 2 may increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Qwen3.8 27B run")
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def create_qwen3_8_preview(
    config: Qwen38Config,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    return _common.create_preview(
        config,
        warnings,
        agent_config_fn=create_qwen3_8_agent_config,
        summarize_fn=summarize_qwen3_8_config,
        gspo_overrides_fn=get_qwen3_8_gspo_overrides,
    )


def load_qwen3_8_config_file(path: str | Path) -> Qwen38Config:
    """Load a Qwen3.8 27B starter config from JSON or YAML."""
    return _common.load_config_file(
        path,
        config_cls=Qwen38Config,
        suffixes=QWEN38_27B_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


def write_qwen3_8_config_file(
    config: Qwen38Config,
    path: str | Path,
    include_preview: bool = False,
) -> Path:
    """Write a Qwen3.8 27B starter config to JSON or YAML."""
    return _common.write_config_file(
        config,
        path,
        include_preview,
        preview_fn=create_qwen3_8_preview,
        suffixes=QWEN38_27B_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def run_qwen3_8_config(
    config: Qwen38Config,
    dry_run: bool = False,
) -> Any:
    """Run or preview a Qwen3.8 27B GSPO job from a resolved config object."""
    return await _common.run_starter_config(
        config,
        dry_run,
        preview_fn=create_qwen3_8_preview,
        gspo_config_fn=get_qwen3_8_gspo_config,
        agent_config_fn=create_qwen3_8_agent_config,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def finetune_qwen3_8(
    model_name: str = QWEN38_27B_BASE_MODEL,
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
    """Run or preview a first GSPO post-training job for Qwen3.8 27B."""
    return await _common.finetune_starter(
        get_config_fn=get_qwen3_8_config,
        run_fn=run_qwen3_8_config,
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
]
