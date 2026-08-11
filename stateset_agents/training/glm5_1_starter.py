"""Packaged GLM 5.1 GSPO starter helpers.

GLM 5.1 (``zai-org/GLM-5.1``) is a 754B-parameter Mixture-of-Experts model
with DeepSeek V3-style Multi-head Latent Attention (MLA) and 256 routed
experts (8 active per token). A private FP8 deployment alias such as
``your-org/GLM-5.1-FP8`` is far beyond what fits on a single GPU, so this
starter assumes:

* QLoRA-only fine-tuning on the routed/dense projection matrices
* vLLM-backed generation during training
* Multi-node serving topology (or single 8x H200/B200 host for the FP8 variant)
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

_FAMILY_LABEL = "GLM"
_DISPLAY_NAME = "GLM 5.1"

GLM5_1_BASE_MODEL = "zai-org/GLM-5.1"
GLM5_1_FP8_MODEL = "your-org/GLM-5.1-FP8"
GLM5_1_SUPPORTED_VARIANTS = [
    GLM5_1_BASE_MODEL,
    GLM5_1_FP8_MODEL,
]
GLM5_1_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
GLM5_1_STARTER_PROFILE_CHOICES = [
    "balanced",
    "memory",
    "quality",
]
GLM5_1_STARTER_PROFILE_DESCRIPTIONS = {
    "balanced": "First GLM 5.1 run with QLoRA defaults, 4-bit quantization, and a moderate context budget.",
    "memory": "Lower-memory GLM 5.1 run with smaller groups and shorter context for tighter multi-node clusters.",
    "quality": "Heavier GLM 5.1 run with larger context and rollout sizes when you have B200/H200 headroom.",
}
GLM5_1_DEFAULT_OUTPUT_DIR = "./outputs/glm5_1_gspo"
# GLM 5.1 uses DeepSeek V3-style MLA attention plus standard SwiGLU FFN.
# These projection names match the ``glm_moe_dsa`` architecture.
GLM5_1_LORA_TARGET_MODULES = [
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
GLM5_1_CONFIG_SUFFIXES = {".json", ".js", ".yaml", ".yml"}

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


def get_glm5_1_system_prompt(task: str = "customer_service") -> str:
    """Return a task-specific system prompt for GLM 5.1."""
    return _common.select_system_prompt(
        task,
        base_intro=(
            "You are GLM, a helpful AI assistant built from the GLM 5.1 reasoning "
            "checkpoint by Zhipu AI."
        ),
        conversational=_common.CONVERSATIONAL_GROUNDED,
    )


def get_glm5_1_profile_overrides(starter_profile: str = "balanced") -> dict[str, Any]:
    """Return preset overrides for a GLM 5.1 starter profile."""
    return _common.select_profile_overrides(
        starter_profile,
        profiles=_PROFILE_OVERRIDES,
        choices=GLM5_1_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def get_glm5_1_profile_description(starter_profile: str = "balanced") -> str:
    """Return the human-readable description for a starter profile."""
    return _common.select_profile_description(
        starter_profile,
        descriptions=GLM5_1_STARTER_PROFILE_DESCRIPTIONS,
        choices=GLM5_1_STARTER_PROFILE_CHOICES,
        family_label=_FAMILY_LABEL,
    )


def summarize_glm5_1_config(config: Glm51Config) -> dict[str, Any]:
    """Summarize the most relevant first-run properties for a resolved config."""
    return _common.summarize_config(config)


def describe_glm5_1_starter_profiles(
    task: str = "customer_service",
    model_name: str = GLM5_1_BASE_MODEL,
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    return _common.describe_starter_profiles(
        task=task,
        model_name=model_name,
        choices=GLM5_1_STARTER_PROFILE_CHOICES,
        get_config=get_glm5_1_config,
        get_description=get_glm5_1_profile_description,
        summarize=summarize_glm5_1_config,
    )


def get_glm5_1_serving_recommendations(
    *,
    use_fp8: bool = False,
    enable_auto_tool_choice: bool = True,
    tensor_parallel_size: int | None = None,
    pipeline_parallel_size: int | None = None,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    """Return the recommended vLLM settings for GLM 5.1 serving."""
    return _common.glm_serving_recommendations(
        use_fp8=use_fp8,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len,
    )


@dataclass
class Glm51Config(_common.StarterConfigMixin):
    """Lightweight configuration container for GLM 5.1 post-training."""

    model_name: str = GLM5_1_BASE_MODEL
    task: str = "customer_service"
    starter_profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 64
    lora_alpha: int | None = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(GLM5_1_LORA_TARGET_MODULES)
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
    output_dir: str = GLM5_1_DEFAULT_OUTPUT_DIR
    save_steps_every: int = 5

    use_wandb: bool = False
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"

    _system_prompt = staticmethod(get_glm5_1_system_prompt)
    _wandb_base_tags = ("glm5.1", "754b", "moe", "gspo")
    _wandb_project_default = "glm5_1-gspo"

    def validate(self) -> list[str]:
        return validate_glm5_1_config(self)


def get_glm5_1_config(
    model_name: str = GLM5_1_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> Glm51Config:
    """Create a tuned first-run GLM 5.1 configuration."""
    return _common.resolve_starter_config(
        Glm51Config,
        get_glm5_1_profile_overrides,
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


def create_glm5_1_agent_config(config: Glm51Config) -> AgentConfig:
    """Create the matching AgentConfig for GLM 5.1."""
    return _common.create_agent_config(config)


def get_glm5_1_gspo_overrides(config: Glm51Config) -> dict[str, Any]:
    """Return the GSPO override payload for GLM 5.1."""
    return _common.build_gspo_overrides(config)


def get_glm5_1_gspo_config(
    config: Glm51Config,
    base_config: TrainingConfig | None = None,
):
    """Create the GSPOConfig used for GLM 5.1 post-training."""
    return _common.build_gspo_config(
        config, base_config, get_config_for_task, get_glm5_1_gspo_overrides
    )


def validate_glm5_1_config(config: Glm51Config) -> list[str]:
    """Validate a GLM 5.1 first-run configuration."""
    warnings: list[str] = []

    installed_transformers = _common.get_transformers_version()
    if installed_transformers is not None and installed_transformers < (5, 4, 0):
        warnings.append(
            "GLM 5.1 requires transformers>=5.4.0 (glm_moe_dsa architecture); upgrade if model loading fails"
        )

    if config.starter_profile not in GLM5_1_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in GLM5_1_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    model_name_lower = config.model_name.lower()
    if "glm" not in model_name_lower:
        warnings.append("model_name does not look like a GLM checkpoint")
    if "glm-5.1" not in model_name_lower and "glm5.1" not in model_name_lower:
        warnings.append(
            "this helper is tuned for zai-org/GLM-5.1; verify overrides carefully"
        )
    if config.learning_rate > 5e-6:
        warnings.append("learning rate is high for a first GLM 5.1 GSPO run")
    if config.learning_rate < 5e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 1:
        warnings.append(
            "per-device batch size above 1 is almost certainly going to OOM on GLM 5.1"
        )
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append(
            "LoRA is mandatory for first GLM 5.1 runs (754B params, full FT not feasible)"
        )
    if not config.use_4bit and not config.use_8bit:
        warnings.append(
            "GLM 5.1 needs 4-bit quantization for any single-node fine-tuning attempt"
        )
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if not config.use_vllm:
        warnings.append(
            "vLLM-backed generation is strongly recommended for GLM 5.1 to keep rollouts tractable"
        )
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def create_glm5_1_preview(
    config: Glm51Config,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    return _common.create_preview(
        config,
        warnings,
        agent_config_fn=create_glm5_1_agent_config,
        summarize_fn=summarize_glm5_1_config,
        gspo_overrides_fn=get_glm5_1_gspo_overrides,
    )


def load_glm5_1_config_file(path: str | Path) -> Glm51Config:
    """Load a GLM 5.1 starter config from JSON or YAML."""
    return _common.load_config_file(
        path,
        config_cls=Glm51Config,
        suffixes=GLM5_1_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


def write_glm5_1_config_file(
    config: Glm51Config,
    path: str | Path,
    include_preview: bool = False,
) -> Path:
    """Write a GLM 5.1 starter config to JSON or YAML."""
    return _common.write_config_file(
        config,
        path,
        include_preview,
        preview_fn=create_glm5_1_preview,
        suffixes=GLM5_1_CONFIG_SUFFIXES,
        family_label=_FAMILY_LABEL,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def run_glm5_1_config(
    config: Glm51Config,
    dry_run: bool = False,
) -> Any:
    """Run or preview a GLM 5.1 GSPO job from a resolved config object."""
    return await _common.run_starter_config(
        config,
        dry_run,
        preview_fn=create_glm5_1_preview,
        gspo_config_fn=get_glm5_1_gspo_config,
        agent_config_fn=create_glm5_1_agent_config,
        display_name=_DISPLAY_NAME,
        logger=logger,
    )


async def finetune_glm5_1(
    model_name: str = GLM5_1_BASE_MODEL,
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
    """Run or preview a first GSPO post-training job for GLM 5.1."""
    return await _common.finetune_starter(
        get_config_fn=get_glm5_1_config,
        run_fn=run_glm5_1_config,
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
