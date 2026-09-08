"""Packaged GLM 5.2 GSPO starter helpers.

GLM 5.2 (``zai-org/GLM-5.2``) is a 754B-parameter Mixture-of-Experts model
with DeepSeek V3-style Multi-head Latent Attention (MLA) and 256 routed
experts (8 active per token). A private FP8 deployment alias such as
``your-org/GLM-5.2-FP8`` is far beyond what fits on a single GPU, so this
starter assumes:

* QLoRA-only fine-tuning on the routed/dense projection matrices
* vLLM-backed generation during training
* Multi-node serving topology (or single 8x H200/B200 host for the FP8 variant)

Built from a :class:`StarterSpec`; see ``starter_factory``.
"""

from __future__ import annotations

import logging
from typing import Any

from stateset_agents.training import starter_common as _common  # noqa: F401
from stateset_agents.training.config import (  # noqa: F401  (patched by tests)
    TrainingConfig,
    get_config_for_task,
)
from stateset_agents.training.starter_factory import (
    StarterSpec,
    build_starter,
    starter_all,
)

logger = logging.getLogger(__name__)

GLM5_2_BASE_MODEL = "zai-org/GLM-5.2"
GLM5_2_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
GLM5_2_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]
GLM5_2_FP8_MODEL = "your-org/GLM-5.2-FP8"


def validate_glm5_2_config(config: Any) -> list[str]:
    """Validate a GLM 5.2 first-run configuration."""
    warnings: list[str] = []

    installed_transformers = _common.get_transformers_version()
    if installed_transformers is not None and installed_transformers < (5, 4, 0):
        warnings.append(
            "GLM 5.2 requires transformers>=5.4.0 (glm_moe_dsa architecture); upgrade if model loading fails"
        )

    if config.starter_profile not in GLM5_2_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in GLM5_2_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    model_name_lower = config.model_name.lower()
    if "glm" not in model_name_lower:
        warnings.append("model_name does not look like a GLM checkpoint")
    if "glm-5.2" not in model_name_lower and "glm5.2" not in model_name_lower:
        warnings.append(
            "this helper is tuned for zai-org/GLM-5.2; verify overrides carefully"
        )
    if config.learning_rate > 5e-6:
        warnings.append("learning rate is high for a first GLM 5.2 GSPO run")
    if config.learning_rate < 5e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 1:
        warnings.append(
            "per-device batch size above 1 is almost certainly going to OOM on GLM 5.2"
        )
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append(
            "LoRA is mandatory for first GLM 5.2 runs (754B params, full FT not feasible)"
        )
    if not config.use_4bit and not config.use_8bit:
        warnings.append(
            "GLM 5.2 needs 4-bit quantization for any single-node fine-tuning attempt"
        )
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if not config.use_vllm:
        warnings.append(
            "vLLM-backed generation is strongly recommended for GLM 5.2 to keep rollouts tractable"
        )
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def get_glm5_2_serving_recommendations(
    *,
    use_fp8: bool = False,
    enable_auto_tool_choice: bool = True,
    tensor_parallel_size: int | None = None,
    pipeline_parallel_size: int | None = None,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    """Return the recommended vLLM settings for GLM 5.2 serving."""
    return _common.glm_serving_recommendations(
        use_fp8=use_fp8,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len,
    )


SPEC = StarterSpec(
    family_label="GLM",
    display_name="GLM 5.2",
    symbol_prefix="GLM5_2",
    fn_infix="glm5_2",
    run_suffix="glm5_2",
    config_class_name="Glm52Config",
    base_model=GLM5_2_BASE_MODEL,
    post_trained_model=None,
    supported_variants=["zai-org/GLM-5.2", "your-org/GLM-5.2-FP8"],
    task_choices=GLM5_2_TASK_CHOICES,
    profile_choices=GLM5_2_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/glm5_2_gspo",
    lora_target_modules=[
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    profile_descriptions={
        "balanced": "First GLM 5.2 run with QLoRA defaults, 4-bit quantization, and a moderate context budget.",
        "memory": "Lower-memory GLM 5.2 run with smaller groups and shorter context for tighter multi-node clusters.",
        "quality": "Heavier GLM 5.2 run with larger context and rollout sizes when you have B200/H200 headroom.",
    },
    profile_overrides={
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
            "learning_rate": 1.5e-06,
        },
    },
    system_prompt_intro="You are GLM, a helpful AI assistant built from the GLM 5.2 reasoning checkpoint by Zhipu AI.",
    config_defaults={
        "lora_r": 64,
        "lora_alpha": 128,
        "use_4bit": True,
        "max_new_tokens": 1536,
        "max_prompt_length": 8192,
        "max_completion_length": 1536,
        "temperature": 1.0,
        "top_p": 0.95,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_generations": 4,
        "learning_rate": 2e-06,
        "num_outer_iterations": 20,
        "generations_per_iteration": 12,
        "clip_range_left": 0.00015,
        "clip_range_right": 0.00025,
    },
    extra_fields=(("use_vllm", "bool", True), ("use_reference_model", "bool", True)),
    wandb_base_tags=("glm5.2", "754b", "moe", "gspo"),
    wandb_project_default="glm5_2-gspo",
    validate=validate_glm5_2_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + [
    "GLM5_2_FP8_MODEL",
    "get_glm5_2_serving_recommendations",
]
