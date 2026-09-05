"""Packaged gpt-oss GSPO starter helpers.

gpt-oss 20B (``openai/gpt-oss-20b``) is OpenAI's open-weight reasoning model:
a Mixture-of-Experts causal LM (``model_type: gpt_oss``,
``GptOssForCausalLM``) with 32 experts (4 active per token), 24 layers,
hidden size 2880, 64 attention heads with 8 KV heads, a 201088-token
vocabulary, and 131072-token context. The family supports adjustable
reasoning effort and the harmony response format, and is published on
HuggingFace under Apache-2.0. The larger ``openai/gpt-oss-120b`` variant is
also listed but needs multi-GPU hardware. LoRA targets are attention-only
(``q_proj``/``k_proj``/``v_proj``/``o_proj``, verified against the
checkpoint's weight map); the MoE expert weights are fused per-layer tensors
and are not standard LoRA targets.

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

GPT_OSS_BASE_MODEL = "openai/gpt-oss-20b"
GPT_OSS_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
GPT_OSS_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]
GPT_OSS_120B_MODEL = "openai/gpt-oss-120b"


def validate_gpt_oss_config(config: Any) -> list[str]:
    """Validate a gpt-oss first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in GPT_OSS_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in GPT_OSS_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    model_name_lower = config.model_name.lower()
    if "gpt-oss" not in model_name_lower:
        warnings.append("model_name does not look like a gpt-oss checkpoint")
    if "gpt-oss-120b" in model_name_lower:
        warnings.append(
            "gpt-oss-120b needs multi-GPU hardware; start with openai/gpt-oss-20b "
            "for single-GPU post-training"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first gpt-oss GSPO run")
    if config.learning_rate < 1e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append("per-device batch size above 2 may increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first gpt-oss run")
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


SPEC = StarterSpec(
    family_label="gpt-oss",
    display_name="gpt-oss",
    symbol_prefix="GPT_OSS",
    fn_infix="gpt_oss",
    run_suffix="gpt_oss",
    config_class_name="GptOssConfig",
    base_model=GPT_OSS_BASE_MODEL,
    post_trained_model=None,
    supported_variants=["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    task_choices=GPT_OSS_TASK_CHOICES,
    profile_choices=GPT_OSS_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/gpt_oss_gspo",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    profile_descriptions={
        "balanced": "Default gpt-oss first run with QLoRA-friendly settings and a moderate context budget.",
        "memory": "Lower-memory gpt-oss first run with smaller rollout groups and shorter context.",
        "quality": "Heavier gpt-oss first run with larger context and rollout sizes when you have more headroom.",
    },
    profile_overrides={
        "balanced": {"use_4bit": True},
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
            "learning_rate": 2e-06,
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
            "learning_rate": 2e-06,
        },
    },
    system_prompt_intro="You are a helpful AI assistant.",
    config_defaults={
        "lora_r": 64,
        "lora_alpha": 128,
        "max_new_tokens": 1024,
        "max_prompt_length": 4096,
        "max_completion_length": 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_generations": 4,
        "learning_rate": 3e-06,
        "num_outer_iterations": 16,
        "generations_per_iteration": 12,
        "clip_range_left": 0.0002,
        "clip_range_right": 0.0003,
        "save_steps_every": 10,
    },
    extra_fields=(),
    wandb_base_tags=("gpt-oss", "gspo"),
    wandb_project_default="gpt_oss-gspo",
    validate=validate_gpt_oss_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + ["GPT_OSS_120B_MODEL"]
