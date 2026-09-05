"""Packaged Qwen3 Coder GSPO starter helpers.

Qwen3 Coder 30B (``Qwen/Qwen3-Coder-30B-A3B-Instruct``) is Alibaba's open
coding model: a Mixture-of-Experts causal LM (``model_type: qwen3_moe``,
``Qwen3MoeForCausalLM``) with 30B total and ~3B active parameters (128
experts, 8 active per token), 48 layers, hidden size 2048, 32 attention
heads with 4 KV heads, a 151936-token vocabulary, and 256K context
(262144 max positions). Weights are published on HuggingFace under
Apache-2.0. The presets below target QLoRA post-training of the BF16
checkpoint (the FP8 variant is inference-oriented). LoRA targets cover the
attention projections only: with 128 experts per layer, the MoE expert MLPs
(``gate_proj``/``up_proj``/``down_proj`` inside every expert) are
impractical LoRA targets, so they are deliberately excluded.

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

QWEN3_CODER_BASE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
QWEN3_CODER_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
QWEN3_CODER_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]


def validate_qwen3_coder_config(config: Any) -> list[str]:
    """Validate a Qwen3 Coder first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in QWEN3_CODER_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in QWEN3_CODER_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    model_name_lower = config.model_name.lower()
    if "qwen" not in model_name_lower:
        warnings.append("model_name does not look like a Qwen checkpoint")
    if "qwen3-coder" not in model_name_lower:
        warnings.append(
            "this helper is tuned for Qwen/Qwen3-Coder-30B-A3B-Instruct; "
            "verify overrides carefully"
        )
    if "fp8" in model_name_lower:
        warnings.append(
            "the FP8 variant is inference-oriented; post-train the BF16 checkpoint instead"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Qwen3 Coder GSPO run")
    if config.learning_rate < 1e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append("per-device batch size above 2 may increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Qwen3 Coder run")
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


SPEC = StarterSpec(
    family_label="Qwen3 Coder",
    display_name="Qwen3 Coder",
    symbol_prefix="QWEN3_CODER",
    fn_infix="qwen3_coder",
    run_suffix="qwen3_coder",
    config_class_name="Qwen3CoderConfig",
    base_model=QWEN3_CODER_BASE_MODEL,
    post_trained_model=None,
    supported_variants=[
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    ],
    task_choices=QWEN3_CODER_TASK_CHOICES,
    profile_choices=QWEN3_CODER_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/qwen3_coder_gspo",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    profile_descriptions={
        "balanced": "Default Qwen3 Coder first run with QLoRA-friendly settings and a moderate context budget.",
        "memory": "Lower-memory Qwen3 Coder first run with smaller rollout groups and shorter context.",
        "quality": "Heavier Qwen3 Coder first run with larger context and rollout sizes when you have more headroom.",
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
    system_prompt_intro="You are Qwen, an AI assistant created by Alibaba Cloud.",
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
    wandb_base_tags=("qwen3-coder", "gspo"),
    wandb_project_default="qwen3_coder-gspo",
    validate=validate_qwen3_coder_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + []
