"""Packaged NVIDIA Nemotron 3.5 Lightning GSPO starter helpers.

Nemotron 3.5 Lightning (``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16``)
is NVIDIA's open model released August 2026: a hybrid Mamba-2 + attention +
MoE causal LM (``model_type: nemotron_h``) with 30B total and ~3B active
parameters (A3B), 52 layers, 131072-token vocabulary, and 256K practical
context (262144 max positions; 1M claimed). Weights are published on
HuggingFace under OpenMDW-1.1. The presets below target QLoRA post-training
of the BF16 checkpoint (the NVFP4 variant is inference-only); the custom
architecture requires ``trust_remote_code=True``.

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

NEMOTRON_3_5_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
NEMOTRON_3_5_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
NEMOTRON_3_5_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]


def validate_nemotron_3_5_config(config: Any) -> list[str]:
    """Validate a Nemotron 3.5 first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in NEMOTRON_3_5_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in NEMOTRON_3_5_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    if "nemotron" not in config.model_name.lower():
        warnings.append("model_name does not look like a Nemotron checkpoint")
    if "nemotron-3.5" not in config.model_name.lower():
        warnings.append(
            "this helper is tuned for nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; "
            "verify overrides carefully"
        )
    if "nvfp4" in config.model_name.lower():
        warnings.append(
            "the NVFP4 variant is inference-only; post-train the BF16 checkpoint instead"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Nemotron 3.5 GSPO run")
    if config.learning_rate < 1e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append("per-device batch size above 2 may increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Nemotron 3.5 run")
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


SPEC = StarterSpec(
    family_label="Nemotron 3.5",
    display_name="Nemotron 3.5",
    symbol_prefix="NEMOTRON_3_5",
    fn_infix="nemotron_3_5",
    run_suffix="nemotron_3_5",
    config_class_name="Nemotron35Config",
    base_model=NEMOTRON_3_5_BASE_MODEL,
    post_trained_model=None,
    supported_variants=[
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16",
    ],
    task_choices=NEMOTRON_3_5_TASK_CHOICES,
    profile_choices=NEMOTRON_3_5_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/nemotron_3_5_gspo",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"],
    profile_descriptions={
        "balanced": "Default Nemotron 3.5 first run with QLoRA-friendly settings and a moderate context budget.",
        "memory": "Lower-memory Nemotron 3.5 first run with smaller rollout groups and shorter context.",
        "quality": "Heavier Nemotron 3.5 first run with larger context and rollout sizes when you have more headroom.",
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
    system_prompt_intro="You are a helpful AI assistant built on NVIDIA Nemotron.",
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
    wandb_base_tags=("nemotron-3-5", "gspo"),
    wandb_project_default="nemotron_3_5-gspo",
    validate=validate_nemotron_3_5_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + []
