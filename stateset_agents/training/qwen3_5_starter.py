"""Packaged Qwen3.5-0.8B GSPO starter helpers (built from a data spec)."""

from __future__ import annotations

import logging
from typing import Any

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

QWEN35_08B_BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
QWEN35_08B_POST_TRAINED_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_08B_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
QWEN35_08B_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]


def validate_qwen3_5_config(config: Any) -> list[str]:
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


SPEC = StarterSpec(
    family_label="Qwen",
    display_name="Qwen3.5-0.8B",
    symbol_prefix="QWEN35_08B",
    fn_infix="qwen3_5",
    run_suffix="qwen3_5_0_8b",
    config_class_name="Qwen35Config",
    base_model=QWEN35_08B_BASE_MODEL,
    post_trained_model=QWEN35_08B_POST_TRAINED_MODEL,
    supported_variants=[QWEN35_08B_BASE_MODEL, QWEN35_08B_POST_TRAINED_MODEL],
    task_choices=QWEN35_08B_TASK_CHOICES,
    profile_choices=QWEN35_08B_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/qwen3_5_0_8b_gspo",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    profile_descriptions={
        "balanced": "Default first run with the standard Qwen 0.8B starter settings.",
        "memory": "Low-memory first run with 4-bit quantization and shorter context/group sizes.",
        "quality": "Heavier first run with larger context/group sizes when you have more headroom.",
    },
    profile_overrides={
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
    },
    system_prompt_intro="You are Qwen, an AI assistant created by Alibaba Cloud.",
    config_defaults={
        "lora_r": 32,
        "lora_alpha": 64,
        "max_new_tokens": 768,
        "max_prompt_length": 1024,
        "max_completion_length": 768,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_generations": 4,
        "learning_rate": 8e-6,
        "num_outer_iterations": 25,
        "generations_per_iteration": 32,
    },
    wandb_base_tags=("qwen3.5", "0.8b", "gspo"),
    wandb_project_default="qwen3_5_0_8b-gspo",
    validate=validate_qwen3_5_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS)
