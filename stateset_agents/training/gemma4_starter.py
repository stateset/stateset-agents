"""Packaged Gemma 4 31B GSPO starter helpers.

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

GEMMA4_31B_BASE_MODEL = "google/gemma-4-31B-it"
GEMMA4_31B_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
GEMMA4_31B_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]


def validate_gemma4_31b_config(config: Any) -> list[str]:
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


SPEC = StarterSpec(
    family_label="Gemma",
    display_name="Gemma 4 31B",
    symbol_prefix="GEMMA4_31B",
    fn_infix="gemma4_31b",
    run_suffix="gemma4_31b",
    config_class_name="Gemma4Config",
    base_model=GEMMA4_31B_BASE_MODEL,
    post_trained_model=None,
    supported_variants=["google/gemma-4-31B-it"],
    task_choices=GEMMA4_31B_TASK_CHOICES,
    profile_choices=GEMMA4_31B_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/gemma4_31b_gspo",
    lora_target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    profile_descriptions={
        "balanced": "First Gemma 4 31B run with QLoRA defaults, 4-bit quantization, and a moderate context budget.",
        "memory": "Lower-memory Gemma 4 31B run with smaller groups and shorter context for tighter GPUs.",
        "quality": "Heavier Gemma 4 31B run with larger context and rollout sizes when you have more headroom.",
    },
    profile_overrides={
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
            "learning_rate": 2e-06,
        },
    },
    system_prompt_intro="You are Gemma, a helpful AI assistant built from the Gemma 4 31B instruction-tuned checkpoint by Google DeepMind.",
    config_defaults={
        "lora_r": 64,
        "lora_alpha": 128,
        "use_4bit": True,
        "max_new_tokens": 1024,
        "max_prompt_length": 4096,
        "max_completion_length": 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_generations": 4,
        "learning_rate": 3e-06,
        "num_outer_iterations": 20,
        "generations_per_iteration": 12,
        "clip_range_left": 0.0002,
        "clip_range_right": 0.0003,
    },
    extra_fields=(),
    agent_config_kwargs={"tokenizer_kwargs": {"padding_side": "left"}},
    wandb_base_tags=("gemma4", "31b", "gspo"),
    wandb_project_default="gemma4_31b-gspo",
    validate=validate_gemma4_31b_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + []
