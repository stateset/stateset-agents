"""Packaged Kimi-K3 GSPO starter helpers.

The released Kimi-K3 checkpoint uses hybrid KDA/MLA attention and native
MXFP4 experts. Targets below match its pinned index, replacing the provisional
K2.6 copy. Training remains blocked pending a validated packed-weight loader
and KDA backward-kernel contract; use ``frontier-models`` to inspect readiness.

Built from a :class:`StarterSpec`; see ``starter_factory``.
"""

from __future__ import annotations

import logging
from typing import Any

from stateset_agents.core.frontier_models import get_frontier_model
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
_KIMI = get_frontier_model("kimi-k3")

KIMI_K3_BASE_MODEL = "moonshotai/Kimi-K3"
KIMI_K3_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
KIMI_K3_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]


def validate_kimi_k3_config(config: Any) -> list[str]:
    """Validate a Kimi-K3 first-run configuration."""
    warnings: list[str] = list(_KIMI.blockers)
    if config.use_4bit or config.use_8bit:
        warnings.append(
            "bitsandbytes re-quantization is unsupported for native MXFP4 Kimi-K3"
        )

    if config.starter_profile not in KIMI_K3_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in KIMI_K3_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    if "kimi" not in config.model_name.lower():
        warnings.append("model_name does not look like a Kimi checkpoint")
    if "kimi-k3" not in config.model_name.lower():
        warnings.append(
            "this helper is tuned for moonshotai/Kimi-K3; verify overrides carefully"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Kimi-K3 GSPO run")
    if config.learning_rate < 1e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append("per-device batch size above 2 may increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Kimi-K3 run")
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


SPEC = StarterSpec(
    family_label="Kimi",
    display_name="Kimi-K3",
    symbol_prefix="KIMI_K3",
    fn_infix="kimi_k3",
    run_suffix="kimi_k3",
    config_class_name="KimiK3Config",
    base_model=KIMI_K3_BASE_MODEL,
    post_trained_model=None,
    supported_variants=["moonshotai/Kimi-K3"],
    task_choices=KIMI_K3_TASK_CHOICES,
    profile_choices=KIMI_K3_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/kimi_k3_gspo",
    lora_target_modules=list(_KIMI.attention_targets),
    profile_descriptions={
        "balanced": "Default Kimi-K3 first run configuration preview; released MXFP4 training loader is blocked.",
        "memory": "Lower-memory Kimi-K3 first run with smaller rollout groups and shorter context.",
        "quality": "Heavier Kimi-K3 first run with larger context and rollout sizes when you have more headroom.",
    },
    profile_overrides={
        "balanced": {"use_4bit": False},
        "memory": {
            "use_4bit": False,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 24,
            "num_generations": 2,
            "num_outer_iterations": 10,
            "generations_per_iteration": 8,
            "max_new_tokens": 768,
            "max_prompt_length": 2048,
            "max_completion_length": 768,
            "learning_rate": 2e-06,
        },
        "quality": {
            "use_4bit": False,
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
    system_prompt_intro="You are Kimi, an AI assistant created by Moonshot AI.",
    config_defaults={
        # Non-native architecture: the HF repo ships its modelling code.
        "trust_remote_code": True,
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
    extra_fields=(("model_revision", "str | None", _KIMI.revision),),
    wandb_base_tags=("kimi-k3", "gspo"),
    wandb_project_default="kimi_k3-gspo",
    validate=validate_kimi_k3_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + []
