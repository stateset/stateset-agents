"""Packaged Inkling 2 Mini GSPO starter helpers.

Inkling 2 Mini (``thinkingmachines/Inkling-2-Mini``, Apache-2.0, released
2026-11-20) is a multimodal looped Mixture-of-Experts model
(``model_type: inkling2_loop``, ``Inkling2LoopForConditionalGeneration``):
26.4B parameters, 24 decoder layers with layers 6-17 run three times, hidden
size 2304, 128 routed experts with 8 active per token, a 201024-token
vocabulary, and up to a 1M-token context window.

Evidence scope: this starter is a configuration integration audited against
the published config and safetensors weight map. No Inkling 2 Mini training
run has been executed with it. In particular:

* The only released checkpoint stores the MoE expert weights in NVFP4 (NVIDIA
  ModelOpt, ``quant_method: modelopt``). There is no BF16 repository, so the
  starter keeps the experts frozen and adapts the attention projections only,
  and it leaves bitsandbytes quantization off (the checkpoint is already
  quantized).
* The architecture needs a Transformers runtime that ships
  ``Inkling2LoopForConditionalGeneration``; the model card does not state a
  minimum version.
* Whether the attention projections are plain linear layers that PEFT can
  wrap is unverified until the model loads in such a runtime.

The attention does not use llama-style ``q_proj``/``k_proj``/``v_proj``
modules; the LoRA targets below use the checkpoint's actual projection names
(``wq_du``/``wk_dv``/``wv_dv``/``wo_ud``), taken from the model's safetensors
weight map. Only the text modality is covered.

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

INKLING2_MINI_BASE_MODEL = "thinkingmachines/Inkling-2-Mini"
INKLING2_MINI_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
INKLING2_MINI_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]
INKLING2_MINI_EXPERT_MODULES = ("w13_weight", "w2_weight")
"""Expert tensors stored in NVFP4 in the released checkpoint; never LoRA targets."""


def validate_inkling2_mini_config(config: Any) -> list[str]:
    """Validate an Inkling 2 Mini first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in INKLING2_MINI_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in INKLING2_MINI_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    model_name_lower = config.model_name.lower()
    if "inkling" not in model_name_lower:
        warnings.append("model_name does not look like an Inkling checkpoint")
    elif "inkling-2-mini" not in model_name_lower:
        warnings.append(
            "this helper is tuned for thinkingmachines/Inkling-2-Mini; "
            "verify overrides carefully"
        )
    if "gguf" in model_name_lower:
        warnings.append(
            "GGUF repos are inference-only; post-train "
            "thinkingmachines/Inkling-2-Mini instead"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Inkling 2 Mini GSPO run")
    if config.learning_rate < 5e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append(
            "per-device batch size above 2 is likely to OOM on Inkling 2 Mini"
        )
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append(
            "LoRA is mandatory for Inkling 2 Mini: the released expert weights "
            "are NVFP4-quantized and cannot be fully fine-tuned"
        )
    if config.use_4bit or config.use_8bit:
        warnings.append(
            "the released checkpoint is already NVFP4-quantized; bitsandbytes "
            "4-bit/8-bit loading cannot be stacked on top of it"
        )
    expert_targets = sorted(
        set(config.lora_target_modules) & set(INKLING2_MINI_EXPERT_MODULES)
    )
    if expert_targets:
        warnings.append(
            "lora_target_modules includes NVFP4 expert tensors "
            f"({', '.join(expert_targets)}); adapt attention projections only"
        )
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


SPEC = StarterSpec(
    family_label="Inkling 2",
    display_name="Inkling 2 Mini",
    symbol_prefix="INKLING2_MINI",
    fn_infix="inkling2_mini",
    run_suffix="inkling2_mini",
    config_class_name="Inkling2MiniConfig",
    base_model=INKLING2_MINI_BASE_MODEL,
    post_trained_model=None,
    supported_variants=["thinkingmachines/Inkling-2-Mini"],
    task_choices=INKLING2_MINI_TASK_CHOICES,
    profile_choices=INKLING2_MINI_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/inkling2_mini_gspo",
    lora_target_modules=["wq_du", "wk_dv", "wv_dv", "wo_ud"],
    profile_descriptions={
        "balanced": "First Inkling 2 Mini run with attention-only LoRA on the released NVFP4 checkpoint and a moderate context budget.",
        "memory": "Lower-memory Inkling 2 Mini run with smaller rollout groups and shorter context for a single 32GB-class device.",
        "quality": "Heavier Inkling 2 Mini run with larger context and rollout sizes when you have more headroom.",
    },
    profile_overrides={
        "balanced": {},
        "memory": {
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
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "num_generations": 6,
            "num_outer_iterations": 24,
            "generations_per_iteration": 16,
            "max_new_tokens": 2048,
            "max_prompt_length": 16384,
            "max_completion_length": 2048,
            "learning_rate": 2e-06,
        },
    },
    system_prompt_intro=(
        "You are Inkling, an AI assistant created by Thinking Machines Lab."
    ),
    config_defaults={
        "lora_r": 32,
        "lora_alpha": 64,
        "use_4bit": False,
        "use_8bit": False,
        "trust_remote_code": False,
        "max_new_tokens": 1024,
        "max_prompt_length": 4096,
        "max_completion_length": 1024,
        "temperature": 1.0,
        "top_p": 1.0,
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
    wandb_base_tags=("inkling-2-mini", "moe", "gspo"),
    wandb_project_default="inkling2_mini-gspo",
    validate=validate_inkling2_mini_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + ["INKLING2_MINI_EXPERT_MODULES"]
