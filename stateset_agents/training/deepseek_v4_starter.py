"""Packaged DeepSeek V4 Flash GSPO starter helpers.

DeepSeek V4 Flash (``deepseek-ai/DeepSeek-V4-Flash``) is a large
Mixture-of-Experts model (``model_type: deepseek_v4``,
``DeepseekV4ForCausalLM``): 43 layers, hidden size 4096, Multi-head Latent
Attention (64 heads, 1 KV latent head), 256 routed experts with 6 active per
token, a 129280-token vocabulary, and up to 1M max position embeddings.
Weights are published on HuggingFace under MIT. The NVFP4/FP8 repos are
inference-only. This is a large MoE, so this starter assumes:

* QLoRA-only fine-tuning on the MLA projection matrices
* vLLM-backed generation during training
* Multi-node (or very-high-memory single-node) topology

The MLA attention does not use llama-style ``q_proj``/``k_proj``/``v_proj``
modules; the LoRA targets below use the checkpoint's actual MLA projection
names (``wq_a``/``wq_b``/``wkv``/``wo_a``/``wo_b``), verified against the
model's safetensors weight map.

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

DEEPSEEK_V4_BASE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_V4_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
DEEPSEEK_V4_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]
DEEPSEEK_V4_FLASH_BASE_MODEL = "deepseek-ai/DeepSeek-V4-Flash-Base"


def validate_deepseek_v4_config(config: Any) -> list[str]:
    """Validate a DeepSeek V4 Flash first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in DEEPSEEK_V4_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in DEEPSEEK_V4_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    model_name_lower = config.model_name.lower()
    if "deepseek" not in model_name_lower:
        warnings.append("model_name does not look like a DeepSeek checkpoint")
    if "v4-flash" not in model_name_lower:
        warnings.append(
            "this helper is tuned for deepseek-ai/DeepSeek-V4-Flash; "
            "verify overrides carefully"
        )
    if "nvfp4" in model_name_lower or "fp8" in model_name_lower:
        warnings.append(
            "the NVFP4/FP8 repos are inference-only; post-train the BF16 "
            "deepseek-ai/DeepSeek-V4-Flash checkpoint instead"
        )
    if config.learning_rate > 5e-6:
        warnings.append("learning rate is high for a first DeepSeek V4 Flash GSPO run")
    if config.learning_rate < 5e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 1:
        warnings.append(
            "per-device batch size above 1 is almost certainly going to OOM on DeepSeek V4 Flash"
        )
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append(
            "LoRA is mandatory for first DeepSeek V4 Flash runs (large MoE, full FT not feasible)"
        )
    if not config.use_4bit and not config.use_8bit:
        warnings.append(
            "DeepSeek V4 Flash needs 4-bit quantization for any single-node fine-tuning attempt"
        )
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if not config.use_vllm:
        warnings.append(
            "vLLM-backed generation is strongly recommended for DeepSeek V4 Flash to keep rollouts tractable"
        )
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


SPEC = StarterSpec(
    family_label="DeepSeek V4",
    display_name="DeepSeek V4 Flash",
    symbol_prefix="DEEPSEEK_V4",
    fn_infix="deepseek_v4",
    run_suffix="deepseek_v4",
    config_class_name="DeepseekV4Config",
    base_model=DEEPSEEK_V4_BASE_MODEL,
    post_trained_model=None,
    supported_variants=[
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Flash-Base",
    ],
    task_choices=DEEPSEEK_V4_TASK_CHOICES,
    profile_choices=DEEPSEEK_V4_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/deepseek_v4_gspo",
    lora_target_modules=["wq_a", "wq_b", "wkv", "wo_a", "wo_b"],
    profile_descriptions={
        "balanced": "First DeepSeek V4 Flash run with QLoRA defaults, 4-bit quantization, and a moderate context budget.",
        "memory": "Lower-memory DeepSeek V4 Flash run with smaller groups and shorter context for tighter multi-node clusters.",
        "quality": "Heavier DeepSeek V4 Flash run with larger context and rollout sizes when you have B200/H200 headroom.",
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
    system_prompt_intro="You are DeepSeek, an AI assistant created by DeepSeek.",
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
    wandb_base_tags=("deepseek-v4", "moe", "gspo"),
    wandb_project_default="deepseek_v4-gspo",
    validate=validate_deepseek_v4_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + ["DEEPSEEK_V4_FLASH_BASE_MODEL"]
