"""Packaged Qwen3.8 27B GSPO starter helpers.

Qwen3.8 27B (``Qwen/Qwen3.8-27B``) is Alibaba's open model released
2026-08-05 under Apache-2.0: a 27.8B-parameter multimodal LM
(``model_type: qwen3_5``, architecture ``Qwen3_5ForConditionalGeneration``)
pairing a vision tower with a 64-layer text stack (hidden 5120, 24 heads /
4 KV heads, 248320-token vocabulary, 262144 max positions = 256K context).
BF16 weights are roughly 56GB, so budget ~160GB of disk and either an 80GB
card or ``--gpu-count 2``. The custom architecture requires
``trust_remote_code=True``; the presets below target QLoRA post-training of
the text stack.

**Why these LoRA targets.** The text stack uses *hybrid* attention,
confirmed against the published weight map:

- a minority of layers use standard attention (``self_attn``: ``q_proj``,
  ``k_proj``, ``v_proj``, ``o_proj`` — 96 tensors);
- most layers use Mamba-style linear attention (``linear_attn``:
  ``in_proj_qkv``, ``in_proj_a``, ``in_proj_b``, ``in_proj_z``,
  ``out_proj``, ``conv1d`` — 432 tensors);
- every layer has an MLP (``gate_proj``, ``up_proj``, ``down_proj`` — 192
  tensors).

Listing only the llama-style names would silently adapt just the minority
standard-attention layers, so ``QWEN38_27B_LORA_TARGET_MODULES`` covers all
three groups. ``conv1d`` is left out because LoRA's low-rank decomposition
targets ``nn.Linear``. The vision tower (``model.visual.*``) is excluded:
text-only SFT sends it no gradient. Note that ``out_proj`` appears in both
stacks; peft matches by leaf name, so adapting the text copies necessarily
reaches the vision ones too.

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

QWEN38_27B_BASE_MODEL = "Qwen/Qwen3.8-27B"
QWEN38_27B_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
QWEN38_27B_STARTER_PROFILE_CHOICES = ["balanced", "memory", "quality"]


def validate_qwen3_8_config(config: Any) -> list[str]:
    """Validate a Qwen3.8 27B first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in QWEN38_27B_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in QWEN38_27B_TASK_CHOICES:
        warnings.append(
            "task is outside the built-in starter presets; default environment fallbacks may be used"
        )
    if "qwen" not in config.model_name.lower():
        warnings.append("model_name does not look like a Qwen checkpoint")
    if "qwen3.8-27b" not in config.model_name.lower():
        warnings.append(
            "this helper is tuned for Qwen/Qwen3.8-27B; verify overrides carefully"
        )
    if "fp8" in config.model_name.lower():
        warnings.append(
            "the FP8 variant is inference-oriented; post-train the BF16 checkpoint instead"
        )
    if config.learning_rate > 1e-5:
        warnings.append("learning rate is high for a first Qwen3.8 27B GSPO run")
    if config.learning_rate < 1e-7:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 2:
        warnings.append("per-device batch size above 2 may increase OOM risk")
    if config.get_effective_batch_size() < 8:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Qwen3.8 27B run")
    if config.max_prompt_length > 32768:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 4096:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


SPEC = StarterSpec(
    family_label="Qwen3.8 27B",
    display_name="Qwen3.8 27B",
    symbol_prefix="QWEN38_27B",
    fn_infix="qwen3_8",
    run_suffix="qwen3_8",
    config_class_name="Qwen38Config",
    base_model=QWEN38_27B_BASE_MODEL,
    post_trained_model=None,
    supported_variants=["Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"],
    task_choices=QWEN38_27B_TASK_CHOICES,
    profile_choices=QWEN38_27B_STARTER_PROFILE_CHOICES,
    default_output_dir="./outputs/qwen3_8_27b_gspo",
    lora_target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    profile_descriptions={
        "balanced": "Default Qwen3.8 27B first run with QLoRA-friendly settings and a moderate context budget.",
        "memory": "Lower-memory Qwen3.8 27B first run with smaller rollout groups and shorter context.",
        "quality": "Heavier Qwen3.8 27B first run with larger context and rollout sizes when you have more headroom.",
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
    wandb_base_tags=("qwen3-8-27b", "gspo"),
    wandb_project_default="qwen3_8_27b-gspo",
    validate=validate_qwen3_8_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + []
