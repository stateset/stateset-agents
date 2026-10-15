"""Text-only GSPO starters for Gemma 4 E2B and E4B.

These are conservative starting points, not benchmark-tuned hyperparameters.
The E sizes describe effective parameters; embeddings add substantial memory.
"""

from __future__ import annotations

import logging
from typing import Any

from stateset_agents.training import starter_common
from stateset_agents.training.starter_factory import (
    StarterSpec,
    build_starter,
    starter_all,
)

logger = logging.getLogger(__name__)

GEMMA4_SMALL_MODELS = ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"]


def validate_gemma4_small_config(config: Any) -> list[str]:
    """Report dependency and memory considerations for small Gemma runs."""
    warnings: list[str] = []
    version = starter_common.get_transformers_version()
    if version is None or version < (5, 5, 0):
        warnings.append(
            "Gemma 4 requires transformers>=5.5.0; install stateset-agents[gemma4]."
        )
    if config.model_name not in GEMMA4_SMALL_MODELS:
        warnings.append("This starter targets Gemma 4 E2B/E4B instruction checkpoints.")
    if not config.use_lora and (config.use_4bit or config.use_8bit):
        raise ValueError("Quantized Gemma training requires use_lora=True.")
    if config.per_device_train_batch_size > 1:
        warnings.append("Start with batch size 1 before increasing GPU memory usage.")
    if config.max_prompt_length + config.max_completion_length > 4096:
        warnings.append(
            "Long rollouts increase training memory; start with shorter context."
        )
    return warnings


SPEC = StarterSpec(
    family_label="Gemma",
    display_name="Gemma 4 E2B/E4B",
    symbol_prefix="GEMMA4_SMALL",
    fn_infix="gemma4_small",
    run_suffix="gemma4_small",
    config_class_name="Gemma4SmallConfig",
    base_model=GEMMA4_SMALL_MODELS[0],
    supported_variants=GEMMA4_SMALL_MODELS,
    default_output_dir="./outputs/gemma4_small_gspo",
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    profile_descriptions={
        "balanced": "Text-only LoRA with batch size 1 and short rollouts.",
        "memory": "QLoRA with 4-bit weights, two generations, and shorter context.",
        "quality": "Longer text rollouts and rank-32 LoRA when memory permits.",
    },
    profile_overrides={
        "balanced": {},
        "memory": {
            "use_4bit": True,
            "num_generations": 2,
            "max_prompt_length": 512,
            "max_completion_length": 256,
            "max_new_tokens": 256,
            "generations_per_iteration": 4,
        },
        "quality": {
            "lora_r": 32,
            "lora_alpha": 64,
            "max_prompt_length": 2048,
            "max_completion_length": 1024,
            "max_new_tokens": 1024,
        },
    },
    system_prompt_intro="You are a helpful assistant built from Google Gemma 4.",
    config_defaults={
        "lora_r": 16,
        "lora_alpha": 32,
        "max_new_tokens": 512,
        "max_prompt_length": 1024,
        "max_completion_length": 512,
        "temperature": 1.0,
        "top_p": 0.95,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_generations": 4,
        "learning_rate": 5e-6,
        "num_outer_iterations": 16,
        "generations_per_iteration": 8,
        "trust_remote_code": False,
    },
    agent_config_kwargs={"tokenizer_kwargs": {"padding_side": "left"}},
    wandb_base_tags=("gemma4", "small", "gspo"),
    wandb_project_default="gemma4-small-gspo",
    validate=validate_gemma4_small_config,
    module=__name__,
)

_SYMBOLS = build_starter(SPEC, logger)
globals().update(_SYMBOLS)

__all__ = starter_all(_SYMBOLS) + ["GEMMA4_SMALL_MODELS"]
