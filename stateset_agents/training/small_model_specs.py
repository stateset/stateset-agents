"""Shared conservative defaults for small-model family starters."""

from __future__ import annotations

from typing import Any

from stateset_agents.core.model_presets import PRESETS
from stateset_agents.training import starter_common
from stateset_agents.training.starter_factory import StarterSpec


def small_model_spec(module: str, infix: str, display: str) -> StarterSpec:
    """Build a family spec using the registry's checkpoint and adapter metadata."""
    variants = [
        p for p in PRESETS.values() if p.starter_module == module.rsplit(".", 1)[-1]
    ]
    first = variants[0]
    model_ids = [p.model_id for p in variants]

    def validate(config: Any) -> list[str]:
        warnings: list[str] = []
        version = starter_common.get_transformers_version()
        if version is None or version < (5, 5, 0):
            warnings.append(
                "Install stateset-agents[small-models] (transformers>=5.5.0)."
            )
        if config.model_name not in model_ids:
            warnings.append(
                "This checkpoint is outside the family's built-in variants."
            )
        if not config.use_lora and (config.use_4bit or config.use_8bit):
            raise ValueError("Quantized training requires use_lora=True.")
        if infix == "lfm2_5":
            warnings.append(
                "Experimental LFM2.5 uses the custom LFM license and always emits reasoning; "
                "allow enough completion tokens and evaluate final-answer quality."
            )
        if infix == "deepseek_r1_small":
            warnings.append(
                "DeepSeek R1 distills need room for reasoning before final answers; "
                "measure completion truncation and increase the token budget as needed."
            )
        if infix == "llama3_2_small":
            warnings.append(
                "Llama 3.2 checkpoints require acceptance of the Llama community "
                "license and Hugging Face access."
            )
        return warnings

    completion = first.max_completion_length
    reasoning = infix in {"lfm2_5", "deepseek_r1_small"}
    return StarterSpec(
        family_label=display,
        display_name=display,
        symbol_prefix=infix.upper(),
        fn_infix=infix,
        run_suffix=infix,
        config_class_name="".join(part.title() for part in infix.split("_")) + "Config",
        base_model=first.model_id,
        supported_variants=model_ids,
        default_output_dir=first.cli_default_output_dir or f"./outputs/{infix}_gspo",
        lora_target_modules=list(first.lora_target_modules),
        profile_descriptions={
            "balanced": "BF16 LoRA, batch size 1, and conservative rollout budgets.",
            "memory": "4-bit NF4 QLoRA, two generations, and shorter prompts.",
            "quality": "Rank-32 LoRA and longer context when memory permits.",
        },
        profile_overrides={
            "balanced": {},
            "memory": {
                "use_4bit": True,
                "num_generations": 2,
                "max_prompt_length": 512,
                "max_completion_length": completion if reasoning else 256,
                "max_new_tokens": completion if reasoning else 256,
                "generations_per_iteration": 4,
            },
            "quality": {
                "lora_r": 32,
                "lora_alpha": 64,
                "max_prompt_length": 2048,
                "max_completion_length": completion * 2 if reasoning else 1024,
                "max_new_tokens": completion * 2 if reasoning else 1024,
            },
        },
        system_prompt_intro=f"You are a helpful assistant built from {display}.",
        config_defaults={
            "lora_r": 16,
            "lora_alpha": 32,
            "max_new_tokens": completion,
            "max_prompt_length": first.max_prompt_length,
            "max_completion_length": completion,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_generations": first.num_generations,
            "learning_rate": first.learning_rate,
            "num_outer_iterations": 16,
            "generations_per_iteration": 8,
            "trust_remote_code": False,
        },
        agent_config_kwargs={"tokenizer_kwargs": {"padding_side": "left"}},
        wandb_base_tags=(infix, "gspo"),
        wandb_project_default=f"{infix}-gspo",
        validate=validate,
        module=module,
    )
