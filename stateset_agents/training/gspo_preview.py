"""Shared GSPO preset configuration and previews without loading model weights."""

from __future__ import annotations

import argparse
import json
from typing import Any

from stateset_agents.core.model_presets import (
    ModelPreset,
    get_preset,
    list_preset_names,
)


def build_gspo_config(
    preset: ModelPreset,
    *,
    task: str,
    output_dir: str,
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_vllm: bool = False,
    learning_rate: float | None = None,
    epochs: int | None = None,
    steps: int | None = None,
    use_wandb: bool = False,
    wandb_project: str | None = None,
) -> Any:
    """Build a ``GSPOConfig`` reproducing the preset's hyperparameters,
    with optional CLI overrides layered on top."""
    from stateset_agents.training.gspo_config import GSPOConfig

    if not preset.supports_gspo:
        raise ValueError(
            "This model supports SFT only; use stateset_agents.training.sft."
        )
    if preset.model_id.startswith("meta-models/Muse-Spark-Open-") and (
        use_4bit or use_8bit
    ):
        raise ValueError(
            "Muse Spark's BF16 text RL preset does not certify 4/8-bit training; "
            "expert tensors are not ordinary quantized Linear modules."
        )

    kwargs: dict[str, Any] = {
        "model_name": preset.model_id,
        "model_revision": preset.model_revision,
        "run_name": f"{preset.model_id}-{task}",
        "trust_remote_code": preset.trust_remote_code,
        "learning_rate": (
            learning_rate if learning_rate is not None else preset.learning_rate
        ),
        "num_generations": preset.num_generations,
        "max_prompt_length": preset.max_prompt_length,
        "max_completion_length": preset.max_completion_length,
        "bf16": preset.bf16,
        "use_lora": use_lora if use_lora is not None else True,
        "lora_target_modules": list(preset.lora_target_modules),
        "use_4bit": use_4bit if use_4bit is not None else preset.use_4bit,
        "use_8bit": use_8bit if use_8bit is not None else preset.use_8bit,
        "use_vllm": use_vllm,
        "output_dir": output_dir,
    }
    if epochs is not None:
        kwargs["num_train_epochs"] = epochs
    if steps is not None:
        kwargs["max_steps"] = steps
    if use_wandb:
        kwargs["report_to"] = "wandb"
        kwargs["wandb_project"] = (
            wandb_project or f"{preset.model_id.split('/')[-1]}-gspo-{task}"
        )
        kwargs["wandb_tags"] = ["gspo", task, preset.model_id.split("/")[-1]]
    else:
        kwargs["report_to"] = "none"
    return GSPOConfig(**kwargs)


def preview_payload(
    preset_name: str, preset: ModelPreset, gspo_config: Any
) -> dict[str, Any]:
    """Describe resolved settings, including caller overrides and CPU fallbacks."""
    return {
        "preset": preset_name,
        "model_id": preset.model_id,
        "tokenizer_id": preset.tokenizer_id,
        "lora_target_modules": list(gspo_config.lora_target_modules or []),
        "max_prompt_length": gspo_config.max_prompt_length,
        "max_completion_length": gspo_config.max_completion_length,
        "learning_rate": gspo_config.learning_rate,
        "num_generations": gspo_config.num_generations,
        "bf16": gspo_config.bf16,
        "use_4bit": gspo_config.use_4bit,
        "use_8bit": gspo_config.use_8bit,
        "output_dir": getattr(gspo_config, "output_dir", None),
        "notes": preset.notes,
    }


def main(argv: list[str] | None = None) -> int:
    """Print a preset configuration without downloading weights or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list_preset_names(), required=True)
    parser.add_argument("--task", default="customer_service")
    parser.add_argument("--output-dir", default="outputs/gspo")
    args = parser.parse_args(argv)
    preset = get_preset(args.model)
    try:
        config = build_gspo_config(preset, task=args.task, output_dir=args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(preview_payload(args.model, preset, config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
