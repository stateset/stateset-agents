"""
Fine-tune Qwen3.5-27B with GSPO and emit vLLM-ready serving artifacts.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.finetune_qwen3_gspo import (
    QWEN35_27B_MODEL,
    finetune_qwen3,
    get_qwen3_config,
    get_qwen3_serving_recommendations,
)
from stateset_agents.training.serving_artifacts import build_serving_manifest


def create_qwen3_5_27b_preview(
    *,
    task: str = "customer_service",
    use_lora: bool = True,
    use_4bit: bool = False,
    use_8bit: bool = False,
    use_vllm: bool = True,
    language_model_only: bool = True,
    enable_auto_tool_choice: bool = True,
    output_dir: str = "./outputs/qwen3_5_27b_gspo",
) -> dict[str, Any]:
    """Return a serializable preview of the planned training and serving setup."""
    gspo_config = get_qwen3_config(
        model_name=QWEN35_27B_MODEL,
        task=task,
        use_lora=use_lora,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        use_vllm=use_vllm,
        output_dir=output_dir,
    )
    serving_manifest = build_serving_manifest(
        output_dir,
        QWEN35_27B_MODEL,
        use_lora=use_lora,
        use_vllm=use_vllm,
        merged_model_dir=f"{output_dir}/merged" if use_lora else None,
        recommended=get_qwen3_serving_recommendations(
            language_model_only=language_model_only,
            enable_auto_tool_choice=enable_auto_tool_choice,
        ),
    )
    return {
        "model_name": QWEN35_27B_MODEL,
        "task": task,
        "gspo_overrides": {
            "use_lora": gspo_config.use_lora,
            "use_vllm": gspo_config.use_vllm,
            "num_generations": gspo_config.num_generations,
            "learning_rate": gspo_config.learning_rate,
            "num_outer_iterations": gspo_config.num_outer_iterations,
            "generations_per_iteration": gspo_config.generations_per_iteration,
            "max_prompt_length": gspo_config.max_prompt_length,
            "max_completion_length": gspo_config.max_completion_length,
            "temperature": gspo_config.temperature,
            "top_p": gspo_config.top_p,
        },
        "warnings": gspo_config.validate(),
        "serving_manifest": serving_manifest,
    }


async def finetune_qwen3_5_27b(
    *,
    task: str = "customer_service",
    use_lora: bool = True,
    use_4bit: bool = False,
    use_8bit: bool = False,
    use_vllm: bool = True,
    export_merged: bool = True,
    language_model_only: bool = True,
    enable_auto_tool_choice: bool = True,
    output_dir: str = "./outputs/qwen3_5_27b_gspo",
    use_wandb: bool = False,
    wandb_project: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any] | Any:
    """Dedicated training entrypoint for Qwen3.5-27B."""
    if dry_run:
        return create_qwen3_5_27b_preview(
            task=task,
            use_lora=use_lora,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            use_vllm=use_vllm,
            language_model_only=language_model_only,
            enable_auto_tool_choice=enable_auto_tool_choice,
            output_dir=output_dir,
        )

    return await finetune_qwen3(
        model_name=QWEN35_27B_MODEL,
        task=task,
        use_lora=use_lora,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        use_vllm=use_vllm,
        export_merged=export_merged,
        language_model_only=language_model_only,
        enable_auto_tool_choice=enable_auto_tool_choice,
        output_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5-27B with GSPO and emit vLLM-ready artifacts.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=["customer_service", "technical_support", "sales"],
        help="Task type",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning is expensive for 27B models)",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization during training",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization during training",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM-backed generation during training",
    )
    parser.add_argument(
        "--no-export-merged",
        action="store_true",
        help="Skip merging LoRA adapters into a standalone serving checkpoint",
    )
    parser.add_argument(
        "--vision-serving",
        action="store_true",
        help="Recommend multimodal serving instead of language-model-only mode",
    )
    parser.add_argument(
        "--disable-auto-tool-choice",
        action="store_true",
        help="Do not recommend vLLM auto tool choice in the serving manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/qwen3_5_27b_gspo",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved GSPO and serving configuration as JSON",
    )

    args = parser.parse_args()

    use_lora = args.use_lora and not args.no_lora
    use_vllm = not args.no_vllm
    export_merged = not args.no_export_merged
    language_model_only = not args.vision_serving

    result = asyncio.run(
        finetune_qwen3_5_27b(
            task=args.task,
            use_lora=use_lora,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            use_vllm=use_vllm,
            export_merged=export_merged,
            language_model_only=language_model_only,
            enable_auto_tool_choice=not args.disable_auto_tool_choice,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            dry_run=args.dry_run,
        )
    )

    if args.dry_run:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
