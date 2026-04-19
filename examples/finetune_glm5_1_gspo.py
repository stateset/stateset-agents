"""Dedicated GLM 5.1 GSPO starter script.

Fine-tune GLM 5.1 (``zai-org/GLM-5.1``) with GSPO and emit vLLM-ready
serving artifacts. GLM 5.1 is a 754B-parameter MoE reasoning model, so
this script defaults to QLoRA + 4-bit quantization + vLLM-backed
generation. Even with these defaults, single-host runs require a
B200/H200-class node and the BF16 weights need pipeline parallelism
across two nodes for full-precision serving.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateset_agents.training.glm5_1_starter import (
    GLM5_1_BASE_MODEL,
    GLM5_1_FP8_MODEL,
    GLM5_1_STARTER_PROFILE_CHOICES,
    GLM5_1_TASK_CHOICES,
    create_glm5_1_preview,
    describe_glm5_1_starter_profiles,
    finetune_glm5_1,
    get_glm5_1_config,
    get_glm5_1_serving_recommendations,
    load_glm5_1_config_file,
    run_glm5_1_config,
    write_glm5_1_config_file,
)
from stateset_agents.training.serving_artifacts import build_serving_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_glm5_1_dry_run_payload(
    *,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    model_name: str = GLM5_1_BASE_MODEL,
    use_fp8_serving: bool = False,
    enable_auto_tool_choice: bool = True,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Return a serializable preview of the planned training and serving setup."""
    config = get_glm5_1_config(
        model_name=model_name,
        task=task,
        starter_profile=starter_profile,
        output_dir=output_dir,
    )
    preview = create_glm5_1_preview(config)

    serving_recommendations = get_glm5_1_serving_recommendations(
        use_fp8=use_fp8_serving,
        enable_auto_tool_choice=enable_auto_tool_choice,
    )
    serving_manifest = build_serving_manifest(
        config.output_dir,
        config.model_name,
        use_lora=config.use_lora,
        use_vllm=config.use_vllm,
        merged_model_dir=f"{config.output_dir}/merged" if config.use_lora else None,
        recommended=serving_recommendations,
    )
    preview["serving_manifest"] = serving_manifest
    preview["model_name"] = config.model_name
    return preview


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Dedicated GSPO starter for zai-org/GLM-5.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview the resolved config without loading a model
  python examples/finetune_glm5_1_gspo.py --dry-run

  # Use the low-memory starter preset for a tighter cluster
  python examples/finetune_glm5_1_gspo.py --starter-profile memory --dry-run

  # Preview against a private FP8 serving alias
  python examples/finetune_glm5_1_gspo.py --model your-org/GLM-5.1-FP8 --fp8-serving --dry-run

  # Save a reusable starter config
  python examples/finetune_glm5_1_gspo.py --write-config ./glm5_1.json

  # Run from a saved config (requires a multi-node training cluster)
  python examples/finetune_glm5_1_gspo.py --config ./glm5_1.json --no-dry-run
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a GLM 5.1 starter config file (JSON/YAML).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GLM5_1_BASE_MODEL,
        help=(
            "Model name. Use zai-org/GLM-5.1 for BF16 training or "
            "your-org/GLM-5.1-FP8 for FP8 serving previews."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=list(GLM5_1_TASK_CHOICES),
        help="Task preset to train against.",
    )
    parser.add_argument(
        "--starter-profile",
        type=str,
        default="balanced",
        choices=list(GLM5_1_STARTER_PROFILE_CHOICES),
        help="Starter profile: balanced, memory, or quality.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the built-in starter profile matrix as JSON and exit.",
    )
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--use-lora",
        dest="use_lora",
        action="store_true",
        default=None,
        help="Force LoRA on (the default).",
    )
    lora_group.add_argument(
        "--no-lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA. Not recommended — full FT of 754B is not feasible.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=None,
        help="Override the starter profile and force 4-bit quantization on.",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        default=None,
        help="Override the starter profile and force 8-bit quantization on.",
    )
    parser.add_argument(
        "--fp8-serving",
        action="store_true",
        help="Recommend FP8 single-host serving (your-org/GLM-5.1-FP8) in the manifest.",
    )
    parser.add_argument(
        "--disable-auto-tool-choice",
        action="store_true",
        help="Do not recommend vLLM auto tool choice in the serving manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory for checkpoints and the final adapter.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the number of outer GSPO iterations.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project name.",
    )
    parser.add_argument(
        "--write-config",
        type=str,
        help="Write the resolved starter config to JSON/YAML and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the resolved training config and exit.",
    )

    args = parser.parse_args()
    if args.list_profiles:
        if args.config is not None:
            parser.error("--list-profiles cannot be combined with --config")
        payload = describe_glm5_1_starter_profiles(task=args.task, model_name=args.model)
        print(json.dumps(payload, indent=2, default=str))
        return

    if args.config:
        conflicting = []
        if args.model != GLM5_1_BASE_MODEL:
            conflicting.append("--model")
        if args.task != "customer_service":
            conflicting.append("--task")
        if args.starter_profile != "balanced":
            conflicting.append("--starter-profile")
        if args.use_lora is not None:
            conflicting.append("--use-lora/--no-lora")
        if args.use_4bit is not None:
            conflicting.append("--use-4bit")
        if args.use_8bit is not None:
            conflicting.append("--use-8bit")
        if args.output_dir is not None:
            conflicting.append("--output-dir")
        if args.iterations is not None:
            conflicting.append("--iterations")
        if args.wandb:
            conflicting.append("--wandb")
        if args.wandb_project is not None:
            conflicting.append("--wandb-project")
        if conflicting:
            parser.error(
                "--config cannot be combined with starter override options: "
                + ", ".join(conflicting)
            )
        resolved_config = load_glm5_1_config_file(args.config)
    else:
        config_overrides: dict[str, Any] = {}
        if args.iterations is not None:
            config_overrides["num_outer_iterations"] = args.iterations
        resolved_config = get_glm5_1_config(
            model_name=args.model,
            task=args.task,
            starter_profile=args.starter_profile,
            use_lora=args.use_lora,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            **config_overrides,
        )

    if args.write_config:
        written_path = write_glm5_1_config_file(resolved_config, args.write_config)
        logger.info("Wrote GLM 5.1 config to %s", written_path)
        return

    if args.dry_run:
        preview = create_glm5_1_preview(resolved_config)
        serving_recommendations = get_glm5_1_serving_recommendations(
            use_fp8=args.fp8_serving or resolved_config.model_name == GLM5_1_FP8_MODEL,
            enable_auto_tool_choice=not args.disable_auto_tool_choice,
        )
        preview["serving_manifest"] = build_serving_manifest(
            resolved_config.output_dir,
            resolved_config.model_name,
            use_lora=resolved_config.use_lora,
            use_vllm=resolved_config.use_vllm,
            merged_model_dir=(
                f"{resolved_config.output_dir}/merged"
                if resolved_config.use_lora
                else None
            ),
            recommended=serving_recommendations,
        )
        preview["model_name"] = resolved_config.model_name
        print(json.dumps(preview, indent=2, default=str))
        return

    asyncio.run(run_glm5_1_config(resolved_config, dry_run=False))
    logger.info("GLM 5.1 starter run complete")


# Re-export the async finetune helper for direct programmatic use.
__all__ = [
    "create_glm5_1_dry_run_payload",
    "finetune_glm5_1",
    "main",
]


if __name__ == "__main__":
    main()
