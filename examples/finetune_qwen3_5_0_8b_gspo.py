"""Dedicated Qwen3.5-0.8B GSPO starter script."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateset_agents.training.qwen3_5_starter import (
    QWEN35_08B_BASE_MODEL,
    QWEN35_08B_STARTER_PROFILE_CHOICES,
    QWEN35_08B_TASK_CHOICES,
    create_qwen3_5_preview,
    describe_qwen3_5_starter_profiles,
    finetune_qwen3_5_0_8b,
    get_qwen3_5_config,
    load_qwen3_5_config_file,
    run_qwen3_5_0_8b_config,
    write_qwen3_5_config_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Dedicated GSPO starter for Qwen/Qwen3.5-0.8B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview the resolved config without loading a model
  python examples/finetune_qwen3_5_0_8b_gspo.py --dry-run

  # Use the low-memory starter preset for a small GPU
  python examples/finetune_qwen3_5_0_8b_gspo.py --starter-profile memory --dry-run

  # Save a reusable starter config
  python examples/finetune_qwen3_5_0_8b_gspo.py --write-config ./qwen3_5_0_8b.json

  # Run from a saved config
  python examples/finetune_qwen3_5_0_8b_gspo.py --config ./qwen3_5_0_8b.json --no-dry-run
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a Qwen/Qwen3.5-0.8B starter config file (JSON/YAML).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=QWEN35_08B_BASE_MODEL,
        help="Model name. For post-training, prefer Qwen/Qwen3.5-0.8B-Base.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=list(QWEN35_08B_TASK_CHOICES),
        help="Task preset to train against.",
    )
    parser.add_argument(
        "--starter-profile",
        type=str,
        default="balanced",
        choices=list(QWEN35_08B_STARTER_PROFILE_CHOICES),
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
        help="Force LoRA on.",
    )
    lora_group.add_argument(
        "--no-lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA and train all parameters.",
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
        payload = describe_qwen3_5_starter_profiles(task=args.task, model_name=args.model)
        print(json.dumps(payload, indent=2, default=str))
        return

    if args.config:
        conflicting = []
        if args.model != QWEN35_08B_BASE_MODEL:
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
            parser.error("--config cannot be combined with starter override options: " + ", ".join(conflicting))
        resolved_config = load_qwen3_5_config_file(args.config)
    else:
        config_overrides = {}
        if args.iterations is not None:
            config_overrides["num_outer_iterations"] = args.iterations
        resolved_config = get_qwen3_5_config(
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

    preview = create_qwen3_5_preview(resolved_config)
    if args.write_config:
        written_path = write_qwen3_5_config_file(resolved_config, args.write_config)
        logger.info("Wrote Qwen3.5-0.8B config to %s", written_path)
        return

    if args.dry_run:
        print(json.dumps(preview, indent=2, default=str))
        return

    asyncio.run(run_qwen3_5_0_8b_config(resolved_config, dry_run=False))
    logger.info("Qwen3.5-0.8B starter run complete")


if __name__ == "__main__":
    main()
