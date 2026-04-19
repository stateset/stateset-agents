"""Dedicated Gemma 4 31B GSPO starter script."""

# ruff: noqa: E402

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

from stateset_agents.training.gemma4_starter import (
    GEMMA4_31B_BASE_MODEL,
    GEMMA4_31B_STARTER_PROFILE_CHOICES,
    GEMMA4_31B_TASK_CHOICES,
    create_gemma4_31b_preview,
    describe_gemma4_31b_starter_profiles,
    finetune_gemma4_31b,
    get_gemma4_31b_config,
    load_gemma4_31b_config_file,
    run_gemma4_31b_config,
    write_gemma4_31b_config_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Dedicated GSPO starter for google/gemma-4-31B-it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview the resolved config without loading a model
  python examples/finetune_gemma4_31b_gspo.py --dry-run

  # Use the low-memory starter preset for a small GPU
  python examples/finetune_gemma4_31b_gspo.py --starter-profile memory --dry-run

  # Save a reusable starter config
  python examples/finetune_gemma4_31b_gspo.py --write-config ./gemma4_31b.json

  # Run from a saved config
  python examples/finetune_gemma4_31b_gspo.py --config ./gemma4_31b.json --no-dry-run
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a Gemma 4 31B starter config file (JSON/YAML).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMMA4_31B_BASE_MODEL,
        help="Model name. For post-training, use google/gemma-4-31B-it.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=list(GEMMA4_31B_TASK_CHOICES),
        help="Task preset to train against.",
    )
    parser.add_argument(
        "--starter-profile",
        type=str,
        default="balanced",
        choices=list(GEMMA4_31B_STARTER_PROFILE_CHOICES),
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
        payload = describe_gemma4_31b_starter_profiles(
            task=args.task, model_name=args.model
        )
        print(json.dumps(payload, indent=2, default=str))
        return

    if args.config:
        conflicting = []
        if args.model != GEMMA4_31B_BASE_MODEL:
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
        resolved_config = load_gemma4_31b_config_file(args.config)
    else:
        config_overrides = {}
        if args.iterations is not None:
            config_overrides["num_outer_iterations"] = args.iterations
        resolved_config = get_gemma4_31b_config(
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

    preview = create_gemma4_31b_preview(resolved_config)
    if args.write_config:
        written_path = write_gemma4_31b_config_file(resolved_config, args.write_config)
        logger.info("Wrote Gemma 4 31B config to %s", written_path)
        return

    if args.dry_run:
        print(json.dumps(preview, indent=2, default=str))
        return

    asyncio.run(run_gemma4_31b_config(resolved_config, dry_run=False))
    logger.info("Gemma 4 31B starter run complete")


if __name__ == "__main__":
    main()


# Re-export the async finetune helper for direct programmatic use.
__all__ = [
    "finetune_gemma4_31b",
    "main",
]
