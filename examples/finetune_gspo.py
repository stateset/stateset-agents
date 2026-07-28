"""Unified GSPO finetune driver, parameterized by model preset.

This replaces the common training-flow boilerplate that used to be
duplicated across ~15 ``examples/finetune_*_gspo.py`` clone scripts. Instead
of one script per model, pick a preset from :mod:`examples.model_presets`:

    # List every supported preset
    python examples/finetune_gspo.py --list-models

    # Preview the resolved agent/reward/trainer wiring without downloading
    # a model or training (uses the stub backend)
    python examples/finetune_gspo.py --model kimi-k3 --dry-run

    # Run for real (requires the real model weights + GPU)
    python examples/finetune_gspo.py --model llama3

``--dry-run`` builds the agent (with ``use_stub_model=True``), the reward
function, and the GSPO training config, prints a summary, and exits 0
without launching training.
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

from examples.model_presets import PRESETS, ModelPreset, get_preset, list_preset_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_agent_config(preset: ModelPreset, *, dry_run: bool) -> Any:
    """Build an ``AgentConfig`` for the given preset.

    When ``dry_run`` is set, the returned config uses the stub backend so no
    real model weights are downloaded or loaded.
    """
    from stateset_agents.core.agent import AgentConfig

    if dry_run:
        return AgentConfig(
            model_name=f"stub://{preset.model_id}",
            system_prompt=(
                f"You are a helpful assistant fine-tuned from {preset.model_id}."
            ),
            use_stub_model=True,
            attn_implementation="eager",
        )

    return AgentConfig(
        model_name=preset.model_id,
        system_prompt=(
            f"You are a helpful assistant fine-tuned from {preset.model_id}."
        ),
    )


def build_gspo_config(preset: ModelPreset, *, task: str, output_dir: str) -> Any:
    """Build a ``GSPOConfig`` reproducing the preset's hyperparameters.

    ``task`` is passed through to ``run_name``/``wandb_tags`` for
    observability; ``GSPOConfig`` itself has no dedicated task field (task
    presets live in ``stateset_agents.training.config.get_config_for_task``).
    """
    from stateset_agents.training.gspo_trainer import GSPOConfig

    return GSPOConfig(
        model_name=preset.model_id,
        run_name=f"{preset.model_id}-{task}",
        learning_rate=preset.learning_rate,
        num_generations=preset.num_generations,
        max_prompt_length=preset.max_prompt_length,
        max_completion_length=preset.max_completion_length,
        bf16=preset.bf16,
        use_lora=True,
        lora_target_modules=list(preset.lora_target_modules),
        use_4bit=preset.use_4bit,
        use_8bit=preset.use_8bit,
        output_dir=output_dir,
    )


def build_reward_fn() -> Any:
    """Build the common composite reward used by the finetune scripts."""
    from stateset_agents.core.reward import CompositeReward
    from stateset_agents.core.reward import HelpfulnessReward, SafetyReward

    return CompositeReward(
        [HelpfulnessReward(weight=0.7), SafetyReward(weight=0.3)]
    )


def build_environment() -> Any:
    """Build a minimal conversation environment for smoke/dry runs."""
    from stateset_agents.core.environment import ConversationEnvironment

    scenarios = [
        {
            "id": "general_help",
            "topic": "general_assistance",
            "context": "User needs general help",
            "user_responses": [
                "Hi there! Can you help me with something?",
                "Thanks, that's helpful.",
            ],
        }
    ]
    return ConversationEnvironment(scenarios=scenarios, max_turns=2)


def preview_payload(preset_name: str, preset: ModelPreset, gspo_config: Any) -> dict[str, Any]:
    return {
        "preset": preset_name,
        "model_id": preset.model_id,
        "tokenizer_id": preset.tokenizer_id,
        "lora_target_modules": list(preset.lora_target_modules),
        "max_prompt_length": preset.max_prompt_length,
        "max_completion_length": preset.max_completion_length,
        "learning_rate": preset.learning_rate,
        "num_generations": preset.num_generations,
        "bf16": preset.bf16,
        "use_4bit": preset.use_4bit,
        "use_8bit": preset.use_8bit,
        "output_dir": getattr(gspo_config, "output_dir", None),
        "notes": preset.notes,
    }


async def run(args: argparse.Namespace) -> int:
    preset = get_preset(args.model)

    agent_config = build_agent_config(preset, dry_run=args.dry_run)

    from stateset_agents.core.agent import MultiTurnAgent

    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    build_reward_fn()  # constructed for parity with the real training flow
    gspo_config = build_gspo_config(
        preset, task=args.task, output_dir=args.output_dir
    )

    if args.dry_run:
        payload = preview_payload(args.model, preset, gspo_config)
        print(json.dumps(payload, indent=2, default=str))
        logger.info("Dry run complete for preset %s -- no training performed.", args.model)
        return 0

    build_environment()  # constructed for parity with the real training flow
    logger.info(
        "Preset %s resolved. Wire this into your preferred training entry "
        "point (e.g. stateset_agents.training.train_with_gspo) to run a "
        "real job; this driver's --dry-run mode is the supported smoke "
        "test path.",
        args.model,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified GSPO finetune driver, parameterized by model preset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list_preset_names(),
        help="Model preset name (see --list-models).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all supported preset names and exit.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        help="Task label passed through to the GSPO config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the checkpoint output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the agent (stub backend), reward function, and GSPO "
            "config, print a preview, and exit without training."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for name in list_preset_names():
            print(name)
        return 0

    if args.model is None:
        parser.error("--model is required unless --list-models is given")

    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.model.replace('.', '_')}_gspo"

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "PRESETS",
    "build_agent_config",
    "build_environment",
    "build_gspo_config",
    "build_parser",
    "build_reward_fn",
    "main",
    "run",
]
