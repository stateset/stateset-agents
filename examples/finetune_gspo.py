"""Unified GSPO finetune driver, parameterized by model preset.

This replaces the common training-flow boilerplate that used to be
duplicated across ~15 ``examples/finetune_*_gspo.py`` clone scripts. Instead
of one script per model, pick a preset from :mod:`examples.model_presets`:

    # List every supported preset
    python examples/finetune_gspo.py --list-models

    # Preview the resolved agent/reward/trainer wiring without downloading
    # a model or training (uses the stub backend)
    python examples/finetune_gspo.py --model kimi-k3 --dry-run

    # Presets backed by a packaged starter (see ``ModelPreset.starter_module``)
    # also accept --starter-profile {balanced,memory,quality}, which delegates
    # to that starter's own config resolution instead of the driver's
    # generic GSPOConfig builder:
    python examples/finetune_gspo.py --model kimi-k3 --starter-profile memory --dry-run

    # Run for real (requires the real model weights + GPU). --dry-run
    # defaults to True, so a real run always requires --no-dry-run.
    python examples/finetune_gspo.py --model llama3 --no-dry-run

``--dry-run`` (the default) builds the agent (with ``use_stub_model=True``),
the reward function, and the GSPO training config, prints a summary, and
exits 0 without launching training. Pass ``--no-dry-run`` to actually train:
the driver then invokes the real training entry point -- for starter-backed
presets that is the packaged starter's own ``run_<name>_config`` coroutine;
for the rest it is ``stateset_agents.training.gspo_entrypoints.train_with_gspo``.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import importlib
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


# Maps ``ModelPreset.starter_module`` to the function-name suffix used by
# that packaged starter (``stateset_agents.training.<module>``). The suffix
# does not always match the preset name 1:1 (e.g. the qwen3.5-0.8b preset's
# starter exposes ``get_qwen3_5_config``/``run_qwen3_5_0_8b_config``), so it
# is recorded explicitly rather than derived by string munging.
STARTER_FN_SUFFIX: dict[str, str] = {
    "muse_glimmer_starter": "muse_glimmer",
    "nemotron_3_5_starter": "nemotron_3_5",
    "qwen3_8_starter": "qwen3_8",
    "qwen3_coder_starter": "qwen3_coder",
    "gpt_oss_starter": "gpt_oss",
    "deepseek_v4_starter": "deepseek_v4",
    "kimi_k3_starter": "kimi_k3",
    "kimi_k2_6_starter": "kimi_k2_6",
    "glm5_1_starter": "glm5_1",
    "glm5_2_starter": "glm5_2",
    "gemma4_starter": "gemma4_31b",
    "qwen3_5_starter": "qwen3_5",
}
# Presets whose starter's ``run_*_config``/``get_*_config`` suffix diverges
# from the shared ``STARTER_FN_SUFFIX`` entry for their module.
STARTER_RUN_FN_OVERRIDE: dict[str, str] = {
    "qwen3.5-0.8b": "qwen3_5_0_8b",
}


def _load_starter_functions(preset: ModelPreset, preset_name: str) -> dict[str, Any]:
    """Resolve a starter module's config/run/write/load functions by name."""
    if preset.starter_module is None:
        raise ValueError(f"Preset {preset_name!r} has no starter_module")

    module = importlib.import_module(
        f"stateset_agents.training.{preset.starter_module}"
    )
    get_suffix = STARTER_FN_SUFFIX[preset.starter_module]
    run_suffix = STARTER_RUN_FN_OVERRIDE.get(preset_name, get_suffix)
    return {
        "get_config": getattr(module, f"get_{get_suffix}_config"),
        "run_config": getattr(module, f"run_{run_suffix}_config"),
        "write_config_file": getattr(module, f"write_{get_suffix}_config_file"),
        "load_config_file": getattr(module, f"load_{get_suffix}_config_file"),
        "describe_profiles": getattr(module, f"describe_{get_suffix}_starter_profiles"),
        "create_preview": getattr(module, f"create_{get_suffix}_preview"),
    }


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
    from stateset_agents.training.gspo_trainer import GSPOConfig

    kwargs: dict[str, Any] = {
        "model_name": preset.model_id,
        "run_name": f"{preset.model_id}-{task}",
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


def build_reward_fn() -> Any:
    """Build the common composite reward used by the finetune scripts."""
    from stateset_agents.core.reward import (
        CompositeReward,
        HelpfulnessReward,
        SafetyReward,
    )

    return CompositeReward([HelpfulnessReward(weight=0.7), SafetyReward(weight=0.3)])


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


def preview_payload(
    preset_name: str, preset: ModelPreset, gspo_config: Any
) -> dict[str, Any]:
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


async def _run_starter_backed(args: argparse.Namespace, preset: ModelPreset) -> int:
    """Delegate to a packaged starter's own config resolution + run path."""
    fns = _load_starter_functions(preset, args.model)

    if args.list_profiles:
        payload = fns["describe_profiles"](task=args.task, model_name=preset.model_id)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.config:
        resolved_config = fns["load_config_file"](args.config)
    else:
        overrides: dict[str, Any] = {}
        if args.steps is not None:
            overrides["max_steps"] = args.steps
        if args.epochs is not None:
            overrides["num_train_epochs"] = args.epochs
        if args.iterations is not None:
            overrides["num_outer_iterations"] = args.iterations
        resolved_config = fns["get_config"](
            model_name=preset.model_id,
            task=args.task,
            starter_profile=args.starter_profile or "balanced",
            use_lora=args.use_lora,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            **(
                {"learning_rate": args.learning_rate}
                if args.learning_rate is not None
                else {}
            ),
            **overrides,
        )

    if args.write_config:
        written_path = fns["write_config_file"](resolved_config, args.write_config)
        logger.info("Wrote %s config to %s", args.model, written_path)
        return 0

    if args.dry_run:
        preview = fns["create_preview"](resolved_config)
        print(json.dumps(preview, indent=2, default=str))
        logger.info(
            "Dry run complete for preset %s -- dry run only, no training performed. Pass --no-dry-run to train.",
            args.model,
        )
        return 0

    logger.info(
        "Starting real %s starter run (starter-profile=%s).",
        args.model,
        args.starter_profile or "balanced",
    )
    await fns["run_config"](resolved_config, dry_run=False)
    logger.info("%s starter run complete.", args.model)
    return 0


async def run(args: argparse.Namespace) -> int:
    preset = get_preset(args.model)

    if args.export_merged and preset.starter_module is not None:
        # None of the packaged starters currently expose a merge-export
        # path (they have no `export_merged` parameter on get_*_config /
        # run_*_config). Fail loudly instead of silently ignoring the flag.
        print(
            f"--export-merged is not supported for preset {args.model!r}: "
            f"its packaged starter (stateset_agents.training."
            f"{preset.starter_module}) has no merge-export path. This flag "
            "is only wired for non-starter-backed presets.",
            file=sys.stderr,
        )
        return 2

    if args.iterations is not None and preset.starter_module is None:
        # --iterations maps to a starter's num_outer_iterations override,
        # which only exists for starter-backed presets. Fail loudly instead
        # of silently dropping it for the rest.
        print(
            f"--iterations is not supported for preset {args.model!r} "
            "(no packaged starter with an outer-iteration count backs it). "
            "Use --epochs or --steps instead.",
            file=sys.stderr,
        )
        return 2

    if preset.starter_module is not None and (
        args.starter_profile is not None
        or args.config
        or args.write_config
        or args.list_profiles
        or args.iterations is not None
        or not args.dry_run
    ):
        # Only take the starter-delegation path when the caller actually
        # exercised a starter-specific surface (or is running for real);
        # this keeps the fast --dry-run smoke-test path (used by the full
        # preset matrix test) on the driver's own lightweight builders.
        return await _run_starter_backed(args, preset)

    agent_config = build_agent_config(preset, dry_run=args.dry_run)

    from stateset_agents.core.agent import MultiTurnAgent

    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    build_reward_fn()  # constructed for parity with the real training flow
    gspo_config = build_gspo_config(
        preset,
        task=args.task,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        use_vllm=args.use_vllm,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        steps=args.steps,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    if args.dry_run:
        payload = preview_payload(args.model, preset, gspo_config)
        print(json.dumps(payload, indent=2, default=str))
        logger.info(
            "Dry run complete for preset %s -- dry run only, no training performed. Pass --no-dry-run to train.",
            args.model,
        )
        return 0

    environment = build_environment()
    reward_fn = build_reward_fn()

    from stateset_agents.training.gspo_entrypoints import train_with_gspo

    logger.info("Starting real training run for preset %s.", args.model)
    await train_with_gspo(
        agent=agent,
        environment=environment,
        reward_model=reward_fn,
        config=gspo_config,
    )
    logger.info("Training run complete for preset %s.", args.model)

    if args.export_merged:
        if gspo_config.use_lora:
            from stateset_agents.training.serving_artifacts import (
                export_merged_model_for_serving,
            )

            merged_dir = export_merged_model_for_serving(
                base_model_name=preset.model_id,
                adapter_dir=gspo_config.output_dir,
                output_dir=f"{gspo_config.output_dir}/merged",
            )
            logger.info("Exported merged checkpoint to %s", merged_dir)
        else:
            logger.warning(
                "Skipping --export-merged because LoRA is disabled for "
                "preset %s; no adapter weights were produced to merge.",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build the agent (stub backend), reward function, and GSPO "
            "config, print a preview, and exit without training."
        ),
    )

    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--use-lora",
        dest="use_lora",
        action="store_true",
        default=None,
        help="Force LoRA on (default varies by preset).",
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
        help="Force 4-bit quantization on, overriding the preset default.",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        default=None,
        help="Force 8-bit quantization on, overriding the preset default.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Enable vLLM-backed generation for the real training run.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name.",
    )
    parser.add_argument(
        "--export-merged",
        action="store_true",
        help="Export a merged (LoRA-folded) checkpoint after training completes.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the preset's learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the maximum number of training steps.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Override the number of outer GSPO iterations. Only supported "
            "for presets backed by a packaged starter (see "
            "ModelPreset.starter_module); fails clearly otherwise."
        ),
    )
    parser.add_argument(
        "--starter-profile",
        type=str,
        default=None,
        choices=["balanced", "memory", "quality"],
        help=(
            "Starter profile for presets backed by a packaged starter "
            "(ModelPreset.starter_module). Ignored for other presets."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a starter config file (JSON/YAML) to load instead of resolving one.",
    )
    parser.add_argument(
        "--write-config",
        type=str,
        default=None,
        help="Write the resolved config to JSON/YAML and exit.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the starter profile matrix as JSON and exit (starter-backed presets only).",
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
