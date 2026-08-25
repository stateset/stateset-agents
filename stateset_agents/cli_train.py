"""Training subcommands for the StateSet Agents CLI.

Split out of ``stateset_agents/cli.py`` to keep that module under ~2200 LOC.
Each command attaches to the parent Typer ``app`` exported by ``cli``;
helpers (``_echo``, ``_load_config``, ``_validate_config``, etc.) are looked up
on the parent module via late binding so that ``unittest.mock.patch`` on
``stateset_agents.cli._<helper>`` continues to work for tests that exercise
these commands.
"""

from __future__ import annotations

import importlib
import json
import typing as t

import typer

# Late-bound access to helpers, constants, and the Typer app. Importing the
# parent ``cli`` module (rather than ``from stateset_agents.cli import ...``)
# means attribute lookup happens at call time, which is what the test patches
# rely on.
from stateset_agents import cli as _cli
from stateset_agents.cli import (
    CLI_CONFIG_EXCEPTIONS,
    CLI_IMPORT_EXCEPTIONS,
    CLI_TRAIN_EXCEPTIONS,
    app,
)
from stateset_agents.core.model_presets import PRESETS, ModelPreset

# Convenience re-exports — these helpers are NOT patched by any test, so a
# local rebinding is fine and keeps function bodies readable.
_echo = _cli._echo
_coerce_positive_int = _cli._coerce_positive_int
_normalize_training_profile = _cli._normalize_training_profile
_load_config = _cli._load_config
_validate_config = _cli._validate_config


@app.command()
def train(
    config: str | None = typer.Option(
        None, help="Path to a training config file (YAML/JSON)."
    ),
    episodes: int | None = typer.Option(None, help="Override number of episodes."),
    save: str | None = typer.Option(
        None, help="Optional checkpoint directory to save the trained agent."
    ),
    dry_run: bool = typer.Option(
        True,
        help="If true, only validates configuration and shows guidance.",
    ),
    stub: bool = typer.Option(
        False,
        "--stub",
        help="Run a lightweight stub demonstration without downloading models.",
    ),
    profile: str = typer.Option(
        "balanced",
        help=(
            "Training profile "
            "(conservative, balanced, aggressive, experimental; aliases: speed, quality)."
        ),
    ),
) -> None:
    """Guide or launch training (lightweight)."""
    cfg = _load_config(config)
    validation_errors, validation_warnings = _validate_config(cfg)
    if validation_errors:
        _echo("Configuration validation failed.")
        for item in validation_errors:
            _echo(f"- error: {item}")
        raise typer.Exit(code=2)
    for item in validation_warnings:
        _echo(f"Warning: {item}")

    agent_cfg = cfg.get("agent", {}) if isinstance(cfg.get("agent"), dict) else {}
    env_cfg = (
        cfg.get("environment", {}) if isinstance(cfg.get("environment"), dict) else {}
    )
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}

    if episodes is not None:
        _ = _coerce_positive_int(episodes, "episodes", 1)

    resolved_profile = _normalize_training_profile(cfg.get("profile", profile))
    if resolved_profile is None:
        _echo(
            "Unsupported profile. Use one of: conservative, balanced, aggressive, "
            "experimental (aliases: speed, quality)."
        )
        raise typer.Exit(code=2)

    if dry_run and not stub:
        if cfg:
            _echo("Dry-run: configuration loaded and validated.")
            if cfg:
                _echo(f"Loaded config keys: {', '.join(sorted(cfg.keys()))}")
        else:
            _echo("Dry-run: environment looks OK. To run a full example:")
        _echo("  python examples/quick_start.py")
        _echo("Or use TRL GRPO (if installed):")
        _echo("  python examples/train_with_trl_grpo.py")
        return

    try:
        from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
        from stateset_agents.core.environment import ConversationEnvironment
        from stateset_agents.core.reward import (
            CompositeReward,
            HelpfulnessReward,
            SafetyReward,
        )
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo(
            "Core agent modules unavailable. Install the package with required extras."
        )
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e

    train_fn = None
    if not stub:
        try:
            from stateset_agents.training.train import train as train_fn  # type: ignore
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Training components unavailable. Falling back to stub mode.")
            _echo(f"Details: {e}")
            stub = True

    ac = AgentConfig(
        model_name=agent_cfg.get("model_name", "stub://demo" if stub else "gpt2"),
        max_new_tokens=agent_cfg.get("max_new_tokens", 64),
        temperature=agent_cfg.get("temperature", 0.7),
        use_stub_model=agent_cfg.get("use_stub_model", stub),
        stub_responses=(
            agent_cfg.get(
                "stub_responses",
                [
                    "Stub backend ready. Install training extras for full GRPO",
                    "Running in offline stub mode.",
                ],
            )
            if stub
            else None
        ),
    )
    agent = MultiTurnAgent(ac)

    if stub:
        import asyncio

        async def _demo() -> None:
            await agent.initialize()
            history = [
                {
                    "role": "user",
                    "content": "Hi there, can you help me troubleshoot an issue?",
                }
            ]
            reply = await agent.generate_response(history)
            _echo("Stub agent conversation:")
            _echo(f"  user: {history[-1]['content']}")
            _echo(f"  assistant: {reply}")

        asyncio.run(_demo())
        _echo(
            "Stub demonstration complete. Install training extras for full GRPO runs."
        )
        raise typer.Exit(code=0)

    if env_cfg.get("type", "conversation") != "conversation":
        _echo("Only conversation environment supported by CLI quick train.")
        raise typer.Exit(code=2)
    scenarios = env_cfg.get("scenarios") or [
        {
            "id": "demo",
            "topic": "general_help",
            "context": "Demo",
            "user_responses": ["Thanks, tell me more.", "Interesting, go on."],
        }
    ]
    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=_coerce_positive_int(
            env_cfg.get("max_turns", train_cfg.get("max_turns", 3)),
            "max_turns",
            3,
        ),
    )

    reward_fn = CompositeReward(
        [
            HelpfulnessReward(weight=0.6),
            SafetyReward(weight=0.4),
        ]
    )

    num_episodes = _coerce_positive_int(
        episodes if episodes is not None else train_cfg.get("num_episodes", 2),
        "episodes",
        2,
    )

    import asyncio

    async def _run():
        await agent.initialize()
        await train_fn(
            agent=agent,
            environment=environment,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            profile=resolved_profile,
            save_path=save or None,
        )

    try:
        asyncio.run(_run())
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Training failed: {e}")
        raise typer.Exit(code=2) from e

    _echo("Training complete.")
    raise typer.Exit(code=0)


def _register_model_command(app: typer.Typer, preset: ModelPreset) -> None:
    """Register one per-model GSPO starter command for ``preset``.

    Every packaged starter exposes the same nine symbols and the commands that
    drive them were byte-identical apart from a handful of labels, so one
    closure covers all of them. The parameter list below is the command's real
    signature — Typer introspects it directly, which keeps ``--help`` output
    (flag names, order, defaults and help strings) identical to the ten
    hand-written commands this replaced.
    """

    display = preset.cli_display_name
    label = preset.cli_echo_label
    module = f"stateset_agents.training.{preset.starter_module}"
    command_name = preset.cli_command
    run_function = preset.cli_run_function or f"run_{preset.cli_symbol_infix}_config"

    def model_command(
        config: str | None = typer.Option(
            None,
            "--config",
            "-c",
            help=f"Path to a {display} starter config file (JSON/YAML).",
        ),
        task: str = typer.Option(
            "customer_service",
            help=f"Task preset for the {display} starter path.",
        ),
        starter_profile: str = typer.Option(
            "balanced",
            "--starter-profile",
            help="Starter profile: balanced, memory, or quality.",
        ),
        list_profiles: bool = typer.Option(
            False,
            "--list-profiles",
            help="Describe all built-in starter profiles and exit.",
        ),
        model: str = typer.Option(
            preset.model_id,
            "--model",
            help=(
                "Model name. For post-training, "
                f"{preset.cli_model_help_verb} {preset.model_id}."
            ),
        ),
        use_lora: bool | None = typer.Option(
            None,
            "--use-lora/--no-lora",
            help="Override LoRA usage. Defaults come from --starter-profile.",
        ),
        use_4bit: bool | None = typer.Option(
            None,
            "--use-4bit/--no-use-4bit",
            help="Override 4-bit quantization. Defaults come from --starter-profile.",
        ),
        use_8bit: bool | None = typer.Option(
            None,
            "--use-8bit/--no-use-8bit",
            help="Override 8-bit quantization. Defaults come from --starter-profile.",
        ),
        output_dir: str | None = typer.Option(
            None,
            "--output-dir",
            help="Override the output directory for checkpoints and adapters.",
        ),
        iterations: int | None = typer.Option(
            None,
            "--iterations",
            help="Override the outer GSPO iteration count for the starter run.",
        ),
        wandb: bool = typer.Option(
            False,
            "--wandb",
            help="Enable Weights & Biases logging.",
        ),
        wandb_project: str | None = typer.Option(
            None,
            "--wandb-project",
            help="Optional W&B project name.",
        ),
        write_config: str | None = typer.Option(
            None,
            "--write-config",
            help=(
                f"Write the resolved {preset.cli_write_label} starter config "
                "to JSON/YAML and exit."
            ),
        ),
        dry_run: bool = typer.Option(
            True,
            "--dry-run/--no-dry-run",
            help="Preview the resolved config instead of loading a model.",
        ),
        json_output: bool = typer.Option(
            False,
            "--json",
            "--json-output",
            help="Output machine-readable JSON.",
        ),
    ) -> None:
        try:
            starter = importlib.import_module(module)
            base_model: str = getattr(starter, f"{preset.cli_symbol_prefix}_BASE_MODEL")
            profile_choices: t.Sequence[str] = getattr(
                starter, f"{preset.cli_symbol_prefix}_STARTER_PROFILE_CHOICES"
            )
            task_choices: t.Sequence[str] = getattr(
                starter, f"{preset.cli_symbol_prefix}_TASK_CHOICES"
            )
            infix = preset.cli_symbol_infix
            create_preview = getattr(starter, f"create_{infix}_preview")
            describe_profiles = getattr(starter, f"describe_{infix}_starter_profiles")
            get_config = getattr(starter, f"get_{infix}_config")
            load_config_file = getattr(starter, f"load_{infix}_config_file")
            run_config = getattr(starter, run_function)
            write_config_file = getattr(starter, f"write_{infix}_config_file")
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo(f"{label} starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if list_profiles:
            if config is not None:
                _echo("`--list-profiles` cannot be combined with `--config`.")
                raise typer.Exit(code=2)
            if task not in task_choices:
                _echo(f"Unsupported task. Use one of: {', '.join(task_choices)}.")
                raise typer.Exit(code=2)

            profile_catalog = describe_profiles(
                task=task,
                model_name=model,
            )
            if json_output:
                _echo(
                    json.dumps(profile_catalog, indent=2, sort_keys=True, default=str)
                )
                return

            _echo(f"Available {label} starter profiles:")
            _echo(f"Model: {profile_catalog['model_name']}")
            _echo(f"Task: {profile_catalog['task']}")
            for profile_name in profile_choices:
                profile_payload = profile_catalog["profiles"][profile_name]
                summary = profile_payload["summary"]
                _echo(f"- {profile_name}: {profile_payload['description']}")
                _echo(
                    "  "
                    f"quantization={summary['quantization_mode']}; effective_batch_size={summary['effective_batch_size']}; "
                    f"prompt/completion={summary['max_prompt_length']}/{summary['max_completion_length']}; "
                    f"generations={summary['num_generations']}; outer_iterations={summary['num_outer_iterations']}"
                )
            return

        if config:
            conflicting_options: list[str] = []
            if task != "customer_service":
                conflicting_options.append("--task")
            if starter_profile != "balanced":
                conflicting_options.append("--starter-profile")
            if model != base_model:
                conflicting_options.append("--model")
            if use_lora is not None:
                conflicting_options.append("--use-lora/--no-lora")
            if use_4bit is not None:
                conflicting_options.append("--use-4bit")
            if use_8bit is not None:
                conflicting_options.append("--use-8bit")
            if output_dir is not None:
                conflicting_options.append("--output-dir")
            if iterations is not None:
                conflicting_options.append("--iterations")
            if wandb:
                conflicting_options.append("--wandb")
            if wandb_project is not None:
                conflicting_options.append("--wandb-project")
            if conflicting_options:
                _echo(
                    "`--config` cannot be combined with starter override options: "
                    + ", ".join(conflicting_options)
                )
                raise typer.Exit(code=2)
            try:
                resolved_config = load_config_file(config)
            except CLI_CONFIG_EXCEPTIONS + (ImportError,) as e:
                _echo(f"Failed to load {label} config: {e}")
                raise typer.Exit(code=2) from e
        else:
            if task not in task_choices:
                _echo(f"Unsupported task. Use one of: {', '.join(task_choices)}.")
                raise typer.Exit(code=2)
            if starter_profile not in profile_choices:
                _echo(
                    f"Unsupported starter profile. Use one of: {', '.join(profile_choices)}."
                )
                raise typer.Exit(code=2)
            config_overrides: dict[str, t.Any] = {}
            if iterations is not None:
                config_overrides["num_outer_iterations"] = _coerce_positive_int(
                    iterations,
                    "iterations",
                    preset.cli_default_iterations,
                )
            resolved_config = get_config(
                model_name=model,
                task=task,
                starter_profile=starter_profile,
                use_lora=use_lora,
                use_4bit=use_4bit,
                use_8bit=use_8bit,
                output_dir=output_dir,
                use_wandb=wandb,
                wandb_project=wandb_project,
                **config_overrides,
            )

        preview = create_preview(resolved_config)

        if write_config:
            try:
                written_path = write_config_file(resolved_config, write_config)
            except CLI_CONFIG_EXCEPTIONS + (ImportError,) as e:
                _echo(f"Failed to write {label} config: {e}")
                raise typer.Exit(code=2) from e

            if json_output:
                payload = dict(preview)
                payload["config_file"] = str(written_path)
                _echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
                return

            _echo(f"Wrote {label} config to {written_path}")
            return

        if dry_run:
            if json_output:
                _echo(json.dumps(preview, indent=2, sort_keys=True, default=str))
                return

            _echo(f"Dry-run: {label} starter config resolved.")
            _echo(f"Model: {preview['config']['model_name']}")
            _echo(f"Task: {preview['config']['task']}")
            _echo(f"Starter profile: {preview['config']['starter_profile']}")
            _echo(f"Output dir: {preview['config']['output_dir']}")
            _echo(f"LoRA: {preview['gspo_overrides']['use_lora']}")
            _echo(
                f"4-bit: {preview['gspo_overrides']['use_4bit']}; 8-bit: {preview['gspo_overrides']['use_8bit']}"
            )
            _echo(
                f"Outer iterations: {preview['gspo_overrides']['num_outer_iterations']}"
            )
            for warning in preview.get("warnings", []):
                _echo(f"Warning: {warning}")
            _echo("Run with:")
            _echo(
                f"  stateset-agents {command_name} --no-dry-run --task customer_service"
            )
            _echo("Or try the low-memory preset:")
            _echo(
                f"  stateset-agents {command_name} --starter-profile memory --json-output"
            )
            _echo("Or save a reusable config:")
            _echo(
                f"  stateset-agents {command_name} --write-config ./{preset.cli_config_stem}.json"
            )
            return

        import asyncio

        try:
            result = asyncio.run(run_config(resolved_config, dry_run=False))
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo(f"{label} training components unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e
        except CLI_TRAIN_EXCEPTIONS as e:
            _echo(f"{label} starter failed: {e}")
            raise typer.Exit(code=2) from e

        if json_output:
            payload = {
                "status": "completed",
                "task": resolved_config.task,
                "starter_profile": resolved_config.starter_profile,
                "model_name": resolved_config.model_name,
                "output_dir": resolved_config.output_dir,
                "result": str(result),
            }
            _echo(json.dumps(payload, indent=2, sort_keys=True))
            return

        _echo(f"{label} starter run complete.")

    assert command_name is not None
    model_command.__name__ = command_name.replace("-", "_")
    model_command.__doc__ = f"Preview or run the dedicated {display} GSPO starter path."
    app.command(command_name)(model_command)


GENERATED_MODEL_COMMANDS: tuple[str, ...] = tuple(
    preset.cli_command for preset in PRESETS.values() if preset.cli_command is not None
)

for _preset in PRESETS.values():
    if _preset.cli_command is not None:
        _register_model_command(app, _preset)
