import importlib
import json
import os
import sys
import typing as t
from pathlib import Path

import typer

from stateset_agents.exceptions import INFERENCE_EXCEPTIONS

app = typer.Typer(add_completion=False, help="StateSet Agents CLI")

CLI_IMPORT_EXCEPTIONS = (AttributeError, ImportError, OSError, RuntimeError)
CLI_CONFIG_EXCEPTIONS = (OSError, TypeError, ValueError)
CLI_TRAIN_EXCEPTIONS = INFERENCE_EXCEPTIONS
TRAIN_PROFILE_CHOICES = (
    "conservative",
    "balanced",
    "aggressive",
    "experimental",
)
TRAIN_PROFILE_ALIASES = {
    "speed": "aggressive",
    "quality": "conservative",
}


def _echo(s: str) -> None:
    typer.echo(s)


def _coerce_positive_int(value: t.Any, name: str, default: int) -> int:
    if value is None:
        value = default

    try:
        value_int = int(value)
    except (TypeError, ValueError):
        _echo(f"{name} must be an integer.")
        raise typer.Exit(code=2) from None

    if value_int <= 0:
        _echo(f"{name} must be a positive integer.")
        raise typer.Exit(code=2)

    return value_int


def _normalize_training_profile(profile: t.Any) -> str | None:
    """Return the canonical training profile name, or ``None`` if invalid."""
    normalized = str(profile).strip().lower()
    if not normalized:
        return None
    if normalized in TRAIN_PROFILE_ALIASES:
        return TRAIN_PROFILE_ALIASES[normalized]
    if normalized in TRAIN_PROFILE_CHOICES:
        return normalized
    return None


def _load_config(config_path: str | None) -> dict[str, t.Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    suffix = path.suffix.lower()
    if suffix and suffix not in {".yaml", ".yml", ".json", ".js"}:
        _echo(f"Unsupported config format: {path.suffix}")
        raise typer.Exit(code=2)

    if not path.exists():
        _echo(f"Config file not found: {config_path}")
        raise typer.Exit(code=2)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            _echo("PyYAML is required for YAML config files. Install with: pip install pyyaml")
            raise typer.Exit(code=2) from exc

        with path.open("r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except CLI_CONFIG_EXCEPTIONS as exc:
                _echo(f"Failed to parse YAML config {config_path}: {exc}")
                raise typer.Exit(code=2) from exc

    if suffix in {".json", ".js"}:
        with path.open("r", encoding="utf-8") as f:
            try:
                return json.load(f) or {}
            except CLI_CONFIG_EXCEPTIONS as exc:
                _echo(f"Failed to parse JSON config {config_path}: {exc}")
                raise typer.Exit(code=2) from exc

    _echo(f"Unsupported config format: {path.suffix}")
    raise typer.Exit(code=2)


def _validate_config(cfg: t.Any) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a training config dictionary."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(cfg, dict):
        return ["Configuration root must be a JSON/YAML object."], warnings

    allowed_keys = {"agent", "environment", "training", "profile", "metadata"}
    unknown_keys = sorted(set(cfg.keys()) - allowed_keys)
    if unknown_keys:
        warnings.append(f"Unknown top-level keys: {', '.join(unknown_keys)}")

    agent_cfg = cfg.get("agent", {})
    env_cfg = cfg.get("environment", {})
    training_cfg = cfg.get("training", {})

    if not isinstance(agent_cfg, dict):
        errors.append("`agent` must be an object.")
    else:
        if "model_name" in agent_cfg and not isinstance(agent_cfg["model_name"], str):
            errors.append("`agent.model_name` must be a string.")
        if "max_new_tokens" in agent_cfg:
            try:
                value = int(agent_cfg["max_new_tokens"])
            except (TypeError, ValueError):
                errors.append("`agent.max_new_tokens` must be an integer.")
            else:
                if value <= 0:
                    errors.append("`agent.max_new_tokens` must be a positive integer.")

        if "temperature" in agent_cfg:
            try:
                float(agent_cfg["temperature"])
            except (TypeError, ValueError):
                errors.append("`agent.temperature` must be a number.")
        if "use_stub_model" in agent_cfg and not isinstance(agent_cfg["use_stub_model"], bool):
            errors.append("`agent.use_stub_model` must be a boolean.")
        if "stub_responses" in agent_cfg:
            if not isinstance(agent_cfg["stub_responses"], list):
                errors.append("`agent.stub_responses` must be a list.")
            else:
                for idx, response in enumerate(agent_cfg["stub_responses"]):
                    if not isinstance(response, str):
                        errors.append(
                            f"`agent.stub_responses[{idx}]` must be a string."
                        )

    if not isinstance(env_cfg, dict):
        errors.append("`environment` must be an object.")
    else:
        if "type" in env_cfg and not isinstance(env_cfg["type"], str):
            errors.append("`environment.type` must be a string.")
        if "scenarios" in env_cfg and not isinstance(env_cfg["scenarios"], list):
            errors.append("`environment.scenarios` must be a list.")
        elif isinstance(env_cfg.get("scenarios"), list):
            scenarios = env_cfg["scenarios"]
            for idx, scenario in enumerate(scenarios):
                if not isinstance(scenario, dict):
                    errors.append(f"`environment.scenarios[{idx}]` must be an object.")
                    continue
                if "id" in scenario and not isinstance(scenario["id"], str):
                    errors.append(
                        f"`environment.scenarios[{idx}].id` must be a string."
                    )
                if "topic" in scenario and not isinstance(scenario["topic"], str):
                    errors.append(
                        f"`environment.scenarios[{idx}].topic` must be a string."
                    )
                if "context" in scenario and not isinstance(scenario["context"], str):
                    errors.append(
                        f"`environment.scenarios[{idx}].context` must be a string."
                    )
                if "user_responses" in scenario:
                    responses = scenario["user_responses"]
                    if not isinstance(responses, list):
                        errors.append(
                            f"`environment.scenarios[{idx}].user_responses` must be a list."
                        )
                    else:
                        for response_idx, response in enumerate(responses):
                            if not isinstance(response, str):
                                errors.append(
                                    f"`environment.scenarios[{idx}].user_responses[{response_idx}]` must be a string."
                                )

    if not isinstance(training_cfg, dict):
        errors.append("`training` must be an object.")
    else:
        if "num_episodes" in training_cfg:
            try:
                value = int(training_cfg["num_episodes"])
            except (TypeError, ValueError):
                errors.append("`training.num_episodes` must be an integer.")
            else:
                if value <= 0:
                    errors.append("`training.num_episodes` must be a positive integer.")

        if "max_turns" in training_cfg:
            try:
                value = int(training_cfg["max_turns"])
            except (TypeError, ValueError):
                errors.append("`training.max_turns` must be an integer.")
            else:
                if value <= 0:
                    errors.append("`training.max_turns` must be a positive integer.")

    if "profile" in cfg and _normalize_training_profile(cfg["profile"]) is None:
        errors.append(
            "`profile` must be one of: conservative, balanced, aggressive, "
            "experimental (aliases: speed, quality)."
        )

    return errors, warnings


def _collect_dependency_status() -> tuple[dict[str, bool], dict[str, bool]]:
    """Collect required and optional dependency availability."""
    import importlib

    required_status: dict[str, bool] = {}
    optional_status: dict[str, bool] = {}

    def _check(mod: str, required: bool = False) -> None:
        try:
            importlib.import_module(mod)
            if required:
                required_status[mod] = True
            else:
                optional_status[mod] = True
        except CLI_IMPORT_EXCEPTIONS:
            if required:
                required_status[mod] = False
            else:
                optional_status[mod] = False

    for mod in ["torch", "transformers", "datasets"]:
        _check(mod, required=True)

    for mod in ["aiohttp", "fastapi", "uvicorn", "trl", "bitsandbytes"]:
        _check(mod, required=False)

    return required_status, optional_status


def _collect_import_status(modules: list[str]) -> dict[str, bool]:
    """Collect import availability for a set of module names."""
    status: dict[str, bool] = {}

    for module in modules:
        try:
            importlib.import_module(module)
            status[module] = True
        except CLI_IMPORT_EXCEPTIONS:
            status[module] = False

    return status


@app.callback()
def main_callback() -> None:
    """StateSet Agents command-line interface."""
    # No-op. Subcommands implement functionality.
    return None


@app.command()
def version(
    json_output: bool = typer.Option(
        False,
        "--json",
        "--json-output",
        help="Output machine-readable JSON",
    )
) -> None:
    """Show installed version, git commit, and key dependency versions.

    Useful for bug reports and verifying installs:

        stateset-agents version          # human-readable
        stateset-agents version --json   # machine-readable
    """
    try:
        from stateset_agents import __version__
    except CLI_IMPORT_EXCEPTIONS:
        __version__ = "unknown"

    # Resolve git commit from the install source tree if we're a -e install.
    git_commit: str | None = None
    try:
        import subprocess
        import pathlib
        pkg_root = pathlib.Path(__import__("stateset_agents").__file__).resolve().parent.parent
        if (pkg_root / ".git").exists():
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=pkg_root, capture_output=True, text=True, check=False, timeout=2,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()
    except Exception:  # noqa: BLE001 — best-effort
        pass

    # Probe key optional deps without failing the call.
    def _safe_version(modname: str) -> str | None:
        try:
            mod = importlib.import_module(modname)
            return getattr(mod, "__version__", None) or "installed"
        except CLI_IMPORT_EXCEPTIONS:
            return None

    deps = {
        "torch": _safe_version("torch"),
        "transformers": _safe_version("transformers"),
        "peft": _safe_version("peft"),
        "trl": _safe_version("trl"),
        "datasets": _safe_version("datasets"),
        "fastapi": _safe_version("fastapi"),
        "vllm": _safe_version("vllm"),
    }

    payload: dict[str, Any] = {
        "name": "stateset-agents",
        "version": __version__,
        "git_commit": git_commit,
        "python": sys.version.split()[0],
        "dependencies": deps,
    }

    if json_output:
        _echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _echo(f"stateset-agents {payload['version']}")
    if git_commit:
        _echo(f"  commit:  {git_commit}")
    _echo(f"  python:  {payload['python']}")
    _echo(f"  deps:")
    for name, ver in deps.items():
        marker = "✓" if ver else "—"
        ver_str = ver if ver else "not installed"
        _echo(f"    {marker} {name:<14} {ver_str}")


@app.command()
def train(
    config: str | None = typer.Option(None, help="Path to a training config file (YAML/JSON)."),
    episodes: int | None = typer.Option(None, help="Override number of episodes."),
    save: str | None = typer.Option(None, help="Optional checkpoint directory to save the trained agent."),
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
    env_cfg = cfg.get("environment", {}) if isinstance(cfg.get("environment"), dict) else {}
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
        _echo("Core agent modules unavailable. Install the package with required extras.")
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
        stub_responses=agent_cfg.get(
            "stub_responses",
            [
                "Stub backend ready. Install training extras for full GRPO",
                "Running in offline stub mode.",
            ],
        )
        if stub
        else None,
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
        _echo("Stub demonstration complete. Install training extras for full GRPO runs.")
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


@app.command("qwen3-5-0-8b")
def qwen3_5_0_8b(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a Qwen/Qwen3.5-0.8B starter config file (JSON/YAML).",
    ),
    task: str = typer.Option(
        "customer_service",
        help="Task preset for the Qwen/Qwen3.5-0.8B starter path.",
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
        "Qwen/Qwen3.5-0.8B-Base",
        "--model",
        help="Model name. For post-training, prefer Qwen/Qwen3.5-0.8B-Base.",
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
        help="Write the resolved Qwen starter config to JSON/YAML and exit.",
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
    """Preview or run the dedicated Qwen/Qwen3.5-0.8B GSPO starter path."""
    try:
        from stateset_agents.training.qwen3_5_starter import (
            QWEN35_08B_BASE_MODEL,
            QWEN35_08B_STARTER_PROFILE_CHOICES,
            QWEN35_08B_TASK_CHOICES,
            create_qwen3_5_preview,
            describe_qwen3_5_starter_profiles,
            get_qwen3_5_config,
            load_qwen3_5_config_file,
            run_qwen3_5_0_8b_config,
            write_qwen3_5_config_file,
        )
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo("Qwen3.5-0.8B starter helpers unavailable. Install training extras.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e

    if list_profiles:
        if config is not None:
            _echo("`--list-profiles` cannot be combined with `--config`.")
            raise typer.Exit(code=2)
        if task not in QWEN35_08B_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(QWEN35_08B_TASK_CHOICES)}.")
            raise typer.Exit(code=2)

        profile_catalog = describe_qwen3_5_starter_profiles(
            task=task,
            model_name=model,
        )
        if json_output:
            _echo(json.dumps(profile_catalog, indent=2, sort_keys=True, default=str))
            return

        _echo("Available Qwen3.5-0.8B starter profiles:")
        _echo(f"Model: {profile_catalog['model_name']}")
        _echo(f"Task: {profile_catalog['task']}")
        for profile_name in QWEN35_08B_STARTER_PROFILE_CHOICES:
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
        if model != QWEN35_08B_BASE_MODEL:
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
            resolved_config = load_qwen3_5_config_file(config)
        except CLI_CONFIG_EXCEPTIONS + (ImportError,) as e:
            _echo(f"Failed to load Qwen3.5-0.8B config: {e}")
            raise typer.Exit(code=2) from e
    else:
        if task not in QWEN35_08B_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(QWEN35_08B_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in QWEN35_08B_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(QWEN35_08B_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)
        config_overrides: dict[str, t.Any] = {}
        if iterations is not None:
            config_overrides["num_outer_iterations"] = _coerce_positive_int(
                iterations,
                "iterations",
                25,
            )
        resolved_config = get_qwen3_5_config(
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

    preview = create_qwen3_5_preview(resolved_config)

    if write_config:
        try:
            written_path = write_qwen3_5_config_file(resolved_config, write_config)
        except CLI_CONFIG_EXCEPTIONS + (ImportError,) as e:
            _echo(f"Failed to write Qwen3.5-0.8B config: {e}")
            raise typer.Exit(code=2) from e

        if json_output:
            payload = dict(preview)
            payload["config_file"] = str(written_path)
            _echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
            return

        _echo(f"Wrote Qwen3.5-0.8B config to {written_path}")
        return

    if dry_run:
        if json_output:
            _echo(json.dumps(preview, indent=2, sort_keys=True, default=str))
            return

        _echo("Dry-run: Qwen3.5-0.8B starter config resolved.")
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
        _echo("  stateset-agents qwen3-5-0-8b --no-dry-run --task customer_service")
        _echo("Or try the low-memory preset:")
        _echo("  stateset-agents qwen3-5-0-8b --starter-profile memory --json-output")
        _echo("Or save a reusable config:")
        _echo("  stateset-agents qwen3-5-0-8b --write-config ./qwen3_5_0_8b.json")
        return

    import asyncio

    try:
        result = asyncio.run(run_qwen3_5_0_8b_config(resolved_config, dry_run=False))
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo("Qwen3.5-0.8B training components unavailable. Install training extras.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Qwen3.5-0.8B starter failed: {e}")
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

    _echo("Qwen3.5-0.8B starter run complete.")


@app.command("kimi-k2-6")
def kimi_k2_6(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a moonshotai/Kimi-K2.6 starter config file (JSON/YAML).",
    ),
    task: str = typer.Option(
        "customer_service",
        help="Task preset for the moonshotai/Kimi-K2.6 starter path.",
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
        "moonshotai/Kimi-K2.6",
        "--model",
        help="Model name. For post-training, prefer moonshotai/Kimi-K2.6.",
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
        help="Write the resolved Kimi starter config to JSON/YAML and exit.",
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
    """Preview or run the dedicated moonshotai/Kimi-K2.6 GSPO starter path."""
    try:
        from stateset_agents.training.kimi_k2_6_starter import (
            KIMI_K26_BASE_MODEL,
            KIMI_K26_STARTER_PROFILE_CHOICES,
            KIMI_K26_TASK_CHOICES,
            create_kimi_k2_6_preview,
            describe_kimi_k2_6_starter_profiles,
            get_kimi_k2_6_config,
            load_kimi_k2_6_config_file,
            run_kimi_k2_6_config,
            write_kimi_k2_6_config_file,
        )
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo("Kimi-K2.6 starter helpers unavailable. Install training extras.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e

    if list_profiles:
        if config is not None:
            _echo("`--list-profiles` cannot be combined with `--config`.")
            raise typer.Exit(code=2)
        if task not in KIMI_K26_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(KIMI_K26_TASK_CHOICES)}.")
            raise typer.Exit(code=2)

        profile_catalog = describe_kimi_k2_6_starter_profiles(
            task=task,
            model_name=model,
        )
        if json_output:
            _echo(json.dumps(profile_catalog, indent=2, sort_keys=True, default=str))
            return

        _echo("Available Kimi-K2.6 starter profiles:")
        _echo(f"Model: {profile_catalog['model_name']}")
        _echo(f"Task: {profile_catalog['task']}")
        for profile_name in KIMI_K26_STARTER_PROFILE_CHOICES:
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
        if model != KIMI_K26_BASE_MODEL:
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
            resolved_config = load_kimi_k2_6_config_file(config)
        except CLI_CONFIG_EXCEPTIONS + (ImportError,) as e:
            _echo(f"Failed to load Kimi-K2.6 config: {e}")
            raise typer.Exit(code=2) from e
    else:
        if task not in KIMI_K26_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(KIMI_K26_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in KIMI_K26_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(KIMI_K26_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)
        config_overrides: dict[str, t.Any] = {}
        if iterations is not None:
            config_overrides["num_outer_iterations"] = _coerce_positive_int(
                iterations,
                "iterations",
                16,
            )
        resolved_config = get_kimi_k2_6_config(
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

    preview = create_kimi_k2_6_preview(resolved_config)

    if write_config:
        try:
            written_path = write_kimi_k2_6_config_file(resolved_config, write_config)
        except CLI_CONFIG_EXCEPTIONS + (ImportError,) as e:
            _echo(f"Failed to write Kimi-K2.6 config: {e}")
            raise typer.Exit(code=2) from e

        if json_output:
            payload = dict(preview)
            payload["config_file"] = str(written_path)
            _echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
            return

        _echo(f"Wrote Kimi-K2.6 config to {written_path}")
        return

    if dry_run:
        if json_output:
            _echo(json.dumps(preview, indent=2, sort_keys=True, default=str))
            return

        _echo("Dry-run: Kimi-K2.6 starter config resolved.")
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
        _echo("  stateset-agents kimi-k2-6 --no-dry-run --task customer_service")
        _echo("Or try the low-memory preset:")
        _echo("  stateset-agents kimi-k2-6 --starter-profile memory --json-output")
        _echo("Or save a reusable config:")
        _echo("  stateset-agents kimi-k2-6 --write-config ./kimi_k2_6.json")
        return

    import asyncio

    try:
        result = asyncio.run(run_kimi_k2_6_config(resolved_config, dry_run=False))
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo("Kimi-K2.6 training components unavailable. Install training extras.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Kimi-K2.6 starter failed: {e}")
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

    _echo("Kimi-K2.6 starter run complete.")


@app.command("gemma-4-31b")
def gemma4_31b(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a Gemma 4 31B starter config file (JSON/YAML).",
    ),
    task: str = typer.Option(
        "customer_service",
        help="Task preset for the Gemma 4 31B starter path.",
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
        "google/gemma-4-31B-it",
        "--model",
        help="Model name. For post-training, use google/gemma-4-31B-it.",
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
        help="Write the resolved Gemma starter config to JSON/YAML and exit.",
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
    """Preview or run the dedicated Gemma 4 31B GSPO starter path."""
    try:
        from stateset_agents.training.gemma4_starter import (
            GEMMA4_31B_BASE_MODEL,
            GEMMA4_31B_STARTER_PROFILE_CHOICES,
            GEMMA4_31B_TASK_CHOICES,
            create_gemma4_31b_preview,
            describe_gemma4_31b_starter_profiles,
            get_gemma4_31b_config,
            load_gemma4_31b_config_file,
            run_gemma4_31b_config,
            write_gemma4_31b_config_file,
        )
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo("Gemma 4 31B starter helpers unavailable. Install training extras.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e

    if list_profiles:
        if config is not None:
            _echo("`--list-profiles` cannot be combined with `--config`.")
            raise typer.Exit(code=2)
        if task not in GEMMA4_31B_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(GEMMA4_31B_TASK_CHOICES)}.")
            raise typer.Exit(code=2)

        profile_catalog = describe_gemma4_31b_starter_profiles(
            task=task,
            model_name=model,
        )
        if json_output:
            _echo(json.dumps(profile_catalog, indent=2, sort_keys=True, default=str))
            return

        _echo("Available Gemma 4 31B starter profiles:")
        _echo(f"Model: {profile_catalog['model_name']}")
        _echo(f"Task: {profile_catalog['task']}")
        for profile_name in GEMMA4_31B_STARTER_PROFILE_CHOICES:
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
        if model != GEMMA4_31B_BASE_MODEL:
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
            resolved_config = load_gemma4_31b_config_file(config)
        except CLI_CONFIG_EXCEPTIONS + (ImportError,) as e:
            _echo(f"Failed to load Gemma 4 31B config: {e}")
            raise typer.Exit(code=2) from e
    else:
        if task not in GEMMA4_31B_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(GEMMA4_31B_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in GEMMA4_31B_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(GEMMA4_31B_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)
        config_overrides: dict[str, t.Any] = {}
        if iterations is not None:
            config_overrides["num_outer_iterations"] = _coerce_positive_int(
                iterations,
                "iterations",
                20,
            )
        resolved_config = get_gemma4_31b_config(
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

    preview = create_gemma4_31b_preview(resolved_config)

    if write_config:
        try:
            written_path = write_gemma4_31b_config_file(resolved_config, write_config)
        except CLI_CONFIG_EXCEPTIONS + (ImportError,) as e:
            _echo(f"Failed to write Gemma 4 31B config: {e}")
            raise typer.Exit(code=2) from e

        if json_output:
            payload = dict(preview)
            payload["config_file"] = str(written_path)
            _echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
            return

        _echo(f"Wrote Gemma 4 31B config to {written_path}")
        return

    if dry_run:
        if json_output:
            _echo(json.dumps(preview, indent=2, sort_keys=True, default=str))
            return

        _echo("Dry-run: Gemma 4 31B starter config resolved.")
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
        _echo("  stateset-agents gemma-4-31b --no-dry-run --task customer_service")
        _echo("Or try the low-memory preset:")
        _echo("  stateset-agents gemma-4-31b --starter-profile memory --json-output")
        _echo("Or save a reusable config:")
        _echo("  stateset-agents gemma-4-31b --write-config ./gemma4_31b.json")
        return

    import asyncio

    try:
        result = asyncio.run(run_gemma4_31b_config(resolved_config, dry_run=False))
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo("Gemma 4 31B training components unavailable. Install training extras.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Gemma 4 31B starter failed: {e}")
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

    _echo("Gemma 4 31B starter run complete.")


@app.command()
def validate_config(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a config file (YAML/JSON).",
    ),
    strict: bool = typer.Option(False, help="Fail if validation errors are found."),
    fail_on_warnings: bool = typer.Option(
        False,
        help="Fail if validation warnings are found.",
    ),
    json_output: bool = typer.Option(False, help="Output machine-readable diagnostics"),
) -> None:
    """Validate a training config file for common CLI-relevant issues."""
    cfg = _load_config(config)
    errors, warnings = _validate_config(cfg)
    has_warnings = bool(warnings)
    has_errors = bool(errors)
    fail = has_errors
    fail = fail or (has_warnings and fail_on_warnings)

    if json_output:
        payload = {
            "name": "stateset-agents",
            "config_path": config,
            "valid": not has_errors,
            "warnings": warnings,
            "errors": errors,
            "strict": strict,
            "fail_on_warnings": fail_on_warnings,
            "failed": fail,
        }
        _echo(json.dumps(payload, indent=2, sort_keys=True))
        if fail:
            raise typer.Exit(code=2)
        return

    if has_errors:
        _echo("Configuration validation failed.")
        for item in errors:
            _echo(f"- error: {item}")
    else:
        _echo("Configuration validation passed.")

    for item in warnings:
        _echo(f"- warning: {item}")

    if fail:
        raise typer.Exit(code=2)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload (development)"),
    dry_run: bool = typer.Option(False, help="Print startup command without running the server."),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to a trained checkpoint (LoRA adapter or full model).",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Base model name when --checkpoint is a LoRA adapter (defaults to the value baked into the adapter).",
    ),
) -> None:
    """Run the FastAPI gateway (`stateset_agents.api.main:app`).

    Pass ``--checkpoint`` to serve a freshly-trained agent — this closes the
    loop from ``make benchmark-phase0`` straight to a running endpoint.
    """
    _ = _coerce_positive_int(port, "port", 8000)

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            print(f"Checkpoint path not found: {checkpoint}", file=sys.stderr)
            raise typer.Exit(code=2)
        os.environ["STATESET_DEFAULT_CHECKPOINT"] = str(ckpt_path.resolve())
        if base_model:
            os.environ["STATESET_DEFAULT_BASE_MODEL"] = base_model
        _echo(f"Will load checkpoint: {ckpt_path.resolve()}")
        if base_model:
            _echo(f"Base model: {base_model}")

    if dry_run:
        _echo("Dry-run: serve command did not start API.")
        _echo(
            f"Preview: uvicorn stateset_agents.api.main:app --host {host} --port {port}"
        )
        if checkpoint:
            _echo(f"Preview: STATESET_DEFAULT_CHECKPOINT={ckpt_path.resolve()}")
        if reload:
            _echo("Preview: --reload")
        return

    try:
        importlib.import_module("stateset_agents.api.main")
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo("API gateway unavailable. Install 'api' extras (fastapi, uvicorn).")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e

    try:
        import uvicorn
    except ImportError as e:
        _echo("uvicorn not installed. Try: pip install 'stateset-agents[api]'")
        raise typer.Exit(code=2) from e

    _echo("Starting StateSet Agents service...")
    uvicorn.run(
        "stateset_agents.api.main:app",
        host=host,
        port=port,
        log_level="info",
        reload=reload,
    )


@app.command()
def doctor(
    strict: bool = typer.Option(False, help="Fail if required dependencies are missing."),
    json_output: bool = typer.Option(
        False,
        "--json",
        "--json-output",
        help="Output machine-readable diagnostics",
    ),
) -> None:
    """Check environment and dependencies for common issues."""
    import platform

    required_status, optional_status = _collect_dependency_status()

    if not json_output:
        for mod in ["torch", "transformers", "datasets"]:
            if required_status.get(mod, False):
                _echo(f"✅ {mod} available")
            else:
                _echo(f"❌ {mod} missing")
        for mod in ["aiohttp", "fastapi", "uvicorn", "trl", "bitsandbytes"]:
            if optional_status.get(mod, False):
                _echo(f"✅ {mod} available")
            else:
                _echo(f"⚠️  {mod} missing")

    cuda = False
    bf16 = False
    gpu_name: str | None = None
    gpu_count = 0

    # GPU info
    try:
        import torch

        cuda = torch.cuda.is_available()
        bf16 = torch.cuda.is_bf16_supported() if cuda else False
        if cuda:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
    except CLI_IMPORT_EXCEPTIONS:
        pass

    # Checkpoint/serve env vars — surfaces "why isn't my checkpoint loading?"
    default_checkpoint = os.environ.get("STATESET_DEFAULT_CHECKPOINT")
    default_base_model = os.environ.get("STATESET_DEFAULT_BASE_MODEL")
    checkpoint_exists: bool | None = None
    if default_checkpoint:
        checkpoint_exists = Path(default_checkpoint).exists()

    if json_output:
        payload = {
            "name": "stateset-agents",
            "required_dependencies": required_status,
            "optional_dependencies": optional_status,
            "strict": strict,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cuda_available": cuda,
            "bfloat16_supported": bf16,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
            "checkpoint": {
                "STATESET_DEFAULT_CHECKPOINT": default_checkpoint,
                "STATESET_DEFAULT_BASE_MODEL": default_base_model,
                "path_exists": checkpoint_exists,
            },
        }
        _echo(json.dumps(payload, sort_keys=True))
        if strict and False in required_status.values():
            raise typer.Exit(code=2)
        return

    _echo(f"CUDA available: {cuda}; bfloat16: {bf16}")
    if cuda and gpu_name is not None:
        _echo(f"GPU count: {gpu_count}; name: {gpu_name}")

    # Surface the checkpoint env vars — common source of "why doesn't this work?"
    if default_checkpoint:
        if checkpoint_exists:
            _echo(f"✅ STATESET_DEFAULT_CHECKPOINT={default_checkpoint} (exists)")
        else:
            _echo(f"❌ STATESET_DEFAULT_CHECKPOINT={default_checkpoint} (path does not exist!)")
        if default_base_model:
            _echo(f"   STATESET_DEFAULT_BASE_MODEL={default_base_model}")
        else:
            _echo("   ⚠️  STATESET_DEFAULT_BASE_MODEL not set — LoRA loading will be skipped.")
    else:
        _echo("ℹ️  No STATESET_DEFAULT_CHECKPOINT set (use `stateset-agents serve --checkpoint` to serve a trained adapter).")

    _echo("StateSet Agents - Environment Doctor")
    _echo(f"python: {sys.version.split()[0]} ({platform.platform()})")

    _echo("Done.")
    missing_required = [name for name, ok in required_status.items() if not ok]
    if strict and missing_required:
        raise typer.Exit(code=2)


@app.command()
def preflight(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional config path to validate during preflight.",
    ),
    strict: bool = typer.Option(False, help="Fail if required dependencies are missing."),
    fail_on_warnings: bool = typer.Option(
        False,
        help="Fail if config warnings are present.",
    ),
    json_output: bool = typer.Option(False, help="Output machine-readable diagnostics"),
) -> None:
    """Run environment and config checks in one command."""
    import platform

    required_status, optional_status = _collect_dependency_status()
    config_errors: list[str] = []
    config_warnings: list[str] = []
    config_valid = True

    if config is not None:
        cfg = _load_config(config)
        config_errors, config_warnings = _validate_config(cfg)
        config_valid = not config_errors

    missing_required = [name for name, ok in required_status.items() if not ok]
    has_warnings = bool(config_warnings)
    fail = bool(missing_required) and strict
    fail = fail or bool(config_errors)
    fail = fail or (has_warnings and fail_on_warnings)

    if json_output:
        payload = {
            "name": "stateset-agents",
            "config": {
                "path": config,
                "valid": config_valid,
                "errors": config_errors,
                "warnings": config_warnings,
            },
            "dependencies": {
                "required": required_status,
                "optional": optional_status,
            },
            "strict": strict,
            "fail_on_warnings": fail_on_warnings,
            "failed": fail,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        _echo(json.dumps(payload, indent=2, sort_keys=True))
        if fail:
            raise typer.Exit(code=2)
        return

    _echo("StateSet Agents - Preflight")
    if missing_required:
        _echo("Dependency check: missing required dependency.")
        for item in missing_required:
            _echo(f"  - {item}")
    else:
        _echo("Dependency check: required dependencies present.")

    if config is None:
        _echo("Config check: skipped (no --config provided).")
    elif config_valid:
        _echo("Config check: valid.")
        if config_warnings:
            _echo(f"  - warning count: {len(config_warnings)}")
            for item in config_warnings:
                _echo(f"  - warning: {item}")
    else:
        _echo("Config check: failed.")
        for item in config_errors:
            _echo(f"  - error: {item}")
        for item in config_warnings:
            _echo(f"  - warning: {item}")

    if fail:
        raise typer.Exit(code=2)


@app.command()
def publish_check(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional config path to validate during publish checks.",
    ),
    strict: bool = typer.Option(False, help="Fail if required checks fail."),
    fail_on_warnings: bool = typer.Option(
        False,
        help="Fail if config warnings are present.",
    ),
    json_output: bool = typer.Option(False, help="Output machine-readable diagnostics"),
) -> None:
    """Run publish-readiness checks for release preparation."""
    import platform

    required_dependency_status, optional_dependency_status = _collect_dependency_status()
    required_import_modules = [
        "stateset_agents",
        "stateset_agents.core",
        "stateset_agents.core.agent",
        "stateset_agents.core.environment",
        "stateset_agents.training",
        "stateset_agents.rewards",
    ]
    optional_import_modules = [
        "stateset_agents.api.main",
        "stateset_agents.cli",
        "stateset_agents.cli_advanced",
        "stateset_agents.utils.wandb_integration",
    ]

    required_import_status = _collect_import_status(required_import_modules)
    optional_import_status = _collect_import_status(optional_import_modules)

    config_errors: list[str] = []
    config_warnings: list[str] = []
    config_valid = True

    if config is not None:
        cfg = _load_config(config)
        config_errors, config_warnings = _validate_config(cfg)
        config_valid = not config_errors

    missing_required_dependencies = [
        name for name, ok in required_dependency_status.items() if not ok
    ]
    missing_required_imports = [
        name for name, ok in required_import_status.items() if not ok
    ]
    has_warnings = bool(config_warnings)
    fail = strict and bool(missing_required_dependencies)
    fail = fail or strict and bool(missing_required_imports)
    fail = fail or bool(config_errors)
    fail = fail or (has_warnings and fail_on_warnings)

    payload = {
        "name": "stateset-agents",
        "publish_ready": not fail,
        "config": {
            "path": config,
            "valid": config_valid,
            "errors": config_errors,
            "warnings": config_warnings,
        },
        "dependencies": {
            "required": required_dependency_status,
            "optional": optional_dependency_status,
        },
        "imports": {
            "required": required_import_status,
            "optional": optional_import_status,
        },
        "strict": strict,
        "fail_on_warnings": fail_on_warnings,
        "failed": fail,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    if json_output:
        _echo(json.dumps(payload, indent=2, sort_keys=True))
        if fail:
            raise typer.Exit(code=2)
        return

    _echo("StateSet Agents - Publish Check")

    if missing_required_dependencies:
        _echo("Required dependency check: failed.")
        for item in missing_required_dependencies:
            _echo(f"  - missing dependency: {item}")
    else:
        _echo("Required dependency check: pass.")

    if missing_required_imports:
        _echo("Required import check: failed.")
        for item in missing_required_imports:
            _echo(f"  - import failed: {item}")
    else:
        _echo("Required import check: pass.")

    if config is None:
        _echo("Config check: skipped (no --config provided).")
    elif config_valid:
        _echo("Config check: valid.")
        if config_warnings:
            _echo(f"  - warning count: {len(config_warnings)}")
            for item in config_warnings:
                _echo(f"  - warning: {item}")
    else:
        _echo("Config check: failed.")
        for item in config_errors:
            _echo(f"  - error: {item}")
        for item in config_warnings:
            _echo(f"  - warning: {item}")

    optional_import_warnings = [
        name for name, ok in optional_import_status.items() if not ok
    ]
    if optional_import_warnings:
        _echo("Optional import check: warnings.")
        for item in optional_import_warnings:
            _echo(f"  - optional import unavailable: {item}")

    if fail:
        raise typer.Exit(code=2)


@app.command()
def evaluate(
    checkpoint: str | None = typer.Option(None, "--checkpoint", help="Path to a saved checkpoint directory"),
    message: str = typer.Option("Hello", help="Message to evaluate"),
    dry_run: bool = typer.Option(False, help="Show evaluation plan without loading checkpoint."),
) -> None:
    """Load a saved agent checkpoint and run a single evaluation message."""
    if dry_run:
        _echo("Dry-run: evaluation was not executed.")
        if checkpoint:
            _echo(f"Checkpoint: {checkpoint}")
        _echo(f"Message: {message}")
        return

    if not checkpoint:
        _echo("checkpoint is required unless --dry-run is used.")
        raise typer.Exit(code=2)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        _echo(f"Checkpoint not found: {checkpoint}")
        raise typer.Exit(code=2)

    try:
        from stateset_agents.core.agent import load_agent_from_checkpoint
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo(f"Failed to import loader: {e}")
        raise typer.Exit(code=2) from e

    import asyncio

    async def _run():
        agent = await load_agent_from_checkpoint(checkpoint, load_model=True)
        resp = await agent.generate_response([{"role": "user", "content": message}])
        return resp

    try:
        resp = asyncio.run(_run())
        _echo(f"Response: {resp}")
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Evaluation failed: {e}")
        raise typer.Exit(code=2) from e


@app.command()
def init(
    path: str = typer.Option(
        "./stateset_agents.yaml", help="Path for a starter config"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing file"),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml or json",
    ),
    preset: str = typer.Option(
        "default",
        "--preset",
        help="Starter preset: default, qwen3-5-0-8b, kimi-k2-6, or gemma-4-31b",
    ),
    task: str = typer.Option(
        "customer_service",
        "--task",
        help="Task preset for model-specific starter presets.",
    ),
    starter_profile: str = typer.Option(
        "balanced",
        "--starter-profile",
        help="Starter profile for model-specific starter presets.",
    ),
) -> None:
    """Scaffold a starter config to get started."""
    if format not in {"yaml", "yml", "json"}:
        _echo("format must be yaml or json")
        raise typer.Exit(code=2)

    if preset not in {"default", "qwen3-5-0-8b", "kimi-k2-6", "gemma-4-31b"}:
        _echo("Unsupported preset. Use one of: default, qwen3-5-0-8b, kimi-k2-6, gemma-4-31b.")
        raise typer.Exit(code=2)

    if preset == "default" and starter_profile != "balanced":
        _echo(
            "`--starter-profile` only applies to --preset qwen3-5-0-8b, kimi-k2-6, or gemma-4-31b."
        )
        raise typer.Exit(code=2)

    config_path = Path(path)
    if config_path.exists() and not overwrite:
        _echo(f"Config already exists: {path}. Use --overwrite to replace it.")
        raise typer.Exit(code=2)

    if preset == "default":
        cfg = {
            "agent": {"model_name": "gpt2", "max_new_tokens": 64, "temperature": 0.7},
            "training": {"num_episodes": 5, "max_turns": 3},
            "environment": {
                "type": "conversation",
                "scenarios": [
                    {
                        "id": "demo",
                        "topic": "general_help",
                        "context": "User needs general assistance",
                        "user_responses": [
                            "Thanks! Can you elaborate?",
                            "Interesting, tell me more.",
                        ],
                    }
                ],
            },
        }

        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            serialized = (
                "# StateSet Agents - Starter Config\n"
                "agent:\n"
                "  model_name: gpt2\n"
                "  max_new_tokens: 64\n"
                "  temperature: 0.7\n"
                "\n"
                "training:\n"
                "  num_episodes: 5\n"
                "  max_turns: 3\n"
                "\n"
                "environment:\n"
                "  type: conversation\n"
                "  scenarios:\n"
                "    - id: demo\n"
                "      topic: general_help\n"
                "      context: User needs general assistance\n"
                "      user_responses:\n"
                "        - Thanks! Can you elaborate?\n"
                "        - Interesting, tell me more.\n"
            )
    elif preset == "qwen3-5-0-8b":
        try:
            from stateset_agents.training.qwen3_5_starter import (
                QWEN35_08B_STARTER_PROFILE_CHOICES,
                QWEN35_08B_TASK_CHOICES,
                get_qwen3_5_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Qwen3.5-0.8B starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in QWEN35_08B_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(QWEN35_08B_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in QWEN35_08B_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(QWEN35_08B_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_qwen3_5_config(task=task, starter_profile=starter_profile).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo("PyYAML is required for YAML starter configs. Install with: pip install pyyaml")
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    elif preset == "kimi-k2-6":
        try:
            from stateset_agents.training.kimi_k2_6_starter import (
                KIMI_K26_STARTER_PROFILE_CHOICES,
                KIMI_K26_TASK_CHOICES,
                get_kimi_k2_6_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Kimi-K2.6 starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in KIMI_K26_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(KIMI_K26_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in KIMI_K26_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(KIMI_K26_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_kimi_k2_6_config(task=task, starter_profile=starter_profile).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo("PyYAML is required for YAML starter configs. Install with: pip install pyyaml")
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    else:
        try:
            from stateset_agents.training.gemma4_starter import (
                GEMMA4_31B_STARTER_PROFILE_CHOICES,
                GEMMA4_31B_TASK_CHOICES,
                get_gemma4_31b_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Gemma 4 31B starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in GEMMA4_31B_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(GEMMA4_31B_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in GEMMA4_31B_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(GEMMA4_31B_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_gemma4_31b_config(task=task, starter_profile=starter_profile).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo("PyYAML is required for YAML starter configs. Install with: pip install pyyaml")
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(serialized, encoding="utf-8")
        _echo(f"Wrote starter config to {path}")
    except OSError as e:
        _echo(f"Failed to write config: {e}")
        raise typer.Exit(code=2) from e


@app.command("init-config")
def init_config(
    path: str = typer.Option(
        "./stateset_agents.yaml", help="Path for a starter config"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing file"),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml or json",
    ),
    preset: str = typer.Option(
        "default",
        "--preset",
        help="Starter preset: default, qwen3-5-0-8b, kimi-k2-6, or gemma-4-31b",
    ),
    task: str = typer.Option(
        "customer_service",
        "--task",
        help="Task preset for model-specific starter presets.",
    ),
    starter_profile: str = typer.Option(
        "balanced",
        "--starter-profile",
        help="Starter profile for model-specific starter presets.",
    ),
) -> None:
    """Alias for `init`."""
    init(
        path=path,
        overwrite=overwrite,
        format=format,
        preset=preset,
        task=task,
        starter_profile=starter_profile,
    )


@app.command("auto-research")
def auto_research(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to auto-research config file (YAML/JSON).",
    ),
    max_experiments: int = typer.Option(
        0,
        "--max-experiments",
        "-n",
        help="Maximum experiments to run (0 = unlimited).",
    ),
    time_budget: int = typer.Option(
        300,
        "--time-budget",
        "-t",
        help="Wall-clock seconds per experiment.",
    ),
    proposer: str = typer.Option(
        "perturbation",
        "--proposer",
        "-p",
        help="Proposer strategy: perturbation, smart, adaptive, random, grid, bayesian, llm.",
    ),
    algorithm: str = typer.Option(
        "gspo",
        "--algorithm",
        "-a",
        help="Training algorithm: gspo, grpo, dapo, vapo.",
    ),
    output_dir: str = typer.Option(
        "./auto_research_results",
        "--output-dir",
        "-o",
        help="Directory for results and checkpoints.",
    ),
    search_space: str = typer.Option(
        "grpo",
        "--search-space",
        "-s",
        help="Search space: grpo, auto_research, quick, reward, model, multi_algorithm, full.",
    ),
    improvement_patience: int = typer.Option(
        0,
        "--improvement-patience",
        help="Stop after this many consecutive non-improvements (0 = disabled).",
    ),
    max_wall_clock: int = typer.Option(
        0,
        "--max-wall-clock",
        help="Total wall-clock budget in seconds (0 = unlimited).",
    ),
    wandb: bool = typer.Option(
        False,
        "--wandb",
        help="Log experiments to Weights & Biases.",
    ),
    wandb_project: str = typer.Option(
        "auto-research",
        "--wandb-project",
        help="W&B project name.",
    ),
    stub: bool = typer.Option(
        False,
        "--stub",
        help="Run with stub model for testing the loop without GPU.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and show plan without running.",
    ),
) -> None:
    """Run the autonomous research loop to optimize agent training.

    The loop autonomously proposes experiments, trains with a time budget,
    evaluates on held-out scenarios, and keeps only improvements.

    Resumes automatically if a previous run exists in the output directory.
    """
    from stateset_agents.training.auto_research.config import AutoResearchConfig

    # Load from config file if provided, then override with CLI args
    if config:
        try:
            ar_config = AutoResearchConfig.from_file(config)
        except (ImportError, ValueError) as exc:
            # Fallback: load as generic dict for non-YAML/JSON
            _echo(f"Warning: {exc}. Falling back to manual config parsing.")
            ar_config = AutoResearchConfig()
    else:
        ar_config = AutoResearchConfig()

    # CLI args override file config (only if explicitly provided / non-default)
    if time_budget != 300:
        ar_config.time_budget = time_budget
    if max_experiments != 0:
        ar_config.max_experiments = max_experiments
    if max_wall_clock != 0:
        ar_config.max_wall_clock = max_wall_clock
    if proposer != "perturbation":
        ar_config.proposer = proposer
    if algorithm != "gspo":
        ar_config.trainer_algorithm = algorithm
    if output_dir != "./auto_research_results":
        ar_config.output_dir = output_dir
    if search_space != "grpo":
        ar_config.search_space_name = search_space
    if improvement_patience != 0:
        ar_config.improvement_patience = improvement_patience
    if wandb:
        ar_config.log_to_wandb = True
    if wandb_project != "auto-research":
        ar_config.wandb_project = wandb_project

    warnings = ar_config.validate()
    for w in warnings:
        _echo(f"Warning: {w}")

    if dry_run:
        _echo("Dry-run: auto-research configuration validated.")
        _echo(f"  Proposer:        {ar_config.proposer}")
        _echo(f"  Algorithm:       {ar_config.trainer_algorithm}")
        _echo(f"  Search space:    {ar_config.search_space_name}")
        _echo(f"  Time budget:     {ar_config.time_budget}s")
        _echo(f"  Max experiments: {ar_config.max_experiments or 'unlimited'}")
        _echo(f"  Max wall clock:  {ar_config.max_wall_clock or 'unlimited'}s")
        _echo(f"  Output dir:      {ar_config.output_dir}")
        _echo(f"  W&B logging:     {ar_config.log_to_wandb}")

        # Show available search spaces
        try:
            from stateset_agents.training.auto_research.search_spaces import (
                list_auto_research_search_spaces,
            )
            from stateset_agents.training.hpo.search_spaces import (
                list_available_search_spaces,
            )

            ar_spaces = list_auto_research_search_spaces()
            hpo_spaces = list_available_search_spaces()
            _echo(f"  Available search spaces: {', '.join(sorted(set(ar_spaces + hpo_spaces)))}")
        except Exception:
            pass

        # Check if resumable
        from pathlib import Path

        jsonl = Path(ar_config.output_dir) / "experiments.jsonl"
        if jsonl.exists():
            count = sum(1 for line in jsonl.open() if line.strip())
            _echo(f"  Resume:          yes ({count} previous experiments found)")
        else:
            _echo("  Resume:          no (fresh run)")

        return

    # Set up agent, environment, reward
    try:
        from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
        from stateset_agents.core.environment import ConversationEnvironment
        from stateset_agents.core.reward import (
            CompositeReward,
            HelpfulnessReward,
            SafetyReward,
        )
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo(f"Core modules unavailable: {e}")
        raise typer.Exit(code=2) from e

    # Load agent/env config from the file if provided
    file_cfg = _load_config(config) if config else {}
    ar_section = file_cfg.get("auto_research", file_cfg)
    agent_cfg = ar_section.get("agent", {}) if isinstance(ar_section, dict) else {}
    env_cfg = ar_section.get("environment", {}) if isinstance(ar_section, dict) else {}

    ac = AgentConfig(
        model_name=agent_cfg.get("model_name", "stub://demo" if stub else "gpt2"),
        max_new_tokens=agent_cfg.get("max_new_tokens", 64),
        temperature=agent_cfg.get("temperature", 0.7),
        use_stub_model=agent_cfg.get("use_stub_model", stub),
        stub_responses=agent_cfg.get(
            "stub_responses",
            ["Stub response for auto-research testing."],
        )
        if stub
        else None,
    )

    scenarios = env_cfg.get("scenarios") or [
        {
            "topic": "general_help",
            "context": "Demo scenario for auto-research",
            "user_responses": ["Thanks, tell me more.", "Interesting, go on."],
        }
    ]
    eval_scenarios = env_cfg.get("eval_scenarios", scenarios)

    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=env_cfg.get("max_turns", 8),
    )

    reward_fn = CompositeReward([
        HelpfulnessReward(weight=0.6),
        SafetyReward(weight=0.4),
    ])

    import asyncio

    from stateset_agents.training.auto_research.experiment_loop import run_auto_research

    async def _run() -> None:
        agent = MultiTurnAgent(ac)
        await agent.initialize()

        tracker = await run_auto_research(
            agent=agent,
            environment=environment,
            eval_scenarios=eval_scenarios,
            reward_fn=reward_fn,
            config=ar_config,
        )

        _echo(f"Done. Results saved to {ar_config.output_dir}")
        if tracker.best_record:
            _echo(
                f"Best {ar_config.objective_metric}: "
                f"{tracker.best_value:.6f} "
                f"(experiment {tracker.best_record.experiment_id})"
            )

    try:
        asyncio.run(_run())
    except CLI_TRAIN_EXCEPTIONS as e:
        _echo(f"Auto-research failed: {e}")
        raise typer.Exit(code=2) from e


@app.command("fine-tune")
def fine_tune(
    curated: str = typer.Argument(
        ...,
        help="Path to a curated JSONL (from `stateset-agents grade-batch ... CURATED=...`).",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen3.5-0.8B", "--base-model", "-m",
        help="HF base model to fine-tune.",
    ),
    output_dir: str = typer.Option(
        "outputs/sft_v1", "--output-dir", "-o",
        help="Where the LoRA adapter is saved.",
    ),
    min_score: float = typer.Option(
        0.7, "--min-score",
        help="Drop curated examples below this score before SFT.",
    ),
    num_epochs: int = typer.Option(
        3, "--num-epochs", "-e",
        help="Training epochs.",
    ),
    lora_r: int = typer.Option(
        16, "--lora-r",
        help="LoRA rank.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the training plan without running it (forced when no GPU).",
    ),
) -> None:
    """Fine-tune from a curated JSONL in one command.

    Convenience wrapper around `make full-loop` — runs prepare_sft_dataset.py
    then sft_from_curated.py. Produces a LoRA adapter consumable by
    `stateset-agents chat --checkpoint` and `stateset-agents serve --checkpoint`.

    Examples:

        stateset-agents fine-tune curated.jsonl
        stateset-agents fine-tune curated.jsonl --base-model Qwen/Qwen3.5-0.8B --num-epochs 5
        stateset-agents fine-tune curated.jsonl --dry-run
    """
    import subprocess
    import tempfile

    curated_path = Path(curated)
    if not curated_path.exists():
        print(f"Curated file not found: {curated}", file=sys.stderr)
        raise typer.Exit(code=2)

    script_dir = Path(__file__).resolve().parents[1] / "scripts"

    with tempfile.TemporaryDirectory() as tmp:
        sft_jsonl = Path(tmp) / "sft_train.jsonl"

        _echo("▶ Step 1/2: prepare-sft (curated → chat format)")
        result = subprocess.run(
            [
                sys.executable, str(script_dir / "prepare_sft_dataset.py"),
                "--input", str(curated_path),
                "--format", "chat",
                "--output", str(sft_jsonl),
                "--min-score", str(min_score),
                "--dedup",
                "--stats",
            ],
            check=False,
        )
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)

        _echo("")
        _echo("▶ Step 2/2: sft-from-curated (chat format → trained adapter)")
        cmd = [
            sys.executable, str(script_dir / "sft_from_curated.py"),
            "--dataset", str(sft_jsonl),
            "--base-model", base_model,
            "--output-dir", output_dir,
            "--num-epochs", str(num_epochs),
            "--lora-r", str(lora_r),
        ]
        if dry_run:
            cmd.append("--dry-run")
        result = subprocess.run(cmd, check=False)
        raise typer.Exit(code=result.returncode)


@app.command("chat")
def chat(
    model: str = typer.Option(
        "stub://chat", "--model", "-m",
        help="HF model name or stub://<id> for the in-process REPL.",
    ),
    checkpoint: str | None = typer.Option(
        None, "--checkpoint", "-c",
        help="Path to a LoRA adapter to load on top of --model.",
    ),
    system_prompt: str | None = typer.Option(
        None, "--system",
        help="Optional system prompt prepended to every conversation.",
    ),
    max_new_tokens: int = typer.Option(
        256, "--max-new-tokens",
        help="Generation length cap per response.",
    ),
    history: str | None = typer.Option(
        None, "--history",
        help="Path to a JSONL file to APPEND each turn (one JSON object per line). "
             "Capture interesting conversations to replay or grade later with "
             "`make grade-transcript`.",
    ),
    replay: str | None = typer.Option(
        None, "--replay",
        help="Path to a JSONL transcript to replay as initial conversation context. "
             "Useful for resuming a debugging session.",
    ),
    grade: str | None = typer.Option(
        None, "--grade",
        help="Score each assistant turn live with the named reward function. "
             "Options: gsm8k, customer_support, tool_calling. "
             "Mismatches between intuition and score surface reward-function bugs.",
    ),
) -> None:
    """Open an interactive REPL against an in-process agent.

    The fastest way to sanity-check a fine-tune locally without spinning up
    the FastAPI gateway. Loads the agent in-process and lets you talk to it
    line by line. Type ``/reset`` to clear conversation history, ``/quit`` or
    Ctrl+D to exit.

    Examples:

        # Stub agent — no GPU, instant
        stateset-agents chat

        # Real HF model
        stateset-agents chat --model Qwen/Qwen3.5-0.8B

        # Fine-tuned LoRA adapter
        stateset-agents chat --model Qwen/Qwen3.5-0.8B --checkpoint outputs/acme_v1

        # With a system prompt
        stateset-agents chat --system "You are a helpful customer support agent."
    """
    import asyncio

    from stateset_agents.core.agent_config import AgentConfig
    from stateset_agents.core.agent import MultiTurnAgent

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            print(f"Checkpoint path not found: {checkpoint}", file=sys.stderr)
            raise typer.Exit(code=2)

    is_stub = model.startswith("stub://") or model == "gpt2"
    config = AgentConfig(
        model_name=model,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        use_stub_model=is_stub,
        peft_path=checkpoint if (checkpoint and not is_stub) else None,
    )

    _echo(f"Initializing agent (model={model}, stub={is_stub})…")
    agent = MultiTurnAgent(config)
    try:
        asyncio.run(agent.initialize())
    except Exception as e:  # noqa: BLE001 — surface the error to the user
        print(f"Failed to initialize agent: {type(e).__name__}: {e}", file=sys.stderr)
        raise typer.Exit(code=2) from e

    # Set up live grader if requested.
    live_reward = None
    if grade:
        try:
            if grade == "gsm8k":
                from stateset_agents.data.gsm8k import GSM8KReward
                live_reward = GSM8KReward()
            elif grade == "customer_support":
                from stateset_agents.data.customer_support_bench import SupportRewardComposite
                live_reward = SupportRewardComposite()
            elif grade == "tool_calling":
                from stateset_agents.data.tool_calling_bench import ToolCallReward
                live_reward = ToolCallReward()
            else:
                print(
                    f"Unknown --grade reward: {grade!r}. "
                    f"Options: gsm8k, customer_support, tool_calling.",
                    file=sys.stderr,
                )
                raise typer.Exit(code=2)
        except ImportError as e:
            print(f"Failed to load grader: {e}", file=sys.stderr)
            raise typer.Exit(code=2) from e

    # Wire up readline for up-arrow recall + persistent input history.
    # The "input history" (one file per user) is separate from the conversation
    # transcript (`--history` flag). Only sets up if readline is available
    # (Linux/macOS by default; Windows ships pyreadline3 separately).
    _readline_history_path: Path | None = None
    try:
        import readline

        # Use XDG_STATE_HOME if set, else ~/.local/state, else ~ as fallback.
        state_dir = os.environ.get("XDG_STATE_HOME")
        if state_dir:
            base = Path(state_dir) / "stateset-agents"
        else:
            base = Path.home() / ".local" / "state" / "stateset-agents"
        base.mkdir(parents=True, exist_ok=True)
        _readline_history_path = base / "chat_input_history"
        if _readline_history_path.exists():
            try:
                readline.read_history_file(str(_readline_history_path))
            except OSError:
                pass
        readline.set_history_length(1000)
    except ImportError:
        pass  # readline unavailable — REPL still works, just no up-arrow

    _echo("")
    _echo("=" * 60)
    _echo("StateSet Agents — Interactive Chat")
    _echo("Type /reset to clear history, /quit or Ctrl+D to exit.")
    if _readline_history_path:
        _echo(f"Input history: {_readline_history_path} (up-arrow recalls)")
    if history:
        _echo(f"Appending each turn to: {history}")
    if grade:
        _echo(f"Live grading enabled: {grade}")
    _echo("=" * 60)

    messages: list[dict[str, str]] = []
    history_path = Path(history).expanduser() if history else None
    if history_path:
        history_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional replay — preload conversation from a saved transcript.
    if replay:
        replay_path = Path(replay).expanduser()
        if not replay_path.exists():
            print(f"Replay path not found: {replay}", file=sys.stderr)
            raise typer.Exit(code=2)
        for line in replay_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "role" in entry and "content" in entry:
                messages.append({"role": entry["role"], "content": entry["content"]})
        _echo(f"Loaded {len(messages)} turn(s) from {replay}")

    def _append_history(role: str, content: str) -> None:
        if history_path is None:
            return
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")

    async def _send(user_text: str) -> str:
        messages.append({"role": "user", "content": user_text})
        _append_history("user", user_text)
        response = await agent.generate_response(messages)
        messages.append({"role": "assistant", "content": response})
        _append_history("assistant", response)
        return response

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/reset":
            messages = []
            _echo("(conversation reset)")
            continue

        try:
            response = asyncio.run(_send(user_input))
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  generation failed: {e}", file=sys.stderr)
            continue

        print(f"agent> {response}")

        # Live grading: score this turn against the reward function.
        if live_reward is not None:
            try:
                from stateset_agents.core.trajectory import ConversationTurn
                # Score using the full conversation so far (matches training-time eval).
                turns_for_reward = [
                    ConversationTurn(role=m["role"], content=m["content"]) for m in messages
                ]
                reward_result = asyncio.run(
                    live_reward.compute_reward(turns_for_reward, context=None)
                )
                score = float(reward_result.score)
                # Color-code: green for ≥0.5, yellow for 0.1–0.5, red for <0.1.
                if score >= 0.5:
                    marker = "✅"
                elif score >= 0.1:
                    marker = "⚠️ "
                else:
                    marker = "❌"
                print(f"  {marker} reward[{grade}] = {score:.3f}")
            except Exception as e:  # noqa: BLE001 — grading failure shouldn't kill the REPL
                print(f"  ⚠️  grading failed: {e}", file=sys.stderr)

    # Persist input history for the next chat session.
    if _readline_history_path is not None:
        try:
            import readline
            readline.write_history_file(str(_readline_history_path))
        except (ImportError, OSError):
            pass

    _echo("Bye.")


@app.command("recipe")
def recipe(
    name: str = typer.Argument(
        "list",
        help="Recipe name to open (e.g. `first-fine-tune`) or `list` to see all.",
    ),
) -> None:
    """Open a cookbook recipe in $PAGER, or `list` them all.

    Recipes are sourced from ``docs/COOKBOOK.md`` — self-contained, copy-paste
    workflows for the 7 most common things you'll want to do with the platform.

    Examples:

        stateset-agents recipe list                     # show every recipe
        stateset-agents recipe first-fine-tune          # open recipe 1
        stateset-agents recipe iterate-from-logs        # open recipe 2
        stateset-agents recipe debug-stuck-reward       # open recipe 5
    """
    import re
    import shutil

    # Locate the cookbook
    candidates = [
        Path(__file__).resolve().parents[1] / "docs" / "COOKBOOK.md",
        Path(__file__).resolve().parent / "docs" / "COOKBOOK.md",
    ]
    cookbook = next((p for p in candidates if p.exists()), None)
    if cookbook is None:
        _echo(
            "Cookbook not bundled in this install. Read it on GitHub:\n"
            "  https://github.com/stateset/stateset-agents/blob/master/docs/COOKBOOK.md"
        )
        return

    body = cookbook.read_text()

    # Extract every recipe by matching "## Recipe N — <title>" headers
    recipe_re = re.compile(r"^## Recipe (\d+) — (.+?)$", re.MULTILINE)
    matches = list(recipe_re.finditer(body))

    def _slug(title: str) -> str:
        s = title.lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        return s.strip("-")

    if name == "list":
        _echo("Available cookbook recipes:")
        for m in matches:
            num, title = m.group(1), m.group(2)
            slug = _slug(title)
            _echo(f"  {num}. {slug:<28} — {title}")
        _echo("")
        _echo("Open one with: stateset-agents recipe <slug>")
        _echo("Read the full cookbook: cat docs/COOKBOOK.md")
        return

    # Find the matching recipe (by slug or numeric prefix)
    name_norm = name.lower().strip()
    chosen_idx: int | None = None
    for i, m in enumerate(matches):
        slug = _slug(m.group(2))
        if slug == name_norm or slug.startswith(name_norm) or m.group(1) == name_norm:
            chosen_idx = i
            break

    if chosen_idx is None:
        print(f"No recipe matches {name!r}. Run `stateset-agents recipe list`.", file=sys.stderr)
        raise typer.Exit(code=2)

    # Extract the recipe content (header through to the next ## or end)
    start = matches[chosen_idx].start()
    end = matches[chosen_idx + 1].start() if chosen_idx + 1 < len(matches) else len(body)
    section = body[start:end]

    # Route through $PAGER if available + TTY
    pager = os.environ.get("PAGER")
    if pager is None:
        pager = "less -R" if shutil.which("less") else None
    if pager and sys.stdout.isatty():
        import subprocess
        try:
            subprocess.run(pager, shell=True, check=False, input=section, text=True)
            return
        except Exception:
            pass
    print(section)


@app.command("tour")
def tour() -> None:
    """Open the platform tour — the one document that walks the full developer journey.

    Tries (in order): the bundled `docs/PLATFORM_TOUR.md` next to the installed
    package, the source-tree copy, and finally a URL to the GitHub copy as a
    fallback. Routes through ``$PAGER`` when stdout is a TTY.
    """
    import shutil

    candidates: list[Path] = []
    # Source-tree copy (developer install).
    src = Path(__file__).resolve().parents[1] / "docs" / "PLATFORM_TOUR.md"
    candidates.append(src)
    # Installed-package copy (if we package the docs).
    pkg = Path(__file__).resolve().parent / "docs" / "PLATFORM_TOUR.md"
    candidates.append(pkg)

    tour_path = next((p for p in candidates if p.exists()), None)
    if tour_path is None:
        _echo(
            "Platform tour not found in this install. Read it on GitHub:\n"
            "  https://github.com/stateset/stateset-agents/blob/master/docs/PLATFORM_TOUR.md"
        )
        return

    pager = os.environ.get("PAGER")
    if pager is None:
        pager = "less -R" if shutil.which("less") else None

    if pager and sys.stdout.isatty():
        # Pipe through PAGER for TTY users.
        import subprocess
        try:
            subprocess.run(f"{pager} {tour_path}", shell=True, check=False)
            return
        except Exception:
            pass

    # Fallback: dump to stdout (CI / non-TTY).
    print(tour_path.read_text())


@app.command("starter")
def starter(
    template: str = typer.Argument(
        ...,
        help="Template name (or `list` to see options).",
    ),
    output: str = typer.Argument(
        "",
        help="Output directory for the scaffolded project (not needed for `list`).",
    ),
    project_name: str | None = typer.Option(
        None, "--name", "-n",
        help="Project name (defaults to the basename of the output directory).",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite an existing non-empty directory.",
    ),
    client_name: str | None = typer.Option(
        None, "--client-name",
        help="Client name (slugified) — patches output_dir paths and the W&B project name throughout the scaffold.",
    ),
) -> None:
    """Scaffold a fork-and-go fine-tuning project.

    Available templates:
      * customer-support  — multi-turn dialogue agent (the differentiator)
      * gsm8k-math        — single-turn math reasoner, verifiable rewards
      * minimal           — bare scaffold

    Examples:

        stateset-agents starter customer-support ./client-acme
        stateset-agents starter gsm8k-math ./math-bench --force
        stateset-agents starter list
    """
    from stateset_agents.scaffolding import (
        SCAFFOLD_TEMPLATES,
        list_templates,
        scaffold_project,
    )

    if template == "list":
        _echo("Available starter templates:")
        for t in list_templates():
            _echo(f"  {t.name:18s}  {t.description}")
        return

    if not output:
        print(
            "OUTPUT directory is required when scaffolding. "
            "Use `stateset-agents starter list` to see available templates.",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    if template not in SCAFFOLD_TEMPLATES:
        available = ", ".join(sorted(SCAFFOLD_TEMPLATES))
        print(
            f"Unknown template {template!r}. Available: {available}",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    try:
        created = scaffold_project(
            template_name=template,
            output_dir=output,
            project_name=project_name,
            force=force,
            client_name=client_name,
        )
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(code=2) from e

    _echo(f"Created {len(created)} files in {output}/")
    _echo("")
    _echo("Next steps:")
    _echo(f"  cd {output}")
    _echo("  pip install -r requirements.txt")
    if template == "customer-support":
        _echo("  # Edit scenarios.jsonl with your customer data, then:")
        _echo("  python train.py")
        _echo("  ./serve.sh outputs/customer_support_v1")
    elif template == "gsm8k-math":
        _echo("  python train.py    # downloads GSM8K and trains")
    else:
        _echo("  python train.py")


benchmark_app = typer.Typer(
    add_completion=False,
    help="Run and aggregate Phase 0 / whitepaper-v1 benchmarks.",
)


@benchmark_app.command("smoke")
def benchmark_smoke() -> None:
    """Quick end-to-end smoke test of the GSM8K benchmark pipeline (no training).

    Verifies that the dataset loads, answers parse, seeds initialize, and the
    runner is importable. Takes about 6 seconds; needs no GPU.
    """
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_phase0_benchmark.py"
    if not script.exists():
        _echo(f"Benchmark script not found at {script}", err=True)
        raise typer.Exit(code=1)

    result = subprocess.run(
        [
            sys.executable, str(script),
            "--trainer", "gspo",
            "--smoke-test",
            "--output", "/tmp/stateset_smoke.json",
        ],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


@benchmark_app.command("phase0")
def benchmark_phase0(
    trainer: str = typer.Option("gspo", "--trainer", "-t",
                                help="Trainer to benchmark: grpo, gspo, dapo."),
    model: str = typer.Option("Qwen/Qwen3.5-0.8B", "--model", "-m"),
    seed: int = typer.Option(42, "--seed", "-s"),
    output: str = typer.Option(
        "benchmark_results/whitepaper_v1/run.json",
        "--output", "-o",
        help="Path for the JSON result file.",
    ),
    num_train_examples: int = typer.Option(200, "--num-train-examples"),
    num_eval_examples: int = typer.Option(100, "--num-eval-examples"),
) -> None:
    """Run a single Phase 0 benchmark and emit a schema-compliant JSON result.

    The result file conforms to benchmark_results/SCHEMA.md and is suitable
    for aggregation via `stateset-agents benchmark aggregate`.
    """
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_phase0_benchmark.py"
    if not script.exists():
        _echo(f"Benchmark script not found at {script}", err=True)
        raise typer.Exit(code=1)

    result = subprocess.run(
        [
            sys.executable, str(script),
            "--trainer", trainer,
            "--model", model,
            "--seed", str(seed),
            "--output", output,
            "--num-train-examples", str(num_train_examples),
            "--num-eval-examples", str(num_eval_examples),
        ],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


@benchmark_app.command("plot")
def benchmark_plot(
    results_dir: str = typer.Option(
        "benchmark_results/whitepaper_v1",
        "--results-dir", "-d",
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o"),
    no_matplotlib: bool = typer.Option(
        False, "--no-matplotlib",
        help="Skip PNG figures; emit only text_plots.md.",
    ),
) -> None:
    """Generate publication figures from aggregated benchmark results.

    Reads ``summary.csv`` from the results directory and writes two PNGs plus
    a text-table fallback. Run ``aggregate`` first to produce the CSV.
    """
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_phase0_results.py"
    if not script.exists():
        _echo(f"Plot script not found at {script}", err=True)
        raise typer.Exit(code=1)

    cmd = [sys.executable, str(script), "--results-dir", results_dir]
    if output_dir:
        cmd += ["--output-dir", output_dir]
    if no_matplotlib:
        cmd += ["--no-matplotlib"]
    result = subprocess.run(cmd, check=False)
    raise typer.Exit(code=result.returncode)


@benchmark_app.command("aggregate")
def benchmark_aggregate(
    results_dir: str = typer.Option(
        "benchmark_results/whitepaper_v1",
        "--results-dir", "-d",
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o"),
    strict: bool = typer.Option(
        False, "--strict",
        help="Exit non-zero if any (trainer, model) group fails publication gates.",
    ),
) -> None:
    """Aggregate all *.json results in a directory into summary.md + summary.csv.

    The publication gates (3 seeds, σ < 0.1, +0.03 improvement) are defined in
    benchmark_results/SCHEMA.md. Use --strict to fail CI on any gate violation.
    """
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_phase0_results.py"
    if not script.exists():
        _echo(f"Aggregate script not found at {script}", err=True)
        raise typer.Exit(code=1)

    cmd = [sys.executable, str(script), "--results-dir", results_dir]
    if output_dir:
        cmd += ["--output-dir", output_dir]
    if strict:
        cmd += ["--strict"]
    result = subprocess.run(cmd, check=False)
    raise typer.Exit(code=result.returncode)


app.add_typer(benchmark_app, name="benchmark")


def _register_advanced_cli() -> None:
    """Register optional advanced CLI only when dependencies are available."""
    try:
        from stateset_agents.cli_advanced import app as advanced_app
    except ImportError:
        @app.command("advanced")
        def advanced() -> None:
            _echo("Advanced CLI requires optional dependencies (rich).")
            _echo("Install with: pip install stateset-agents[dev]")
            _echo("Tip: use 'stateset-agents advanced --help' after installing.")
            raise typer.Exit(code=2)

        return

    app.add_typer(
        advanced_app,
        name="advanced",
        help="Advanced StateSet Agents commands",
    )


_register_advanced_cli()


def run() -> None:
    app()


if __name__ == "__main__":
    run()
