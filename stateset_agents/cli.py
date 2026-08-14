import importlib
import json
import os
import sys
import typing as t
from pathlib import Path
from typing import Any

import typer

# When invoked via ``python -m stateset_agents.cli`` this module loads as
# ``__main__``. Subcommand modules (``cli_train``, …) reach back via
# ``from stateset_agents import cli`` which would otherwise trigger a *second*
# import of this file under the canonical name — producing two ``app``
# instances and dropping subcommand registrations on the floor. Aliasing the
# running module under its canonical name keeps a single app instance.
sys.modules.setdefault("stateset_agents.cli", sys.modules[__name__])

from stateset_agents.exceptions import (  # noqa: E402 — must follow the sys.modules alias above
    INFERENCE_EXCEPTIONS,
)

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


def _echo(s: str, err: bool = False) -> None:
    typer.echo(s, err=err)


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
            import yaml
        except ImportError as exc:
            _echo(
                "PyYAML is required for YAML config files. Install with: pip install pyyaml"
            )
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
        if "use_stub_model" in agent_cfg and not isinstance(
            agent_cfg["use_stub_model"], bool
        ):
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
    ),
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
        import pathlib
        import subprocess

        _pkg_file = __import__("stateset_agents").__file__
        if _pkg_file is None:
            raise RuntimeError("stateset_agents.__file__ is unset")
        pkg_root = pathlib.Path(_pkg_file).resolve().parent.parent
        if (pkg_root / ".git").exists():
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=pkg_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
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
    _echo("  deps:")
    for name, ver in deps.items():
        marker = "✓" if ver else "—"
        ver_str = ver if ver else "not installed"
        _echo(f"    {marker} {name:<14} {ver_str}")


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
    host: str = typer.Option("0.0.0.0", help="Bind host"),  # nosec: B104
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload (development)"),
    dry_run: bool = typer.Option(
        False, help="Print startup command without running the server."
    ),
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
def evaluate(
    checkpoint: str | None = typer.Option(
        None, "--checkpoint", help="Path to a saved checkpoint directory"
    ),
    message: str = typer.Option(
        "Hello", help="Single message to evaluate (ignored when --scenarios is set)"
    ),
    scenarios: str | None = typer.Option(
        None,
        "--scenarios",
        help='JSONL of scenarios for batch mode. Each line: {"user_query": ..., <reward-specific context>}.',
    ),
    reward: str | None = typer.Option(
        None,
        "--reward",
        help="Reward name for batch mode: gsm8k, customer_support, tool_calling.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the batch-mode markdown report to this path (default: stdout).",
    ),
    threshold: float = typer.Option(
        0.7,
        "--threshold",
        help="Pass/fail threshold for the markdown summary (just informational).",
    ),
    dry_run: bool = typer.Option(
        False, help="Show evaluation plan without loading checkpoint."
    ),
) -> None:
    """Evaluate a saved checkpoint — single message or batch with reward.

    Single-message mode (the original behavior):

        stateset-agents evaluate --checkpoint outputs/v1 --message "Hello"

    Batch mode — score every scenario in a JSONL against a reward function:

        stateset-agents evaluate --checkpoint outputs/v1 \\
            --scenarios eval_set.jsonl --reward customer_support \\
            --output eval_report.md

    The batch markdown report shows mean score, perfect/zero counts, and
    a per-scenario table. Same shape as ``grade_transcript`` output so the
    two reports compose naturally.
    """
    import asyncio

    if dry_run:
        _echo("Dry-run: evaluation was not executed.")
        if checkpoint:
            _echo(f"Checkpoint: {checkpoint}")
        if scenarios:
            _echo(f"Scenarios: {scenarios}")
            _echo(f"Reward: {reward}")
            _echo(f"Output: {output or '(stdout)'}")
        else:
            _echo(f"Message: {message}")
        return

    # Argument validation before filesystem checks — gives clearer errors.
    if scenarios:
        if not reward:
            print("--reward is required with --scenarios.", file=sys.stderr)
            raise typer.Exit(code=2)
        if reward not in {"gsm8k", "customer_support", "tool_calling"}:
            print(
                f"Unknown reward: {reward!r}. Options: gsm8k, customer_support, tool_calling.",
                file=sys.stderr,
            )
            raise typer.Exit(code=2)

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

    # Batch mode — score every scenario.
    if scenarios:
        scenarios_path = Path(scenarios)
        if not scenarios_path.exists():
            print(f"Scenarios file not found: {scenarios}", file=sys.stderr)
            raise typer.Exit(code=2)

        rows = [
            json.loads(line)
            for line in scenarios_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            print("No scenarios loaded.", file=sys.stderr)
            raise typer.Exit(code=2)

        # Build the reward function.
        from stateset_agents.core.reward_base import RewardFunction

        reward_fn: RewardFunction
        if reward == "gsm8k":
            from stateset_agents.data.gsm8k import GSM8KReward

            reward_fn = GSM8KReward()
        elif reward == "customer_support":
            from stateset_agents.data.customer_support_bench import (
                SupportRewardComposite,
            )

            reward_fn = SupportRewardComposite()
        else:
            from stateset_agents.data.tool_calling_bench import ToolCallReward

            reward_fn = ToolCallReward()

        async def _run_batch():
            from stateset_agents.core.trajectory import ConversationTurn

            agent = await load_agent_from_checkpoint(checkpoint, load_model=True)
            results = []
            for row in rows:
                query = (
                    row.get("user_query")
                    or row.get("question")
                    or row.get("prompt")
                    or ""
                )
                response = await agent.generate_response(
                    [{"role": "user", "content": query}]
                )
                turns = [ConversationTurn(role="assistant", content=response)]
                result = await reward_fn.compute_reward(turns, context=row)
                results.append(
                    {
                        "query": query,
                        "response": response,
                        "score": float(result.score),
                    }
                )
            return results

        try:
            results = asyncio.run(_run_batch())
        except CLI_TRAIN_EXCEPTIONS as e:
            _echo(f"Batch evaluation failed: {e}")
            raise typer.Exit(code=2) from e

        # Render markdown summary.
        scores = [r["score"] for r in results]
        mean = sum(scores) / len(scores) if scores else 0.0
        import statistics

        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        n_pass = sum(1 for s in scores if s >= threshold)

        lines = [
            f"# Batch evaluation — `{reward}`",
            "",
            f"**Checkpoint:** `{checkpoint}`",
            f"**Scenarios:** {len(results)}",
            f"**Mean score:** {mean:.3f} ± {std:.3f}",
            f"**Pass rate (≥ {threshold}):** {n_pass}/{len(results)} ({100 * n_pass / len(results):.1f}%)",
            "",
            "| # | Score | Query | Response (head) |",
            "|---|-------|-------|-----------------|",
        ]
        for i, r in enumerate(results):
            marker = (
                "✅"
                if r["score"] >= threshold
                else ("⚠️ " if r["score"] >= 0.1 else "❌")
            )
            q_preview = r["query"][:50].replace("|", "\\|").replace("\n", " ")
            r_preview = r["response"][:50].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i} | {marker} {r['score']:.3f} | {q_preview} | {r_preview} |"
            )

        md = "\n".join(lines) + "\n"
        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
            _echo(f"Wrote batch eval report → {output}")
        else:
            print(md)
        return

    # Single-message mode (preserved behavior).
    async def _run() -> str:
        agent = await load_agent_from_checkpoint(checkpoint, load_model=True)
        resp = await agent.generate_response([{"role": "user", "content": message}])
        return str(resp)

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
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing file"
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml or json",
    ),
    preset: str = typer.Option(
        "default",
        "--preset",
        help="Starter preset: default, qwen3-5-0-8b, kimi-k2-6, kimi-k3, gemma-4-31b, muse-glimmer, nemotron-3-5, qwen3.8-27b, qwen3-coder, gpt-oss, or deepseek-v4",
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

    if preset not in {
        "default",
        "qwen3-5-0-8b",
        "kimi-k2-6",
        "kimi-k3",
        "gemma-4-31b",
        "muse-glimmer",
        "nemotron-3-5",
        "qwen3.8-27b",
        "qwen3-coder",
        "gpt-oss",
        "deepseek-v4",
    }:
        _echo(
            "Unsupported preset. Use one of: default, qwen3-5-0-8b, kimi-k2-6, kimi-k3, gemma-4-31b, muse-glimmer, nemotron-3-5, qwen3.8-27b, qwen3-coder, gpt-oss, deepseek-v4."
        )
        raise typer.Exit(code=2)

    if preset == "default" and starter_profile != "balanced":
        _echo(
            "`--starter-profile` only applies to --preset qwen3-5-0-8b, kimi-k2-6, kimi-k3, gemma-4-31b, muse-glimmer, nemotron-3-5, qwen3.8-27b, qwen3-coder, gpt-oss, or deepseek-v4."
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
            _echo(
                f"Unsupported task. Use one of: {', '.join(QWEN35_08B_TASK_CHOICES)}."
            )
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
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
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
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    elif preset == "kimi-k3":
        try:
            from stateset_agents.training.kimi_k3_starter import (
                KIMI_K3_STARTER_PROFILE_CHOICES,
                KIMI_K3_TASK_CHOICES,
                get_kimi_k3_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Kimi-K3 starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in KIMI_K3_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(KIMI_K3_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in KIMI_K3_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(KIMI_K3_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_kimi_k3_config(task=task, starter_profile=starter_profile).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    elif preset == "muse-glimmer":
        try:
            from stateset_agents.training.muse_glimmer_starter import (
                MUSE_GLIMMER_STARTER_PROFILE_CHOICES,
                MUSE_GLIMMER_TASK_CHOICES,
                get_muse_glimmer_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Muse Glimmer starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in MUSE_GLIMMER_TASK_CHOICES:
            _echo(
                f"Unsupported task. Use one of: {', '.join(MUSE_GLIMMER_TASK_CHOICES)}."
            )
            raise typer.Exit(code=2)
        if starter_profile not in MUSE_GLIMMER_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(MUSE_GLIMMER_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_muse_glimmer_config(
            task=task, starter_profile=starter_profile
        ).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    elif preset == "nemotron-3-5":
        try:
            from stateset_agents.training.nemotron_3_5_starter import (
                NEMOTRON_3_5_STARTER_PROFILE_CHOICES,
                NEMOTRON_3_5_TASK_CHOICES,
                get_nemotron_3_5_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Nemotron 3.5 starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in NEMOTRON_3_5_TASK_CHOICES:
            _echo(
                f"Unsupported task. Use one of: {', '.join(NEMOTRON_3_5_TASK_CHOICES)}."
            )
            raise typer.Exit(code=2)
        if starter_profile not in NEMOTRON_3_5_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(NEMOTRON_3_5_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_nemotron_3_5_config(
            task=task, starter_profile=starter_profile
        ).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    elif preset == "qwen3.8-27b":
        try:
            from stateset_agents.training.qwen3_8_starter import (
                QWEN38_27B_STARTER_PROFILE_CHOICES,
                QWEN38_27B_TASK_CHOICES,
                get_qwen3_8_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Qwen3.8 27B starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in QWEN38_27B_TASK_CHOICES:
            _echo(
                f"Unsupported task. Use one of: {', '.join(QWEN38_27B_TASK_CHOICES)}."
            )
            raise typer.Exit(code=2)
        if starter_profile not in QWEN38_27B_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(QWEN38_27B_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_qwen3_8_config(task=task, starter_profile=starter_profile).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    elif preset == "qwen3-coder":
        try:
            from stateset_agents.training.qwen3_coder_starter import (
                QWEN3_CODER_STARTER_PROFILE_CHOICES,
                QWEN3_CODER_TASK_CHOICES,
                get_qwen3_coder_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("Qwen3 Coder starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in QWEN3_CODER_TASK_CHOICES:
            _echo(
                f"Unsupported task. Use one of: {', '.join(QWEN3_CODER_TASK_CHOICES)}."
            )
            raise typer.Exit(code=2)
        if starter_profile not in QWEN3_CODER_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(QWEN3_CODER_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_qwen3_coder_config(
            task=task, starter_profile=starter_profile
        ).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    elif preset == "gpt-oss":
        try:
            from stateset_agents.training.gpt_oss_starter import (
                GPT_OSS_STARTER_PROFILE_CHOICES,
                GPT_OSS_TASK_CHOICES,
                get_gpt_oss_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("gpt-oss starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in GPT_OSS_TASK_CHOICES:
            _echo(f"Unsupported task. Use one of: {', '.join(GPT_OSS_TASK_CHOICES)}.")
            raise typer.Exit(code=2)
        if starter_profile not in GPT_OSS_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(GPT_OSS_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_gpt_oss_config(task=task, starter_profile=starter_profile).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
                raise typer.Exit(code=2) from e
            serialized = yaml.safe_dump(cfg, sort_keys=False)
    elif preset == "deepseek-v4":
        try:
            from stateset_agents.training.deepseek_v4_starter import (
                DEEPSEEK_V4_STARTER_PROFILE_CHOICES,
                DEEPSEEK_V4_TASK_CHOICES,
                get_deepseek_v4_config,
            )
        except CLI_IMPORT_EXCEPTIONS as e:
            _echo("deepseek-v4 starter helpers unavailable. Install training extras.")
            _echo(f"Details: {e}")
            raise typer.Exit(code=2) from e

        if task not in DEEPSEEK_V4_TASK_CHOICES:
            _echo(
                f"Unsupported task. Use one of: {', '.join(DEEPSEEK_V4_TASK_CHOICES)}."
            )
            raise typer.Exit(code=2)
        if starter_profile not in DEEPSEEK_V4_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(DEEPSEEK_V4_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_deepseek_v4_config(
            task=task, starter_profile=starter_profile
        ).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
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
            _echo(
                f"Unsupported task. Use one of: {', '.join(GEMMA4_31B_TASK_CHOICES)}."
            )
            raise typer.Exit(code=2)
        if starter_profile not in GEMMA4_31B_STARTER_PROFILE_CHOICES:
            _echo(
                f"Unsupported starter profile. Use one of: {', '.join(GEMMA4_31B_STARTER_PROFILE_CHOICES)}."
            )
            raise typer.Exit(code=2)

        cfg = get_gemma4_31b_config(
            task=task, starter_profile=starter_profile
        ).to_dict()
        if format == "json":
            serialized = json.dumps(cfg, indent=2) + "\n"
        else:
            try:
                import yaml
            except ImportError as e:
                _echo(
                    "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
                )
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
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing file"
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml or json",
    ),
    preset: str = typer.Option(
        "default",
        "--preset",
        help="Starter preset: default, qwen3-5-0-8b, kimi-k2-6, kimi-k3, gemma-4-31b, muse-glimmer, nemotron-3-5, qwen3.8-27b, qwen3-coder, gpt-oss, or deepseek-v4",
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

    body = cookbook.read_text(encoding="utf-8")

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
        print(
            f"No recipe matches {name!r}. Run `stateset-agents recipe list`.",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    # Extract the recipe content (header through to the next ## or end)
    start = matches[chosen_idx].start()
    end = (
        matches[chosen_idx + 1].start() if chosen_idx + 1 < len(matches) else len(body)
    )
    section = body[start:end]

    # Route through $PAGER if available + TTY
    pager = os.environ.get("PAGER")
    if pager is None:
        pager = "less -R" if shutil.which("less") else None
    if pager and sys.stdout.isatty():
        import shlex
        import subprocess

        try:
            subprocess.run(shlex.split(pager), check=False, input=section, text=True)
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
        import shlex
        import subprocess

        try:
            subprocess.run([*shlex.split(pager), str(tour_path)], check=False)
            return
        except Exception:
            pass

    # Fallback: dump to stdout (CI / non-TTY).
    print(tour_path.read_text(encoding="utf-8"))


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
        None,
        "--name",
        "-n",
        help="Project name (defaults to the basename of the output directory).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an existing non-empty directory.",
    ),
    client_name: str | None = typer.Option(
        None,
        "--client-name",
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


# Subcommand modules. Each module imports `app` (the same instance defined
# above — see the sys.modules alias at the top of this file) and attaches its
# commands via `@app.command()` decorators at import time. Import order is
# irrelevant for correctness but is kept stable so `--help` output is
# reproducible.
from stateset_agents import cli_benchmark  # noqa: E402, F401 — benchmark sub-app
from stateset_agents import cli_chat  # noqa: E402, F401 — chat
from stateset_agents import cli_ingest  # noqa: E402, F401 — ingest third-party logs
from stateset_agents import cli_mcp  # noqa: E402, F401 — run the MCP server
from stateset_agents import cli_remote  # noqa: E402, F401 — train-remote
from stateset_agents import cli_research  # noqa: E402, F401 — auto-research, fine-tune
from stateset_agents import (  # noqa: E402, F401 — grade -> curate -> retrain loop; noqa: E402, F401 — train, qwen3-5-0-8b, kimi-k2-6, gemma-4-31b; noqa: E402, F401 — doctor, preflight, publish-check
    cli_improve,
    cli_meta,
    cli_train,
)


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
    # Windows consoles default to a legacy codepage (cp1252) that cannot
    # encode the CLI's unicode output (checkmarks, arrows); force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    app()


if __name__ == "__main__":
    run()
