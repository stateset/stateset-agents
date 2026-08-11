"""Diagnostic & release-prep meta commands subcommands for the StateSet Agents CLI.

Split out of stateset_agents/cli.py. Each command attaches to the parent
Typer app exported by cli; helpers _echo, _load_config, etc. are
re-bound locally for readability. Helpers that tests patch on
stateset_agents.cli (_collect_dependency_status, _collect_import_status)
are looked up via late binding through the _cli module reference so the
patches still propagate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import CLI_IMPORT_EXCEPTIONS, app

# Late-bound helper aliases — these are unpatched in tests.
_echo = _cli._echo
_load_config = _cli._load_config
_validate_config = _cli._validate_config

# NOTE: _collect_dependency_status and _collect_import_status are NOT
# imported as local names — tests patch them via stateset_agents.cli, so
# call sites use _cli._collect_... for late binding.


@app.command()
def doctor(
    strict: bool = typer.Option(
        False, help="Fail if required dependencies are missing."
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "--json-output",
        help="Output machine-readable diagnostics",
    ),
) -> None:
    """Check environment and dependencies for common issues."""
    import platform

    required_status, optional_status = _cli._collect_dependency_status()

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
            _echo(
                f"❌ STATESET_DEFAULT_CHECKPOINT={default_checkpoint} (path does not exist!)"
            )
        if default_base_model:
            _echo(f"   STATESET_DEFAULT_BASE_MODEL={default_base_model}")
        else:
            _echo(
                "   ⚠️  STATESET_DEFAULT_BASE_MODEL not set — LoRA loading will be skipped."
            )
    else:
        _echo(
            "ℹ️  No STATESET_DEFAULT_CHECKPOINT set (use `stateset-agents serve --checkpoint` to serve a trained adapter)."
        )

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
    strict: bool = typer.Option(
        False, help="Fail if required dependencies are missing."
    ),
    fail_on_warnings: bool = typer.Option(
        False,
        help="Fail if config warnings are present.",
    ),
    json_output: bool = typer.Option(False, help="Output machine-readable diagnostics"),
) -> None:
    """Run environment and config checks in one command."""
    import platform

    required_status, optional_status = _cli._collect_dependency_status()
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

    (
        required_dependency_status,
        optional_dependency_status,
    ) = _cli._collect_dependency_status()
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

    required_import_status = _cli._collect_import_status(required_import_modules)
    optional_import_status = _cli._collect_import_status(optional_import_modules)

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
