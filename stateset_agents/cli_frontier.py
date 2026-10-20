"""Frontier model readiness and distributed text fine-tuning commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from stateset_agents.cli import app
from stateset_agents.core.frontier_models import FRONTIER_MODELS, get_frontier_model
from stateset_agents.training.frontier_sft import (
    FrontierSFTConfig,
    describe_frontier_sft,
    train_frontier_sft,
)


@app.command("frontier-models")
def frontier_models(
    model: str | None = typer.Option(
        None, "--model", help="Preset key or official model ID"
    ),
) -> None:
    """List pinned checkpoints, adapter targets and explicit training blockers."""
    try:
        models = (get_frontier_model(model),) if model else FRONTIER_MODELS
        typer.echo(json.dumps([item.describe() for item in models], indent=2))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("frontier-sft")
def frontier_sft(
    model: str | None = typer.Option(
        None, "--model", help="Preset key or official model ID"
    ),
    config: Path | None = typer.Option(None, "--config", help="SFT JSON configuration"),
    train_data: Path | None = typer.Option(None, "--train-data"),
    validation_data: Path | None = typer.Option(None, "--validation-data"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    write_config: Path | None = typer.Option(None, "--write-config"),
    check_data: bool = typer.Option(False, "--check-data"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Preview or train a pinned frontier checkpoint with distributed text LoRA."""
    try:
        values = json.loads(config.read_text(encoding="utf-8")) if config else {}
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        values.update(
            {
                name: str(value)
                for name, value in (
                    ("model", model),
                    ("train_data", train_data),
                    ("validation_data", validation_data),
                    ("output_dir", output_dir),
                )
                if value is not None
            }
        )
        settings = FrontierSFTConfig.model_validate(values)
        if write_config:
            write_config.parent.mkdir(parents=True, exist_ok=True)
            write_config.write_text(
                settings.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            typer.echo(str(write_config))
            return
        report = (
            describe_frontier_sft(settings, check_data=check_data)
            if dry_run
            else train_frontier_sft(settings)
        )
        typer.echo(json.dumps(report, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
