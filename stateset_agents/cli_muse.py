"""CLI for the revision-pinned Muse Spark distributed text SFT path."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from stateset_agents.cli import app
from stateset_agents.training.muse_spark import (
    MuseSparkSFTConfig,
    describe_muse_spark,
    train_muse_spark,
)


@app.command("muse-spark-open")
def muse_spark_open(
    config: Path | None = typer.Option(None, "--config", help="SFT JSON configuration"),
    train_data: Path | None = typer.Option(None, "--train-data"),
    validation_data: Path | None = typer.Option(None, "--validation-data"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    write_config: Path | None = typer.Option(None, "--write-config"),
    check_data: bool = typer.Option(False, "--check-data"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Configure Muse Spark Open assistant-only, distributed BF16 LoRA SFT."""
    try:
        settings = (
            MuseSparkSFTConfig.model_validate_json(config.read_text(encoding="utf-8"))
            if config
            else MuseSparkSFTConfig()
        )
        changes = {
            name: str(value)
            for name, value in (
                ("train_data", train_data),
                ("validation_data", validation_data),
                ("output_dir", output_dir),
            )
            if value is not None
        }
        settings = MuseSparkSFTConfig.model_validate(
            {**settings.model_dump(), **changes}
        )
        if write_config:
            write_config.parent.mkdir(parents=True, exist_ok=True)
            write_config.write_text(settings.model_dump_json(indent=2) + "\n")
            typer.echo(str(write_config))
            return
        report = (
            describe_muse_spark(settings, check_data=check_data)
            if dry_run
            else train_muse_spark(settings)
        )
        typer.echo(json.dumps(report, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
