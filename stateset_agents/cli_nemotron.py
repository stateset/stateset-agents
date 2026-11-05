"""Multimodal Nemotron fine-tuning commands (optional GPU dependencies)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from stateset_agents.cli import app


def _register_nemotron_vl(command: str, model: str) -> None:
    @app.command(command)
    def fine_tune(
        config: Path | None = typer.Option(None, help="Load a JSON configuration."),
        train_data: str | None = typer.Option(
            None, help="Local image/chat training JSONL."
        ),
        validation_data: str | None = typer.Option(
            None, help="Separate validation JSONL."
        ),
        image_root: str | None = typer.Option(
            None, help="Root directory for local images."
        ),
        output_dir: str | None = typer.Option(
            None, help="Adapter and processor output directory."
        ),
        max_steps: int | None = typer.Option(None, help="Number of optimizer steps."),
        fsdp: bool | None = typer.Option(
            None, help="Enable sharding when launched with torchrun."
        ),
        write_config: Path | None = typer.Option(
            None, help="Write settings without loading weights."
        ),
        check_data: bool = typer.Option(
            False, help="Validate JSONL and local image paths."
        ),
        dry_run: bool = typer.Option(
            True, help="Describe settings; --no-dry-run loads weights and trains."
        ),
    ) -> None:
        """Fine-tune BF16 weights with image-conditioned, assistant-only LoRA SFT."""
        from stateset_agents.training.nemotron_vl import (
            NemotronVLConfig,
            describe_nemotron_vl,
            train_nemotron_vl,
        )

        try:
            settings: dict[str, Any] = (
                json.loads(config.read_text(encoding="utf-8"))
                if config
                else {"model": model}
            )
            if not isinstance(settings, dict):
                raise ValueError("configuration must be a JSON object")
            if settings.get("model", model) != model:
                raise ValueError(f"{command} requires model={model!r}")
            settings["model"] = model
            for name, value in (
                ("train_data", train_data),
                ("validation_data", validation_data),
                ("image_root", image_root),
                ("output_dir", output_dir),
                ("max_steps", max_steps),
                ("fsdp", fsdp),
            ):
                if value is not None:
                    settings[name] = value
            resolved = NemotronVLConfig.model_validate(settings)
            if write_config:
                write_config.parent.mkdir(parents=True, exist_ok=True)
                write_config.write_text(
                    resolved.model_copy(
                        update={"revision": resolved.model_revision}
                    ).model_dump_json(indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                typer.echo(f"Wrote {write_config}")
                return
            report = (
                describe_nemotron_vl(resolved, check_data=check_data)
                if dry_run
                else train_nemotron_vl(resolved)
            )
            typer.echo(json.dumps(report, indent=2))
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            typer.echo(f"Nemotron fine-tuning failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc


_register_nemotron_vl("nemotron-3-5-super-vl", "super-vl")
_register_nemotron_vl("nemotron-3-5-bolt", "bolt")
