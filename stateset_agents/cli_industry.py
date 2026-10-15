"""Industry dataset preparation and SFT commands, with lazy training imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from stateset_agents.cli import app

industry_app = typer.Typer(help="Prepare industry datasets and train LoRA adapters.")
app.add_typer(industry_app, name="industry")


def _emit(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


def _error(exc: Exception) -> None:
    typer.echo(str(exc), err=True)
    raise typer.Exit(code=2) from exc


@industry_app.command("list")
def list_recipes() -> None:
    """List the nine supported industry starter recipes."""
    from dataclasses import asdict

    from stateset_agents.training.industry import list_industry_recipes

    _emit([asdict(recipe) for recipe in list_industry_recipes()])


@industry_app.command("show")
def show_recipe(industry: str = typer.Argument(...)) -> None:
    """Show a recipe's scope, system prompt, and evaluation criteria."""
    from dataclasses import asdict

    from stateset_agents.training.industry import get_industry_recipe

    try:
        recipe = get_industry_recipe(industry)
        _emit({**asdict(recipe), "system_prompt": recipe.system_prompt})
    except ValueError as exc:
        _error(exc)


@industry_app.command("init")
def init_project(
    industry: str = typer.Argument(...), output: Path = typer.Argument(...)
) -> None:
    """Write synthetic format examples and guidance into a new directory."""
    from stateset_agents.training.industry import init_industry_project

    try:
        path = init_industry_project(industry, output)
        _emit({"directory": str(path), "status": "created", "synthetic_examples": True})
    except (OSError, ValueError) as exc:
        _error(exc)


@industry_app.command("validate")
def validate_dataset(dataset: Path = typer.Argument(...)) -> None:
    """Strictly validate chat JSONL and tool-call/result structure."""
    from stateset_agents.data.finetuning import load_finetuning_data

    try:
        rows = load_finetuning_data(dataset)
        _emit({"status": "valid", "rows": len(rows)})
    except (OSError, ValueError) as exc:
        _error(exc)


@industry_app.command("prepare")
def prepare_project(
    industry: str = typer.Argument(...),
    dataset: Path = typer.Argument(...),
    output: Path = typer.Argument(...),
    model: str = typer.Option("qwen3.5-2b", help="Registered model preset name."),
    validation_fraction: float = typer.Option(
        0.2, help="Fraction of source groups held out."
    ),
    seed: int = typer.Option(42, help="Deterministic dataset split seed."),
    model_revision: str | None = typer.Option(
        None, help="Immutable 40-character Hugging Face model commit."
    ),
) -> None:
    """Validate, deduplicate, split, and hash data in a new training directory."""
    from stateset_agents.training.industry import prepare_industry_training

    try:
        _emit(
            prepare_industry_training(
                industry,
                dataset,
                output,
                model=model,
                validation_fraction=validation_fraction,
                seed=seed,
                model_revision=model_revision,
            )
        )
    except (OSError, ValueError) as exc:
        _error(exc)


@industry_app.command("train")
def train_project(
    project: Path = typer.Argument(...),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview without loading any model."
    ),
    num_epochs: int = typer.Option(3, min=1),
    max_length: int = typer.Option(1024, min=2),
) -> None:
    """Verify prepared data and run BF16 LoRA SFT on CUDA."""
    from stateset_agents.training.industry import train_industry_project

    try:
        _emit(
            train_industry_project(
                project,
                dry_run=dry_run,
                num_epochs=num_epochs,
                max_length=max_length,
            )
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        _error(exc)


@industry_app.command("eval-export")
def export_evaluation(
    project: Path = typer.Argument(...), output: Path = typer.Argument(...)
) -> None:
    """Export held-out assistant prefixes for paired model inference."""
    from stateset_agents.evaluation.industry import export_industry_evaluation

    try:
        requests = export_industry_evaluation(project)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(requests, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        _emit(
            {
                "output": str(output),
                "cases": len(requests["cases"]),
                "suite_sha256": requests["suite_sha256"],
            }
        )
    except (OSError, ValueError) as exc:
        _error(exc)


@industry_app.command("evaluate")
def evaluate_project(
    project: Path = typer.Argument(...),
    baseline: Path = typer.Option(..., help="Base-model prediction bundle JSON."),
    candidate: Path = typer.Option(..., help="Adapter prediction bundle JSON."),
    output: Path = typer.Option(..., help="New comparison report path."),
    min_groups: int = typer.Option(30, min=1),
    min_success_rate: float = typer.Option(
        0.9,
        min=0,
        max=1,
        help="Minimum fraction of source groups matching every reference turn.",
    ),
    min_improvement: float = typer.Option(0.0, min=0, max=1),
    max_regression_rate: float = typer.Option(0.0, min=0, max=1),
    max_mean_latency_seconds: float | None = typer.Option(None, min=0),
    max_mean_cost_usd: float | None = typer.Option(None, min=0),
) -> None:
    """Compare reference agreement; exit 1 for failed gates, 2 for invalid input."""
    from stateset_agents.evaluation.industry import (
        IndustryEvaluationPolicy,
        evaluate_industry_project,
    )

    try:
        report = evaluate_industry_project(
            project,
            json.loads(baseline.read_text(encoding="utf-8")),
            json.loads(candidate.read_text(encoding="utf-8")),
            policy=IndustryEvaluationPolicy(
                min_groups=min_groups,
                min_success_rate=min_success_rate,
                min_improvement=min_improvement,
                max_regression_rate=max_regression_rate,
                max_mean_latency_seconds=max_mean_latency_seconds,
                max_mean_cost_usd=max_mean_cost_usd,
            ),
        )
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        _emit(
            {
                "output": str(output),
                "gate": report["gate"],
                "group_reference_match_delta": report["group_reference_match_delta"],
            }
        )
    except (OSError, ValueError) as exc:
        _error(exc)
    if not report["gate"]["passed"]:
        raise typer.Exit(code=1)
