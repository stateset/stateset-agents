"""``stateset-agents preference``: offline preference optimization and reward models.

Heavy imports happen inside the command so the CLI (and ``--dry-run``) stay
torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from stateset_agents.cli import app

METHOD_CHOICES = ("dpo", "ipo", "simpo", "kto", "orpo", "rm")
#: Settings that only apply to policy methods (not the ``rm`` reward model).
_POLICY_ONLY = (
    "beta",
    "label_smoothing",
    "simpo_gamma",
    "precompute_reference_logps",
)


@app.command("preference")
def preference(
    train_data: Path = typer.Option(
        ...,
        "--train-data",
        help="Preference JSONL: prompt/messages with chosen/rejected (or "
        "completion/label for KTO).",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Hugging Face model id or local model directory."
    ),
    method: str = typer.Option(
        "dpo",
        "--method",
        help="dpo, ipo, simpo, kto, orpo, or rm (Bradley-Terry reward model).",
    ),
    eval_data: Path | None = typer.Option(
        None, "--eval-data", help="Held-out JSONL; must not share prompts."
    ),
    validation_fraction: float | None = typer.Option(
        None,
        "--validation-fraction",
        help="Hold out this fraction of prompt groups when --eval-data is absent.",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Where the final checkpoint and metrics go."
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="JSON object of PreferenceConfig (or RewardModelConfig) fields.",
    ),
    model_revision: str | None = typer.Option(None, "--model-revision"),
    beta: float | None = typer.Option(None, "--beta"),
    learning_rate: float | None = typer.Option(None, "--learning-rate"),
    num_epochs: int | None = typer.Option(None, "--epochs"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    gradient_accumulation_steps: int | None = typer.Option(
        None, "--gradient-accumulation-steps"
    ),
    max_length: int | None = typer.Option(None, "--max-length"),
    max_prompt_length: int | None = typer.Option(None, "--max-prompt-length"),
    label_smoothing: float | None = typer.Option(
        None, "--label-smoothing", help="Conservative DPO / SimPO label noise."
    ),
    simpo_gamma: float | None = typer.Option(None, "--simpo-gamma"),
    use_lora: bool | None = typer.Option(None, "--lora/--no-lora"),
    precompute_reference: bool | None = typer.Option(
        None,
        "--precompute-reference/--no-precompute-reference",
        help="Score the reference once up front and release it.",
    ),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the data, split and settings; print the plan without "
        "loading a model.",
    ),
) -> None:
    """Train a policy with DPO/IPO/SimPO/KTO/ORPO, or a Bradley-Terry reward model."""
    from stateset_agents.training.preference.config import (
        PreferenceConfig,
        RewardModelConfig,
    )
    from stateset_agents.training.preference.data import resolve_splits

    try:
        if method not in METHOD_CHOICES:
            raise ValueError(f"--method must be one of {', '.join(METHOD_CHOICES)}")
        values: Any = json.loads(config.read_text(encoding="utf-8")) if config else {}
        if not isinstance(values, dict):
            raise ValueError("--config must hold a JSON object")
        flags = {
            "beta": beta,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
            "label_smoothing": label_smoothing,
            "simpo_gamma": simpo_gamma,
            "use_lora": use_lora,
            "precompute_reference_logps": precompute_reference,
            "seed": seed,
            "device": device,
            "validation_fraction": validation_fraction,
            "output_dir": str(output_dir) if output_dir is not None else None,
        }
        overrides = {name: value for name, value in flags.items() if value is not None}
        settings: PreferenceConfig | RewardModelConfig
        if method == "rm":
            misplaced = sorted(set(overrides) & set(_POLICY_ONLY))
            if misplaced:
                raise ValueError(f"--method rm does not take {misplaced}")
            settings = RewardModelConfig.from_dict({**values, **overrides})
        else:
            settings = PreferenceConfig.from_dict(
                {**values, **overrides, "method": method}
            )
        train, evaluation = resolve_splits(
            train_data, eval_data, settings.validation_fraction, settings.seed
        )
        for split in (train, evaluation):
            if split is not None and split.unpaired and method != "kto":
                raise ValueError(
                    "unpaired (completion/label) records need --method kto"
                )
            if method != "kto" and split is not None and not split.pairs:
                raise ValueError(f"--method {method} needs chosen/rejected pairs")
        plan = {
            "method": method,
            "model": model,
            "train": train.summary(),
            "eval": evaluation.summary() if evaluation is not None else None,
            "config": settings.to_dict(),
        }
        if dry_run:
            typer.echo(json.dumps(plan, indent=2))
            return
        if not model:
            raise ValueError("--model is required unless --dry-run")
        report = _train(method, model, train, evaluation, settings, model_revision)
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        typer.echo(f"Preference training could not run: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(report, indent=2))


def _train(
    method: str,
    model: str,
    train: Any,
    evaluation: Any,
    settings: Any,
    model_revision: str | None,
) -> dict[str, Any]:
    result: Any
    if method == "rm":
        from stateset_agents.training.preference.reward_model import (
            train_reward_model,
        )

        result = train_reward_model(
            model,
            train,
            eval_data=evaluation,
            config=settings,
            model_revision=model_revision,
        )
    else:
        from stateset_agents.training.preference.trainer import train_preference

        result = train_preference(
            model,
            train,
            eval_data=evaluation,
            config=settings,
            model_revision=model_revision,
        )
    summary = result.summary
    return {
        "method": method,
        "model": model,
        "checkpoint": str(result.checkpoint) if result.checkpoint else None,
        "global_step": summary.global_step,
        "skipped_steps": summary.skipped_steps,
        "final_train_metrics": summary.history[-1] if summary.history else None,
        "eval_metrics": summary.eval_metrics,
        "data": summary.data,
    }
