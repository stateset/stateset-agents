"""``stateset-agents distill``: knowledge distillation from a frozen teacher.

Offline logit KD (forward/reverse KL or generalized JSD on dataset
completions, optionally from a top-k teacher cache) and on-policy GKD
(``--lmbda``). Heavy imports happen inside the command so the CLI (and
``--dry-run``) stay torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from stateset_agents.cli import app

#: CLI flag -> DistillationConfig field for the plain overrides.
_FIELDS = {
    "divergence": "divergence",
    "jsd_beta": "jsd_beta",
    "temperature": "temperature",
    "lmbda": "lmbda",
    "on_policy_loss": "on_policy_loss",
    "max_new_tokens": "max_new_tokens",
    "teacher_top_k": "teacher_top_k",
    "vocab_alignment": "vocab_alignment",
    "learning_rate": "learning_rate",
    "num_epochs": "num_epochs",
    "max_steps": "max_steps",
    "batch_size": "batch_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "max_length": "max_length",
    "max_prompt_length": "max_prompt_length",
    "use_lora": "use_lora",
    "seed": "seed",
    "device": "device",
    "validation_fraction": "validation_fraction",
}


@app.command("distill")
def distill(
    train_data: Path = typer.Option(
        ...,
        "--train-data",
        help="Distillation JSONL: prompt/messages with an optional completion.",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Student: Hugging Face model id or local directory."
    ),
    teacher: str | None = typer.Option(
        None, "--teacher", help="Teacher: Hugging Face model id or local directory."
    ),
    eval_data: Path | None = typer.Option(
        None, "--eval-data", help="Held-out JSONL; must not share prompts."
    ),
    validation_fraction: float | None = typer.Option(None, "--validation-fraction"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    config: Path | None = typer.Option(
        None, "--config", help="JSON object of DistillationConfig fields."
    ),
    divergence: str | None = typer.Option(
        None, "--divergence", help="forward_kl (default), reverse_kl or jsd."
    ),
    jsd_beta: float | None = typer.Option(None, "--jsd-beta"),
    temperature: float | None = typer.Option(None, "--temperature"),
    lmbda: float | None = typer.Option(
        None, "--lmbda", help="Fraction of batches on student samples (GKD λ)."
    ),
    on_policy_loss: str | None = typer.Option(
        None, "--on-policy-loss", help="full (GKD) or sampled (sampled-token RKL)."
    ),
    max_new_tokens: int | None = typer.Option(None, "--max-new-tokens"),
    teacher_top_k: int | None = typer.Option(
        None, "--teacher-top-k", help="Train from a top-k teacher cache."
    ),
    teacher_cache: Path | None = typer.Option(
        None, "--teacher-cache", help="Load a saved top-k teacher cache."
    ),
    save_teacher_cache: Path | None = typer.Option(
        None, "--save-teacher-cache", help="Write the top-k cache built here."
    ),
    vocab_alignment: str | None = typer.Option(
        None, "--vocab-alignment", help="strict (default) or shared."
    ),
    learning_rate: float | None = typer.Option(None, "--learning-rate"),
    num_epochs: int | None = typer.Option(None, "--epochs"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    gradient_accumulation_steps: int | None = typer.Option(
        None, "--gradient-accumulation-steps"
    ),
    max_length: int | None = typer.Option(None, "--max-length"),
    max_prompt_length: int | None = typer.Option(None, "--max-prompt-length"),
    use_lora: bool | None = typer.Option(None, "--lora/--no-lora"),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    model_revision: str | None = typer.Option(None, "--model-revision"),
    teacher_revision: str | None = typer.Option(None, "--teacher-revision"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the data, split and settings; print the plan without "
        "loading a model.",
    ),
) -> None:
    """Distil a teacher into a student: offline KD or on-policy GKD."""
    flags = {
        name: value
        for name, value in locals().items()
        if name in _FIELDS and value is not None
    }
    if output_dir is not None:
        flags["output_dir"] = str(output_dir)
    try:
        settings = _settings(config, flags)
        train, evaluation = _splits(train_data, eval_data, settings)
        plan = {
            "model": model,
            "teacher": teacher,
            "teacher_cache": str(teacher_cache) if teacher_cache else None,
            "train": train.summary(),
            "eval": evaluation.summary() if evaluation is not None else None,
            "config": settings.to_dict(),
        }
        if dry_run:
            typer.echo(json.dumps(plan, indent=2))
            return
        report = _train(
            plan,
            train,
            evaluation,
            settings,
            revisions=(model_revision, teacher_revision),
            save_cache=save_teacher_cache,
        )
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        typer.echo(f"Distillation could not run: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(report, indent=2))


def _settings(config: Path | None, flags: dict[str, Any]) -> Any:
    from stateset_agents.training.distillation.config import DistillationConfig

    values: Any = json.loads(config.read_text(encoding="utf-8")) if config else {}
    if not isinstance(values, dict):
        raise ValueError("--config must hold a JSON object")
    overrides = {_FIELDS.get(name, name): value for name, value in flags.items()}
    return DistillationConfig.from_dict({**values, **overrides})


def _splits(train_data: Path, eval_data: Path | None, settings: Any) -> Any:
    from stateset_agents.training.distillation.data import (
        resolve_distillation_splits,
    )

    train, evaluation = resolve_distillation_splits(
        train_data, eval_data, settings.validation_fraction, settings.seed
    )
    if float(settings.lmbda) < 1.0 and not train.has_completions:
        raise ValueError(
            "every training record needs a completion unless --lmbda 1 "
            "(on-policy only)"
        )
    return train, evaluation


def _train(
    plan: dict[str, Any],
    train: Any,
    evaluation: Any,
    settings: Any,
    *,
    revisions: tuple[str | None, str | None],
    save_cache: Path | None,
) -> dict[str, Any]:
    from stateset_agents.training.distillation.trainer import train_distillation

    if not plan["model"]:
        raise ValueError("--model is required unless --dry-run")
    if not plan["teacher"] and not plan["teacher_cache"]:
        raise ValueError("--teacher (or --teacher-cache) is required")
    if save_cache is not None and settings.teacher_top_k is None:
        raise ValueError("--save-teacher-cache needs --teacher-top-k")
    result = train_distillation(
        plan["model"],
        train,
        teacher=plan["teacher"],
        eval_data=evaluation,
        config=settings,
        teacher_cache=plan["teacher_cache"],
        model_revision=revisions[0],
        teacher_revision=revisions[1],
    )
    if save_cache is not None and result.trainer.teacher_cache is not None:
        result.trainer.teacher_cache.save(save_cache)
    summary = result.summary
    return {
        "model": plan["model"],
        "teacher": plan["teacher"],
        "checkpoint": str(result.checkpoint) if result.checkpoint else None,
        "global_step": summary.global_step,
        "skipped_steps": summary.skipped_steps,
        "final_train_metrics": summary.history[-1] if summary.history else None,
        "eval_metrics": summary.eval_metrics,
        "data": summary.data,
    }
