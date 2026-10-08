"""Run reviewed reward-function checks without loading a training model."""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import typer

from stateset_agents.cli import app


@app.command("reward-audit")
def reward_audit(
    suite: Path = typer.Argument(..., help="Reviewed JSON candidate suite."),
    reward: str = typer.Option(
        ..., "--reward", help="Python module:factory returning a reward instance."
    ),
    output: Path = typer.Option(
        ..., "--output", help="New JSON report file; never overwritten."
    ),
    repeats: int = typer.Option(3, min=2),
    score_tolerance: float = typer.Option(1e-8),
    min_informative_fraction: float = typer.Option(1.0),
    timeout_seconds: float = typer.Option(30.0),
) -> None:
    """Check reward signal, repeatability, and expected candidate rankings."""
    from stateset_agents.evaluation.reward_audit import (
        RewardAuditPolicy,
        audit_reward,
        load_reward_suite,
    )

    try:
        policy = RewardAuditPolicy(
            repeats, score_tolerance, min_informative_fraction, timeout_seconds
        )
        cases = load_reward_suite(suite)
        if output.exists() or output.is_symlink():
            raise ValueError(f"Report already exists: {output}")
        if not output.parent.is_dir():
            raise ValueError("Report parent directory must already exist")
        module_name, separator, factory_name = reward.partition(":")
        if (
            not separator
            or not all(part.isidentifier() for part in module_name.split("."))
            or not factory_name.isidentifier()
        ):
            raise ValueError("--reward must be module:factory")
        try:
            factory = getattr(importlib.import_module(module_name), factory_name)
            instance = factory()
        except Exception as exc:
            raise ValueError(
                f"Reward factory failed: {type(exc).__name__}: {exc}"
            ) from exc
        report = asyncio.run(
            audit_reward(instance, cases, policy=policy, reward_name=reward)
        )
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError, ImportError, AttributeError) as exc:
        typer.echo(f"Reward audit could not run: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "passed": report["passed"],
                "failure_reasons": report["failure_reasons"],
                "summary": report["summary"],
                "report": str(output),
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise typer.Exit(1)
