"""Private dataset, experiment and staging deployment commands."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

import typer

from stateset_agents.improvement import (
    Deployment,
    Experiment,
    ExperimentConfig,
    evidence_catalog,
    prepare_dataset,
)
from stateset_agents.improvement.storage import read_json

app = typer.Typer(
    help="Private improvement workflows; keep all customer artifacts outside Git."
)


def _emit(value: dict[str, Any]) -> None:
    typer.echo(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except Exception as error:  # Public boundary: never expose private exception text.
        # Provider/parser messages can contain customer text or credentials.
        typer.echo(
            f"Workflow operation failed ({type(error).__name__}). Check the private workspace and documented prerequisites; do not retry an uncertain paid attempt.",
            err=True,
        )
        raise typer.Exit(code=2) from None


def _paid(acknowledge_river_billing: bool) -> None:
    if not acknowledge_river_billing:
        typer.echo(
            "Execution requires --acknowledge-river-billing. River costs are unknown; this workflow enforces no dollar ceiling.",
            err=True,
        )
        raise typer.Exit(code=2)


@app.command("dataset")
def dataset(
    source: Path,
    destination: Path,
    validation_after: str = typer.Option(...),
    test_after: str = typer.Option(...),
    review: Path | None = typer.Option(None),
    synthetic: bool = typer.Option(False),
) -> None:
    """Prepare reviewed data outside Git; print only split counts and source kind."""
    with _errors():
        manifest = prepare_dataset(
            source,
            destination,
            validation_after=validation_after,
            test_after=test_after,
            review=review,
            synthetic=synthetic,
        )
        _emit(
            {
                "source_kind": manifest["source_kind"],
                "production_quality_established": False,
                "counts": {
                    s: len(read_json(destination / f"{s}.json"))
                    for s in ("train", "validation", "test")
                },
            }
        )


@app.command("plan")
def plan(workspace: Path, dataset: Path, config: Path) -> None:
    """Freeze the experiment without contacting River."""
    with _errors():
        experiment = Experiment.create(
            workspace, dataset, ExperimentConfig(**read_json(config))
        )
        _emit(experiment.status())


@app.command("status")
def status(workspace: Path) -> None:
    """Show content-free counts, evidence classification and uncertain attempts."""
    with _errors():
        _emit(Experiment(workspace).status())


@app.command("train")
def train(
    workspace: Path,
    acknowledge_river_billing: bool = typer.Option(False),
    allow_provider_upload: bool = typer.Option(False),
) -> None:
    """Upload the private training split to River and train each fixed seed once."""
    _paid(acknowledge_river_billing)
    if not allow_provider_upload:
        typer.echo(
            "Training uploads data to River; pass --allow-provider-upload after approving that use.",
            err=True,
        )
        raise typer.Exit(code=2)
    with _errors():
        from stateset_agents.improvement.river import RiverBackend

        experiment = Experiment(workspace)
        experiment.train(RiverBackend())
        _emit(experiment.status())


@app.command("evaluate")
def evaluate(
    workspace: Path,
    split: str = typer.Option("validation"),
    acknowledge_river_billing: bool = typer.Option(False),
    allow_provider_upload: bool = typer.Option(False),
) -> None:
    """Run synthetic refund-policy evaluation; all tool traces are replayed."""
    _paid(acknowledge_river_billing)
    if not allow_provider_upload:
        typer.echo(
            "Evaluation uploads prompts to River; pass --allow-provider-upload after approving that use.",
            err=True,
        )
        raise typer.Exit(code=2)
    with _errors():
        from stateset_agents.improvement.river import RiverBackend

        experiment = Experiment(workspace)
        experiment.evaluate(RiverBackend(), split=split)
        _emit(experiment.status())


@app.command("compare")
def compare(workspace: Path, split: str = typer.Option("test")) -> None:
    """Recompute paired gates; print no transcripts, paths, case IDs or reviewers."""
    with _errors():
        experiment = Experiment(workspace)
        result = experiment.compare(split=split)
        summary = experiment.status()
        _emit(
            {
                "passed": result["passed"],
                "gates": result["gates"],
                "mean_success_gain": result["mean_success_gain"],
                "production_quality_established": False,
                "source_kind": summary["source_kind"],
                "evidence_mode": summary["evidence_mode"],
                "evaluation_scope": summary["evaluation_scope"],
            }
        )


@app.command("approve")
def approve(workspace: Path, reviewer: str = typer.Option(...)) -> None:
    """Approve the prespecified candidate for staging only, after passing test."""
    with _errors():
        approval = Experiment(workspace).approve(reviewer=reviewer)
        _emit(
            {
                "approved": True,
                "evidence_mode": approval["evidence_mode"],
                "scope": "staging_only",
                "production_quality_established": False,
            }
        )


@app.command("prepare-deployment")
def prepare_deployment(
    workspace: Path,
    experiment: Path,
    previous_deployment_id: str | None = typer.Option(None),
) -> None:
    """Prepare a one-replica staging deployment without provider calls."""
    with _errors():
        run = Experiment(experiment)
        Deployment.prepare(
            workspace,
            run,
            previous_deployment_id=previous_deployment_id,
        )
        _emit(
            {
                "prepared": True,
                "replicas": 1,
                "cost_usd": None,
                "scope": "staging_only",
                "evidence_mode": run.status()["evidence_mode"],
            }
        )


@app.command("deployment")
def deployment(
    action: str,
    workspace: Path,
    acknowledge_river_billing: bool = typer.Option(False),
    rollback_on_failure: bool = typer.Option(False),
) -> None:
    """Deploy/reconcile/activate/monitor/rollback/stop a retained staging deployment."""
    if action not in ("deploy", "reconcile", "activate", "monitor", "rollback", "stop"):
        raise typer.BadParameter(
            "Expected deploy, reconcile, activate, monitor, rollback or stop"
        )
    if action in ("deploy", "reconcile"):
        _paid(acknowledge_river_billing)
    with _errors():
        from stateset_agents.improvement.deployments import RiverDeploymentProvider
        from stateset_agents.improvement.river import river_client

        manager = Deployment(workspace)
        with closing(river_client()) as client:
            provider = RiverDeploymentProvider(client)
            if action in ("deploy", "reconcile"):
                result = manager.deploy(provider, reconcile=action == "reconcile")
            elif action == "activate":
                result = manager.activate(provider)
            elif action == "monitor":
                result = manager.monitor(
                    provider, rollback_on_failure=rollback_on_failure
                )
            elif action == "rollback":
                result = manager.rollback(provider)
            else:
                manager.stop(provider)
                result = {"ready": False}
            _emit(
                {
                    "operation_completed": True,
                    "evidence_mode": provider.evidence_mode,
                    "ready": result["ready"],
                    "cost_usd": None,
                    "scope": "staging_only",
                    "shutdown_confirmed": result.get("phase") == "scaled_to_zero",
                }
            )


@app.command("evidence")
def evidence(repository: Path | None = typer.Option(None)) -> None:
    """List only curated, already-public synthetic evidence; optionally verify hashes."""
    with _errors():
        _emit(evidence_catalog(repository=repository))


@app.command("synthetic-fixture")
def synthetic_fixture(workspace: Path, count: int = typer.Option(32)) -> None:
    """Generate offline fictional conversations outside Git for the tutorial."""
    with _errors():
        from stateset_agents.improvement.fixtures import create_synthetic_fixture

        create_synthetic_fixture(workspace, count=count)
        _emit(
            {
                "source_kind": "synthetic",
                "cases_per_split": count,
                "provider_calls": 0,
                "production_quality_established": False,
            }
        )
