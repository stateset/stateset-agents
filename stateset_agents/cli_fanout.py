"""``stateset-agents fanout``: run independent RL runs in parallel on sandboxes.

Thin over :mod:`stateset_agents.fanout`. Its one real decision is the spend gate:
paid compute starts only against a confirmed worst case, computed from the
per-sandbox budget the platform enforces.

    stateset-agents fanout plan sweep.yaml
    stateset-agents fanout run sweep.yaml --run-dir runs/sweep            # local
    stateset-agents fanout run sweep.yaml --run-dir runs/sweep \\
        --backend sandbox --confirm-max-cost-usd 12
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from stateset_agents.cli import app

fanout_app = typer.Typer(
    help="Fan RL workloads out into independent, parallel runs.",
    no_args_is_help=True,
)
app.add_typer(fanout_app, name="fanout")

BACKENDS = ("local", "sandbox")

_MANIFEST_ARGUMENT = typer.Argument(..., help="Fan-out manifest (YAML or JSON).")
_RUN_DIR_ARGUMENT = typer.Argument(..., help="A fan-out's run directory.")
_CONFIRM_OPTION = typer.Option(
    None,
    "--confirm-max-cost-usd",
    help="Approve up to this many dollars. Must cover the computed worst case.",
)
_POLL_OPTION = typer.Option(5.0, "--poll-interval", help="Seconds between polls.")
_JSON_OPTION = typer.Option(False, "--json", help="Print JSON.")
_RATE_OPTION = typer.Option(
    None,
    "--max-requests-per-minute",
    min=1,
    help=(
        "Sandbox API rate across all runs. The default suits the smallest plan "
        "(60/min); on larger ones raise it so that many runs are polled sooner."
    ),
)


def _refuse(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code=2)


def _load_manifest(path: Path) -> Any:
    from stateset_agents.fanout.manifest import FanoutManifest, ManifestError

    try:
        return FanoutManifest.from_file(path)
    except ManifestError as exc:
        raise _refuse(f"Invalid manifest: {exc}") from exc


def _run_info(run_dir: Path) -> dict[str, Any]:
    from stateset_agents.fanout.journal import load_run_file

    try:
        return load_run_file(run_dir)
    except ValueError as exc:
        raise _refuse(str(exc)) from exc


def _make_backend(
    name: str, run_dir: Path, requests_per_minute: int | None = None
) -> Any:
    if name == "local":
        from stateset_agents.fanout.local_backend import LocalBackend

        return LocalBackend(run_dir / "scratch")
    from stateset_agents.fanout.sandbox_backend import (
        DEFAULT_REQUESTS_PER_MINUTE,
        SandboxBackend,
    )

    return SandboxBackend(
        requests_per_minute=requests_per_minute or DEFAULT_REQUESTS_PER_MINUTE
    )


def _check_spend(
    manifest: Any,
    run_dir: Path,
    backend: str,
    confirm: float | None,
    *,
    retry_failed: bool = False,
) -> None:
    """Paid compute needs a known worst case, and approval to spend it."""
    if backend == "local":
        return
    from stateset_agents.fanout.orchestrator import remaining_worst_case_usd

    worst = remaining_worst_case_usd(manifest, run_dir, retry_failed=retry_failed)
    if worst is None:
        raise _refuse(
            "Set resources.max_cost_cents in the manifest: a fan-out multiplies "
            "spend, so every run needs a ceiling the platform enforces."
        )
    if confirm is None or confirm < worst:
        raise _refuse(
            f"This could spend up to ${worst:.2f} on compute and LLM calls: every "
            f"remaining run, retried to its limit, at its ceiling. Network transfer "
            f"past the plan's allowance is billed on top. Approve it with "
            f"--confirm-max-cost-usd {worst:.2f}"
        )


def _print_event(event: dict[str, Any]) -> None:
    kind = event["type"]
    run = event.get("run_id", "")
    if kind == "attempt_started":
        typer.echo(f"[{run}] attempt {event['attempt']} on {event['allocation_id']}")
    elif kind == "attempt_finished":
        reason = f": {event['reason']}" if event.get("reason") else ""
        typer.echo(f"[{run}] attempt {event['attempt']} {event['outcome']}{reason}")
    elif kind == "capacity_wait":
        typer.echo(f"[{run}] waiting for capacity: {event['reason']}")
    elif kind == "time_limit_shortened":
        typer.echo(
            f"[{run}] the sandbox was granted {event['granted']:.0f}s, so this run's "
            f"limit is {event['limit']}s rather than {event['requested']}s",
            err=True,
        )
    elif kind == "run_reattached":
        typer.echo(f"[{run}] reattached to attempt {event['attempt']}")
    elif kind == "release_failed":
        typer.echo(
            f"[{run}] could not stop {event['allocation_id']}: {event['error']}",
            err=True,
        )
    elif kind == "fanout_aborted":
        typer.echo(f"Stopped launching: {event['reason']}", err=True)


def _execute(
    manifest: Any,
    backend: str,
    run_dir: Path,
    *,
    poll_interval: float,
    requests_per_minute: int | None = None,
    retry_failed: bool = False,
) -> None:
    from stateset_agents.fanout.backend import FatalBackendError
    from stateset_agents.fanout.orchestrator import FanoutRunner
    from stateset_agents.fanout.report import render_report

    try:
        summary = FanoutRunner(
            manifest,
            _make_backend(backend, run_dir, requests_per_minute),
            run_dir,
            retry_failed=retry_failed,
            poll_interval_s=poll_interval,
            on_event=_print_event,
        ).run()
    except (FatalBackendError, ValueError) as exc:
        raise _refuse(str(exc)) from exc
    # Warnings first: printing the report can raise if stdout has gone away, and
    # these are the lines that cost money to miss.
    if summary.unreleased:
        typer.echo(
            f"WARNING: these may still be running and billing: "
            f"{', '.join(summary.unreleased)}. "
            f"Retry with: stateset-agents fanout resume {run_dir}",
            err=True,
        )
    if summary.aborted_reason:
        typer.echo(
            f"Unfinished runs can continue with: stateset-agents fanout resume "
            f"{run_dir}",
            err=True,
        )
    typer.echo("")
    typer.echo(render_report(summary.report))
    typer.echo(f"Run directory: {run_dir}")
    if not summary.ok:
        raise typer.Exit(code=1)


@fanout_app.command("plan")
def plan(
    manifest_path: Path = _MANIFEST_ARGUMENT, as_json: bool = _JSON_OPTION
) -> None:
    """Validate a manifest and show what it would run. Free: allocates nothing."""
    manifest = _load_manifest(manifest_path)
    runs = manifest.expand()
    arms = {run.arm_id: run.params for run in runs}
    worst = manifest.worst_case_cost_usd()
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "name": manifest.name,
                    "digest": manifest.digest(),
                    "total_runs": len(runs),
                    "arms": len(arms),
                    "seeds": list(manifest.seeds),
                    "concurrency": manifest.concurrency,
                    "max_attempts": manifest.max_attempts,
                    "worst_case_usd": worst,
                    "runs": [
                        {
                            "run_id": run.run_id,
                            "arm_id": run.arm_id,
                            "seed": run.seed,
                            "params": run.params,
                        }
                        for run in runs
                    ],
                },
                indent=2,
            )
        )
        return
    resources = manifest.resources
    typer.echo(
        f"{manifest.name}: {len(runs)} runs "
        f"({len(arms)} arms x {len(manifest.seeds)} seeds), "
        f"{manifest.concurrency} at a time, up to {manifest.max_attempts} attempts each"
    )
    typer.echo(f"Workload: {manifest.workload['kind']}")
    typer.echo(
        f"Each run: {resources.cpus} CPU, {resources.memory}, "
        f"{resources.resource_class or 'default'} class, "
        f"{resources.timeout_seconds}s limit"
        + (f", {resources.gpu.count} GPU" if resources.gpu else "")
    )
    for arm_id, params in sorted(arms.items()):
        label = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        typer.echo(f"  {arm_id}  {label or '(no parameters)'}")
    if worst is None:
        typer.echo(
            "Worst case on the sandbox backend: unbounded. "
            "Set resources.max_cost_cents; sandbox runs require it."
        )
    else:
        typer.echo(
            f"Worst case on the sandbox backend: ${worst:.2f} "
            f"({len(runs)} runs x {manifest.max_attempts} attempts x "
            f"${resources.max_cost_cents / 100:.2f})"
        )


@fanout_app.command("run")
def run(
    manifest_path: Path = _MANIFEST_ARGUMENT,
    run_dir: Path = typer.Option(
        ..., "--run-dir", help="Where this fan-out's records go."
    ),
    backend: str = typer.Option("local", "--backend", help="local or sandbox."),
    confirm_max_cost_usd: float | None = _CONFIRM_OPTION,
    poll_interval: float = _POLL_OPTION,
    requests_per_minute: int | None = _RATE_OPTION,
) -> None:
    """Start a fan-out. Exits nonzero unless every run succeeded and came back."""
    from stateset_agents.fanout.journal import RUN_FILE

    manifest = _load_manifest(manifest_path)
    if backend not in BACKENDS:
        raise _refuse(f"Unknown backend {backend!r}; use one of: {', '.join(BACKENDS)}")
    if (run_dir / RUN_FILE).exists():
        raise _refuse(
            f"{run_dir} already holds a fan-out. Continue it with: "
            f"stateset-agents fanout resume {run_dir}"
        )
    _check_spend(manifest, run_dir, backend, confirm_max_cost_usd)
    _execute(
        manifest,
        backend,
        run_dir,
        poll_interval=poll_interval,
        requests_per_minute=requests_per_minute,
    )


@fanout_app.command("resume")
def resume(
    run_dir: Path = _RUN_DIR_ARGUMENT,
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Also rerun failed and cancelled runs."
    ),
    confirm_max_cost_usd: float | None = _CONFIRM_OPTION,
    poll_interval: float = _POLL_OPTION,
    requests_per_minute: int | None = _RATE_OPTION,
) -> None:
    """Continue a fan-out: reattach to what is running, start what has not run."""
    from stateset_agents.fanout.manifest import FanoutManifest, ManifestError

    info = _run_info(run_dir)
    try:
        manifest = FanoutManifest.from_dict(info["manifest"], base_dir=info["base_dir"])
    except ManifestError as exc:
        raise _refuse(f"Cannot restore the manifest: {exc}") from exc
    backend = str(info["backend"])
    _check_spend(
        manifest, run_dir, backend, confirm_max_cost_usd, retry_failed=retry_failed
    )
    _execute(
        manifest,
        backend,
        run_dir,
        poll_interval=poll_interval,
        requests_per_minute=requests_per_minute,
        retry_failed=retry_failed,
    )


@fanout_app.command("status")
def status(run_dir: Path = _RUN_DIR_ARGUMENT, as_json: bool = _JSON_OPTION) -> None:
    """Show every run's state. Safe while a driver is running."""
    from stateset_agents.fanout.journal import load_records

    info = _run_info(run_dir)
    records = load_records(run_dir)
    states = [record.status for record in records]
    counts = {
        "succeeded": states.count("succeeded"),
        "failed": states.count("failed"),
        "cancelled": states.count("cancelled"),
        "pending": states.count("pending"),
        "running": states.count("running"),
    }
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "name": info["name"],
                    "digest": info["digest"],
                    "backend": info["backend"],
                    "counts": counts,
                    "runs": [
                        {
                            "run_id": record.run_id,
                            "status": record.status,
                            "attempts": len(record.attempts),
                            "allocation_id": (
                                record.attempts[-1].allocation_id
                                if record.attempts
                                else None
                            ),
                            "error": record.error,
                            "cost_cents": record.cost_cents,
                            "metrics": record.metrics,
                            "last_event": record.last_event,
                        }
                        for record in records
                    ],
                },
                indent=2,
            )
        )
        return
    summary = ", ".join(f"{count} {name}" for name, count in counts.items() if count)
    typer.echo(f"{info['name']} on {info['backend']}: {summary}")
    for record in records:
        detail = record.error or ""
        if record.status == "running" and record.attempts:
            detail = f"on {record.attempts[-1].allocation_id}"
        cost = (
            f"${record.cost_cents / 100:.4f}" if record.cost_cents is not None else "?"
        )
        typer.echo(
            f"  {record.run_id}  {record.status}  "
            f"attempts={len(record.attempts)}  cost={cost}  {detail}".rstrip()
        )


@fanout_app.command("logs")
def logs(
    run_dir: Path = _RUN_DIR_ARGUMENT,
    run_id: str = typer.Argument(..., help="Run id, as shown by status."),
    stream: str = typer.Option("stdout", "--stream", help="stdout or stderr."),
    attempt: int | None = typer.Option(None, "--attempt", help="Default: the latest."),
) -> None:
    """Print a run's mirrored logs. They stay available after the sandbox stops."""
    from stateset_agents.fanout.journal import load_records

    _run_info(run_dir)
    record = next((r for r in load_records(run_dir) if r.run_id == run_id), None)
    if record is None:
        raise _refuse(f"No run {run_id!r} in {run_dir}")
    if stream not in ("stdout", "stderr"):
        raise _refuse("--stream must be stdout or stderr")
    if not record.attempts:
        raise _refuse(f"Run {run_id} has not started")
    number = attempt if attempt is not None else record.attempts[-1].number
    path = run_dir / "runs" / record.run_id / f"attempt-{number}" / f"{stream}.log"
    if not path.is_file():
        raise _refuse(f"No {stream} log for attempt {number}")
    typer.echo(path.read_text(encoding="utf-8", errors="replace"), nl=False)


@fanout_app.command("report")
def report(run_dir: Path = _RUN_DIR_ARGUMENT, as_json: bool = _JSON_OPTION) -> None:
    """Show per-arm statistics and the verdict, from the journal alone."""
    from stateset_agents.fanout.orchestrator import rebuild_report
    from stateset_agents.fanout.report import render_report

    _run_info(run_dir)
    built = rebuild_report(run_dir)
    typer.echo(json.dumps(built, indent=2) if as_json else render_report(built))


@fanout_app.command("cancel")
def cancel(run_dir: Path = _RUN_DIR_ARGUMENT) -> None:
    """Stop a fan-out and its sandboxes, even if the driver has died."""
    from stateset_agents.fanout.backend import FatalBackendError
    from stateset_agents.fanout.orchestrator import cancel_run

    info = _run_info(run_dir)
    try:
        outcome = cancel_run(
            run_dir, lambda: _make_backend(str(info["backend"]), run_dir)
        )
    except FatalBackendError as exc:
        raise _refuse(str(exc)) from exc
    if outcome.driver_active:
        typer.echo("Cancel requested: the running driver will stop its sandboxes.")
        return
    typer.echo(
        f"Stopped: {', '.join(outcome.stopped)}"
        if outcome.stopped
        else "No sandboxes were running."
    )
    if outcome.unreleased:
        typer.echo(
            f"WARNING: these may still be running and billing: "
            f"{', '.join(outcome.unreleased)}. Each is bounded by its own time "
            f"limit and budget. Retry with: stateset-agents fanout cancel "
            f"{run_dir}, or fanout resume to settle an unresolved create.",
            err=True,
        )
        raise typer.Exit(code=1)
