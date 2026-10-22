"""Versioned JSON fine-tuning commands for AI agents, sharing the studio workspace."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from stateset_agents.data.text_sft import DatasetFormatError
from stateset_agents.finetune_evaluation import (
    EvaluationConfig,
    EvaluationProtocolError,
    evaluation_status,
    inspect_evaluation,
    launch_evaluation,
    list_evaluations,
    plan_evaluation,
    prepare_evaluation,
    stop_evaluation,
)
from stateset_agents.finetune_inspection import (
    compare_runs,
    inspect_readiness,
    inspect_training_data,
    inspect_workflow,
    metric_history,
)
from stateset_agents.finetune_nomination import (
    list_nominations,
    nomination_config,
    nomination_plan,
    nomination_status,
    prepare_nomination,
)
from stateset_agents.finetune_quality import inspect_quality, record_quality
from stateset_agents.finetune_recovery import inspect_checkpoints, prepare_resume
from stateset_agents.finetune_validation import (
    ValidationComparisonError,
    comparison_path,
    comparison_status,
    submit_comparison,
)
from stateset_agents.finetune_workspace import (
    LOG_SOURCES,
    METHODS,
    FinetuneConfig,
    check_config_values,
    create_run,
    failure_info,
    finetune_schema,
    launch_run,
    list_runs,
    plan_run,
    read_log_snapshot,
    read_run_plan,
    redact_finetune_secrets,
    request_stop,
    require_backend_credentials,
    run_status,
    run_terminal_status,
    verified_run_settings,
)

app = typer.Typer(
    add_completion=False, help="JSON fine-tuning interface for AI agents."
)
DEFAULT_WORKSPACE = Path("outputs/finetune")


class _RequestError(ValueError):
    def __init__(
        self, code: str, message: str, diagnostic: dict[str, Any] | None = None
    ) -> None:
        self.code = code
        self.diagnostic = diagnostic
        super().__init__(message)


def _redact(value: Any) -> Any:
    return redact_finetune_secrets(value)


def _perform(command: str, action: Callable[[], dict[str, Any]]) -> None:
    try:
        # Library diagnostics belong on stderr; stdout is one JSON object.
        with redirect_stdout(sys.stderr):
            result = action()
        response: dict[str, Any] = {
            "schema_version": 1,
            "ok": True,
            "command": command,
            "result": result,
        }
    except (_RequestError, EvaluationProtocolError, ValidationComparisonError) as error:
        response = {
            "schema_version": 1,
            "ok": False,
            "command": command,
            "error": {"code": error.code, "message": str(error)},
        }
        if error.diagnostic is not None:
            response["error"]["diagnostic"] = error.diagnostic
    except OSError:
        response = {
            "schema_version": 1,
            "ok": False,
            "command": command,
            "error": {
                "code": "io_error",
                "message": "Unable to access the requested files or start the worker. Check paths, permissions and retained run status.",
            },
        }
    except Exception:
        response = {
            "schema_version": 1,
            "ok": False,
            "command": command,
            "error": {
                "code": "operation_failed",
                "message": "The operation failed. Inspect retained local records; no automatic retry was attempted.",
            },
        }
    try:
        encoded = json.dumps(_redact(response), allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        response = {
            "schema_version": 1,
            "ok": False,
            "command": command,
            "error": {
                "code": "invalid_result",
                "message": "The retained result could not be represented as finite JSON.",
            },
        }
        encoded = json.dumps(response)
    typer.echo(encoded)
    if not response["ok"]:
        raise typer.Exit(1)


def _object(path: Path, *, limit: int = 128 * 1024) -> tuple[dict[str, Any], str]:
    """Read exact bounded record bytes without blocking on special files."""

    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite JSON number")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON number")
        return number

    message = "Expected a bounded JSON object with finite numbers."
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink():
        raise _RequestError("invalid_record", message)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise _RequestError("invalid_record", message)
        raw = stream.read(limit + 1)
    try:
        value = (
            json.loads(raw, parse_constant=reject_constant, parse_float=finite_float)
            if len(raw) <= limit
            else None
        )
    except (ValueError, RecursionError):
        value = None
    if not isinstance(value, dict):
        raise _RequestError("invalid_record", message)
    return value, hashlib.sha256(raw).hexdigest()


def _config(path: Path) -> FinetuneConfig:
    record, _ = _object(path.expanduser(), limit=64 * 1024)
    if set(record) - asdict(FinetuneConfig()).keys():
        raise _RequestError(
            "invalid_config",
            "Unknown configuration fields; use finetune schema for supported fields.",
        )
    try:
        config = FinetuneConfig(**record)
        config.validate()
    except ValueError as error:
        raise _RequestError("invalid_config", str(error)) from None
    except TypeError:
        raise _RequestError(
            "invalid_config",
            "Invalid configuration values; use finetune schema and check the studio constraints.",
        ) from None
    return config


def _run(workspace: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise _RequestError(
            "invalid_run_id", "Run ID must be a single workspace child name."
        )
    run = workspace.expanduser().resolve() / run_id
    if run.is_symlink() or not run.is_dir() or (run / "plan.json").is_symlink():
        raise _RequestError(
            "run_not_found",
            "No regular retained run with that ID exists in this workspace.",
        )
    return run


def _scalar(value: Any) -> bool:
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and math.isfinite(value))
    )


def _summary(run: Path) -> dict[str, Any]:
    unreadable = []
    digest: str | None
    try:
        plan, digest = _object(run / "plan.json")
    except (OSError, _RequestError):
        plan, digest = {}, None
        unreadable.append("plan.json")
    try:
        state = run_status(run)
        if not isinstance(state.get("status"), str):
            raise ValueError("Invalid state")
    except (OSError, ValueError, RecursionError):
        state = {"status": "unreadable"}
        unreadable.append("state.json")
    fields = (
        "status",
        "exit_code",
        "created_at",
        "started_at",
        "training_started_at",
        "finished_at",
        "committed_updates",
        "stop_requested",
        "execution_unconfirmed",
        "recovery_checkpoint",
        "recovery_unavailable",
        "teardown_error",
        "teardown_exit_code",
        "error",
    )
    config = plan.get("config", {})
    history = metric_history(run, limit=1)
    for event in history:
        event["metrics"] = {
            _redact(key): value for key, value in event["metrics"].items()
        }
        if not _scalar(event.get("updated_at")):
            event["updated_at"] = None
    result: dict[str, Any] = {
        "run_id": run.name,
        "path": str(run),
        "plan_sha256": digest,
        "config": (
            {
                key: value
                for key, value in config.items()
                if key in asdict(FinetuneConfig()) and _scalar(value)
            }
            if isinstance(config, dict)
            else {}
        ),
        "state": {
            key: state[key] for key in fields if key in state and _scalar(state[key])
        },
        "terminal": run_terminal_status(state),
        "latest_metrics": history,
        "failure": failure_info(run, state),
        "quality_gain_verified": False,
    }
    if isinstance(plan.get("resume"), dict):
        result["resume"] = {
            key: value
            for key, value in plan["resume"].items()
            if key
            in {
                "parent_run_id",
                "parent_plan_sha256",
                "checkpoint",
                "checkpoint_sha256",
                "kind",
                "step",
                "epoch",
                "recovery_sha256",
            }
            and _scalar(value)
        }
    for filename, name, keys in (
        ("provider-job.json", "provider", ("provider", "job_id")),
        ("provider-result.json", "provider_result", ("cost_usd", "duration_s")),
        (
            "completion.json",
            "completion",
            ("output_dir", "committed_updates", "result_kind", "checkpoint"),
        ),
    ):
        if not (run / filename).exists():
            continue
        try:
            record, _ = _object(run / filename)
            result[name] = {
                key: record[key]
                for key in keys
                if key in record and _scalar(record[key])
            }
        except (OSError, _RequestError):
            unreadable.append(filename)
    if (run / "provider-submission.json").exists() or (
        run / "provider-submission.json"
    ).is_symlink():
        from stateset_agents.finetune_provider_submission import inspect_submission

        try:
            result["provider_submission"] = inspect_submission(run)
        except (OSError, ValueError, TypeError, RecursionError):
            unreadable.append("provider-submission.json")
    result["unreadable_records"] = unreadable
    return result


@app.command("schema")
def schema() -> None:
    """Describe the shared agent protocol without reading data or authenticating."""
    _perform("schema", finetune_schema)


@app.command("validate")
def validate_config(
    config: Path = typer.Option(..., "--config"),
) -> None:
    """Report all configuration issues without reading data or contacting providers."""

    def check() -> dict[str, Any]:
        record, _ = _object(config.expanduser(), limit=64 * 1024)
        _, issues = check_config_values(record)
        return {
            "valid": not issues,
            "issues": [asdict(issue) for issue in issues],
            "dataset_contents_checked": False,
            "provider_access_checked": False,
            "training_started": False,
        }

    _perform("validate", check)


@app.command("models")
def models() -> None:
    """List catalog IDs and maturity without certifying hardware or backend fit."""

    def describe() -> dict[str, Any]:
        from stateset_agents.tui import model_choices

        return {
            "models": [
                {"model": model, "maturity": maturity}
                for model, maturity in model_choices()
            ],
            "backend_fit_verified": False,
        }

    _perform("models", describe)


@app.command("readiness")
def readiness(
    config: Path = typer.Option(..., "--config"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
) -> None:
    """Inspect local readiness; never authenticate a provider or provision compute."""
    _perform("readiness", lambda: inspect_readiness(_config(config), workspace))


@app.command("prepare")
def prepare(
    config: Path = typer.Option(..., "--config"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
) -> None:
    """Validate and retain an immutable input snapshot without starting training."""

    def create() -> dict[str, Any]:
        try:
            plan = plan_run(_config(config))
        except _RequestError:
            raise
        except (ValueError, TypeError):
            raise _RequestError(
                "invalid_dataset_or_plan",
                "Dataset or plan validation failed. Inspect the documented SFT/RL data format and provider constraints.",
            ) from None
        run = create_run(workspace.expanduser().resolve(), plan)
        summary = _summary(run)
        return {
            "run": summary,
            "review": plan.describe(),
            "requires_paid_opt_in": plan.config.provider != "local",
            "launch_argv": [
                sys.executable,
                "-m",
                "stateset_agents.cli",
                "finetune",
                "launch",
                run.name,
                "--workspace",
                str(run.parent),
                "--expected-plan-sha256",
                summary["plan_sha256"],
            ],
        }

    _perform("prepare", create)


@app.command("provider-status")
def provider_status(
    run_id: str,
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Read retained allocation receipts without provider access or automatic retry."""
    from stateset_agents.finetune_provider_submission import inspect_submission

    def inspect() -> dict[str, Any]:
        run = _run(workspace, run_id)
        try:
            return {"run_id": run_id, **inspect_submission(run)}
        except (OSError, ValueError, TypeError, RecursionError):
            raise _RequestError(
                "invalid_provider_receipt",
                "Cannot verify retained provider allocation receipts; inspect the private workspace before retrying.",
            ) from None

    _perform("provider-status", inspect)


@app.command("launch")
def launch(
    run_id: str = typer.Argument(...),
    expected_plan_sha256: str = typer.Option(..., "--expected-plan-sha256"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    allow_paid: bool = typer.Option(False, "--allow-paid"),
) -> None:
    """Launch exactly the reviewed plan once; a repeated submission starts no work."""

    def submit() -> dict[str, Any]:
        run = _run(workspace, run_id)
        _, digest = _object(run / "plan.json")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256)
            or digest != expected_plan_sha256
        ):
            raise _RequestError(
                "plan_mismatch",
                "Plan checksum differs from the reviewed plan. Inspect status and prepare a fresh review.",
            )
        if (run / "launch.lock").exists() or (run / "launch.lock").is_symlink():
            return {"launched": False, "already_submitted": True, "run": _summary(run)}
        try:
            _, settings = verified_run_settings(run)
        except (OSError, ValueError, TypeError, KeyError):
            raise _RequestError(
                "review_invalid",
                "Reviewed data, package identity or configuration no longer matches. Prepare a fresh run.",
            ) from None
        if settings.provider != "local":
            if not allow_paid:
                raise _RequestError(
                    "paid_opt_in_required",
                    "This plan is billable. Use --allow-paid only within your authorized spending budget.",
                )
            try:
                require_backend_credentials(settings.provider)
            except ValueError as error:
                raise _RequestError(
                    "credential_missing",
                    str(error),
                ) from None
        try:
            pid = launch_run(run, expected_plan_sha256=expected_plan_sha256)
        except FileExistsError:
            return {"launched": False, "already_submitted": True, "run": _summary(run)}
        return {
            "launched": True,
            "already_submitted": False,
            "supervisor_pid": pid,
            "run": _summary(run),
        }

    _perform("launch", submit)


@app.command("status")
def status(
    run_id: str | None = typer.Argument(None),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Return retained status, latest metrics, failures and whitelisted receipts."""

    def describe() -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise _RequestError(
                "invalid_limit", "Status limit must be between 1 and 100."
            )
        if run_id is not None:
            return {"run": _summary(_run(workspace, run_id))}
        return {
            "runs": [
                _summary(Path(row["path"]))
                for row in list_runs(workspace.expanduser().resolve(), limit=limit)
            ]
        }

    _perform("status", describe)


@app.command("logs")
def logs(
    run_id: str = typer.Argument(...),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    limit: int = typer.Option(80, "--limit"),
    offset: int = typer.Option(0, "--offset"),
    source: str = typer.Option("auto", "--source", help="auto, training or supervisor"),
) -> None:
    """Read bounded private log lines, removing terminal controls and known keys."""

    def describe() -> dict[str, Any]:
        if not 1 <= limit <= 200 or not 0 <= offset <= 1_000_000:
            raise _RequestError(
                "invalid_limit", "Log limit must be 1..200 and offset 0..1000000."
            )
        if source not in LOG_SOURCES:
            raise _RequestError(
                "invalid_source", "Choose auto, training or supervisor log source."
            )
        snapshot = read_log_snapshot(_run(workspace, run_id), source=source)
        end = max(0, len(snapshot.lines) - offset)
        return {
            "run_id": run_id,
            "lines": list(snapshot.lines[max(0, end - limit) : end]),
            "limit": limit,
            "offset": offset,
            "requested_source": source,
            "source": snapshot.source,
            "filename": snapshot.filename,
            "bytes_read": snapshot.bytes_read,
            "window_truncated": snapshot.window_truncated,
            "available": snapshot.available,
            "error": snapshot.error,
        }

    _perform("logs", describe)


@app.command("compare")
def compare(
    reference_id: str = typer.Argument(...),
    candidate_id: str = typer.Argument(...),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
) -> None:
    """Compare saved settings and latest metrics for two runs, without ranking quality."""

    def describe() -> dict[str, Any]:
        reference, candidate = _run(workspace, reference_id), _run(
            workspace, candidate_id
        )
        if reference == candidate:
            raise _RequestError(
                "invalid_comparison", "Choose two different retained runs."
            )
        return compare_runs(reference, candidate)

    _perform("compare", describe)


@app.command("inspect-data")
def inspect_data(
    dataset: Path = typer.Option(..., "--dataset"),
    method: str = typer.Option("sft", "--method"),
    include_samples: bool = typer.Option(False, "--include-samples"),
) -> None:
    """Read dataset statistics; private text excerpts require explicit opt-in."""

    def describe() -> dict[str, Any]:
        if method not in METHODS:
            raise _RequestError("invalid_method", "Choose sft, grpo, gspo or dapo.")
        try:
            return inspect_training_data(
                dataset, method, include_samples=include_samples
            )
        except DatasetFormatError as error:
            raise _RequestError(
                "invalid_dataset", str(error), diagnostic=error.diagnostic
            ) from None
        except (ValueError, TypeError, KeyError, RecursionError):
            raise _RequestError(
                "invalid_dataset",
                "Dataset must be a regular file within 64 MiB matching the documented training format.",
            ) from None

    _perform("inspect-data", describe)


@app.command("workflow")
def workflow(
    config: Path = typer.Option(..., "--config"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Inspect guided setup stages without preparing or starting training."""

    def describe() -> dict[str, Any]:
        record, _ = _object(config.expanduser(), limit=64 * 1024)
        return inspect_workflow(record, workspace)

    _perform("workflow", describe)


@app.command("quality")
def quality(
    run_id: str,
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
    evidence: Path | None = typer.Option(None, "--evidence"),
) -> None:
    """Verify recorded base/trained outcomes against completed artifact bytes."""

    def describe() -> dict[str, Any]:
        run = _run(workspace, run_id)
        try:
            return record_quality(run, evidence) if evidence else inspect_quality(run)
        except ValueError as error:
            raise _RequestError("invalid_quality_evidence", str(error)) from None

    _perform("quality", describe)


@app.command("stop")
def stop(
    run_id: str,
    expected_plan_sha256: str = typer.Option(..., "--expected-plan-sha256"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Request a cooperative local stop for an explicitly reviewed run."""

    def describe() -> dict[str, Any]:
        run = _run(workspace, run_id)
        try:
            read_run_plan(run)
            return request_stop(run, expected_plan_sha256=expected_plan_sha256)
        except ValueError as error:
            raise _RequestError("stop_unavailable", str(error)) from None

    _perform("stop", describe)


@app.command("checkpoints")
def checkpoints(
    run_id: str,
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Hash published local checkpoints and report verified recovery choices."""

    def describe() -> dict[str, Any]:
        try:
            return inspect_checkpoints(_run(workspace, run_id))
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            raise _RequestError(
                "recovery_unavailable",
                "Recovery requires a terminal local run, stopped workers, matching code, pinned model commit and intact published checkpoint state.",
            ) from None

    _perform("checkpoints", describe)


@app.command("prepare-resume")
def prepare_recovery(
    run_id: str,
    checkpoint: str = typer.Option(..., "--checkpoint"),
    expected_recovery_sha256: str = typer.Option(..., "--expected-recovery-sha256"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Prepare a new private recovery run; launch remains a separate command."""

    def prepare() -> dict[str, Any]:
        try:
            child = prepare_resume(
                _run(workspace, run_id),
                checkpoint,
                expected_recovery_sha256=expected_recovery_sha256,
            )
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            raise _RequestError(
                "recovery_unavailable",
                "Cannot prepare the reviewed checkpoint. Reinspect checkpoint bytes, remaining budget, worker lifecycle, code identity and disk headroom.",
            ) from None
        return {
            "prepared": True,
            "launched": False,
            "parent_run_id": run_id,
            "plan_sha256": read_run_plan(child)[1],
            "run": _summary(child),
        }

    _perform("prepare-resume", prepare)


def _evaluation_config(path: Path) -> EvaluationConfig:
    record, _ = _object(path.expanduser(), limit=64 * 1024)
    try:
        config = EvaluationConfig(**record)
        config.validate()
        return config
    except EvaluationProtocolError:
        raise
    except (TypeError, ValueError, OverflowError):
        raise _RequestError(
            "invalid_evaluation_config",
            "Check evaluation fields, numeric limits and immutable judge pin; see finetune schema.",
        ) from None


def _evaluation_path(workspace: Path, run_id: str, evaluation_id: str) -> Path:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{8}", evaluation_id):
        raise _RequestError(
            "invalid_evaluation_id", "Expected a retained evaluation ID"
        )
    directory = _run(workspace, run_id) / "evaluations"
    path = directory / evaluation_id
    if directory.is_symlink() or path.is_symlink() or not path.is_dir():
        raise _RequestError(
            "evaluation_not_found", "No regular retained evaluation with that ID exists"
        )
    return path


@app.command("evaluation-plan")
def evaluation_plan(
    run_id: str,
    config: Path = typer.Option(..., "--config"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Capture a content-free paired evaluation review; load no models or jobs."""

    def describe() -> dict[str, Any]:
        review = plan_evaluation(_run(workspace, run_id), _evaluation_config(config))
        return {
            "plan": review.plan,
            "plan_sha256": review.sha256,
            "prepared": False,
            "launched": False,
        }

    _perform("evaluation-plan", describe)


@app.command("evaluation-prepare")
def evaluation_prepare(
    run_id: str,
    config: Path = typer.Option(..., "--config"),
    expected_plan_sha256: str = typer.Option(..., "--expected-plan-sha256"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Snapshot exact reviewed cases and evaluation settings; start no inference."""

    def prepare() -> dict[str, Any]:
        review = plan_evaluation(_run(workspace, run_id), _evaluation_config(config))
        path = prepare_evaluation(review, expected_plan_sha256=expected_plan_sha256)
        return {**evaluation_status(path), "prepared": True, "launched": False}

    _perform("evaluation-prepare", prepare)


@app.command("evaluation-launch")
def evaluation_launch(
    run_id: str,
    evaluation_id: str,
    expected_plan_sha256: str = typer.Option(..., "--expected-plan-sha256"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Submit a retained local evaluation once through the shared Buildr queue."""

    def launch() -> dict[str, Any]:
        path = _evaluation_path(workspace, run_id, evaluation_id)
        pid = launch_evaluation(path, expected_plan_sha256=expected_plan_sha256)
        return {"evaluation_id": evaluation_id, "launched": True, "supervisor_pid": pid}

    _perform("evaluation-launch", launch)


@app.command("evaluations")
def evaluations(
    run_id: str,
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """List small retained evaluation progress records without model hashing."""
    _perform(
        "evaluations",
        lambda: {"evaluations": list_evaluations(_run(workspace, run_id))},
    )


@app.command("evaluation-targets")
def evaluation_targets_command(
    run_id: str, workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace")
) -> None:
    """List final/published checkpoint metadata without hashing weights or models."""
    from stateset_agents.finetune_evaluation import evaluation_targets

    _perform("evaluation-targets", lambda: evaluation_targets(_run(workspace, run_id)))


@app.command("evaluation-status")
def evaluation_inspect(
    run_id: str,
    evaluation_id: str,
    verify: bool = typer.Option(False, "--verify"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Inspect progress; --verify rehashes weights and replays completed judge data."""

    def describe() -> dict[str, Any]:
        path = _evaluation_path(workspace, run_id, evaluation_id)
        return inspect_evaluation(path) if verify else evaluation_status(path)

    _perform("evaluation-status", describe)


@app.command("evaluation-stop")
def evaluation_stop(
    run_id: str,
    evaluation_id: str,
    expected_plan_sha256: str = typer.Option(..., "--expected-plan-sha256"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Request cooperative stop of an exact evaluation; never control training."""

    def request() -> dict[str, Any]:
        path = _evaluation_path(workspace, run_id, evaluation_id)
        stop_evaluation(path, expected_plan_sha256=expected_plan_sha256)
        return {"evaluation_id": evaluation_id, "stop_requested": True}

    _perform("evaluation-stop", request)


@app.command("validation-compare")
def validation_compare(
    run_id: str,
    anchor_id: str,
    evaluation_id: list[str] | None = typer.Option(None, "--evaluation-id"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Queue replay/byte verification of matching checkpoint validations; no inference."""

    def submit() -> dict[str, Any]:
        path = submit_comparison(
            _run(workspace, run_id), anchor_id, evaluation_id or None
        )
        return comparison_status(path)

    _perform("validation-compare", submit)


@app.command("validation-comparison-status")
def validation_comparison_status(
    run_id: str,
    comparison_id: str,
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Reconnect to an existing comparison without submitting or hashing weights."""
    _perform(
        "validation-comparison-status",
        lambda: comparison_status(
            comparison_path(_run(workspace, run_id), comparison_id)
        ),
    )


@app.command("nomination-plan")
def checkpoint_nomination_plan(
    run_id: str,
    comparison_id: str,
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Review a checkpoint nomination without writing or reading weights."""

    def inspect() -> dict[str, Any]:
        from stateset_agents.evaluation.agent_runs import content_hash

        plan = nomination_plan(_run(workspace, run_id), comparison_id)
        return {
            "plan": plan,
            "plan_sha256": content_hash(plan),
            "prepared": False,
            "inference_started": False,
        }

    _perform("nomination-plan", inspect)


@app.command("nomination-prepare")
def checkpoint_nomination_prepare(
    run_id: str,
    comparison_id: str,
    expected_plan_sha256: str = typer.Option(..., "--expected-plan-sha256"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Retain a reviewed selection receipt; evaluation requires a separate review."""
    _perform(
        "nomination-prepare",
        lambda: prepare_nomination(
            _run(workspace, run_id),
            comparison_id,
            expected_plan_sha256=expected_plan_sha256,
        ),
    )


@app.command("nominations")
def checkpoint_nominations(
    run_id: str,
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """List saved nomination metadata without weight reads, writes or submission."""
    _perform("nominations", lambda: list_nominations(_run(workspace, run_id)))


@app.command("nomination-status")
def checkpoint_nomination_status(
    run_id: str,
    nomination_id: str,
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Inspect one saved selection's lineage; checkpoint bytes remain unchecked."""
    _perform(
        "nomination-status",
        lambda: nomination_status(_run(workspace, run_id), nomination_id),
    )


@app.command("nomination-config")
def checkpoint_nomination_config(
    run_id: str,
    nomination_id: str,
    expected_plan_sha256: str = typer.Option(..., "--expected-plan-sha256"),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace"),
) -> None:
    """Restore a selected nomination into an unprepared config with blank test cases."""
    _perform(
        "nomination-config",
        lambda: nomination_config(
            _run(workspace, run_id),
            nomination_id,
            expected_plan_sha256=expected_plan_sha256,
        ),
    )
