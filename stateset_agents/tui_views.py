"""Read-only studio views: run details, metrics, comparisons and launch reviews.

Pure renderers over retained records and the draft; the studio decides which
one is on screen. They never start work or change state.
"""

from __future__ import annotations

import math
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeGuard

from stateset_agents.finetune_inspection import metric_history, metric_sparkline
from stateset_agents.finetune_provider_submission import inspect_submission
from stateset_agents.finetune_workspace import (
    BACKENDS,
    RunPlan,
    failure_info,
    queue_executable,
    read_json,
    redact_finetune_secrets,
)
from stateset_agents.tui_text import _safe


def _teardown_note(state: dict[str, Any]) -> str:
    """Describe a failure observed after the worker committed its success."""
    code = state.get("teardown_exit_code")
    return (
        "Note: the worker exited after committing a verified result"
        + (f" (exit {code})" if isinstance(code, int) else "")
        + f"; teardown: {_safe(str(state.get('teardown_error')))}. The result stands."
    )


def _run_config(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("plan", {}).get("config", {})
    return config if isinstance(config, dict) else {}


def _finite(value: Any) -> TypeGuard[int | float]:
    """Reject malformed retained counters, including integers too large for floats."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def progress_lines(
    row: dict[str, Any], history: list[dict[str, Any]], width: int
) -> list[str]:
    config = _run_config(row)
    current: Any = row["state"].get("committed_updates")
    total: Any = config.get("steps")
    label = "Committed updates"
    if config.get("method") == "sft":
        current = next(
            (
                record["metrics"]["epoch"]
                for record in reversed(history)
                if "epoch" in record["metrics"]
            ),
            None,
        )
        total = config.get("epochs")
        label = "Reported epoch"
    rows = []
    if _finite(current) and current >= 0 and _finite(total) and total > 0:
        fraction = current / total
        cells = max(8, min(24, width - 45))
        fill = int(min(1.0, max(0.0, fraction)) * cells)
        percent = f"{fraction:.0%}" if fraction < 1000 else "exceeds budget"
        rows.append(
            f"{label}: {current:g}/{total:g} ["
            + "#" * fill
            + "." * (cells - fill)
            + "] "
            + percent
        )
    else:
        rows.append(label + ": not reported yet")
    if not history:
        return [
            *rows,
            "No trainer metrics yet; initial loading or queue wait may precede logs.",
        ]
    last = history[-1]
    timestamp = last.get("updated_at")
    if _finite(timestamp) and timestamp >= 0 and timestamp <= time.time():
        rows.append(
            f"Trainer step {last['step']} | Metrics last reported {int(time.time() - timestamp)}s ago"
        )
    else:
        rows.append(f"Trainer step {last['step']} | Metric freshness unavailable")
    keys = (
        "loss",
        "total_loss",
        "policy_loss",
        "average_reward",
        "mean_reward",
        "reward",
        "kl",
        "entropy",
    )
    values = last["metrics"]
    visible = [key for key in keys if key in values][:4]
    if not visible:
        visible = list(values)[:4]
    rows.append(" | ".join(f"{key}={values[key]:.4g}" for key in visible))
    return rows


def comparison_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["No comparison snapshot. Esc returns to Runs."]
    left, right = report["reference"], report["candidate"]
    rows = [
        "RUN COMPARISON / candidate minus reference",
        "Reference: " + left["run_id"],
        "Candidate: " + right["run_id"],
        f"Saved status: {left['status']} -> {right['status']}",
        "Dataset hashes: " + report["dataset_identity"] + " (recorded, not reverified)",
        "Training snapshots do not establish quality or a winner.",
        "",
        "CHANGED SETTINGS",
    ]
    for change in report["config_differences"]:
        a = change["reference"] if change["reference_present"] else "(not recorded)"
        b = change["candidate"] if change["candidate_present"] else "(not recorded)"
        rows += [change["field"] + ":", f"  reference: {a}", f"  candidate: {b}"]
    if not report["config_differences"]:
        rows.append("No differences among recorded configuration fields.")
    rows += [
        "",
        "LATEST TRAINING METRICS / metrics.json",
        f"Reported step: {left['latest_metrics']['step']} -> {right['latest_metrics']['step']}",
        f"Reported time: {left['latest_metrics']['updated_at']} -> {right['latest_metrics']['updated_at']}",
        "Snapshots may differ in step, time, data and objective.",
    ]
    for metric in report["metrics"]:
        a, b, delta = metric["reference"], metric["candidate"], metric["delta"]
        rows += [
            metric["metric"] + ":",
            "  reference: " + (f"{a:.6g}" if a is not None else "not reported"),
            "  candidate: " + (f"{b:.6g}" if b is not None else "not reported"),
            "  delta: " + (f"{delta:+.6g}" if delta is not None else "unavailable"),
        ]
    if not report["metrics"]:
        rows.append("No finite metrics reported in these snapshots.")
    for label, snapshot in (("REFERENCE", left), ("CANDIDATE", right)):
        rows += [
            "",
            label + " PROVENANCE",
            "Plan SHA256: " + str(snapshot["plan_sha256"]),
            "Input SHA256: " + str(snapshot["dataset_sha256"]),
            "Examples: " + str(snapshot["examples"]),
            "Framework: " + str(snapshot["framework_version"]),
            "Package SHA256: " + str(snapshot["package_implementation_sha256"]),
        ]
        if snapshot["unreadable_records"]:
            rows.append(
                "Unavailable records: " + ", ".join(snapshot["unreadable_records"])
            )
        if snapshot["latest_metrics"]["omitted_fields"]:
            rows.append(
                "Metric fields omitted (invalid or over limit): "
                + str(snapshot["latest_metrics"]["omitted_fields"])
            )
    rows += ["", "INTERPRETATION", *report["notes"]]
    return rows


def metric_lines(row: dict[str, Any] | None, width: int) -> list[str]:
    if row is None:
        return ["No retained run selected."]
    history = metric_history(Path(row["path"]), limit=120)
    rows = [
        "TRAINING METRICS / reported samples only",
        "Run: " + row["id"],
        "",
        *progress_lines(row, history, width),
        "",
    ]
    if not history:
        return [
            *rows,
            "History starts when the trainer reports metrics.",
            "Older runs can show their latest snapshot without a trend.",
        ]
    rows.append(
        f"{len(history)} retained events | reported steps {history[0]['step']}..{history[-1]['step']}"
    )
    keys = [
        key
        for key in (
            "loss",
            "total_loss",
            "policy_loss",
            "average_reward",
            "mean_reward",
            "reward",
            "kl",
            "entropy",
            "grad_norm",
            "learning_rate",
        )
        if any(key in record["metrics"] for record in history)
    ]
    if not keys:
        keys = list(history[-1]["metrics"])[:8]
    for key in keys:
        values = [
            record["metrics"][key] for record in history if key in record["metrics"]
        ]
        rows += [
            "",
            f"{key}: last {values[-1]:.4g} | range {min(values):.4g}..{max(values):.4g} | {len(values)} samples",
        ]
        rows.append(
            metric_sparkline(values, width=max(12, min(72, width - 4)))
            if len(values) > 1
            else "One sample; more logs are needed for a trend."
        )
    rows += [
        "",
        "Charts sample reported values; intervals are not assumed uniform.",
        "No ETA is inferred. Training reward is not independent quality evidence.",
    ]
    return rows


def run_details_lines(row: dict[str, Any] | None, workspace: Path) -> list[str]:
    if row is None:
        return ["No retained run is selected."]
    config = _run_config(row)
    state = row["state"]
    run = Path(row["path"])
    result = [
        "RUN DETAILS / c reuses settings for a NEW run",
        "",
        "Run: " + row["id"],
        f"Status: {state.get('status', 'unknown')}",
        "Model: " + str(config.get("model", "unknown")),
        "Model commit: " + str(config.get("revision") or "mutable / unpinned"),
        f"Method: {str(config.get('method', '?')).upper()} | Provider: {config.get('provider', '?')}",
    ]
    if state.get("teardown_error"):
        result.append(_teardown_note(state))
    if state.get("recovery_unavailable"):
        result.append(
            "Recovery checkpoints: unavailable ("
            + _safe(str(state["recovery_unavailable"]))
            + "); training continued without recovery receipts."
        )
    if state.get("execution_unconfirmed"):
        result += [
            "Execution: unconfirmed; detached work may still be queued or running.",
            (
                "Local stop receipt is saved; worker exit is not yet confirmed."
                if state.get("stop_requested")
                else (
                    "k reviews a cooperative stop for a local run."
                    if config.get("provider") == "local"
                    else "Check the retained provider handle and account before retrying."
                )
            ),
        ]
    start = state.get("started_at", state.get("created_at"))
    end = state.get("finished_at", time.time())
    if (
        isinstance(start, (int, float))
        and isinstance(end, (int, float))
        and 0 <= end - start < 10**9
    ):
        seconds = int(end - start)
        result.append(
            f"Elapsed (including queue wait): {seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
        )
    if state.get("error"):
        result += ["", "Error: " + str(state["error"])]
    diagnosis = failure_info(run, state)
    if diagnosis:
        result += [
            "",
            "FAILURE AND NEXT STEPS",
            diagnosis["title"],
            f"Code: {diagnosis['code']} | Stage: {diagnosis['stage']}",
            diagnosis["action"],
        ]
        if not diagnosis["recorded"]:
            result.append(
                "No valid structured diagnosis is retained; the cause has not been inferred from logs."
            )
        if config.get("provider") != "local":
            result.append(
                "Check the retained provider handle and your provider account for active resources or charges before a new paid launch."
            )
        result.append(
            "c reuses settings for a NEW reviewed run; it does not resume a checkpoint or retry a provider job."
        )
    if config.get("provider") != "local":
        result += ["", "PROVIDER SUBMISSION / RETAINED OBSERVATIONS"]
        try:
            provider = redact_finetune_secrets(inspect_submission(run))
            if not provider["available"]:
                result.append(
                    "No submission receipt retained; provider identity and cleanup are unknown."
                )
            else:
                result += [
                    _safe(
                        f"Logical job: {provider.get('job_id') or 'not yet recorded'}"
                    ),
                    _safe(
                        f"Submission: {provider['submission_state']} | Observed result: {provider.get('observed_status') or 'not reported'}"
                    ),
                ]
                for resource in provider["resources"]:
                    result.append(
                        _safe(
                            f"{resource['kind']}: {resource['id']} | {resource['state'].replace('_', ' ')}"
                        )
                    )
                    if resource.get("name"):
                        result.append(_safe("Owned name: " + resource["name"]))
                for intent in provider["unresolved_intents"]:
                    result.append(
                        _safe(
                            f"Unresolved allocation: {intent['resource_kind']} | {intent.get('resource_name') or 'identity not returned'}"
                        )
                    )
                result.append(
                    "Cleanup required or unconfirmed; reconcile provider records before retrying."
                    if provider["cleanup_required_or_unconfirmed"]
                    else "No unresolved cleanup in this receipt; live provider state is unchecked."
                )
            result += [
                "Provider status is not checked live; this view makes no provider request.",
                "Cleanup acknowledgement does not prove billing stopped. No automatic restart.",
            ]
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RecursionError,
            OverflowError,
        ):
            result.append(
                "Provider receipt unavailable; retained identity and cleanup could not be verified."
            )
    result += [
        "",
        "INPUT AND SETTINGS",
        f"Examples: {row['plan'].get('examples', 'unknown')}",
        "Input SHA256: " + str(row["plan"].get("dataset_sha256", "unavailable")),
        f"Learning rate: {config.get('learning_rate', '?')} | LoRA rank: {config.get('lora_rank', '?')}",
        f"Training seed: {config.get('seed', 'not recorded')}",
        (
            f"Budget: {config.get('epochs', '?')} epochs"
            if config.get("method") == "sft"
            else f"Budget: {config.get('steps', '?')} updates | Recorded: {state.get('committed_updates', 'not reported')}"
        ),
        "Dataset snapshot: " + str(run / "dataset"),
    ]
    resume = row["plan"].get("resume")
    if isinstance(resume, dict):
        result += [
            "",
            "RECOVERY LINEAGE",
            "Parent: " + str(resume.get("parent_run_id", "unknown")),
            "Checkpoint: " + str(resume.get("checkpoint", "unknown")),
            "Checkpoint SHA256: " + str(resume.get("checkpoint_sha256", "unknown")),
            "Original total training budget retained.",
        ]
    for filename, label, keys in (
        (
            "completion.json",
            "COMPLETION",
            (
                "output_dir",
                "committed_updates",
                "quality_gain_verified",
                "result_kind",
                "checkpoint",
            ),
        ),
        ("provider-job.json", "PROVIDER HANDLE", ("provider", "job_id")),
        (
            "provider-result.json",
            "PROVIDER-REPORTED COST",
            ("cost_usd", "duration_s"),
        ),
    ):
        if not (run / filename).exists():
            continue
        result += ["", label]
        try:
            record = read_json(run / filename)
            result += [f"{key}: {record[key]}" for key in keys if key in record]
        except (OSError, ValueError, TypeError):
            result.append(filename + " is unreadable; inspect the retained record.")
    try:
        metrics = read_json(run / "metrics.json")
        if isinstance(metrics.get("metrics"), dict):
            result += [
                "",
                "LATEST METRICS",
                f"Reported step: {metrics.get('step', '?')}",
            ]
            result += [f"{key}: {value}" for key, value in metrics["metrics"].items()]
    except (OSError, ValueError, TypeError):
        pass
    result += [
        "",
        "FILES",
        "Workspace: " + str(run),
        "Training log: " + str(run / "run.log"),
        "Supervisor log: " + str(run / "supervisor.log"),
        "",
        "AI AGENT ACCESS / JSON",
        shlex.join(
            [
                sys.executable,
                "-m",
                "stateset_agents.cli",
                "finetune",
                "status",
                row["id"],
                "--workspace",
                str(workspace),
            ]
        ),
        "Use finetune schema for agent commands and config fields.",
        "",
        "Reusing settings creates a new configuration from this private input snapshot.",
        "No job starts until you validate and confirm a new launch review.",
        "Completion alone does not establish a quality gain.",
    ]
    return result


def resource_lines(plan: RunPlan | None, draft: dict[str, Any]) -> list[str]:
    values = asdict(plan.config) if plan else draft
    if values["provider"] != "local":
        lines = ["Local queue: not held during remote training"]
        if plan and values["provider"] == "runpod":
            resources = plan.resources
            lines += [
                f"GPU: {resources['gpu_count']} x {resources['gpu']}",
                f"Disk: {resources['container_disk_gb']} GB",
                "Resource evidence: " + resources["recommendation_evidence"],
            ]
        elif values["provider"] == "modal":
            resources = plan.resources if plan else {}
            lines += [
                f"GPU: {resources.get('gpu_count', values['gpu_count'] or 1)} x {resources.get('gpu', values['gpu'] or 'A10')}",
                "Artifacts: downloaded weights",
                "Billing: no enforced dollar ceiling",
            ]
        elif values["provider"] == "river":
            lines += [
                "Hardware: managed by River",
                "Artifacts: hosted river:// checkpoint pointer",
                "Billing: no enforced dollar ceiling",
            ]
        return lines
    try:
        queue = queue_executable()
    except ValueError as error:
        return [str(error)]
    lines = [
        "Queue: " + (Path(queue).name if queue else "not installed (direct worker)"),
        f"Local reservation: {values['cpu_slots']} CPUs / {values['memory_gb']} GiB RAM",
        f"Scratch reservation: {values['scratch_gb']} GiB",
    ]
    return lines


def review_lines(plan: RunPlan | None, draft: dict[str, Any]) -> list[str]:
    if plan is None:
        return ["No validated launch plan. Esc returns to configuration."]
    cfg = plan.config
    return [
        "REVIEW LAUNCH / Enter launches, Esc returns",
        "",
        f"{cfg.method.upper()} | {cfg.provider}",
        "Model: " + cfg.model,
        "Model commit: " + (cfg.revision or "mutable / unpinned"),
        f"Input snapshot: {plan.examples} examples",
        "SHA256: " + plan.describe()["dataset_sha256"],
        f"Learning rate: {cfg.learning_rate:g} | LoRA rank: {cfg.lora_rank}",
        f"Training seed: {cfg.seed}",
        (
            f"Budget: {cfg.epochs} epochs"
            if cfg.method == "sft"
            else f"Budget: {cfg.steps} updates | KL: {cfg.kl_coefficient:g}"
        ),
        (
            "Timeout: provider request settings; no whole-run deadline"
            if cfg.provider == "river"
            else f"Timeout: {cfg.timeout_seconds}s ({'function execution only' if cfg.provider == 'modal' else 'queue wait is additional'})"
        ),
        (
            f"RunPod pod ceiling: ${cfg.max_cost_usd:.2f}"
            if cfg.provider == "runpod"
            else (
                "Compute: local CUDA"
                if cfg.provider == "local"
                else "Provider billed: NO enforced dollar ceiling"
            )
        ),
        "Result: " + str(BACKENDS[cfg.provider]["result_kind"]),
        "",
        *resource_lines(plan, draft),
        "",
        *plan.notes,
        *(
            [
                "",
                "RECOVERY LINEAGE",
                "Parent: " + str(plan.resume["parent_run_id"]),
                "Checkpoint: " + str(plan.resume["checkpoint"]),
                "Checkpoint SHA256: " + str(plan.resume["checkpoint_sha256"]),
                "Original total training budget retained; no fresh-training fallback.",
            ]
            if plan.resume is not None
            else []
        ),
        "",
        "Closing this terminal will not stop a launched job.",
    ]
