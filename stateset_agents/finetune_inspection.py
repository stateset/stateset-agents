"""Read-only readiness and retained metrics for the fine-tuning studio.

No trainers, model downloads, provider authentication, or allocations are used.
The optional local subprocesses inspect NVIDIA hardware and the local queue.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents.data.text_sft import DatasetFormatError
from stateset_agents.finetune_workspace import (
    MAX_DATA_BYTES,
    METHODS,
    FinetuneConfig,
    check_config_values,
    queue_executable,
    read_scenarios,
    read_training_data,
)


def inspect_workflow(
    values: Mapping[str, Any], workspace: Path, *, parse_text: bool = False
) -> dict[str, Any]:
    """Combine setup checks into explicit next actions without preparing a run."""
    parsed, issues = check_config_values(values, parse_text=parse_text)
    steps: list[dict[str, Any]] = [
        {
            "name": "Settings",
            "action": "settings",
            "status": "blocked" if issues else "ready",
            "detail": issues[0].message if issues else "Configuration rules pass.",
        }
    ]
    data_ok = False
    try:
        if (
            parsed["method"] not in METHODS
            or not isinstance(parsed["dataset"], str)
            or not parsed["dataset"].strip()
        ):
            raise ValueError("Choose a dataset")
        data = inspect_training_data(Path(parsed["dataset"]), parsed["method"])
        data_ok = True
        detail = f"{data['examples']} examples parsed; {data['duplicate_examples']} duplicates. No tokenizer loaded."
        synthetic = data.get("dataset_provenance", {}).get("synthetic_examples", 0)
        if synthetic:
            detail += (
                f" {synthetic} declared synthetic examples; not independently reviewed."
            )
    except DatasetFormatError as error:
        detail = str(error)
    except (OSError, ValueError, TypeError, RecursionError):
        detail = (
            "Choose a regular dataset within 64 MiB; inspect its documented format."
        )
    steps.append(
        {
            "name": "Dataset",
            "action": "dataset",
            "status": "ready" if data_ok else "blocked",
            "detail": detail,
        }
    )
    readiness = (
        None if issues else inspect_readiness(FinetuneConfig(**parsed), workspace)
    )
    runtime_status = (
        "unchecked"
        if readiness is None
        else (
            "blocked"
            if readiness["failed"]
            else "warning" if readiness["warnings"] else "ready"
        )
    )
    steps.append(
        {
            "name": "Runtime",
            "action": "runtime",
            "status": runtime_status,
            "detail": (
                "Fix settings before checking runtime."
                if readiness is None
                else f"{readiness['failed']} failed checks; {readiness['warnings']} warnings. Model fit and provider access remain unverified."
            ),
        }
    )
    steps.append(
        {
            "name": "Review launch",
            "action": "review",
            "status": "available" if not issues and data_ok else "blocked",
            "detail": "Capture fresh input bytes and review resources. A separate Enter in launch review starts training; this guide starts nothing.",
        }
    )
    steps.append(
        {
            "name": "Evaluate quality",
            "action": "quality",
            "status": "unchecked",
            "detail": "After training, import paired held-out base/trained reports with e on Runs. Completion and training reward alone do not show improvement.",
        }
    )
    return {
        "steps": steps,
        "training_started": False,
        "prepared": False,
        "notes": [
            "Read-only setup snapshot; recheck after editing. Launch revalidates data and resources.",
            "Remote studio backends currently support SFT. Dedicated RL workflows use their own reviewed contracts.",
        ],
    }


def inspect_training_data(
    dataset: Path, method: str = "sft", *, include_samples: bool = False
) -> dict[str, Any]:
    """Inspect one captured dataset without tokenization or training dependencies.

    Samples are opt-in and limited to three excerpts of 4,096 characters each.
    Character lengths are Unicode code points, not model tokens. Duplicate
    checks neither remove examples nor establish their training quality.
    """
    if method not in METHODS:
        raise ValueError("Choose sft, grpo, gspo or dapo")
    if type(include_samples) is not bool:
        raise ValueError("Sample inclusion must be a boolean")
    raw = read_training_data(dataset.expanduser())
    provenance: dict[str, Any] = {}
    try:
        if method == "sft":
            from stateset_agents.data.text_sft import parse_text_sft_records

            rows, provenance = parse_text_sft_records(raw, "dataset")
        else:
            rows = read_scenarios(raw)
    except DatasetFormatError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError):
        raise ValueError(
            "Dataset does not match the documented training format"
        ) from None

    lengths: list[int] = []
    turns: list[int] = []
    identities: Counter[str] = Counter()
    replies: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    samples = []
    for index, row in enumerate(rows):
        if method == "sft":
            canonical = row
            fields = [
                (message["role"], message["content"]) for message in row["messages"]
            ]
            turns.append(sum(role == "assistant" for role, _ in fields))
            roles.update(role for role, _ in fields)
            replies.update(
                hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
                for role, text in fields
                if role == "assistant"
            )
        else:
            canonical = {
                key: row.get(key, [])
                for key in (
                    "context",
                    "policy_facts",
                    "expected_resolution",
                    "user_responses",
                    "must_avoid",
                )
            }
            fields = [
                ("context", row["context"]),
                ("expected resolution", row["expected_resolution"]),
            ]
            for key in ("policy_facts", "user_responses", "must_avoid"):
                fields += [(key, value) for value in row.get(key, [])]
            turns.append(len(row["user_responses"]))
        lengths.append(sum(len(text) for _, text in fields))
        identity = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        identities[
            hashlib.sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()
        ] += 1
        if include_samples and index < 3:
            parts: list[str] = []
            remaining = 4096
            total = 0
            for label, text in fields:
                total += len(label) + 2 + len(text) + 1
                if remaining:
                    part = (label + ": " + text[:remaining] + "\n")[:remaining]
                    parts.append(part)
                    remaining -= len(part)
            samples.append(
                {
                    "example": index + 1,
                    "text": "".join(parts),
                    "truncated": total > 4096,
                }
            )

    def distribution(values: list[int]) -> dict[str, int | float]:
        ordered = sorted(values)
        count = len(ordered)
        return {
            "min": ordered[0],
            "median": (ordered[(count - 1) // 2] + ordered[count // 2]) / 2,
            "p95": ordered[math.ceil(0.95 * count) - 1],
            "max": ordered[-1],
        }

    duplicates = sum(count - 1 for count in identities.values())
    reply_count = sum(replies.values())
    most_common = max(replies.values(), default=0)
    warnings = []
    if provenance.get("synthetic_examples"):
        warnings.append(
            f"{provenance['synthetic_examples']} declared synthetic industry examples; not independently reviewed customer data."
        )
    if duplicates:
        warnings.append(
            f"{duplicates} repeated examples under the stated duplicate definition; no rows were removed."
        )
    if reply_count >= 4 and most_common / reply_count >= 0.5:
        warnings.append(
            f"One exact assistant reply appears {most_common}/{reply_count} times; inspect whether this repetition is intended."
        )
    report: dict[str, Any] = {
        "method": method,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "examples": len(rows),
        "characters_per_example": distribution(lengths),
        "turns_per_example": distribution(turns),
        "turn_definition": (
            "assistant replies" if method == "sft" else "scripted user responses"
        ),
        "role_counts": dict(roles),
        "duplicate_examples": duplicates,
        "duplicate_definition": (
            "parsed messages; IDs ignored"
            if method == "sft"
            else "documented core scenario fields; IDs and extra metadata ignored"
        ),
        "warnings": warnings,
        "samples_included": include_samples,
        "notes": [
            "Lengths count Unicode characters, not tokens; no tokenizer or model was loaded.",
            "p95 uses nearest rank. Turn counts are dataset structure, not completed rollouts.",
            "This report does not establish data quality, model fit or launch readiness.",
            "The file may change after this snapshot; preparation captures and validates it again.",
        ],
    }
    if include_samples:
        report["samples"] = samples
    if provenance:
        report["dataset_provenance"] = provenance
    return report


def inspect_readiness(config: FinetuneConfig, workspace: Path) -> dict[str, Any]:
    """Collect bounded local checks without importing a training dependency."""
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    try:
        config.validate()
        add("Settings", "pass", "Configuration values are valid.")
    except (TypeError, ValueError) as error:
        add("Settings", "fail", str(error))
    source = Path(config.dataset).expanduser()
    if config.dataset and source.is_file():
        size = source.stat().st_size
        add(
            "Dataset",
            "pass" if size <= MAX_DATA_BYTES else "fail",
            f"File available ({size:,} bytes); limit is 64 MiB. Press p to validate its contents.",
        )
    else:
        add(
            "Dataset",
            "fail",
            "Choose an existing dataset with f. Contents have not been validated.",
        )
    target = workspace.expanduser().resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    free = shutil.disk_usage(target).free / 2**30
    add(
        "Local disk",
        (
            "warning"
            if config.provider == "local" and free < config.scratch_gb
            else "pass"
        ),
        (
            f"{free:.1f} GiB free; {config.scratch_gb:g} GiB requested scratch. Queue headroom and model storage are additional."
            if config.provider == "local"
            else f"{free:.1f} GiB free locally. Remote storage and artifact-download space need separate review."
        ),
    )
    if config.provider != "local":
        if config.provider in {"runpod", "river"}:
            name = "RUNPOD_API_KEY" if config.provider == "runpod" else "RIVER_API_KEY"
            present = bool(os.environ.get(name, "").strip())
            add(
                (
                    "RunPod credential"
                    if config.provider == "runpod"
                    else "River credential"
                ),
                "pass" if present else "fail",
                (
                    f"{name} is present (value hidden)."
                    if present
                    else f"{name} is missing from the launching environment."
                ),
            )
        if config.provider == "modal":
            add(
                "Modal authentication",
                "warning",
                "Modal resolves environment credentials or its active profile at launch; this inspection does not read or validate stored tokens.",
            )
        if config.provider in {"modal", "river"}:
            packages = (
                ("modal",)
                if config.provider == "modal"
                else ("river_client", "transformers")
            )
            for package in packages:
                try:
                    available = importlib.util.find_spec(package) is not None
                except (ImportError, ValueError):
                    available = False
                add(
                    "Backend package: " + package,
                    "pass" if available else "fail",
                    f"{package} {'installed; runtime compatibility unchecked' if available else 'missing; install the backend extra and tokenizer dependencies'}.",
                )
            if config.provider == "river":
                add(
                    "River Python",
                    "pass" if sys.version_info >= (3, 12) else "fail",
                    "River requires Python 3.12 or newer.",
                )
            add(
                "Provider billing",
                "warning",
                "Provider billed; this integration enforces no dollar ceiling. Review provider account limits and current pricing.",
            )
        add(
            "Provider readiness",
            "warning",
            "Credential validity, GPU availability, model compatibility and the current full-cost quote are unchecked. No provider request was made.",
        )
    else:
        missing = []
        for name in ("torch", "transformers", "peft", "datasets"):
            try:
                available = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                available = False
            if not available:
                missing.append(name)
        add(
            "Training packages",
            "fail" if missing else "pass",
            (
                "Missing: "
                + ", ".join(missing)
                + ". Install stateset-agents[training]."
                if missing
                else "Training packages are installed; runtime/model version compatibility is unchecked."
            ),
        )
        binary = shutil.which("nvidia-smi")
        if binary:
            try:
                result = subprocess.run(
                    [
                        binary,
                        "--query-gpu=name,memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                rows = result.stdout.strip().splitlines()
                if result.returncode == 0 and rows:
                    add(
                        "NVIDIA hardware",
                        "pass",
                        "; ".join(rows[:8])[:1800]
                        + " (name, total MiB, free MiB). CUDA training compatibility and model fit are unchecked.",
                    )
                else:
                    add(
                        "NVIDIA hardware",
                        "warning",
                        "The hardware query did not return a GPU. The worker still checks CUDA before training.",
                    )
            except (OSError, subprocess.TimeoutExpired):
                add(
                    "NVIDIA hardware",
                    "warning",
                    "The local GPU query was unavailable or timed out. CUDA readiness is unchecked.",
                )
        else:
            add(
                "NVIDIA hardware",
                "warning",
                "nvidia-smi is unavailable. GPU readiness is unchecked; the worker requires CUDA.",
            )
        try:
            queue = queue_executable()
            if queue:
                result = subprocess.run(
                    [queue, "status", "--summary", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if result.returncode:
                    raise ValueError("Queue inspection failed")
                snapshot = json.loads(result.stdout)
                budget = snapshot["config"]
                counts = snapshot["counts"]
                oversized = (
                    config.cpu_slots > budget["cpu_slots"]
                    or config.memory_gb > budget["memory_gb"]
                )
                health = snapshot.get("queue_health", {})
                blocked = (
                    isinstance(health, dict) and health.get("status") == "disk_blocked"
                )
                health_detail = (
                    " " + str(health.get("summary", ""))[:600] if blocked else ""
                )
                add(
                    "Shared queue",
                    "fail" if oversized else ("warning" if blocked else "pass"),
                    f"Requested {config.cpu_slots} CPUs/{config.memory_gb:g} GiB; configured {budget['cpu_slots']} CPUs/{budget['memory_gb']} GiB. {counts['running']} running, {counts['queued']} queued. Snapshot only; admission may wait."
                    + health_detail,
                )
            else:
                add(
                    "Shared queue",
                    "warning",
                    "No Buildr executable found; a direct detached worker will be used.",
                )
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
            add(
                "Shared queue",
                "warning",
                "Queue inspection is unavailable. No settings or jobs were changed.",
            )
    return {
        "checked_at": time.time(),
        "model": config.model,
        "provider": config.provider,
        "checks": checks,
        "failed": sum(row["status"] == "fail" for row in checks),
        "warnings": sum(row["status"] == "warning" for row in checks),
        "provisions_hardware": False,
    }


def _read_metric_bytes(path: Path, *, tail: bool = False) -> bytes | None:
    """Capture a bounded regular-file window without following links or FIFOs."""
    if path.is_symlink():
        return None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return None
            end = stream.seek(0, os.SEEK_END)
            if not tail and end > 128 * 1024:
                return None
            start = max(0, end - 128 * 1024) if tail else 0
            stream.seek(start)
            remaining = end - start
            if start:
                # Ignore a possible partial first record. Concurrent appends
                # never enlarge the byte window captured by the initial seek.
                remaining -= len(stream.readline(remaining))
            return stream.read(remaining)
    except OSError:
        return None


def metric_history(run: Path, *, limit: int = 120) -> list[dict[str, Any]]:
    """Read regular metric files safely, tolerating partial or unavailable data."""
    if type(limit) is not int or limit < 1:
        raise ValueError("History limit must be a positive integer")
    if run.is_symlink():
        return []
    rows: list[Any] = []
    raw = _read_metric_bytes(run / "metrics.jsonl", tail=True)
    if raw is not None:
        for line in raw.decode("utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except (ValueError, RecursionError):
                continue

    def normalized(records: list[Any]) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        for row in records:
            if (
                not isinstance(row, dict)
                or type(row.get("step")) is not int
                or row["step"] < 0
                or not isinstance(row.get("metrics"), dict)
            ):
                continue
            values: dict[str, float] = {}
            for key, value in row["metrics"].items():
                if (
                    not isinstance(key, str)
                    or not isinstance(value, (int, float))
                    or isinstance(value, bool)
                ):
                    continue
                try:
                    numeric = float(value)
                except (OverflowError, ValueError):
                    continue
                if math.isfinite(numeric):
                    values[key] = numeric
            valid.append(
                {
                    "step": row["step"],
                    "metrics": values,
                    "updated_at": row.get("updated_at"),
                }
            )
        return valid

    valid = normalized(rows)
    # History is appended before the atomic latest snapshot. Do not append an
    # older snapshot during that write window or duplicate samples in a chart.
    if not valid:
        try:
            snapshot = _read_metric_bytes(run / "metrics.json")
            if snapshot is not None:
                valid = normalized([json.loads(snapshot)])
        except (ValueError, RecursionError):
            pass
    return valid[-min(limit, 1000) :]


def metric_sparkline(values: list[float], *, width: int = 44) -> str:
    """Render finite reported samples as a compact chart; never interpolate data."""
    if width < 1 or not values or any(not math.isfinite(value) for value in values):
        return ""
    if len(values) > width:
        values = (
            [values[-1]]
            if width == 1
            else [
                values[round(index * (len(values) - 1) / (width - 1))]
                for index in range(width)
            ]
        )
    low, high = min(values), max(values)
    levels = ".:-=+*#%@"
    if low == high:
        return "=" * len(values)
    scale = max(abs(low), abs(high), 1.0)
    span = high / scale - low / scale
    return "".join(
        levels[
            min(
                len(levels) - 1,
                max(0, round((value / scale - low / scale) / span * (len(levels) - 1))),
            )
        ]
        for value in values
    )


def _comparison_record(run: Path, filename: str) -> tuple[dict[str, Any], str | None]:
    """Read one bounded regular record without following links or opening a FIFO."""
    path = run / filename
    if run.is_symlink() or path.is_symlink():
        return {}, None
    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(os.open(path, flags), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return {}, None
            raw = stream.read(128 * 1024 + 1)
        if len(raw) > 128 * 1024:
            return {}, None
        record = json.loads(raw)
        if not isinstance(record, dict):
            return {}, None
        return record, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, RecursionError):
        return {}, None


def _comparison_scalar(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) <= 4096
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return False


def _comparison_snapshot(run: Path) -> dict[str, Any]:
    records = {
        name: _comparison_record(run, name)
        for name in ("plan.json", "state.json", "metrics.json")
    }
    plan, digest = records["plan.json"]
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        plan, digest = {}, None
    state, _ = records["state.json"]
    metrics, _ = records["metrics.json"]
    config = plan.get("config")
    known = asdict(FinetuneConfig())
    config = (
        {
            key: value
            for key, value in config.items()
            if key in known and _comparison_scalar(value)
        }
        if isinstance(config, dict)
        else {}
    )
    values = metrics.get("metrics")
    numeric: dict[str, float] = {}
    omitted = 0
    valid_step = type(metrics.get("step")) is int and 0 <= metrics["step"] <= 2**53
    if valid_step and isinstance(values, dict):
        for key, value in values.items():
            if (
                len(numeric) < 64
                and isinstance(key, str)
                and len(key) <= 128
                and type(value) in {int, float}
                and _comparison_scalar(value)
            ):
                numeric[key] = float(value)
            else:
                omitted += 1
    else:
        metrics = {}
    dataset = plan.get("dataset_sha256")
    implementation = plan.get("package_implementation_sha256")
    status = state.get("status")
    return {
        "run_id": run.name,
        "status": (
            status
            if isinstance(status, str)
            and status
            in {
                "prepared",
                "starting",
                "queued",
                "running",
                "succeeded",
                "failed",
                "interrupted",
            }
            else "unknown" if isinstance(status, str) else "unreadable"
        ),
        "plan_sha256": digest,
        "dataset_sha256": (
            dataset
            if isinstance(dataset, str) and re.fullmatch(r"[0-9a-f]{64}", dataset)
            else None
        ),
        "examples": (
            plan.get("examples")
            if type(plan.get("examples")) is int and 0 < plan["examples"] <= 2**53
            else None
        ),
        "framework_version": (
            plan.get("framework_version")
            if isinstance(plan.get("framework_version"), str)
            and len(plan["framework_version"]) <= 128
            else None
        ),
        "package_implementation_sha256": (
            implementation
            if isinstance(implementation, str)
            and re.fullmatch(r"[0-9a-f]{64}", implementation)
            else None
        ),
        "config": config,
        "latest_metrics": {
            "step": metrics.get("step"),
            "updated_at": (
                metrics.get("updated_at")
                if _comparison_scalar(metrics.get("updated_at"))
                else None
            ),
            "metrics": numeric,
            "omitted_fields": omitted,
        },
        "unreadable_records": [
            name
            for name, (_, checksum) in records.items()
            if checksum is None or (name == "plan.json" and digest is None)
        ],
    }


def compare_runs(reference: Path, candidate: Path) -> dict[str, Any]:
    """Compare saved settings and latest training snapshots, without ranking quality.

    Dataset identities are recorded hashes, not rechecks of retained contents.
    Metrics are the latest atomic snapshots, not aligned evaluation observations.
    This read never opens datasets, logs, credentials, workers or provider APIs.
    """
    if reference.absolute() == candidate.absolute():
        raise ValueError("Choose two different retained runs")
    left, right = _comparison_snapshot(reference), _comparison_snapshot(candidate)
    settings = []
    for key in asdict(FinetuneConfig()):
        a, b = left["config"], right["config"]
        if (
            (key in a) != (key in b)
            or a.get(key) != b.get(key)
            or type(a.get(key)) is not type(b.get(key))
        ):
            settings.append(
                {
                    "field": key,
                    "reference": a.get(key),
                    "candidate": b.get(key),
                    "reference_present": key in a,
                    "candidate_present": key in b,
                }
            )
    metric_rows = []
    a, b = left["latest_metrics"]["metrics"], right["latest_metrics"]["metrics"]
    for key in sorted(a.keys() | b.keys()):
        delta = b[key] - a[key] if key in a and key in b else None
        metric_rows.append(
            {
                "metric": key,
                "reference": a.get(key),
                "candidate": b.get(key),
                "delta": delta if delta is not None and math.isfinite(delta) else None,
            }
        )
    identity = (
        "unknown"
        if not left["dataset_sha256"] or not right["dataset_sha256"]
        else (
            "same" if left["dataset_sha256"] == right["dataset_sha256"] else "different"
        )
    )
    return {
        "comparison_kind": "recorded_training_snapshots",
        "reference": left,
        "candidate": right,
        "dataset_identity": identity,
        "config_differences": settings,
        "metrics": metric_rows,
        "quality_gain_verified": False,
        "notes": [
            "Recorded dataset hashes are not reverified against retained files.",
            "Persisted statuses may lag running jobs; no process or provider was polled.",
            "Latest metrics.json snapshots may differ in step, time, data and objective.",
            "Deltas are candidate minus reference; missing or nonfinite deltas are null.",
            "Training reward and completion do not establish independent quality or a winner.",
        ],
    }
