"""Durable queued comparison of replayable checkpoint validation diagnostics."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Collection, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.finetune_evaluation import (
    MAX_TRANSCRIPT_BYTES,
    REQUIRED_CHECKS,
    EvaluationReview,
    inspect_evaluation,
    report_settings,
)
from stateset_agents.finetune_recovery import CHECKPOINT_NAME_PATTERN
from stateset_agents.finetune_workspace import (
    owner_alive,
    private_file_lock,
    process_token,
    queue_executable,
    read_json,
    read_run_plan,
    transition_state,
    write_json,
)

RECORD_ID = r"[0-9]{8}-[0-9]{6}-[0-9a-f]{8}"
MAX_COMPARISON_RECORDS = 100
#: A queued verification no worker has claimed after this long is reported as
#: ``stale``: it no longer blocks a new submission for its anchor. The queue
#: job itself is not cancelled; if it starts later it simply verifies again.
QUEUED_STALE_AFTER_SECONDS = 6 * 3600
#: Derived statuses that still represent work that may complete.
ACTIVE_COMPARISON_STATUSES = frozenset({"queued", "running", "submission_uncertain"})
DIRECTORY = "validation-comparisons"
GUARDS = (
    "no_success_regressions",
    "no_new_policy_violations",
    "no_new_truncations",
    "no_new_critical_check_failures",
)
NOTES = (
    "Validation diagnostic only; fresh test assessment and independent training seeds remain required.",
    "Rank by success rate, judge score, then earliest saved step. New critical-check, success, policy or truncation regressions disqualify a candidate.",
    "Incompatible protocols are excluded; unverifiable, conflicting or baseline-mismatched evidence prevents a recommendation.",
    "Only the captured history is compared. Verification records bytes at execution time; review rechecks them before another evaluation.",
)


class ValidationComparisonError(ValueError):
    """A controlled comparison failure without private source values."""

    code = "invalid_validation_comparison"
    diagnostic = {
        "action": "Inspect completed checkpoint validation records and retry explicitly."
    }


def _record(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not re.fullmatch(RECORD_ID, name):
        raise ValidationComparisonError("Choose a retained evaluation/comparison ID.")
    path = directory / name
    if directory.is_symlink() or path.is_symlink() or not path.is_dir():
        raise ValidationComparisonError(
            "The retained record must be a regular directory."
        )
    return path


def comparison_path(run: Path, comparison_id: str) -> Path:
    """Resolve only a regular retained comparison belonging to this run."""
    return _record(run / DIRECTORY, comparison_id)


def _owned_run(path: Path) -> Path:
    if path.parent.name != DIRECTORY or path.parent.is_symlink():
        raise ValidationComparisonError(
            "Expected an own retained comparison directory."
        )
    run = path.parent.parent
    read_run_plan(run)
    _record(run / DIRECTORY, path.name)
    return run


def _validate_plan(plan: dict[str, Any], run: Path) -> None:
    if (
        type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 1
        or plan.get("kind") != "stateset-validation-comparison"
        or plan.get("run_id") != run.name
        or not isinstance(plan.get("anchor_id"), str)
        or not re.fullmatch(RECORD_ID, plan["anchor_id"])
    ):
        raise ValidationComparisonError("Comparison plan metadata is unreadable.")
    for key in ["parent_plan_sha256", "implementation_sha256"]:
        if not isinstance(plan.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", plan[key]
        ):
            raise ValidationComparisonError("Comparison checksums are unreadable.")
    snapshots = plan.get("evaluations")
    if (
        not isinstance(snapshots, list)
        or not 1 <= len(snapshots) <= MAX_COMPARISON_RECORDS
    ):
        raise ValidationComparisonError("Comparison snapshot metadata is unreadable.")
    ids = set()
    for row in snapshots:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("evaluation_id"), str)
            or not re.fullmatch(RECORD_ID, row["evaluation_id"])
            or row["evaluation_id"] in ids
            or (
                row.get("plan_sha256") is not None
                and (
                    not isinstance(row["plan_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", row["plan_sha256"])
                )
            )
        ):
            raise ValidationComparisonError(
                "Comparison snapshot identity is unreadable."
            )
        ids.add(row["evaluation_id"])
    if plan["anchor_id"] not in ids:
        raise ValidationComparisonError("Comparison anchor is missing.")


def _protocol(run: Path, plan: dict[str, Any]) -> str:
    binding = plan["binding"]
    return content_hash(
        {
            "parent": {
                key: binding.get(key)
                for key in [
                    "parent_plan_sha256",
                    "parent_state_sha256",
                    "completion_sha256",
                    "training_dataset_sha256",
                    "model",
                    "model_revision",
                    "method",
                    "training_seed",
                ]
            },
            "settings": report_settings(EvaluationReview(run, plan, {})),
            "cases_sha256": plan["cases_sha256"],
            "rubric_sha256": plan["rubric_sha256"],
            "rubric_revision": plan["rubric_revision"],
            "implementation_sha256": plan["implementation_sha256"],
            "decoding": plan["decoding"],
            "thresholds": plan["thresholds"],
        }
    )


def comparison_plan(
    run: Path, anchor_id: str, evaluation_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Capture bounded metadata only; checkpoint bytes are checked in the queue."""
    run = run.resolve()
    _, parent_sha = read_run_plan(run)
    directory = run / "evaluations"
    anchor = _record(directory, anchor_id)
    anchor_plan = read_json(anchor / "plan.json")
    if (
        read_json(anchor / "state.json").get("status") != "succeeded"
        or anchor_plan["config"].get("split") != "validation"
        or not anchor_plan["config"].get("checkpoint")
    ):
        raise ValidationComparisonError(
            "Choose a completed published-checkpoint validation as the anchor."
        )
    if evaluation_ids is None:
        paths = list(islice(directory.iterdir(), MAX_COMPARISON_RECORDS + 1))
        if len(paths) > MAX_COMPARISON_RECORDS:
            raise ValidationComparisonError(
                "History exceeds 100 entries; provide explicit evaluation IDs."
            )
        ids = sorted(path.name for path in paths if re.fullmatch(RECORD_ID, path.name))
    else:
        if isinstance(evaluation_ids, (str, bytes)) or not isinstance(
            evaluation_ids, Sequence
        ):
            raise ValidationComparisonError(
                "Provide a sequence of retained evaluation IDs."
            )
        selected = tuple(evaluation_ids)
        if any(
            not isinstance(name, str) or not re.fullmatch(RECORD_ID, name)
            for name in selected
        ):
            raise ValidationComparisonError(
                "Choose retained evaluation IDs for the comparison subset."
            )
        ids = sorted({anchor_id, *selected})
        if len(ids) > MAX_COMPARISON_RECORDS:
            raise ValidationComparisonError("Compare at most 100 retained evaluations.")
    snapshots: list[dict[str, Any]] = []
    for name in ids:
        try:
            path = _record(directory, name)
            plan = read_json(path / "plan.json")
            if read_json(path / "state.json").get("status") != "succeeded":
                continue
            if plan["config"].get("split") != "validation" or not plan["config"].get(
                "checkpoint"
            ):
                continue
            snapshots.append({"evaluation_id": name, "plan_sha256": content_hash(plan)})
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            snapshots.append({"evaluation_id": name, "plan_sha256": None})
    if not any(
        row["evaluation_id"] == anchor_id and row["plan_sha256"] is not None
        for row in snapshots
    ):
        raise ValidationComparisonError("The validation anchor could not be captured.")
    return {
        "schema_version": 1,
        "kind": "stateset-validation-comparison",
        "run_id": run.name,
        "parent_plan_sha256": parent_sha,
        "anchor_id": anchor_id,
        "evaluations": snapshots,
        "implementation_sha256": content_hash(package_implementation()),
    }


def _details(path: Path, verified: dict[str, Any]) -> dict[str, Any]:
    from stateset_agents.evaluation.judge_evidence import replay_judge_evidence

    reports = read_json(path / "paired-reports.json", max_bytes=4 * 1024 * 1024)
    matrices = {}
    for index, arm in enumerate(("baseline", "candidate")):
        result = read_json(path / "result.json", max_bytes=4 * 1024 * 1024)
        replay = replay_judge_evidence(
            path / "judge-recordings" / f"batch-{index:06d}",
            expected_manifest_sha256=result["judge_recordings"][arm]["manifest_sha256"],
        )
        matrices[arm] = {
            row["id"]: [
                [checks[key] for key in REQUIRED_CHECKS] for checks in row["checks"]
            ]
            for row in replay["results"]
        }
    base = reports["baselines"][0]
    candidate = reports["candidates"][0]
    before = {row["case_id"]: row for row in base["outcomes"]}
    after = {row["case_id"]: row for row in candidate["outcomes"]}
    pair = verified["summary"]["pairs"][0]
    guards = {
        "no_success_regressions": pair["regressions"] == 0,
        "no_new_policy_violations": all(
            after[key]["policy_violations"] <= row["policy_violations"]
            for key, row in before.items()
        ),
        "no_new_truncations": all(
            not after[key]["truncated"] or row["truncated"] is not None
            for key, row in before.items()
        ),
        "no_new_critical_check_failures": all(
            not value or matrices["candidate"][key][i][j]
            for key, passes in matrices["baseline"].items()
            for i, checks in enumerate(passes)
            for j, value in enumerate(checks)
        ),
    }
    transcript = read_json(path / "baseline.json", max_bytes=MAX_TRANSCRIPT_BYTES)
    fingerprint = content_hash(
        {
            "outcomes": [
                {key: value for key, value in row.items() if key != "elapsed_seconds"}
                for row in base["outcomes"]
            ],
            "transcripts": [
                {"case_id": row["case_id"], "transcript": row["transcript"]}
                for row in transcript["rows"]
            ],
            "checks": matrices["baseline"],
        }
    )
    candidate_fingerprint = content_hash(
        {
            "outcomes": [
                {key: value for key, value in row.items() if key != "elapsed_seconds"}
                for row in candidate["outcomes"]
            ],
            "checks": matrices["candidate"],
        }
    )
    return {
        "baseline_fingerprint": fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "success_rate": pair["candidate"]["success_rate"],
        "judge_score": verified["mean_judge_scores"]["candidate"],
        "baseline_success_rate": pair["baseline"]["success_rate"],
        "baseline_judge_score": verified["mean_judge_scores"]["baseline"],
        "regressions": pair["regressions"],
        "guards": guards,
    }


def compare_validations(
    run: Path,
    plan: dict[str, Any],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Rehash checkpoint bytes and replay one frozen validation cohort.

    This performs potentially large disk reads. The CLI/TUI always execute it
    through Buildr; Python callers should use the same resource discipline.
    """
    _validate_plan(plan, run)
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "stateset-validation-comparison"
        or plan.get("run_id") != run.name
        or read_run_plan(run)[1] != plan.get("parent_plan_sha256")
        or content_hash(package_implementation()) != plan.get("implementation_sha256")
    ):
        raise ValidationComparisonError(
            "Comparison inputs or implementation changed; capture a fresh request."
        )
    snapshots = plan["evaluations"]
    if (
        not isinstance(snapshots, list)
        or not 1 <= len(snapshots) <= MAX_COMPARISON_RECORDS
    ):
        raise ValidationComparisonError("The comparison snapshot is invalid.")
    anchor_path = _record(run / "evaluations", plan["anchor_id"])
    anchor_plan = read_json(anchor_path / "plan.json")
    if content_hash(anchor_plan) != next(
        row["plan_sha256"]
        for row in snapshots
        if row["evaluation_id"] == plan["anchor_id"]
    ):
        raise ValidationComparisonError(
            "The anchor changed after comparison submission."
        )
    anchor_verified = inspect_evaluation(anchor_path)
    if anchor_verified["split"] != "validation" or not anchor_verified["checkpoint"]:
        raise ValidationComparisonError("The anchor is not a checkpoint validation.")
    reference = _details(anchor_path, anchor_verified)
    protocol = _protocol(run, anchor_plan)
    rows, excluded = [], []
    complete = True
    for index, snapshot in enumerate(snapshots):
        if progress:
            progress(index, len(snapshots))
        name = snapshot["evaluation_id"]
        try:
            path = _record(run / "evaluations", name)
            current = read_json(path / "plan.json")
            if (
                snapshot["plan_sha256"] is None
                or content_hash(current) != snapshot["plan_sha256"]
            ):
                raise ValueError("Changed input")
            if _protocol(run, current) != protocol:
                excluded.append({"evaluation_id": name, "reason": "different_protocol"})
                continue
            verified = (
                anchor_verified
                if name == plan["anchor_id"]
                else inspect_evaluation(path)
            )
            detail = (
                reference if name == plan["anchor_id"] else _details(path, verified)
            )
            if detail["baseline_fingerprint"] != reference["baseline_fingerprint"]:
                excluded.append({"evaluation_id": name, "reason": "different_baseline"})
                complete = False
                continue
            binding = current["binding"]
            if (
                not binding.get("checkpoint")
                or type(binding.get("checkpoint_step")) is not int
            ):
                raise ValueError("Missing publication")
            rows.append(
                {
                    "evaluation_id": name,
                    "checkpoint": binding["checkpoint"],
                    "step": binding["checkpoint_step"],
                    "artifact_sha256": binding["artifact_sha256"],
                    **detail,
                }
            )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RecursionError,
            OverflowError,
        ):
            excluded.append({"evaluation_id": name, "reason": "unverifiable"})
            complete = False
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["evaluation_id"]):
        previous = unique.get(row["checkpoint"])
        if previous is None:
            unique[row["checkpoint"]] = {**row, "replicates": 1}
        elif any(
            previous[key] != row[key]
            for key in ["artifact_sha256", "candidate_fingerprint"]
        ):
            complete = False
            excluded.append(
                {
                    "evaluation_id": row["evaluation_id"],
                    "reason": "conflicting_replicate",
                }
            )
        else:
            previous["replicates"] += 1
    ranked = sorted(
        unique.values(),
        key=lambda row: (
            -row["success_rate"],
            -row["judge_score"],
            row["step"],
            row["checkpoint"],
        ),
    )
    base_key = (reference["baseline_success_rate"], reference["baseline_judge_score"])
    for row in ranked:
        row["eligible"] = (
            all(row["guards"].values())
            and (row["success_rate"], row["judge_score"]) > base_key
        )
        for key in [
            "baseline_fingerprint",
            "candidate_fingerprint",
            "baseline_success_rate",
            "baseline_judge_score",
        ]:
            row.pop(key)
    winner = (
        next((row for row in ranked if row["eligible"]), None) if complete else None
    )
    if progress:
        progress(len(snapshots), len(snapshots))
    return {
        "schema_version": 1,
        "run_id": run.name,
        "anchor_id": plan["anchor_id"],
        "protocol_sha256": protocol,
        "case_count": anchor_plan["cases"],
        "comparison_plan_sha256": content_hash(plan),
        "comparison_complete": complete,
        "verified_at": time.time(),
        "rows": ranked,
        "excluded": excluded,
        "baseline": {"success_rate": base_key[0], "judge_score": base_key[1]},
        "recommendation": (
            {
                "kind": "checkpoint",
                "checkpoint": winner["checkpoint"],
                "evaluation_id": winner["evaluation_id"],
            }
            if winner
            else {"kind": "base" if complete else "none", "checkpoint": ""}
        ),
        "quality_gain_verified": False,
        "notes": list(NOTES),
    }


def submit_comparison(
    run: Path,
    anchor_id: str,
    evaluation_ids: Sequence[str] | None = None,
    *,
    resubmit: bool = False,
) -> Path:
    """Retain and submit read-only verification once, without model inference.

    An active comparison for the same anchor is returned instead of queuing a
    duplicate, unless ``resubmit`` explicitly asks for a fresh one (for
    example after an uncertain submission). Stale, interrupted and failed
    comparisons never block a new submission.
    """
    run = run.resolve()
    plan = comparison_plan(run, anchor_id, evaluation_ids)
    queue = queue_executable()
    if not queue:
        raise ValidationComparisonError(
            "Install Buildr before comparing checkpoint bytes."
        )
    directory = run / DIRECTORY
    if directory.is_symlink():
        raise ValidationComparisonError(
            "Comparison history must be a regular directory."
        )
    directory.mkdir(mode=0o700, exist_ok=True)
    # This lock is released by the OS on process death. Never nest queue jobs;
    # submission itself is brief and only the detached verifier owns resources.
    if sys.platform != "linux":
        raise ValidationComparisonError(
            "Published-checkpoint comparison currently requires Linux."
        )
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(
                private_file_lock(
                    directory / ".submit.lock",
                    timeout_s=60.0,
                    unavailable="Comparison submission lock must be a private regular file.",
                    busy="Another comparison submission is still in progress; retry shortly.",
                )
            )
        except ValueError as error:
            raise ValidationComparisonError(str(error)) from None
        latest = None if resubmit else latest_comparison(run)
        if latest is not None:
            previous = comparison_status(latest)
            if (
                previous["anchor_id"] == anchor_id
                and previous["state"]["status"] in ACTIVE_COMPARISON_STATUSES
            ):
                return latest
        _prune_history(run, directory)
        return _submit_request(run, plan, queue, directory)


def _submit_request(
    run: Path, plan: dict[str, Any], queue: str, directory: Path
) -> Path:
    path = directory / (
        time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    )
    path.mkdir(mode=0o700)
    write_json(path / "plan.json", plan)
    write_json(path / "state.json", {"status": "queued", "created_at": time.time()})
    command = [
        queue,
        "run",
        "--detach",
        "--backend",
        "local",
        "--lane",
        "heavy",
        "--jobs",
        "1",
        "--memory-gb",
        "2",
        "--scratch-gb",
        "1",
        "--timeout",
        "3600",
        "--resource-key",
        "stateset-validation-comparison:" + str(run),
        "--",
        sys.executable,
        "-m",
        "stateset_agents.finetune_validation",
        "--execute",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=20, start_new_session=True
        )
        write_json(
            path / "submission-diagnostic.json",
            {
                "stdout": result.stdout[:4096],
                "stderr": result.stderr[:4096],
                "exit_code": result.returncode,
            },
        )
        match = re.fullmatch(
            r"submitted ([0-9a-f]{12}) detached \| log [^\n]+\n?", result.stdout
        )
        if result.returncode != 0 or match is None:
            write_json(path / "submission.json", {"status": "uncertain"})
        else:
            write_json(
                path / "submission.json",
                {"status": "submitted", "job_id": match.group(1)},
            )
    except subprocess.TimeoutExpired as error:

        def captured(value: str | bytes | None) -> str:
            return (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value or ""
            )[:4096]

        stdout, stderr = captured(error.output), captured(error.stderr)
        write_json(
            path / "submission-diagnostic.json",
            {"stdout": stdout, "stderr": stderr, "exception_type": "TimeoutExpired"},
        )
        match = re.fullmatch(
            r"submitted ([0-9a-f]{12}) detached \| log [^\n]+\n?", stdout
        )
        write_json(
            path / "submission.json",
            {"status": "uncertain", **({"job_id": match.group(1)} if match else {})},
        )
    except OSError as error:
        write_json(
            path / "submission-diagnostic.json",
            {"exception_type": type(error).__name__},
        )
        write_json(path / "submission.json", {"status": "uncertain"})
    return path


def _state(
    path: Path, *, allowed_from: Collection[str] | None = None, **changes: Any
) -> bool:
    """Change a verification's state through the shared locked transition.

    A recorded outcome is never overwritten, and a success whose result is
    retained is final, so a late progress update cannot reopen it.
    """
    return transition_state(
        path,
        allowed_from=allowed_from,
        success_is_final=lambda directory: (directory / "result.json").is_file(),
        **changes,
    )


def _fail_unclaimed(path: Path) -> None:
    """Record that a queued verification's worker could not start.

    Only an own comparison record that is still queued changes; without this
    the record stayed ``queued`` forever and blocked its anchor.
    """
    try:
        if (
            path.parent.name == DIRECTORY
            and not path.is_symlink()
            and (path / "state.json").is_file()
        ):
            _state(
                path,
                allowed_from={"queued"},
                status="failed",
                error="The verification worker could not start; inspect inputs and private queue logs.",
                finished_at=time.time(),
            )
    except (OSError, ValueError):
        pass


def execute_comparison(path: Path) -> int:
    """Own one queued verification and publish a bounded content-free result."""
    os.umask(0o077)
    try:
        run = _owned_run(path)
        fd = os.open(
            path / "execution.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
        with os.fdopen(fd, "w") as stream:
            json.dump(
                {
                    "pid": os.getpid(),
                    "process_token": process_token(os.getpid()),
                    "module_path": __file__,
                },
                stream,
            )
    except FileExistsError:
        return 1  # Another worker owns this verification; keep its state.
    except (OSError, ValueError):
        _fail_unclaimed(path)
        return 1
    try:
        plan = read_json(path / "plan.json")
        _state(path, status="running", phase="verifying_checkpoints")

        def progress(done: int, total: int) -> None:
            _state(path, completed=done, total=total)

        result = compare_validations(run, plan, progress=progress)
        write_json(path / "result.json", result)
        _state(
            path,
            status="succeeded",
            phase="complete",
            result_sha256=content_hash(result),
            finished_at=time.time(),
        )
        return 0
    except Exception as error:
        write_json(
            path / "worker-error.json",
            {
                "exception_type": type(error).__name__,
                "frames": [
                    {
                        "module": Path(frame.filename).name,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback.extract_tb(error.__traceback__)
                ],
            },
        )
        _state(
            path,
            status="failed",
            error="Verification failed; inspect inputs, code and private queue logs.",
            finished_at=time.time(),
        )
        return 1


def comparison_status(path: Path) -> dict[str, Any]:
    """Read own metadata and retained results, without rehashing models or querying Buildr."""
    run = _owned_run(path)
    plan = read_json(path / "plan.json")
    _validate_plan(plan, run)
    state = read_json(path / "state.json")
    status = state.get("status")
    if not isinstance(status, str) or status not in {
        "queued",
        "running",
        "succeeded",
        "failed",
    }:
        raise ValidationComparisonError("Comparison status is unreadable.")
    submission = (
        read_json(path / "submission.json")
        if (path / "submission.json").exists()
        else {}
    )
    job = submission.get("job_id")
    if job is not None and (
        not isinstance(job, str) or not re.fullmatch(r"[0-9a-f]{12}", job)
    ):
        raise ValidationComparisonError("Comparison queue receipt is unreadable.")
    claimed = (path / "execution.json").exists()
    if status in {"queued", "running"} and claimed:
        # Only an owner that is provably gone (or replaced) interrupts it.
        if owner_alive(path / "execution.json") is False:
            status = "interrupted"
    elif status == "queued":
        created = state.get("created_at")
        if (
            isinstance(created, (int, float))
            and not isinstance(created, bool)
            and math.isfinite(created)
            and time.time() - created > QUEUED_STALE_AFTER_SECONDS
        ):
            status = "stale"
        elif submission.get("status") == "uncertain":
            status = "submission_uncertain"
    result = None
    if status == "succeeded":
        result = read_json(path / "result.json", max_bytes=256 * 1024)
        if result.get("comparison_plan_sha256") != content_hash(plan) or content_hash(
            result
        ) != state.get("result_sha256"):
            raise ValidationComparisonError(
                "Comparison results changed after verification."
            )
        _validate_result(result)
        if result["anchor_id"] != plan["anchor_id"] or result["run_id"] != run.name:
            raise ValidationComparisonError("Comparison result identity changed.")
        result = _public_result(result)
    public: dict[str, Any] = {"status": status}
    for key in ["completed", "total"]:
        value = state.get(key)
        if type(value) is int and 0 <= value <= MAX_COMPARISON_RECORDS:
            public[key] = value
    return {
        "comparison_id": path.name,
        "anchor_id": plan["anchor_id"],
        "state": public,
        "job_id": job,
        "result": result,
        "training_started": False,
        "inference_started": False,
        "notes": [
            "Queue work survives leaving the TUI. Reopening does not resubmit.",
            "No cancellation or automatic retry is performed; inspect the retained job handle if progress is uncertain, then resubmit explicitly.",
        ],
    }


def _validate_result(result: dict[str, Any]) -> None:
    if type(result.get("case_count")) is not int or not 1 <= result["case_count"] <= 64:
        raise ValidationComparisonError("Comparison case count is unreadable.")
    if (
        type(result.get("schema_version")) is not int
        or result["schema_version"] != 1
        or type(result.get("comparison_complete")) is not bool
        or not isinstance(result.get("anchor_id"), str)
        or not re.fullmatch(RECORD_ID, result["anchor_id"])
        or not isinstance(result.get("run_id"), str)
        or not re.fullmatch(RECORD_ID, result["run_id"])
    ):
        raise ValidationComparisonError("Comparison result identity is unreadable.")
    for key in ["protocol_sha256", "comparison_plan_sha256"]:
        if not isinstance(result.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", result[key]
        ):
            raise ValidationComparisonError("Comparison result checksum is unreadable.")
    stamp = result.get("verified_at")
    if (
        not isinstance(stamp, (int, float))
        or isinstance(stamp, bool)
        or not 0 <= stamp <= 10**12
        or not math.isfinite(stamp)
    ):
        raise ValidationComparisonError("Comparison timestamp is unreadable.")
    rows = result.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) > MAX_COMPARISON_RECORDS
        or not isinstance(result.get("baseline"), dict)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValidationComparisonError("Comparison rows are unreadable.")
    for row in [result["baseline"], *rows]:
        for key in ["success_rate", "judge_score"]:
            value = row.get(key)
            if (
                type(value) not in {int, float}
                or not 0 <= value <= 1
                or not math.isfinite(value)
            ):
                raise ValidationComparisonError("Comparison metrics are unreadable.")
    for row in rows:
        if (
            not isinstance(row.get("checkpoint"), str)
            or not re.fullmatch(CHECKPOINT_NAME_PATTERN, row["checkpoint"])
            or not isinstance(row.get("evaluation_id"), str)
            or not re.fullmatch(RECORD_ID, row["evaluation_id"])
            or type(row["eligible"]) is not bool
            or type(row["step"]) is not int
            or row["step"] < 1
        ):
            raise ValidationComparisonError("Comparison target metadata is unreadable.")
        if (
            not isinstance(row.get("artifact_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["artifact_sha256"])
            or type(row.get("regressions")) is not int
            or not 0 <= row["regressions"] <= 64
            or type(row.get("replicates")) is not int
            or not 1 <= row["replicates"] <= MAX_COMPARISON_RECORDS
            or not isinstance(row.get("guards"), dict)
            or set(row["guards"]) != set(GUARDS)
            or any(type(row["guards"][key]) is not bool for key in GUARDS)
        ):
            raise ValidationComparisonError("Comparison guard metadata is unreadable.")
        base = result["baseline"]
        if row["eligible"] != (
            all(row["guards"].values())
            and (row["success_rate"], row["judge_score"])
            > (base["success_rate"], base["judge_score"])
        ):
            raise ValidationComparisonError(
                "Comparison eligibility differs from its guards."
            )
    excluded = result.get("excluded")
    if not isinstance(excluded, list) or len(excluded) > MAX_COMPARISON_RECORDS:
        raise ValidationComparisonError("Comparison exclusions are unreadable.")
    for row in excluded:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("evaluation_id"), str)
            or not re.fullmatch(RECORD_ID, row["evaluation_id"])
            or row.get("reason")
            not in {
                "different_protocol",
                "different_baseline",
                "unverifiable",
                "conflicting_replicate",
            }
        ):
            raise ValidationComparisonError(
                "Comparison exclusion identity is unreadable."
            )
    choice = result.get("recommendation")
    if not isinstance(choice, dict) or choice.get("kind") not in {
        "base",
        "checkpoint",
        "none",
    }:
        raise ValidationComparisonError("Comparison recommendation is unreadable.")
    if choice["kind"] == "checkpoint" and (
        result.get("comparison_complete") is not True
        or not any(
            row["checkpoint"] == choice.get("checkpoint")
            and row["evaluation_id"] == choice.get("evaluation_id")
            and row["eligible"]
            for row in rows
        )
    ):
        raise ValidationComparisonError(
            "Comparison recommendation differs from its eligible rows."
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["success_rate"],
            -row["judge_score"],
            row["step"],
            row["checkpoint"],
        ),
    )
    winner = (
        next((row for row in ranked if row["eligible"]), None)
        if result["comparison_complete"]
        else None
    )
    expected = (
        {
            "kind": "checkpoint",
            "checkpoint": winner["checkpoint"],
            "evaluation_id": winner["evaluation_id"],
        }
        if winner
        else {
            "kind": "base" if result["comparison_complete"] else "none",
            "checkpoint": "",
        }
    )
    if rows != ranked or choice != expected:
        raise ValidationComparisonError(
            "Comparison recommendation differs from its ranking."
        )


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: result[key]
        for key in [
            "schema_version",
            "run_id",
            "anchor_id",
            "protocol_sha256",
            "case_count",
            "comparison_plan_sha256",
            "comparison_complete",
            "verified_at",
        ]
    }
    public["baseline"] = {
        key: result["baseline"][key] for key in ["success_rate", "judge_score"]
    }
    public["rows"] = [
        {
            key: row[key]
            for key in [
                "evaluation_id",
                "checkpoint",
                "step",
                "artifact_sha256",
                "success_rate",
                "judge_score",
                "regressions",
                "guards",
                "replicates",
                "eligible",
            ]
        }
        for row in result["rows"]
    ]
    public["excluded"] = [
        {key: row[key] for key in ["evaluation_id", "reason"]}
        for row in result["excluded"]
    ]
    public["recommendation"] = {
        key: result["recommendation"][key]
        for key in ["kind", "checkpoint", "evaluation_id"]
        if key in result["recommendation"]
    }
    public["quality_gain_verified"] = False
    public["notes"] = list(NOTES)
    return public


#: Directory entries read from the history (records plus the submit lock and
#: any stray files); a larger directory is reported rather than scanned.
_HISTORY_SCAN_LIMIT = 4 * MAX_COMPARISON_RECORDS


def _history_records(directory: Path) -> list[Path]:
    """Own comparison record directories, oldest first; other entries ignored."""
    entries = list(islice(directory.iterdir(), _HISTORY_SCAN_LIMIT + 1))
    if len(entries) > _HISTORY_SCAN_LIMIT:
        raise ValidationComparisonError(
            f"Comparison history has more than {_HISTORY_SCAN_LIMIT} entries; "
            f"remove stray files from {DIRECTORY}."
        )
    return sorted(
        path
        for path in entries
        if re.fullmatch(RECORD_ID, path.name)
        and path.is_dir()
        and not path.is_symlink()
    )


def _nominated_comparisons(run: Path) -> set[str] | None:
    """Comparison IDs that retained nominations depend on; ``None`` if unknown."""
    directory = run / "nominations"
    if not directory.exists():
        return set()
    if directory.is_symlink():
        return None
    referenced: set[str] = set()
    try:
        for path in islice(directory.iterdir(), _HISTORY_SCAN_LIMIT + 1):
            if not re.fullmatch(RECORD_ID, path.name) or path.is_symlink():
                continue
            receipt = path / "receipt.json"
            if not receipt.exists():
                continue  # an interrupted nomination authorizes nothing
            plan = read_json(receipt).get("plan")
            identifier = plan.get("comparison_id") if isinstance(plan, dict) else None
            if not isinstance(identifier, str):
                return None
            referenced.add(identifier)
    except (OSError, ValueError, RecursionError):
        return None
    return referenced


def _prune_history(run: Path, directory: Path) -> None:
    """Make room for one more record by deleting the oldest finished ones.

    Active verifications, and successful ones a retained nomination depends
    on, are kept. If nothing can be removed the submission is refused with a
    clear message instead of failing later as unreadable history.
    """
    records = _history_records(directory)
    excess = len(records) - (MAX_COMPARISON_RECORDS - 1)
    if excess <= 0:
        return
    nominated = _nominated_comparisons(run)
    for path in records:
        if excess <= 0:
            return
        try:
            status = comparison_status(path)["state"]["status"]
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            status = "unreadable"
        if status in ACTIVE_COMPARISON_STATUSES or (
            status == "succeeded" and (nominated is None or path.name in nominated)
        ):
            continue
        shutil.rmtree(path)
        excess -= 1
    if excess > 0:
        raise ValidationComparisonError(
            f"Comparison history already holds {MAX_COMPARISON_RECORDS} records "
            "that are still active or back a nomination; wait for active "
            "verifications to finish before submitting another."
        )


def latest_comparison(run: Path) -> Path | None:
    """Find the newest own request without loading results or model libraries."""
    directory = run / DIRECTORY
    if not directory.exists():
        return None
    if directory.is_symlink():
        raise ValidationComparisonError("Comparison history must be regular.")
    return max(_history_records(directory), default=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path, required=True)
    raise SystemExit(execute_comparison(parser.parse_args().execute))
