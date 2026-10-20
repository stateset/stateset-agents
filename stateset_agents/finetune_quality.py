"""Bounded, artifact-bound inspection of recorded held-out comparisons.

This module runs no models and makes no provider requests. Recorded outcomes
remain attestations; hashing them does not establish evaluator independence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import compare_runs
from stateset_agents.finetune_workspace import (
    TERMINAL_STATES,
    read_run_plan,
    run_status,
    write_json,
)

MAX_EVIDENCE_BYTES = 8 * 1024 * 1024


def _read(path: Path, limit: int = MAX_EVIDENCE_BYTES) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink():
        raise ValueError("Evidence must be a regular file")
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Evidence exceeds the regular-file limit")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Evidence exceeds the regular-file limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected evidence object")
    return value, hashlib.sha256(raw).hexdigest()


def _artifact(run: Path, completion: dict[str, Any]) -> Path:
    output = Path(completion["output_dir"])
    if (
        not output.is_absolute()
        or ".." in output.parts
        or not output.is_relative_to(run / "artifacts")
        or not output.resolve().is_relative_to(run / "artifacts")
    ):
        raise ValueError("Artifact must belong to this run")
    for path in (output, *output.parents):
        if path == run:
            break
        if path.is_symlink():
            raise ValueError("Artifact path cannot contain symlinks")
    return output


def _summarize(
    run: Path,
    evidence: dict[str, Any],
    *,
    expected_split: str = "test",
    artifact: Path | None = None,
    nomination_id: str = "",
) -> dict[str, Any]:
    from stateset_agents.training.promotion import artifact_digest

    if not isinstance(evidence, dict):
        raise ValueError("Expected paired report object")
    plan, plan_sha = read_run_plan(run)
    state, _ = _read(run / "state.json", 128 * 1024)
    if expected_split not in {"test", "validation"} or (
        artifact is not None and expected_split != "validation" and not nomination_id
    ):
        raise ValueError("Published artifact overrides require validation reports")
    if state.get("status") != "succeeded" and not (
        artifact is not None and run_status(run).get("status") in TERMINAL_STATES
    ):
        raise ValueError("Quality inspection requires a completed successful run")
    completion_path = run / "completion.json"
    completion: dict[str, Any]
    completion_sha: str | None
    if (
        artifact is not None
        and not completion_path.exists()
        and not completion_path.is_symlink()
    ):
        completion, completion_sha = {}, None
    else:
        completion, completion_sha = _read(completion_path, 128 * 1024)
    if completion.get("result_kind") == "hosted_pointer":
        raise ValueError("Hosted artifacts need provider-bound evaluation support")
    output = _artifact(
        run, {"output_dir": str(artifact)} if artifact is not None else completion
    )
    digest = artifact_digest(output)
    if nomination_id:
        from stateset_agents.finetune_nomination import verify_nomination

        if artifact is None or expected_split != "test":
            raise ValueError("Nominated artifacts require a fresh test assessment")
        verify_nomination(
            run,
            nomination_id,
            {
                "checkpoint": output.name,
                "artifact_sha256": digest,
                "parent_plan_sha256": plan_sha,
            },
        )
    baselines, candidates = evidence["baselines"], evidence["candidates"]
    if (
        type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != 1
        or not isinstance(baselines, list)
        or not isinstance(candidates, list)
        or not 1 <= len(baselines) <= 64
        or len(baselines) != len(candidates)
        or any(not isinstance(row, dict) for row in baselines + candidates)
    ):
        raise ValueError("Expected paired held-out reports")
    comparison = compare_runs(
        baselines, candidates, min_seeds=1, expected_split=expected_split
    )
    if comparison["protocol"]["base_model"] != plan["config"]["model"]:
        raise ValueError("Evaluation model differs from the reviewed model")
    bound = [
        report
        for report in candidates
        if report["selected_checkpoint"].get("inventory_sha256") == digest
        and Path(report["selected_checkpoint"]["path"]).resolve() == output.resolve()
    ]
    if len(bound) != 1:
        raise ValueError("Exactly one candidate must match the completed artifact")
    training_seed = plan["config"].get("seed")
    if training_seed is not None and (
        type(training_seed) is not int
        or not 0 <= training_seed <= 2**32 - 1
        or bound[0]["seed"] != training_seed
    ):
        raise ValueError("Bound report seed differs from the recorded training seed")
    regressions: list[dict[str, Any]] = []
    pairs = []
    for pair in comparison["pairs"]:
        pairs.append(
            {
                **{
                    key: pair[key]
                    for key in ("seed", "success_gain", "wins", "regressions")
                },
                **{
                    arm: {
                        key: value
                        for key, value in pair[arm].items()
                        if value is None or type(value) in {int, float}
                    }
                    for arm in ("baseline", "candidate")
                },
            }
        )
        before = next(row for row in baselines if row["seed"] == pair["seed"])
        after = next(row for row in candidates if row["seed"] == pair["seed"])
        old = {row["case_id"]: row for row in before["outcomes"]}
        for row in after["outcomes"]:
            previous = old[row["case_id"]]
            if previous["success"] and not row["success"] and len(regressions) < 50:
                regressions.append(
                    {
                        "seed": pair["seed"],
                        "case_handle": hashlib.sha256(
                            row["case_id"].encode()
                        ).hexdigest()[:16],
                        "baseline_reward": previous["reward"],
                        "candidate_reward": row["reward"],
                        "policy_violations": row["policy_violations"],
                    }
                )
    # Show the stronger gate separately; a one-seed diagnostic is never promoted.
    study = compare_runs(
        baselines,
        candidates,
        min_seeds=6,
        require_significance=True,
        expected_split=expected_split,
    )
    inventories = [
        row["selected_checkpoint"].get("inventory_sha256") for row in candidates
    ]
    distinct = all(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
        for value in inventories
    ) and len(set(inventories)) == len(inventories)
    sufficient_cases = all(
        len(set(row["case_hashes"].values())) >= 32 for row in candidates
    )
    protocol = comparison["protocol"]["settings"]
    revision = plan["config"].get("revision")
    pinned_model = (
        isinstance(revision, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", revision))
        and revision != "0" * 40
        and protocol.get("model_revision") == revision
    )
    pinned_judge = (
        isinstance(protocol.get("judge_model"), str)
        and bool(protocol["judge_model"].strip())
        and isinstance(protocol.get("judge_revision"), str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", protocol["judge_revision"]))
        and protocol["judge_revision"] != "0" * 40
        and isinstance(protocol.get("rubric_revision"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", protocol["rubric_revision"]))
    )
    review = evidence.get("review")
    reviewed = (
        isinstance(review, dict)
        and review.get("source") == "reviewed"
        and isinstance(review.get("reviewer"), str)
        and bool(review["reviewer"].strip())
    )
    return {
        "schema_version": 1,
        "evaluation_split": expected_split,
        "training_seed": training_seed,
        "plan_sha256": plan_sha,
        "completion_sha256": completion_sha,
        "artifact_sha256": digest,
        "seeds": len(pairs),
        "pairs": pairs,
        "mean_success_gain": comparison["mean_success_gain"],
        "descriptive_gates": comparison["gates"],
        "thresholds": study["thresholds"],
        "seed_inference": study["inference"],
        "study_gates": {
            "held_out_test_partition": expected_split == "test",
            "bound_training_seed_recorded": training_seed is not None,
            **study["gates"],
            "distinct_artifact_hashes": distinct,
            "at_least_32_distinct_cases": sufficient_cases,
            "pinned_model_revision": pinned_model,
            "pinned_judge_and_rubric": pinned_judge,
            "reviewed_provenance": reviewed,
        },
        "recorded_study_passed": expected_split == "test"
        and training_seed is not None
        and study["passed"]
        and distinct
        and sufficient_cases
        and pinned_model
        and pinned_judge
        and reviewed,
        "regressions": regressions,
        "regressions_total": sum(pair["regressions"] for pair in pairs),
        "bound_seed": bound[0]["seed"],
        "notes": [
            (
                "Validation supports checkpoint selection; final quality requires separate fresh test cases."
                if expected_split == "validation"
                else "Held-out test outcomes are recorded claims; evaluator identity and training independence need review."
            ),
            "Only this run's selected artifact was rehashed; other seed artifacts remain recorded claims. This summarizer runs no model or judge.",
            "The bound report must match its recorded training seed; unknown historical seeds cannot establish training independence.",
            "Regression handles hash case IDs; raw IDs and transcripts are hidden.",
            "The study gate requires six seeds, distinct checkpoints, 32 distinct cases per seed, significance and safety gates.",
            "This is not production promotion or automatic checkpoint selection.",
        ],
    }


def record_quality(run: Path, source: Path) -> dict[str, Any]:
    """Validate paired reports against completed weights and retain them privately."""
    try:
        run = run.resolve()
        evidence, evidence_sha = _read(source.expanduser(), MAX_EVIDENCE_BYTES)
        return _record_quality(run, evidence, evidence_sha)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ValueError(
            "Cannot verify quality evidence. Use paired test reports for this model and completed local weights; check formats, hashes and file limits."
        ) from None


def _record_quality(
    run: Path, evidence: dict[str, Any], evidence_sha: str
) -> dict[str, Any]:
    summary = _summarize(run, evidence)
    record = {
        "schema_version": 1,
        "recorded_at": time.time(),
        "source_sha256": evidence_sha,
        "evidence": evidence,
        "binding": {
            key: summary[key]
            for key in ("plan_sha256", "completion_sha256", "artifact_sha256")
        },
    }
    if (
        len(json.dumps(record, indent=2, allow_nan=False).encode())
        > MAX_EVIDENCE_BYTES + 128 * 1024
    ):
        raise ValueError("Retained evidence exceeds size limit")
    write_json(run / "quality.json", record)
    return summary


def record_quality_document(run: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Retain a bounded JSON upload using the same artifact checks as file imports."""
    try:
        raw = json.dumps(evidence, allow_nan=False).encode()
        if len(raw) > MAX_EVIDENCE_BYTES:
            raise ValueError("Evidence exceeds the upload limit")
        # Capture the input before checks so a caller cannot mutate the retained snapshot.
        captured = json.loads(raw)
        return _record_quality(run.resolve(), captured, hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ValueError(
            "Cannot verify quality evidence. Use paired test reports for this model and completed local weights; check formats, hashes and file limits."
        ) from None


def inspect_quality(run: Path) -> dict[str, Any]:
    """Recompute retained comparisons and verify current artifact and plan bytes."""
    try:
        record, _ = _read(run / "quality.json", MAX_EVIDENCE_BYTES + 128 * 1024)
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
        ):
            raise ValueError("Unsupported receipt")
        summary = _summarize(run.resolve(), record["evidence"])
        if record["binding"] != {
            key: summary[key]
            for key in ("plan_sha256", "completion_sha256", "artifact_sha256")
        }:
            raise ValueError("Evidence binding changed")
        return summary
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ValueError(
            "No verifiable quality evidence. Import paired reports with e then f, or use finetune quality --evidence PATH."
        ) from None
