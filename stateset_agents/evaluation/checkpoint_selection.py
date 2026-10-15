"""Validation-only checkpoint selection with complete, unambiguous evidence."""

from __future__ import annotations

import math
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash


def validate_validation_cases(evidence: Any, *, cases: dict[str, Any]) -> None:
    """Require one finite reward per planned case and an accurate aggregate.

    Native benchmark validation uses exactly one rollout per case. Case hashes
    bind the evidence to the planned split, including its private scenario data.
    """
    if not cases or any(not isinstance(key, str) or not key for key in cases):
        raise ValueError("Validation cases must have nonempty identifiers")
    if not isinstance(evidence, dict) or evidence.get("case_hashes") != {
        key: content_hash(row) for key, row in cases.items()
    }:
        raise ValueError("Validation cases differ from plan")
    outcomes = evidence.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(cases):
        raise ValueError("Incomplete validation case coverage")
    seen = set()
    rewards = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise ValueError("Invalid validation outcome")
        case_id = outcome.get("case_id")
        if not isinstance(case_id, str) or case_id not in cases or case_id in seen:
            raise ValueError("Invalid or duplicate validation case")
        if outcome.get("case_hash") != evidence["case_hashes"][case_id]:
            raise ValueError("Validation outcome reset input differs from planned case")
        seen.add(case_id)
        rewards.append(_finite_reward(outcome.get("reward")))
    metrics = evidence.get("metrics")
    reward = _finite_reward(
        metrics.get("reward_mean") if isinstance(metrics, dict) else None
    )
    # Scale before summing to avoid overflow for finite, same-sign rewards.
    mean = math.fsum(value / len(rewards) for value in rewards)
    if not math.isclose(reward, mean, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Validation reward mean differs from case outcomes")


def _finite_reward(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Validation reward must be finite")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("Validation reward must be finite") from exc
    if not math.isfinite(result):
        raise ValueError("Validation reward must be finite")
    return result


def select_validation_checkpoint(
    evaluations: Any, *, steps: int, cases: dict[str, Any]
) -> dict[str, Any]:
    """Require every planned validation step, then select with earliest-step ties.

    Duplicate steps are rejected: recovery must replace a replayed evaluation,
    rather than leave conflicting checkpoints available for selection.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("Planned training steps must be positive")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("Missing validation selection evidence")
    seen = set()
    for item in evaluations:
        if not isinstance(item, dict):
            raise ValueError("Invalid validation evidence")
        step = item.get("step")
        checkpoint = item.get("checkpoint")
        if (
            type(step) is not int
            or not 0 <= step <= steps
            or not isinstance(checkpoint, dict)
            or not isinstance(checkpoint.get("path"), str)
            or not checkpoint["path"].startswith("river://")
            or not checkpoint["path"][len("river://") :].strip()
        ):
            raise ValueError("Invalid validation evidence")
        validate_validation_cases(item, cases=cases)
        if step in seen:
            raise ValueError("Duplicate validation step")
        seen.add(step)
    if seen != set(range(steps + 1)):
        raise ValueError("Incomplete validation checkpoint coverage")
    return max(
        sorted(evaluations, key=lambda entry: entry["step"]),
        key=lambda entry: entry["metrics"]["reward_mean"],
    )


def require_training_progress(
    progress: Any,
    *,
    steps: int,
    recovery_receipts: dict[str, Any] | None = None,
    run_manifest_hash: str | None = None,
) -> None:
    """Reject missing, duplicated, or out-of-order committed training records."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Planned training steps must be positive")
    if (
        not isinstance(progress, list)
        or any(
            not isinstance(p, dict) or type(p.get("step")) is not int for p in progress
        )
        or [p["step"] for p in progress] != list(range(1, steps + 1))
    ):
        raise ValueError("Incomplete RL training progress; test split remains sealed")
    from stateset_agents.training.river_progress import (
        validate_recovery_receipt,
        validate_training_metrics,
    )

    if recovery_receipts is not None:
        if not isinstance(recovery_receipts, dict):
            raise ValueError("Invalid recovery receipts")
        if recovery_receipts and run_manifest_hash is None:
            raise ValueError("Recovery receipts require a run identity")
        for key, receipt in recovery_receipts.items():
            validate_recovery_receipt(receipt, key, run_manifest_hash or "", steps)
    for entry in progress:
        if isinstance(entry.get("metrics"), dict):
            if entry.get("source") == "river_recovery":
                raise ValueError("Recovered completion cannot invent measured metrics")
            validate_training_metrics(entry["metrics"])
            continue
        if (
            entry.get("metrics", "missing") is not None
            or entry.get("source") != "river_recovery"
            or not isinstance(entry.get("recovery_receipt_hash"), str)
            or recovery_receipts is None
            or run_manifest_hash is None
        ):
            raise ValueError("Missing training metrics or recovery receipt")
        key = entry["recovery_receipt_hash"]
        receipt = validate_recovery_receipt(
            recovery_receipts.get(key), key, run_manifest_hash, steps
        )
        if entry["step"] > receipt["completed_batches"]:
            raise ValueError("Recovery receipt does not cover training step")
