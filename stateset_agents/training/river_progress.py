"""Reconcile committed River batches without inventing historical metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_rl import atomic_json

DEFAULT_ZERO_UPDATE_PATIENCE = 5


def zero_update_stop_step(records: list[dict[str, Any]], patience: int) -> int | None:
    """Return the first step reaching a consecutive observed-skip threshold.

    Zero disables the guard. Unknown activity and gaps reset the streak; SDK
    recovery receipts alone cannot establish skipped optimizer updates.
    """
    if type(patience) is not int or patience < 0:
        raise ValueError("zero_update_patience must be an integer >= 0")
    streak = 0
    previous = 0
    for record in records:
        step = record.get("step")
        if type(step) is not int or step < 1:
            raise ValueError("Training activity requires positive integer steps")
        if step != previous + 1:
            streak = 0
        previous = step
        metrics = record.get("metrics")
        if metrics is not None:
            validate_training_metrics(metrics)
        if metrics is not None and metrics.get("train/updated") == 0:
            streak += 1
        else:
            streak = 0
        if patience and streak >= patience:
            return step
    return None


def validate_training_metrics(metrics: Any) -> None:
    """Require nonempty, named, finite scalar observations from River's Step."""
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Training metrics must be a nonempty dictionary")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Training metrics require nonempty string names")
        if type(value) not in (int, float):
            raise ValueError("Training metrics must contain finite numeric scalars")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError("Training metrics exceed finite numeric range") from exc
        if not finite:
            raise ValueError("Training metrics must contain finite numeric scalars")
    if "train/updated" in metrics and metrics["train/updated"] not in (0, 1):
        raise ValueError("Training metrics train/updated must be zero or one")
    if "train/datums" in metrics:
        datums = metrics["train/datums"]
        if datums < 0 or datums != int(datums):
            raise ValueError(
                "Training metrics train/datums must be a nonnegative count"
            )
        if "train/updated" in metrics and bool(datums) != bool(
            metrics["train/updated"]
        ):
            raise ValueError("Training metrics update flag contradicts train/datums")


def summarize_training_activity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize SDK-reported updates after validating batch completion separately.

    Recovery receipts and legacy metrics without ``train/updated`` leave activity
    unknown. Batch completion is not evidence of an optimizer update, and an
    update acknowledgement does not prove gradients or learned improvement.
    """
    updates = skipped = unknown = 0
    for record in records:
        metrics = record.get("metrics")
        if metrics is None:
            unknown += 1
            continue
        validate_training_metrics(metrics)
        if "train/updated" not in metrics:
            unknown += 1
        elif metrics["train/updated"] == 1:
            updates += 1
        else:
            skipped += 1
    status = (
        "updates_observed"
        if updates
        else "unknown" if unknown or not records else "no_updates_observed"
    )
    return {
        "schema_version": 1,
        "status": status,
        "completed_batches": len(records),
        "observed_optimizer_updates": updates,
        "observed_skipped_batches": skipped,
        "unknown_update_batches": unknown,
        "scope": "SDK-reported run activity; not proof of nonzero gradients, selected-checkpoint changes, learning gains, or provider cost",
    }


class RiverTrainingProgress:
    """Durable observation records plus explicit SDK recovery receipts.

    The caller owns the run directory lock. Only River's ``after_recovery``
    callback may authorize reconciliation; file presence alone proves no commit.
    Receipts record recovered batch counts, not measured losses or provider cost.
    """

    def __init__(self, directory: Path, *, steps: int, run_manifest_hash: str):
        if type(steps) is not int or steps < 1:
            raise ValueError("Planned training steps must be positive")
        if not isinstance(run_manifest_hash, str) or not run_manifest_hash:
            raise ValueError("Training progress requires a run identity")
        self.directory = directory
        self.steps = steps
        self.run_manifest_hash = run_manifest_hash
        self.path = directory / "training_metrics.json"
        self.receipts_path = directory / "recovery_receipts.json"
        self.records: list[dict[str, Any]] = (
            json.loads(self.path.read_text()) if self.path.exists() else []
        )
        self.receipts: dict[str, Any] = (
            json.loads(self.receipts_path.read_text())
            if self.receipts_path.exists()
            else {}
        )
        if not isinstance(self.records, list) or any(
            not isinstance(entry, dict)
            or type(entry.get("step")) is not int
            or not 1 <= entry["step"] <= steps
            for entry in self.records
        ):
            raise ValueError("Invalid persisted training progress")
        if len({entry["step"] for entry in self.records}) != len(self.records):
            raise ValueError("Duplicate persisted training step")
        for entry in self.records:
            if entry.get("metrics") is not None:
                validate_training_metrics(entry["metrics"])
        if not isinstance(self.receipts, dict):
            raise ValueError("Invalid recovery receipts")
        for key, receipt in self.receipts.items():
            validate_recovery_receipt(receipt, key, run_manifest_hash, steps)

    def observe(self, step: int, metrics: dict[str, Any]) -> None:
        """Persist an emitted batch, replacing any replayed observation suffix."""
        if type(step) is not int or not 1 <= step <= self.steps:
            raise ValueError("Emitted batch is outside the planned run")
        validate_training_metrics(metrics)
        # Step dictionaries belong to the SDK. A later mutation must not
        # change an earlier observation when the next batch rewrites this file.
        snapshot = dict(metrics)
        records = [entry for entry in self.records if entry["step"] < step]
        records.append({"step": step, "metrics": snapshot})
        atomic_json(self.path, records)
        self.records = records

    def reconcile(self, completed_batches: int) -> None:
        """Record an SDK-confirmed prefix, retaining gaps as unknown metrics.

        Receipt-first persistence is idempotent across a crash between writes.
        An observed suffix past the restored commit is discarded, never trusted.
        """
        if (
            type(completed_batches) is not int
            or not 0 <= completed_batches <= self.steps
        ):
            raise ValueError("Recovered batch count is outside the planned run")
        receipt = {
            "schema_version": 1,
            "source": "river_after_recovery",
            "run_manifest_hash": self.run_manifest_hash,
            "completed_batches": completed_batches,
        }
        receipt_hash = content_hash(receipt)
        receipts = {**self.receipts, receipt_hash: receipt}
        atomic_json(self.receipts_path, receipts)
        self.receipts = receipts
        observed = {entry["step"]: entry for entry in self.records}
        records = [
            observed.get(
                step,
                {
                    "step": step,
                    "metrics": None,
                    "source": "river_recovery",
                    "recovery_receipt_hash": receipt_hash,
                },
            )
            for step in range(1, completed_batches + 1)
        ]
        atomic_json(self.path, records)
        self.records = records


def validate_recovery_receipt(
    receipt: Any, receipt_hash: str, run_manifest_hash: str, steps: int
) -> dict[str, Any]:
    """Verify the identity and bounds of one local SDK recovery receipt."""
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("source") != "river_after_recovery"
        or receipt.get("run_manifest_hash") != run_manifest_hash
        or type(receipt.get("completed_batches")) is not int
        or not 0 <= receipt["completed_batches"] <= steps
        or content_hash(receipt) != receipt_hash
    ):
        raise ValueError("Invalid or mismatched recovery receipt")
    return receipt
