"""Recovery must recover commits, never invent observations or trust stale tails."""

import json

import pytest

from stateset_agents.evaluation.checkpoint_selection import require_training_progress
from stateset_agents.training.river_progress import (
    RiverTrainingProgress,
    summarize_training_activity,
    zero_update_stop_step,
)


@pytest.mark.parametrize(
    "flags,patience,expected",
    [
        ([], 2, None),
        ([0, 0, 1], 2, 2),
        ([1, 0, 0], 2, 3),
        ([0, 1, 0], 2, None),
        ([0, None, 0], 2, None),
        ([0, "legacy", 0], 2, None),
        ([None, 0, 0], 2, 3),
        ([0, 0, 0], 0, None),
    ],
)
def test_zero_update_guard_requires_consecutive_known_skips(flags, patience, expected):
    records = [
        {
            "step": step,
            "metrics": (
                None
                if flag is None
                else {"loss": 0.0} if flag == "legacy" else {"train/updated": flag}
            ),
        }
        for step, flag in enumerate(flags, 1)
    ]
    assert zero_update_stop_step(records, patience) == expected


def test_missing_batch_resets_zero_update_streak():
    records = [{"step": step, "metrics": {"train/updated": 0}} for step in (1, 3)]
    assert zero_update_stop_step(records, 2) is None


@pytest.mark.parametrize("patience", [-1, True, 1.5, "2", None])
def test_zero_update_patience_rejects_invalid_configuration(patience):
    with pytest.raises(ValueError, match="zero_update_patience"):
        zero_update_stop_step([], patience)


@pytest.mark.parametrize(
    "metrics",
    [
        {"train/updated": -1},
        {"train/updated": 0.5},
        {"train/updated": 2},
        {"train/datums": -1},
        {"train/datums": 1.5},
        {"train/updated": 0, "train/datums": 2},
        {"train/updated": 1, "train/datums": 0},
    ],
)
def test_invalid_update_activity_cannot_be_logged_or_authorize_completion(
    tmp_path, metrics
):
    progress = RiverTrainingProgress(tmp_path, steps=1, run_manifest_hash="run")
    with pytest.raises(ValueError, match="Training metrics"):
        progress.observe(1, metrics)
    assert progress.records == [] and not progress.path.exists()
    with pytest.raises(ValueError, match="Training metrics"):
        require_training_progress([{"step": 1, "metrics": metrics}], steps=1)
    with pytest.raises(ValueError, match="Training metrics"):
        summarize_training_activity([{"step": 1, "metrics": metrics}])


def test_activity_summary_never_counts_missing_metrics_as_updates_or_skips(tmp_path):
    progress = RiverTrainingProgress(tmp_path, steps=4, run_manifest_hash="run")
    progress.observe(1, {"train/updated": 1.0, "train/datums": 2.0})
    progress.observe(2, {"train/updated": 0.0, "train/datums": 0.0})
    progress.observe(3, {"loss": 0.5})
    progress.reconcile(4)
    require_complete(progress)
    activity = summarize_training_activity(progress.records)
    assert activity["status"] == "updates_observed"
    assert activity["completed_batches"] == 4
    assert activity["observed_optimizer_updates"] == 1
    assert activity["observed_skipped_batches"] == 1
    assert activity["unknown_update_batches"] == 2
    assert summarize_training_activity(ledger(tmp_path, steps=4).records) == activity


@pytest.mark.parametrize(
    "metrics,status",
    [
        (None, "unknown"),
        ({"loss": 0.0}, "unknown"),
        ({"train/updated": 0, "train/datums": 0}, "no_updates_observed"),
        ({"train/updated": 1, "train/datums": 1}, "updates_observed"),
    ],
)
def test_activity_summary_distinguishes_unknown_from_observed_zero(metrics, status):
    summary = summarize_training_activity([{"step": 1, "metrics": metrics}])
    assert summary["status"] == status
    assert (
        sum(
            summary[key]
            for key in (
                "observed_optimizer_updates",
                "observed_skipped_batches",
                "unknown_update_batches",
            )
        )
        == 1
    )
    assert summarize_training_activity([])["status"] == "unknown"


def ledger(path, steps=3):
    return RiverTrainingProgress(path, steps=steps, run_manifest_hash="run")


def require_complete(progress):
    require_training_progress(
        progress.records,
        steps=progress.steps,
        recovery_receipts=progress.receipts,
        run_manifest_hash=progress.run_manifest_hash,
    )


def test_recovery_fills_commit_gap_without_fabricating_metrics(tmp_path):
    progress = ledger(tmp_path)
    progress.observe(1, {"loss": 0.25})
    progress.reconcile(2)
    assert progress.records[0] == {"step": 1, "metrics": {"loss": 0.25}}
    assert progress.records[1]["metrics"] is None
    assert progress.records[1]["source"] == "river_recovery"
    with pytest.raises(ValueError, match="Incomplete"):
        require_complete(progress)
    progress.observe(3, {"loss": 0.1})
    require_complete(ledger(tmp_path))


def test_recovery_removes_uncommitted_observation_suffix(tmp_path):
    progress = ledger(tmp_path)
    for step in range(1, 4):
        progress.observe(step, {"loss": step})
    progress.reconcile(1)
    assert progress.records == [{"step": 1, "metrics": {"loss": 1}}]
    progress.observe(2, {"loss": 0.5})
    progress.observe(3, {"loss": 0.25})
    require_complete(progress)
    assert progress.records[-1]["metrics"]["loss"] == 0.25


def test_reconciliation_is_idempotent_across_partial_local_write(tmp_path, monkeypatch):
    import stateset_agents.training.river_progress as module

    progress = ledger(tmp_path)
    original = module.atomic_json

    def write(path, value):
        if path.name == "training_metrics.json":
            raise OSError("disk full")
        original(path, value)

    monkeypatch.setattr(module, "atomic_json", write)
    with pytest.raises(OSError):
        progress.reconcile(3)
    assert not progress.path.exists()
    assert len(json.loads(progress.receipts_path.read_text())) == 1
    monkeypatch.setattr(module, "atomic_json", original)
    resumed = ledger(tmp_path)
    resumed.reconcile(3)
    resumed.reconcile(3)
    assert len(resumed.receipts) == 1
    assert all(entry["metrics"] is None for entry in resumed.records)
    require_complete(resumed)


@pytest.mark.parametrize("count", [-1, True, 4, 1.5, "2"])
def test_invalid_recovered_count_never_writes_progress(tmp_path, count):
    progress = ledger(tmp_path)
    with pytest.raises(ValueError):
        progress.reconcile(count)
    assert not progress.path.exists() and not progress.receipts_path.exists()


@pytest.mark.parametrize(
    "mutation", ["missing", "different_run", "uncovered", "fabricated"]
)
def test_recovered_progress_requires_matching_receipt(tmp_path, mutation):
    progress = ledger(tmp_path)
    progress.reconcile(3)
    if mutation == "missing":
        progress.receipts.clear()
    elif mutation == "different_run":
        progress.run_manifest_hash = "other"
    elif mutation == "uncovered":
        from stateset_agents.evaluation.agent_runs import content_hash

        receipt = next(iter(progress.receipts.values()))
        receipt["completed_batches"] = 1
        key = content_hash(receipt)
        progress.receipts = {key: receipt}
        for entry in progress.records:
            entry["recovery_receipt_hash"] = key
    else:
        progress.records[0]["metrics"] = {"loss": 0}
    with pytest.raises(ValueError):
        require_complete(progress)


def test_progress_without_metrics_cannot_self_certify_recovery():
    with pytest.raises(ValueError, match="receipt"):
        require_training_progress([{"step": 1, "metrics": None}], steps=1)


def test_corrupt_or_foreign_receipts_fail_on_open(tmp_path):
    progress = ledger(tmp_path)
    progress.reconcile(1)
    with pytest.raises(ValueError, match="receipt"):
        RiverTrainingProgress(tmp_path, steps=3, run_manifest_hash="other")
    progress.receipts_path.write_text("[]")
    with pytest.raises(ValueError, match="receipts"):
        ledger(tmp_path)


INVALID_METRICS = [
    {},
    {"loss": float("nan")},
    {"loss": float("inf")},
    {"loss": -float("inf")},
    {"loss": True},
    {"loss": "0.5"},
    {"loss": None},
    {"loss": [0.5]},
    {"loss": {"value": 0.5}},
    {"loss": 10**400},
    {"": 0.5},
    {" ": 0.5},
    {1: 0.5},
]


@pytest.mark.parametrize("metrics", INVALID_METRICS)
def test_invalid_observation_preserves_prior_durable_progress(tmp_path, metrics):
    progress = ledger(tmp_path)
    progress.observe(1, {"loss": 0.5})
    before = progress.path.read_bytes()
    with pytest.raises(ValueError, match="Training metrics"):
        progress.observe(2, metrics)
    assert progress.path.read_bytes() == before
    assert progress.records == [{"step": 1, "metrics": {"loss": 0.5}}]


@pytest.mark.parametrize("metrics", INVALID_METRICS)
def test_invalid_observation_cannot_pass_completion_audit(metrics):
    with pytest.raises(ValueError, match="Training metrics"):
        require_training_progress([{"step": 1, "metrics": metrics}], steps=1)


@pytest.mark.parametrize("metrics", INVALID_METRICS[:-1])
def test_invalid_persisted_metrics_fail_on_open(tmp_path, metrics):
    # Deliberately use the permissive stdlib writer to simulate external or
    # legacy evidence. Non-string JSON keys are normalized by that writer.
    (tmp_path / "training_metrics.json").write_text(
        json.dumps([{"step": 1, "metrics": metrics}])
    )
    with pytest.raises(ValueError, match="Training metrics"):
        ledger(tmp_path)


def test_sdk_dictionary_reuse_cannot_rewrite_prior_observations(tmp_path):
    progress = ledger(tmp_path)
    metrics = {"loss": 0.5, "sampling/generated_tokens": 2**53 + 1}
    progress.observe(1, metrics)
    metrics["loss"] = 0.25
    progress.observe(2, metrics)
    metrics.clear()
    progress.reconcile(2)
    resumed = ledger(tmp_path)
    assert resumed.records == [
        {"step": 1, "metrics": {"loss": 0.5, "sampling/generated_tokens": 2**53 + 1}},
        {"step": 2, "metrics": {"loss": 0.25, "sampling/generated_tokens": 2**53 + 1}},
    ]
    resumed.observe(3, {"loss": -0.1, "reward_mean": 0.0})
    require_complete(resumed)


def test_failed_metric_write_does_not_advance_in_memory_progress(tmp_path, monkeypatch):
    import stateset_agents.training.river_progress as module

    progress = ledger(tmp_path)
    progress.observe(1, {"loss": 0.5})
    before = progress.path.read_bytes()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(module, "atomic_json", fail)
    with pytest.raises(OSError, match="disk full"):
        progress.observe(2, {"loss": 0.25})
    assert progress.path.read_bytes() == before
    assert progress.records == [{"step": 1, "metrics": {"loss": 0.5}}]
