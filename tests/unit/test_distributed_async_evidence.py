"""Tests for the multi-node asynchronous RL publication gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.distributed_async_evidence import (
    REQUIRED_SCENARIOS,
    DistributedAsyncEvidenceError,
    summarize,
    validate_matrix,
    validate_run,
)


def _run(scenario: str, seed: int, *, duration: float = 43_200.0) -> dict[str, Any]:
    config = {"max_policy_lag": 1, "queue_capacity": 1024}
    accepted = 43_200
    return {
        "schema_version": 1,
        "kind": "stateset-distributed-async-evidence",
        "status": "completed",
        "measured": True,
        "run_id": f"{scenario}-{seed}",
        "framework_version": "0.47.0",
        "harness_commit": "a" * 40,
        "protocol": "stateset-distributed-async-v1",
        "timestamp": "2026-08-31T00:00:00Z",
        "provider": "runpod",
        "seed": seed,
        "scenario": scenario,
        "duration_seconds": duration,
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "topology": {
            "node_count": 2,
            "worker_count": 4,
            "node_ids": ["machine-a", "machine-b"],
            "accelerator": "NVIDIA H100 80GB HBM3",
            "accelerator_driver": "CUDA 12.9",
            "interconnect": "100 GbE",
        },
        "metrics": {
            "rollouts_attempted": accepted + 10,
            "rollouts_accepted": accepted,
            "optimizer_updates": 100,
            "policy_versions": 101,
            "max_observed_policy_lag": 1,
            "lost_optimizer_updates": 0,
            "duplicate_optimizer_updates": 0,
            "artifact_digest_mismatches": 0,
            "accepted_rollouts_per_second": accepted / duration,
            "weight_sync_latency_ms": {"p50": 20.0, "p95": 40.0, "max": 80.0},
        },
        "cost": {
            "currency": "USD",
            "source": "provider_invoice",
            "total": 12.96,
            "per_accepted_rollout": 12.96 / accepted,
        },
        "recovery": {
            "recovered": True,
            "recovery_seconds": 2.5,
            "resources_remaining": 0,
        },
        "artifact_sha256": "b" * 64,
    }


def _matrix() -> list[dict[str, Any]]:
    return [
        _run(scenario, seed)
        for scenario in REQUIRED_SCENARIOS
        for seed in (42, 1337, 2026)
    ]


def test_complete_matrix_passes_and_preserves_cost_and_tail_latency() -> None:
    runs = [
        validate_run(run, Path(f"run-{index}.json"))
        for index, run in enumerate(_matrix())
    ]
    validate_matrix(runs)
    report = summarize(runs)
    assert report["passed"] is True
    assert report["run_count"] == 12
    assert report["node_count"] == 2
    assert report["max_weight_sync_p95_ms"] == 40.0
    assert report["total_cost_usd"] == pytest.approx(155.52)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lost_optimizer_updates", 1),
        ("duplicate_optimizer_updates", 1),
        ("artifact_digest_mismatches", 1),
        ("max_observed_policy_lag", 2),
    ],
)
def test_run_rejects_safety_invariant_violations(field: str, value: int) -> None:
    run = _run("worker_exit", 42)
    run["metrics"][field] = value
    with pytest.raises(DistributedAsyncEvidenceError):
        validate_run(run, Path("bad.json"))


def test_run_rejects_single_node_or_unaccounted_throughput_and_cost() -> None:
    run = _run("worker_exit", 42)
    run["topology"]["node_count"] = 1
    run["topology"]["node_ids"] = ["machine-a"]
    with pytest.raises(DistributedAsyncEvidenceError):
        validate_run(run, Path("single-node.json"))

    run = _run("worker_exit", 42)
    run["metrics"]["accepted_rollouts_per_second"] = 99.0
    with pytest.raises(DistributedAsyncEvidenceError):
        validate_run(run, Path("throughput.json"))

    run = _run("worker_exit", 42)
    run["cost"]["per_accepted_rollout"] = 99.0
    with pytest.raises(DistributedAsyncEvidenceError):
        validate_run(run, Path("cost.json"))


def test_matrix_rejects_missing_scenario_seed_drift_and_short_soak() -> None:
    runs = _matrix()
    with pytest.raises(DistributedAsyncEvidenceError, match="scenario matrix"):
        validate_matrix([run for run in runs if run["scenario"] != "worker_exit"])

    drifted = _matrix()
    drifted[-1]["seed"] = 7
    with pytest.raises(DistributedAsyncEvidenceError, match="seed set mismatch"):
        validate_matrix(drifted)

    short = _matrix()
    short[0]["duration_seconds"] = 60.0
    short[0]["metrics"]["accepted_rollouts_per_second"] = 720.0
    with pytest.raises(DistributedAsyncEvidenceError, match="duration"):
        validate_matrix(short)


def test_run_rejects_synthetic_or_digest_drift() -> None:
    run = _run("steady_state_soak", 42)
    run["measured"] = False
    with pytest.raises(DistributedAsyncEvidenceError, match="measured"):
        validate_run(run, Path("synthetic.json"))

    run = _run("steady_state_soak", 42)
    run["config"]["queue_capacity"] = 2048
    with pytest.raises(DistributedAsyncEvidenceError, match="config_sha256"):
        validate_run(run, Path("drift.json"))
