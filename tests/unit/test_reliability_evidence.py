"""Tests for measured checkpoint/fault-recovery evidence gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[2] / "benchmarks" / "reliability_evidence.py"
SPEC = importlib.util.spec_from_file_location("reliability_evidence", MODULE)
assert SPEC is not None and SPEC.loader is not None
reliability = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reliability
SPEC.loader.exec_module(reliability)

EvidenceError = reliability.ReliabilityEvidenceError


def _run(fault: str, seed: int, **overrides: Any) -> dict[str, Any]:
    config = {
        "batch_size": 32,
        "checkpoint_interval_steps": 10,
        "final_step": 100,
    }
    run: dict[str, Any] = {
        "schema_version": 1,
        "measured": True,
        "run_id": f"{fault}-{seed}",
        "framework_version": "0.42.3",
        "harness_commit": "a" * 40,
        "protocol": "fault-recovery-v1",
        "model": "Qwen/Qwen3.5-0.8B",
        "model_revision": "b" * 40,
        "seed": seed,
        "timestamp": "2026-08-26T21:00:00Z",
        "command": f"inject-fault --type {fault} --seed {seed}",
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "hardware": {"accelerator": "NVIDIA H100", "cuda": "12.8"},
        "software": {"python": "3.12.12", "torch": "2.8.0"},
        "fault": {"type": fault, "injected_at_step": 51, "target": "worker-0"},
        "recovery": {
            "resumed": True,
            "completed": True,
            "checkpoint_step": 50,
            "resumed_step": 50,
            "duplicate_updates": 0,
            "data_loss_steps": 1,
            "recovery_seconds": 12.5,
            "final_step": 100,
            "expected_final_step": 100,
            "resources_remaining": 0,
        },
        "artifact_sha256": "c" * 64,
    }
    run.update(overrides)
    return run


def _matrix() -> list[dict[str, Any]]:
    return [
        _run(fault, seed)
        for fault in reliability.REQUIRED_FAULTS
        for seed in (42, 1337, 2026)
    ]


def test_complete_fault_matrix_passes() -> None:
    runs = _matrix()
    reliability.validate_matrix(runs)
    report = reliability.summarize(runs)
    assert report["passed"] is True
    assert report["faults"]["worker_exit"]["max_data_loss_steps"] == 1


def test_rejects_synthetic_or_incomplete_run(tmp_path: Path) -> None:
    run = _run("worker_exit", 42, measured=False)
    with pytest.raises(EvidenceError, match="measured=true"):
        reliability.validate_run(run, tmp_path / "run.json")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("duplicate_updates", 1, "duplicate optimizer"),
        ("final_step", 99, "expected final step"),
        ("resources_remaining", 1, "resources remain"),
        ("resumed_step", 49, "must equal checkpoint_step"),
    ],
)
def test_rejects_corrupt_recovery_invariants(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    run = _run("worker_exit", 42)
    run["recovery"][field] = value
    with pytest.raises(EvidenceError, match=message):
        reliability.validate_run(run, tmp_path / "run.json")


def test_rejects_missing_fault_or_seed(tmp_path: Path) -> None:
    runs = _matrix()
    without_network = [
        run for run in runs if run["fault"]["type"] != "network_interruption"
    ]
    with pytest.raises(EvidenceError, match="fault matrix mismatch"):
        reliability.validate_matrix(without_network)

    missing_seed = [
        run
        for run in runs
        if not (run["fault"]["type"] == "worker_exit" and run["seed"] == 2026)
    ]
    with pytest.raises(EvidenceError, match="only 2 seeds"):
        reliability.validate_matrix(missing_seed)


def test_rejects_excessive_checkpoint_gap() -> None:
    runs = _matrix()
    for run in runs:
        if run["fault"]["type"] == "controller_restart":
            run["fault"]["injected_at_step"] = 70
            run["recovery"]["data_loss_steps"] = 20
    with pytest.raises(EvidenceError, match="exceeds 10 steps"):
        reliability.validate_matrix(runs)


def test_rejects_config_digest_mismatch(tmp_path: Path) -> None:
    run = _run("worker_exit", 42)
    run["config"]["batch_size"] = 64
    with pytest.raises(EvidenceError, match="does not match canonical config"):
        reliability.validate_run(run, tmp_path / "run.json")


def test_rejects_mixed_hardware() -> None:
    runs = _matrix()
    runs[-1]["hardware"] = {"accelerator": "NVIDIA A100", "cuda": "12.8"}
    with pytest.raises(EvidenceError, match="hardware.accelerator"):
        reliability.validate_matrix(runs)
