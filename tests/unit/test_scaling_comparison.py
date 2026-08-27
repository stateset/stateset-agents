"""Tests for measured distributed-scaling evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "scaling_comparison", BENCHMARKS / "scaling_comparison.py"
)
assert SPEC is not None and SPEC.loader is not None
scaling_comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scaling_comparison
SPEC.loader.exec_module(scaling_comparison)

EvidenceError = scaling_comparison.EvidenceError


def _document(gpu_count: int, seed: int, **overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": 1,
        "measured": True,
        "framework": "stateset-agents",
        "framework_version": "0.42.3",
        "harness_commit": "a" * 40,
        "protocol": "distributed-scaling-v1",
        "cache_policy": "prewarmed-v1",
        "algorithm": "gspo",
        "algorithm_revision": "stateset-gspo-v1",
        "model": "Qwen/Qwen3.5-8B-Instruct",
        "model_revision": "b" * 40,
        "task": "customer-support-multiturn-v1",
        "dataset_revision": "c" * 40,
        "workload_config_sha256": "e" * 64,
        "seed": seed,
        "timestamp": "2026-08-26T21:00:00Z",
        "command": f"torchrun --nproc-per-node {gpu_count} train.py --seed {seed}",
        "config": {"global_batch_size": 32, "steps": 100},
        "hardware": {"gpu": "NVIDIA H100", "gpu_count": gpu_count, "cuda": "12.8"},
        "metrics": {
            "samples_per_second": gpu_count * 10.0,
            "wall_clock_seconds": 1000.0 / gpu_count,
            "peak_vram_mb": 70000.0,
            "eval_score_baseline": 0.5,
            "eval_score_final": 0.6,
        },
        "artifact_sha256": "d" * 64,
    }
    document.update(overrides)
    return document


def _runs(tmp_path: Path) -> list[Any]:
    paths = []
    for gpu_count in (1, 2, 4, 8):
        for seed in (42, 1337, 2026):
            path = tmp_path / f"gpu{gpu_count}-{seed}.json"
            path.write_text(json.dumps(_document(gpu_count, seed)), encoding="utf-8")
            paths.append(path)
    return scaling_comparison.load_scaling_evidence(paths)


def test_complete_matrix_reports_efficiency(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    scaling_comparison.validate_scaling_comparison(runs)
    summary = scaling_comparison.summarize_scaling(runs)
    assert summary["topologies"]["8"]["speedup_vs_1_gpu"] == pytest.approx(8.0)
    assert summary["topologies"]["8"]["scaling_efficiency"] == pytest.approx(1.0)
    scaling_comparison.validate_scaling_performance(summary)


def test_performance_gate_rejects_low_efficiency(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    for index, run in enumerate(runs):
        if run.data["hardware"]["gpu_count"] == 8:
            changed = dict(run.data)
            changed["metrics"] = dict(run.metrics, samples_per_second=20.0)
            runs[index] = scaling_comparison.RunEvidence(run.source, changed)
    summary = scaling_comparison.summarize_scaling(runs)

    with pytest.raises(EvidenceError, match="below 50.0%"):
        scaling_comparison.validate_scaling_performance(
            summary, require_monotonic=False
        )


def test_performance_gate_rejects_non_monotonic_throughput(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    for index, run in enumerate(runs):
        if run.data["hardware"]["gpu_count"] == 4:
            changed = dict(run.data)
            changed["metrics"] = dict(run.metrics, samples_per_second=15.0)
            runs[index] = scaling_comparison.RunEvidence(run.source, changed)
    summary = scaling_comparison.summarize_scaling(runs)

    with pytest.raises(EvidenceError, match="not monotonic"):
        scaling_comparison.validate_scaling_performance(summary)


def test_rejects_missing_topology(tmp_path: Path) -> None:
    runs = [run for run in _runs(tmp_path) if run.data["hardware"]["gpu_count"] != 4]
    with pytest.raises(EvidenceError, match=r"missing=\[4\]"):
        scaling_comparison.validate_scaling_comparison(runs)


def test_rejects_mismatched_seed_sets(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    runs = [
        run
        for run in runs
        if not (run.data["hardware"]["gpu_count"] == 8 and run.seed == 2026)
    ]
    with pytest.raises(EvidenceError, match="only 2 seeds"):
        scaling_comparison.validate_scaling_comparison(runs)


def test_rejects_changed_workload(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    changed = dict(runs[-1].data)
    changed["workload_config_sha256"] = "f" * 64
    runs[-1] = scaling_comparison.RunEvidence(runs[-1].source, changed)
    with pytest.raises(EvidenceError, match="workload_config_sha256"):
        scaling_comparison.validate_scaling_comparison(runs)


def test_rejects_invalid_workload_digest(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(_document(1, 42, workload_config_sha256="not-a-digest")),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="64 hex"):
        scaling_comparison.load_scaling_evidence([path])
