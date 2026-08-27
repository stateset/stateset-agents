"""Tests for measured-only algorithm comparisons."""

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
    "algorithm_comparison", BENCHMARKS / "algorithm_comparison.py"
)
assert SPEC is not None and SPEC.loader is not None
algorithm_comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = algorithm_comparison
SPEC.loader.exec_module(algorithm_comparison)

EvidenceError = algorithm_comparison.EvidenceError


def _document(algorithm: str, seed: int, **overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": 1,
        "measured": True,
        "framework": "stateset-agents",
        "framework_version": "0.42.3",
        "harness_commit": "a" * 40,
        "protocol": "agent-rl-algorithms-v1",
        "cache_policy": "prewarmed-v1",
        "algorithm": algorithm,
        "algorithm_revision": f"{algorithm}-objective-v1",
        "model": "Qwen/Qwen3.5-8B-Instruct",
        "model_revision": "b" * 40,
        "task": "customer-support-multiturn-v1",
        "dataset_revision": "c" * 40,
        "seed": seed,
        "timestamp": "2026-08-26T21:00:00Z",
        "command": f"train --algorithm {algorithm} --seed {seed}",
        "config": {"learning_rate": 5e-6},
        "hardware": {"gpu": "NVIDIA H100", "gpu_count": 1, "cuda": "12.8"},
        "metrics": {
            "samples_per_second": 1.0,
            "wall_clock_seconds": 100.0,
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
    for algorithm in ("grpo", "gspo"):
        for seed in (42, 1337, 2026):
            path = tmp_path / f"{algorithm}-{seed}.json"
            path.write_text(json.dumps(_document(algorithm, seed)), encoding="utf-8")
            paths.append(path)
    return algorithm_comparison.load_algorithm_evidence(paths)


def test_accepts_measured_matched_algorithms(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    algorithm_comparison.validate_algorithm_comparison(runs)
    adapted = algorithm_comparison.as_algorithm_runs(runs)
    assert {run.framework for run in adapted} == {"grpo", "gspo"}


def test_rejects_mixed_models(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    changed = dict(runs[-1].data)
    changed["model"] = "different/model"
    runs[-1] = algorithm_comparison.RunEvidence(runs[-1].source, changed)
    with pytest.raises(EvidenceError, match="model"):
        algorithm_comparison.validate_algorithm_comparison(runs)


def test_rejects_too_few_seeds(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(EvidenceError, match="at least 3 required"):
        algorithm_comparison.validate_algorithm_comparison([runs[0], runs[3]])


def test_report_has_no_synthetic_rankings(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    output = tmp_path / "report"
    algorithm_comparison.write_algorithm_report(runs, output)
    markdown = (output / "comparison.md").read_text(encoding="utf-8")
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert "Measured algorithm comparison" in markdown
    assert "| Algorithm |" in markdown
    assert "| Framework |" not in markdown
    assert "Winner" not in markdown
    assert set(report["algorithms"]) == {"grpo", "gspo"}
    assert "frameworks" not in report
    assert report["comparison"]["algorithms"] == ["grpo", "gspo"]
    assert "algorithm" not in report["comparison"]
    assert len(report["evidence_sha256"]) == 64


def test_requires_identical_seed_sets(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    changed = dict(runs[-1].data)
    changed["seed"] = 7
    runs[-1] = algorithm_comparison.RunEvidence(runs[-1].source, changed)
    with pytest.raises(EvidenceError, match="seed set"):
        algorithm_comparison.validate_algorithm_comparison(runs)


def test_rejects_mismatched_cuda(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    changed = dict(runs[-1].data)
    changed["hardware"] = dict(changed["hardware"], cuda="12.9")
    runs[-1] = algorithm_comparison.RunEvidence(runs[-1].source, changed)
    with pytest.raises(EvidenceError, match="hardware.cuda"):
        algorithm_comparison.validate_algorithm_comparison(runs)


def test_required_algorithm_roster_fails_closed(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(EvidenceError, match="dapo, gepo, vapo"):
        algorithm_comparison.validate_algorithm_comparison(
            runs,
            required_algorithms=("grpo", "gspo", "dapo", "vapo", "gepo"),
        )
