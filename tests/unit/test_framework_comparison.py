"""Tests for provenance-enforced cross-framework benchmark reports."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "framework_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("framework_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
framework_comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = framework_comparison
SPEC.loader.exec_module(framework_comparison)

EvidenceError = framework_comparison.EvidenceError


def _document(framework: str, seed: int, **overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": 1,
        "measured": True,
        "framework": framework,
        "framework_version": "1.2.3",
        "harness_commit": "a" * 40,
        "protocol": "agent-rl-shootout-v1",
        "cache_policy": "prewarmed-v1",
        "algorithm": "gspo",
        "algorithm_revision": "stateset-gspo-v1",
        "model": "Qwen/Qwen3.5-8B-Instruct",
        "model_revision": "b" * 40,
        "task": "customer-support-multiturn-v1",
        "dataset_revision": "c" * 40,
        "seed": seed,
        "timestamp": "2026-08-26T21:00:00Z",
        "command": f"train --framework {framework} --seed {seed}",
        "config": {"learning_rate": 5e-6, "num_generations": 4},
        "hardware": {"gpu": "NVIDIA H100 80GB HBM3", "gpu_count": 1, "cuda": "12.8"},
        "metrics": {
            "samples_per_second": 1.0 + seed / 1000,
            "wall_clock_seconds": 3600.0,
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
    for framework in ("stateset-agents", "trl"):
        for seed in (42, 1337, 2026):
            path = tmp_path / f"{framework}-{seed}.json"
            path.write_text(json.dumps(_document(framework, seed)), encoding="utf-8")
            paths.append(path)
    return framework_comparison.load_evidence(paths)


def test_valid_matched_three_seed_comparison(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    framework_comparison.validate_comparison(runs)
    summary = framework_comparison.summarize(runs)

    assert set(summary["frameworks"]) == {"stateset-agents", "trl"}
    assert summary["frameworks"]["stateset-agents"]["improvement"][
        "mean"
    ] == pytest.approx(0.1)
    assert summary["frameworks"]["stateset-agents"]["evidence"] == [
        "stateset-agents-42.json",
        "stateset-agents-1337.json",
        "stateset-agents-2026.json",
    ]
    assert all(
        not Path(source).is_absolute()
        for values in summary["frameworks"].values()
        for source in values["evidence"]
    )


def test_rejects_simulated_evidence(tmp_path: Path) -> None:
    path = tmp_path / "fake.json"
    path.write_text(json.dumps(_document("stateset-agents", 42, measured=False)))

    with pytest.raises(EvidenceError, match="simulated or estimated"):
        framework_comparison.load_evidence([path])


def test_rejects_non_finite_metrics(tmp_path: Path) -> None:
    data = _document("stateset-agents", 42)
    data["metrics"]["samples_per_second"] = float("nan")

    with pytest.raises(EvidenceError, match="must be finite"):
        framework_comparison.validate_document(data, tmp_path / "run.json")


def test_rejects_zero_throughput(tmp_path: Path) -> None:
    data = _document("stateset-agents", 42)
    data["metrics"]["samples_per_second"] = 0.0

    with pytest.raises(EvidenceError, match="greater than zero"):
        framework_comparison.validate_document(data, tmp_path / "run.json")


def test_rejects_mismatched_hardware(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    mismatched = dict(runs[-1].data)
    mismatched["hardware"] = {"gpu": "NVIDIA A100", "gpu_count": 1, "cuda": "12.8"}
    runs[-1] = framework_comparison.RunEvidence(runs[-1].source, mismatched)

    with pytest.raises(EvidenceError, match="hardware.gpu"):
        framework_comparison.validate_comparison(runs)


def test_rejects_mismatched_cuda(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    mismatched = dict(runs[-1].data)
    mismatched["hardware"] = dict(mismatched["hardware"], cuda="12.9")
    runs[-1] = framework_comparison.RunEvidence(runs[-1].source, mismatched)

    with pytest.raises(EvidenceError, match="hardware.cuda"):
        framework_comparison.validate_comparison(runs)


def test_rejects_mismatched_config(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    mismatched = dict(runs[-1].data)
    mismatched["config"] = dict(mismatched["config"], learning_rate=9e-5)
    runs[-1] = framework_comparison.RunEvidence(runs[-1].source, mismatched)

    with pytest.raises(EvidenceError, match="config"):
        framework_comparison.validate_comparison(runs)


def test_requires_identical_seed_sets(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    changed = dict(runs[-1].data)
    changed["seed"] = 7
    runs[-1] = framework_comparison.RunEvidence(runs[-1].source, changed)

    with pytest.raises(EvidenceError, match="seed set"):
        framework_comparison.validate_comparison(runs)


def test_required_framework_roster_fails_closed(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(EvidenceError, match="nemo-rl, openrlhf, verl"):
        framework_comparison.validate_comparison(
            runs,
            required_frameworks=(
                "stateset-agents",
                "trl",
                "verl",
                "nemo-rl",
                "openrlhf",
            ),
        )


def test_retained_pair_does_not_satisfy_full_competitive_roster() -> None:
    root = MODULE_PATH.parents[1]
    runs = framework_comparison.load_evidence(
        [root / "benchmark_results" / "framework_comparison" / "evidence"]
    )
    with pytest.raises(EvidenceError, match="nemo-rl, openrlhf, verl"):
        framework_comparison.validate_comparison(
            runs,
            required_frameworks=(
                "stateset-agents",
                "trl",
                "verl",
                "nemo-rl",
                "openrlhf",
            ),
        )


def test_rejects_duplicate_or_insufficient_seeds(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    duplicate = framework_comparison.RunEvidence(runs[0].source, dict(runs[0].data))
    with pytest.raises(EvidenceError, match="duplicate seed"):
        framework_comparison.validate_comparison([*runs, duplicate])

    with pytest.raises(EvidenceError, match="at least 3 required"):
        framework_comparison.validate_comparison([runs[0], runs[3]])


def test_report_contains_digest_and_no_subjective_winner(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    output = tmp_path / "report"
    framework_comparison.write_report(runs, output)

    payload = json.loads((output / "comparison.json").read_text())
    markdown = (output / "comparison.md").read_text()
    assert len(payload["evidence_sha256"]) == 64
    assert "Descriptive results only" in markdown
    assert "Winner" not in markdown


def test_cli_fails_closed_on_bad_input(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    assert framework_comparison.main([str(path), "--validate-only"]) == 2
