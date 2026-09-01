"""Behavioral tests for the standard agent-quality evidence gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.agent_quality_evidence import (
    REQUIRED_SUITES,
    AgentQualityEvidenceError,
    summarize,
    validate_matrix,
    validate_run,
)


def _run(suite: str, seed: int, improvement: float = 0.08) -> dict[str, Any]:
    config = {"temperature": 0.0, "max_turns": 20}
    return {
        "schema_version": 1,
        "kind": "stateset-agent-quality-evidence",
        "status": "completed",
        "measured": True,
        "run_id": f"{suite}-{seed}",
        "suite": suite,
        "suite_revision": "a" * 40,
        "framework_version": "0.47.0",
        "model": "example/model-8b",
        "model_revision": "b" * 40,
        "harness_commit": "c" * 40,
        "split": "test",
        "timestamp": "2026-08-31T00:00:00Z",
        "seed": seed,
        "evaluation_config": config,
        "evaluation_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "tasks": 100,
        "successful_episodes": 58,
        "baseline_score": 0.50,
        "trained_score": 0.50 + improvement,
        "evaluation_seconds": 120.0,
        "evaluation_cost_usd": 5.80,
        "cost_per_successful_episode_usd": 0.10,
        "artifact_sha256": "d" * 64,
    }


def _matrix(
    improvements: tuple[float, float, float] = (0.08, 0.08, 0.08),
) -> list[dict[str, Any]]:
    return [
        _run(suite, seed, improvements[index])
        for suite in REQUIRED_SUITES
        for index, seed in enumerate((42, 1337, 2026))
    ]


def test_matched_standard_suite_matrix_passes() -> None:
    runs = [validate_run(run, Path("run.json")) for run in _matrix()]
    validate_matrix(runs)
    report = summarize(runs)
    assert report["passed"] is True
    assert set(report["suites"]) == set(REQUIRED_SUITES)
    assert report["suites"]["tau-bench"]["mean_improvement"] == pytest.approx(0.08)


def test_run_rejects_synthetic_bad_digest_and_bad_cost() -> None:
    run = _run("tau-bench", 42)
    run["measured"] = False
    with pytest.raises(AgentQualityEvidenceError, match="measured"):
        validate_run(run, Path("synthetic.json"))

    run = _run("tau-bench", 42)
    run["evaluation_config"]["max_turns"] = 99
    with pytest.raises(AgentQualityEvidenceError, match="config"):
        validate_run(run, Path("digest.json"))

    run = _run("tau-bench", 42)
    run["cost_per_successful_episode_usd"] = 1.0
    with pytest.raises(AgentQualityEvidenceError, match="cost"):
        validate_run(run, Path("cost.json"))


def test_matrix_rejects_missing_suite_seed_drift_and_weak_effect() -> None:
    runs = _matrix()
    with pytest.raises(AgentQualityEvidenceError, match="suite matrix"):
        validate_matrix([run for run in runs if run["suite"] != "bfcl"])

    drifted = _matrix()
    drifted[-1]["seed"] = 7
    with pytest.raises(AgentQualityEvidenceError, match="seed set mismatch"):
        validate_matrix(drifted)

    with pytest.raises(AgentQualityEvidenceError, match="below the floor"):
        validate_matrix(_matrix((0.01, 0.01, 0.01)))

    with pytest.raises(AgentQualityEvidenceError, match="confidence bound"):
        validate_matrix(_matrix((0.20, 0.03, 0.03)))
