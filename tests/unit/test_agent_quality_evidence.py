"""Behavioral tests for the standard agent-quality evidence gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.agent_quality_evidence import (
    OPTIONAL_SUITES,
    REQUIRED_SUITES,
    SUPPORTED_SUITES,
    AgentQualityEvidenceError,
    hash_artifact,
    load_runs,
    summarize,
    validate_matrix,
    validate_run,
    validate_suite_roster,
)


def _run(suite: str, seed: int, improvement: float = 0.08) -> dict[str, Any]:
    config = {"temperature": 0.0, "max_turns": 20}
    trained_successes = round((0.50 + improvement) * 100)
    return {
        "schema_version": 3,
        "kind": "stateset-agent-quality-evidence",
        "status": "completed",
        "measured": True,
        "run_id": f"{suite}-{seed}",
        "protocol": "stateset-standard-agent-quality-v2",
        "suite": suite,
        "suite_revision": "a" * 40,
        "framework_version": "0.47.0",
        "baseline_model": "example/model-8b",
        "baseline_model_revision": "b" * 40,
        "trained_model": "example/model-8b-stateset",
        "trained_model_revision": "e" * 40,
        "training_artifact_sha256": "f" * 64,
        "harness_commit": "c" * 40,
        "split": "test",
        "timestamp": "2026-08-31T00:00:00Z",
        "seed": seed,
        "evaluation_config": config,
        "evaluation_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "paired_task_ids_sha256": "9" * 64,
        "tasks": 100,
        "baseline_successful_episodes": 50,
        "trained_successful_episodes": trained_successes,
        "baseline_score": 50 / 100,
        "trained_score": trained_successes / 100,
        "evaluation_seconds": 120.0,
        "evaluation_cost_usd": 5.80,
        "cost_source": "provider-api",
        "cost_per_successful_episode_usd": (
            5.80 / trained_successes if trained_successes else 0.0
        ),
        "artifact_sha256": "d" * 64,
        "artifact_path": "../runs/artifact",
    }


def _matrix(
    improvements: tuple[float, float, float] = (0.08, 0.08, 0.08),
    suites: tuple[str, ...] = REQUIRED_SUITES,
) -> list[dict[str, Any]]:
    return [
        _run(suite, seed, improvements[index])
        for suite in suites
        for index, seed in enumerate((42, 1337, 2026))
    ]


def test_suite_policy_constants() -> None:
    assert REQUIRED_SUITES == ("tau3-bench", "bfcl-v4")
    assert OPTIONAL_SUITES == ("swe-bench-verified",)
    assert SUPPORTED_SUITES == REQUIRED_SUITES + OPTIONAL_SUITES


def test_required_only_matrix_passes() -> None:
    runs = [validate_run(run, Path("run.json")) for run in _matrix()]
    validate_matrix(runs)
    validate_matrix(runs, expected_suites=REQUIRED_SUITES)
    report = summarize(runs)
    assert list(report["suites"]) == list(REQUIRED_SUITES)
    assert report["required_suites"] == list(REQUIRED_SUITES)


def test_matrix_missing_required_suite_fails() -> None:
    runs = _matrix(suites=("tau3-bench", "swe-bench-verified"))
    with pytest.raises(AgentQualityEvidenceError, match="missing required.*bfcl-v4"):
        validate_matrix(runs)


def test_optional_suite_in_roster_but_missing_from_evidence_fails() -> None:
    with pytest.raises(AgentQualityEvidenceError, match="exactly the executed roster"):
        validate_matrix(_matrix(), expected_suites=SUPPORTED_SUITES)


def test_failing_optional_suite_fails_whole_gate() -> None:
    runs = _matrix(suites=REQUIRED_SUITES) + _matrix(
        (0.0, 0.0, 0.0), suites=OPTIONAL_SUITES
    )
    with pytest.raises(
        AgentQualityEvidenceError, match="swe-bench-verified: mean improvement"
    ):
        validate_matrix(runs)
    with pytest.raises(AgentQualityEvidenceError, match="swe-bench-verified"):
        validate_matrix(runs, expected_suites=SUPPORTED_SUITES)


def test_passing_optional_suite_passes_and_is_reported() -> None:
    runs = [
        validate_run(run, Path("run.json")) for run in _matrix(suites=SUPPORTED_SUITES)
    ]
    validate_matrix(runs)
    validate_matrix(runs, expected_suites=SUPPORTED_SUITES)
    report = summarize(runs)
    assert list(report["suites"]) == list(SUPPORTED_SUITES)
    assert report["suites"]["swe-bench-verified"]["seeds"] == [42, 1337, 2026]


def test_optional_suite_must_share_trained_artifact() -> None:
    runs = _matrix(suites=SUPPORTED_SUITES)
    runs[-1]["training_artifact_sha256"] = "7" * 64
    with pytest.raises(AgentQualityEvidenceError, match="training_artifact"):
        validate_matrix(runs)


def test_unknown_suite_rejected() -> None:
    with pytest.raises(AgentQualityEvidenceError, match="unsupported suite"):
        validate_run(_run("webarena", 42), Path("unknown.json"))
    with pytest.raises(AgentQualityEvidenceError, match="unsupported suites"):
        validate_matrix(_matrix() + _matrix(suites=("webarena",)))
    with pytest.raises(AgentQualityEvidenceError, match="unsupported suites"):
        validate_suite_roster([*REQUIRED_SUITES, "webarena"])
    with pytest.raises(AgentQualityEvidenceError, match="duplicate suites"):
        validate_suite_roster([*REQUIRED_SUITES, "bfcl-v4"])
    assert validate_suite_roster(["swe-bench-verified", "bfcl-v4", "tau3-bench"]) == (
        "tau3-bench",
        "bfcl-v4",
        "swe-bench-verified",
    )


def test_matched_standard_suite_matrix_passes() -> None:
    runs = [validate_run(run, Path("run.json")) for run in _matrix()]
    validate_matrix(runs)
    report = summarize(runs)
    assert report["passed"] is True
    assert set(report["suites"]) == set(REQUIRED_SUITES)
    assert report["suites"]["tau3-bench"]["mean_improvement"] == pytest.approx(0.08)


def test_run_rejects_synthetic_bad_digest_and_bad_cost() -> None:
    run = _run("tau3-bench", 42)
    run["schema_version"] = 2
    with pytest.raises(AgentQualityEvidenceError, match="schema_version=3"):
        validate_run(run, Path("legacy.json"))

    run = _run("tau3-bench", 42)
    run["measured"] = False
    with pytest.raises(AgentQualityEvidenceError, match="measured"):
        validate_run(run, Path("synthetic.json"))

    run = _run("tau3-bench", 42)
    run["evaluation_config"]["max_turns"] = 99
    with pytest.raises(AgentQualityEvidenceError, match="config"):
        validate_run(run, Path("digest.json"))

    run = _run("tau3-bench", 42)
    run["cost_per_successful_episode_usd"] = 1.0
    with pytest.raises(AgentQualityEvidenceError, match="cost"):
        validate_run(run, Path("cost.json"))

    run = _run("tau3-bench", 42)
    run["training_artifact_sha256"] = "unbound"
    with pytest.raises(AgentQualityEvidenceError, match="training_artifact"):
        validate_run(run, Path("checkpoint.json"))

    run = _run("tau3-bench", 42)
    run["paired_task_ids_sha256"] = "different-tasks"
    with pytest.raises(AgentQualityEvidenceError, match="paired_task"):
        validate_run(run, Path("pairing.json"))


def _write_task_records(
    artifact: Path, baseline: list[bool], trained: list[bool]
) -> list[str]:
    """Write the paired harness's retained per-policy task records."""
    task_ids = [f"task-{index}" for index in range(len(baseline))]
    for policy, outcomes in (("baseline", baseline), ("trained", trained)):
        directory = artifact / policy
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tasks.jsonl").write_text(
            "".join(
                json.dumps({"task_id": task_id, "success": success, "cost_usd": 0.5})
                + "\n"
                for task_id, success in zip(task_ids, outcomes, strict=True)
            ),
            encoding="utf-8",
        )
    return task_ids


def _bind_records(run: dict[str, Any], artifact: Path, task_ids: list[str]) -> None:
    run["tasks"] = len(task_ids)
    run["baseline_successful_episodes"] = 1
    run["trained_successful_episodes"] = 3
    run["baseline_score"] = 1 / 4
    run["trained_score"] = 3 / 4
    run["cost_per_successful_episode_usd"] = run["evaluation_cost_usd"] / 3
    run["paired_task_ids_sha256"] = hashlib.sha256(
        json.dumps(task_ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run["artifact_sha256"] = hash_artifact(artifact)


def test_validate_run_rejects_score_inconsistent_with_counts() -> None:
    run = _run("tau3-bench", 42)
    validate_run(run, Path("consistent.json"))
    run["trained_score"] = 0.90
    with pytest.raises(AgentQualityEvidenceError, match="trained_score"):
        validate_run(run, Path("tampered.json"))
    run = _run("tau3-bench", 42)
    run["baseline_score"] = 0.49
    with pytest.raises(AgentQualityEvidenceError, match="baseline_score"):
        validate_run(run, Path("tampered.json"))


def test_load_runs_recomputes_scores_from_retained_task_records(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    artifact = tmp_path / "runs" / "artifact"
    evidence_dir.mkdir()
    task_ids = _write_task_records(
        artifact, [True, False, False, False], [True, True, True, False]
    )
    run = _run("tau3-bench", 42)
    _bind_records(run, artifact, task_ids)
    source = evidence_dir / "run.json"
    source.write_text(json.dumps(run), encoding="utf-8")
    loaded = load_runs([source])[0]
    assert (loaded["baseline_score"], loaded["trained_score"]) == (0.25, 0.75)

    # Internally consistent counts and score, but not what the records say.
    run["trained_successful_episodes"] = 4
    run["trained_score"] = 1.0
    run["cost_per_successful_episode_usd"] = run["evaluation_cost_usd"] / 4
    source.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(AgentQualityEvidenceError, match="per-task records"):
        load_runs([source])


def test_load_runs_requires_retained_task_records(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    artifact = tmp_path / "runs" / "artifact"
    evidence_dir.mkdir()
    artifact.mkdir(parents=True)
    (artifact / "paired-summary.json").write_text("{}", encoding="utf-8")
    run = _run("tau3-bench", 42)
    run["artifact_sha256"] = hash_artifact(artifact)
    source = evidence_dir / "run.json"
    source.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(AgentQualityEvidenceError, match="tasks.jsonl"):
        load_runs([source])


def test_load_runs_verifies_retained_artifact_bytes(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    artifact = tmp_path / "runs" / "artifact"
    evidence_dir.mkdir()
    task_ids = _write_task_records(
        artifact, [True, False, False, False], [True, True, True, False]
    )
    run = _run("tau3-bench", 42)
    _bind_records(run, artifact, task_ids)
    source = evidence_dir / "run.json"
    source.write_text(json.dumps(run), encoding="utf-8")

    assert load_runs([source])[0]["run_id"] == "tau3-bench-42"

    (artifact / "extra.jsonl").write_text('{"task_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(AgentQualityEvidenceError, match="digest does not match"):
        load_runs([source])


def test_retained_artifact_rejects_escape_and_symlink(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "bundle" / "evidence"
    artifact = tmp_path / "bundle" / "runs" / "artifact"
    evidence_dir.mkdir(parents=True)
    artifact.mkdir(parents=True)
    payload = artifact / "tasks.jsonl"
    payload.write_text("{}\n", encoding="utf-8")
    run = _run("tau3-bench", 42)
    run["artifact_sha256"] = hash_artifact(artifact)
    source = evidence_dir / "run.json"

    run["artifact_path"] = "../../../outside"
    source.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(AgentQualityEvidenceError, match="escapes"):
        load_runs([source])

    run["artifact_path"] = "../runs/artifact"
    (artifact / "alias").symlink_to(payload)
    source.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(AgentQualityEvidenceError, match="symlink"):
        load_runs([source])


def test_matrix_rejects_missing_suite_seed_drift_and_weak_effect() -> None:
    runs = _matrix()
    with pytest.raises(AgentQualityEvidenceError, match="suite matrix"):
        validate_matrix([run for run in runs if run["suite"] != "bfcl-v4"])

    drifted = _matrix()
    drifted[-1]["seed"] = 7
    with pytest.raises(AgentQualityEvidenceError, match="seed set mismatch"):
        validate_matrix(drifted)

    with pytest.raises(AgentQualityEvidenceError, match="below the floor"):
        validate_matrix(_matrix((0.01, 0.01, 0.01)))

    with pytest.raises(AgentQualityEvidenceError, match="confidence bound"):
        validate_matrix(_matrix((0.20, 0.03, 0.03)))


def test_matrix_rejects_config_task_and_checkpoint_drift() -> None:
    drifted = _matrix()
    drifted[-1]["evaluation_config"] = {"temperature": 0.2, "max_turns": 20}
    drifted[-1]["evaluation_config_sha256"] = hashlib.sha256(
        json.dumps(
            drifted[-1]["evaluation_config"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(AgentQualityEvidenceError, match="evaluation_config"):
        validate_matrix(drifted)

    drifted = _matrix()
    drifted[-1]["paired_task_ids_sha256"] = "8" * 64
    with pytest.raises(AgentQualityEvidenceError, match="paired_task_ids"):
        validate_matrix(drifted)

    drifted = _matrix()
    drifted[-1]["training_artifact_sha256"] = "7" * 64
    with pytest.raises(AgentQualityEvidenceError, match="training_artifact"):
        validate_matrix(drifted)
