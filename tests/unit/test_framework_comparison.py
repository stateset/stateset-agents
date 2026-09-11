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


def test_schema_v2_rehashes_portable_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "runs" / "stateset" / "artifact"
    artifact.mkdir(parents=True)
    payload = artifact / "metrics.json"
    payload.write_text("{}\n", encoding="utf-8")
    data = _document(
        "stateset-agents",
        42,
        schema_version=2,
        artifact_path="runs/stateset/artifact",
        artifact_sha256=framework_comparison.hash_artifact(artifact),
        manifest_sha256="e" * 64,
    )
    source = tmp_path / "stateset-agents-42.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    assert framework_comparison.load_evidence([source])[0].seed == 42

    payload.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(EvidenceError, match="digest does not match"):
        framework_comparison.load_evidence([source])


def test_comparison_rejects_mixed_evidence_schemas(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    changed = dict(runs[-1].data)
    changed["schema_version"] = 2
    runs[-1] = framework_comparison.RunEvidence(runs[-1].source, changed)
    with pytest.raises(EvidenceError, match="schema_version"):
        framework_comparison.validate_comparison(runs)


def test_provider_cost_bundle_is_bound_and_arithmetically_verified(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "runs" / "stateset" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "metrics.json").write_text("{}\n", encoding="utf-8")
    document = _document(
        "stateset-agents",
        42,
        schema_version=2,
        manifest_sha256="e" * 64,
        artifact_path="runs/stateset/artifact",
        artifact_sha256=framework_comparison.hash_artifact(artifact),
    )
    evidence = tmp_path / "stateset-agents-42.json"
    evidence.write_text(json.dumps(document), encoding="utf-8")
    runs = framework_comparison.load_evidence([evidence])
    provider = {
        "schema_version": 2,
        "kind": "stateset-runpod-shootout-provider-record",
        "provider": "runpod",
        "cost_source": framework_comparison.PROVIDER_COST_SOURCE,
        "status": "completed",
        "shootout_manifest_sha256": "e" * 64,
        "harness_revision": "a" * 40,
        "termination_confirmed": True,
        "pod_id": "pod-1",
        "authoritative_pod_cost_per_hr_usd": 1.0,
        "pod_lifetime_seconds": 3600.0,
        "estimated_cost_usd": 1.0,
    }
    path = tmp_path / "runpod-provider.json"
    path.write_text(json.dumps(provider), encoding="utf-8")

    cost = framework_comparison.load_provider_cost_bundle(tmp_path, runs)
    assert cost["records"] == 1
    assert cost["total_cost_usd"] == pytest.approx(1.0)
    report = tmp_path / "report"
    framework_comparison.write_report(runs, report, cost)
    payload = json.loads((report / "comparison.json").read_text(encoding="utf-8"))
    markdown = (report / "comparison.md").read_text(encoding="utf-8")
    assert payload["provider_cost"]["total_cost_usd"] == pytest.approx(1.0)
    assert "Provider cost: $1.0000" in markdown
    assert framework_comparison.PROVIDER_COST_SOURCE in markdown

    provider["estimated_cost_usd"] = 2.0
    path.write_text(json.dumps(provider), encoding="utf-8")
    with pytest.raises(EvidenceError, match="arithmetic"):
        framework_comparison.load_provider_cost_bundle(tmp_path, runs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shootout_manifest_sha256", "f" * 64, "manifest mismatch"),
        ("harness_revision", "b" * 40, "harness mismatch"),
        ("termination_confirmed", False, "cleanup is not confirmed"),
        ("status", "cleanup-pending", "lifecycle status is incomplete"),
    ],
)
def test_provider_cost_bundle_rejects_unbound_or_unclosed_lifecycle(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_text("measured\n", encoding="utf-8")
    document = _document(
        "stateset-agents",
        42,
        schema_version=2,
        manifest_sha256="e" * 64,
        artifact_path="artifact",
        artifact_sha256=framework_comparison.hash_artifact(artifact),
    )
    evidence = tmp_path / "stateset-agents-42.json"
    evidence.write_text(json.dumps(document), encoding="utf-8")
    runs = framework_comparison.load_evidence([evidence])
    provider = {
        "schema_version": 2,
        "kind": "stateset-runpod-shootout-provider-record",
        "provider": "runpod",
        "cost_source": framework_comparison.PROVIDER_COST_SOURCE,
        "status": "completed",
        "shootout_manifest_sha256": "e" * 64,
        "harness_revision": "a" * 40,
        "termination_confirmed": True,
        "pod_id": "pod-1",
        "authoritative_pod_cost_per_hr_usd": 1.0,
        "pod_lifetime_seconds": 1800.0,
        "estimated_cost_usd": 0.5,
    }
    provider[field] = value
    (tmp_path / "runpod-provider.json").write_text(
        json.dumps(provider), encoding="utf-8"
    )

    with pytest.raises(EvidenceError, match=message):
        framework_comparison.load_provider_cost_bundle(tmp_path, runs)


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


def test_rejects_mismatched_harness_commit(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    mismatched = dict(runs[-1].data)
    mismatched["harness_commit"] = "d" * 40
    runs[-1] = framework_comparison.RunEvidence(runs[-1].source, mismatched)

    with pytest.raises(EvidenceError, match="harness_commit"):
        framework_comparison.validate_comparison(runs)


def test_rejects_non_hex_immutable_revision(tmp_path: Path) -> None:
    data = _document("stateset-agents", 42)
    data["model_revision"] = "z" * 40

    with pytest.raises(EvidenceError, match="lowercase hex commit"):
        framework_comparison.validate_document(data, tmp_path / "run.json")


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
    assert payload["comparison"]["harness_commit"] == "a" * 40
    assert "Descriptive results only" in markdown
    assert "Harness commit" in markdown
    assert "Winner" not in markdown


def test_cli_fails_closed_on_bad_input(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    assert framework_comparison.main([str(path), "--validate-only"]) == 2


def test_cli_requires_verifiable_schema_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _runs(tmp_path)
    assert framework_comparison.main([str(tmp_path), "--validate-only"]) == 2
    assert "schema_version>=2" in capsys.readouterr().err
    assert (
        framework_comparison.main(
            [str(tmp_path), "--validate-only", "--allow-legacy-schema-v1"]
        )
        == 0
    )


def test_directory_discovery_skips_launcher_records(tmp_path):
    """A RunPod provider record or accounting summary beside the evidence must
    not be mistaken for a run (it lacks ``measured`` and would fail closed)."""
    import json as _json

    (tmp_path / "runpod-provider.json").write_text(
        _json.dumps(
            {"kind": "stateset-runpod-shootout-provider-record", "status": "completed"}
        )
    )
    (tmp_path / "accounting.json").write_text(
        _json.dumps({"kind": "framework-shootout-accounting", "attempted": 1})
    )
    (tmp_path / "trl-seed42.json").write_text(_json.dumps({"framework": "trl"}))
    found = [p.name for p in framework_comparison.discover_inputs([tmp_path])]
    assert found == ["trl-seed42.json"]


def test_rejects_evidence_whose_training_reward_was_identically_zero(
    tmp_path: Path,
) -> None:
    data = _document("stateset-agents-gspo", 42)
    data["metrics"]["train_reward_zero_fraction"] = 1.0
    with pytest.raises(EvidenceError, match="no learning signal"):
        framework_comparison.validate_document(data, tmp_path / "run.json")
    data["metrics"]["train_reward_zero_fraction"] = 0.2
    framework_comparison.validate_document(data, tmp_path / "run.json")
