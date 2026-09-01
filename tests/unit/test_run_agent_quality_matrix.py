"""Tests for executable standard-agent benchmark collection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.agent_quality_evidence import validate_run
from benchmarks.run_agent_quality_matrix import (
    AgentQualityRunnerError,
    canonical_digest,
    load_manifest,
    run_suite,
    validate_adapter_result,
)


def _command() -> list[str]:
    return [
        "adapter",
        "{seed}",
        "{suite}",
        "{suite_revision}",
        "{split}",
        "{baseline_model}",
        "{baseline_revision}",
        "{trained_model}",
        "{trained_revision}",
        "{evaluation_config_json}",
        "{adapter_output}",
        "{artifact_dir}",
    ]


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "stateset-agent-quality-manifest",
        "protocol": "stateset-standard-agent-quality-v1",
        "framework_version": "0.47.0",
        "baseline_policy": {"model": "example/base", "revision": "a" * 40},
        "trained_policy": {
            "model": "example/trained",
            "revision": "b" * 40,
            "artifact_sha256": "c" * 64,
        },
        "evaluation_config": {"temperature": 0.0, "max_turns": 20},
        "seeds": [42, 1337, 2026],
        "suites": [
            {
                "name": name,
                "revision": str(index) * 40,
                "split": "test",
                "command": _command(),
            }
            for index, name in enumerate(
                ("tau-bench", "bfcl", "swe-bench-verified"), start=1
            )
        ],
    }


def _adapter(manifest: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "completed",
        "measured": True,
        "suite": suite["name"],
        "suite_revision": suite["revision"],
        "split": suite["split"],
        "seed": 42,
        "framework_version": manifest["framework_version"],
        "baseline_model": manifest["baseline_policy"]["model"],
        "baseline_model_revision": manifest["baseline_policy"]["revision"],
        "trained_model": manifest["trained_policy"]["model"],
        "trained_model_revision": manifest["trained_policy"]["revision"],
        "evaluation_config_sha256": canonical_digest(manifest["evaluation_config"]),
        "paired_task_ids_sha256": "d" * 64,
        "tasks": 100,
        "baseline_successful_episodes": 40,
        "trained_successful_episodes": 50,
        "baseline_score": 0.40,
        "trained_score": 0.50,
        "evaluation_cost_usd": 5.0,
        "cost_source": "provider-api",
        "artifact_path": "filled-by-test",
    }


def test_manifest_requires_exact_roster_policy_digests_and_placeholders(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert len(load_manifest(path)["suites"]) == 3

    invalid = _manifest()
    invalid["trained_policy"]["artifact_sha256"] = "unknown"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(AgentQualityRunnerError, match="artifact_sha256"):
        load_manifest(path)

    invalid = _manifest()
    invalid["suites"][0]["command"].remove("{trained_revision}")
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(AgentQualityRunnerError, match="missing placeholders"):
        load_manifest(path)


def test_adapter_rejects_unpaired_or_drifted_results(tmp_path: Path) -> None:
    manifest = _manifest()
    suite = manifest["suites"][0]
    result = _adapter(manifest, suite)
    validate_adapter_result(
        result,
        manifest=manifest,
        suite=suite,
        seed=42,
        source=tmp_path / "result.json",
    )

    result["paired_task_ids_sha256"] = "not-paired"
    with pytest.raises(AgentQualityRunnerError, match="paired_task"):
        validate_adapter_result(
            result,
            manifest=manifest,
            suite=suite,
            seed=42,
            source=tmp_path / "result.json",
        )

    result = _adapter(manifest, suite)
    result["trained_model_revision"] = "e" * 40
    with pytest.raises(AgentQualityRunnerError, match="trained_model_revision"):
        validate_adapter_result(
            result,
            manifest=manifest,
            suite=suite,
            seed=42,
            source=tmp_path / "result.json",
        )


def test_run_suite_emits_v2_evidence_and_hashes_retained_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    suite = manifest["suites"][0]
    output_dir = tmp_path / "output"

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        adapter_output = Path(command[-2])
        artifact_dir = Path(command[-1])
        artifact = artifact_dir / "paired-results.jsonl"
        artifact.write_text('{"task":"one","base":0,"trained":1}\n')
        result = _adapter(manifest, suite)
        result["artifact_path"] = str(artifact)
        adapter_output.write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("benchmarks.run_agent_quality_matrix.subprocess.run", fake_run)
    destination = run_suite(
        manifest,
        suite,
        42,
        output_dir=output_dir,
        root=tmp_path,
        harness_commit="f" * 40,
        timeout_seconds=30,
    )
    evidence = json.loads(destination.read_text(encoding="utf-8"))
    validate_run(evidence, destination)
    assert evidence["schema_version"] == 2
    assert evidence["training_artifact_sha256"] == "c" * 64
    assert evidence["cost_per_successful_episode_usd"] == pytest.approx(0.1)


def test_run_suite_retains_failure_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    suite = manifest["suites"][0]

    monkeypatch.setattr(
        "benchmarks.run_agent_quality_matrix.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7, stdout="", stderr="failed"
        ),
    )
    with pytest.raises(AgentQualityRunnerError, match="exited 7"):
        run_suite(
            manifest,
            suite,
            42,
            output_dir=tmp_path / "output",
            root=tmp_path,
            harness_commit="f" * 40,
            timeout_seconds=30,
        )
    failure = tmp_path / "output/runs/tau-bench-seed42/failure.json"
    assert json.loads(failure.read_text(encoding="utf-8"))["returncode"] == 7


def test_ci_validates_agent_quality_manifest_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "make benchmark-agent-quality-contract" in workflow
