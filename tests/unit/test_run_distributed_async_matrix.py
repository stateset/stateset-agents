"""Tests for executable multi-node asynchronous evidence collection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.distributed_async_evidence import validate_run
from benchmarks.run_distributed_async_matrix import (
    DistributedAsyncRunnerError,
    canonical_digest,
    load_manifest,
    main,
    run_scenario,
)


def _command() -> list[str]:
    return [
        "adapter",
        "{scenario}",
        "{seed}",
        "{mode}",
        "{protocol}",
        "{framework_version}",
        "{config_json}",
        "{config_sha256}",
        "{adapter_output}",
        "{artifact_dir}",
    ]


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "stateset-distributed-async-manifest",
        "protocol": "stateset-distributed-async-v1",
        "framework_version": "0.47.1",
        "provider": "runpod",
        "cost_source": "provider-api",
        "seeds": [42, 1337, 2026],
        "scenarios": [
            "steady_state_soak",
            "worker_exit",
            "controller_restart",
            "network_interruption",
        ],
        "minimum_duration_seconds": {
            "steady_state_soak": 43_200,
            "worker_exit": 1,
            "controller_restart": 1,
            "network_interruption": 1,
        },
        "config": {"max_policy_lag": 1, "queue_capacity": 1024},
        "topology": {
            "node_count": 2,
            "worker_count": 4,
            "accelerator": "NVIDIA H100 80GB HBM3",
            "accelerator_driver": "CUDA 12.9",
            "interconnect": "100 GbE",
        },
        "command": _command(),
    }


def _adapter(manifest: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    duration = 5.0
    return {
        "status": "completed",
        "measured": True,
        "scenario": "worker_exit",
        "seed": 42,
        "protocol": manifest["protocol"],
        "framework_version": manifest["framework_version"],
        "provider": manifest["provider"],
        "config_sha256": canonical_digest(manifest["config"]),
        "duration_seconds": duration,
        "topology": {
            **manifest["topology"],
            "node_ids": ["machine-a", "machine-b"],
        },
        "metrics": {
            "rollouts_attempted": 6,
            "rollouts_accepted": 5,
            "optimizer_updates": 2,
            "policy_versions": 3,
            "max_observed_policy_lag": 1,
            "lost_optimizer_updates": 0,
            "duplicate_optimizer_updates": 0,
            "artifact_digest_mismatches": 0,
            "accepted_rollouts_per_second": 1.0,
            "weight_sync_latency_ms": {"p50": 10.0, "p95": 20.0, "max": 30.0},
        },
        "cost": {
            "currency": "USD",
            "source": "provider-api",
            "total": 0.5,
            "per_accepted_rollout": 0.1,
        },
        "recovery": {
            "recovered": True,
            "recovery_seconds": 1.0,
            "resources_remaining": 0,
        },
        "artifact_path": str(artifact_path),
    }


def test_manifest_requires_full_matrix_soak_and_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert load_manifest(path)["topology"]["node_count"] == 2

    invalid = _manifest()
    invalid["minimum_duration_seconds"]["steady_state_soak"] = 3600
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(DistributedAsyncRunnerError, match="43200"):
        load_manifest(path)

    invalid = _manifest()
    invalid["command"].remove("{artifact_dir}")
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(DistributedAsyncRunnerError, match="missing placeholders"):
        load_manifest(path)

    invalid = _manifest()
    invalid["config"]["provider_api_key"] = "must-not-be-here"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(DistributedAsyncRunnerError, match="environment"):
        load_manifest(path)


def test_run_scenario_hashes_artifact_and_emits_valid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        adapter_output = Path(command[-2])
        artifact_dir = Path(command[-1])
        artifact = artifact_dir / "audit.jsonl"
        artifact.write_text('{"rollout":"accepted"}\n', encoding="utf-8")
        adapter_output.write_text(
            json.dumps(_adapter(manifest, artifact)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    ticks = iter((10.0, 15.0))
    monkeypatch.setattr(
        "benchmarks.run_distributed_async_matrix.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "benchmarks.run_distributed_async_matrix.time.monotonic", lambda: next(ticks)
    )
    destination = run_scenario(
        manifest,
        "worker_exit",
        42,
        output_dir=tmp_path / "output",
        root=tmp_path,
        harness_commit="f" * 40,
        timeout_seconds=30,
    )
    evidence = json.loads(destination.read_text(encoding="utf-8"))
    validate_run(evidence, destination)
    assert evidence["external_wall_seconds"] == 5.0
    assert evidence["artifact_sha256"] != ""


def test_run_scenario_rejects_impossible_reported_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        adapter_output = Path(command[-2])
        artifact_dir = Path(command[-1])
        artifact = artifact_dir / "audit.jsonl"
        artifact.write_text("measured\n", encoding="utf-8")
        adapter_output.write_text(
            json.dumps(_adapter(manifest, artifact)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    ticks = iter((10.0, 11.0))
    monkeypatch.setattr(
        "benchmarks.run_distributed_async_matrix.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "benchmarks.run_distributed_async_matrix.time.monotonic", lambda: next(ticks)
    )
    with pytest.raises(DistributedAsyncRunnerError, match="external wall time"):
        run_scenario(
            manifest,
            "worker_exit",
            42,
            output_dir=tmp_path / "output",
            root=tmp_path,
            harness_commit="f" * 40,
            timeout_seconds=30,
        )


def test_run_scenario_rejects_artifact_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    escaped = tmp_path / "outside.json"
    escaped.write_text("outside\n", encoding="utf-8")

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        Path(command[-2]).write_text(
            json.dumps(_adapter(manifest, escaped)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    ticks = iter((10.0, 15.0))
    monkeypatch.setattr(
        "benchmarks.run_distributed_async_matrix.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "benchmarks.run_distributed_async_matrix.time.monotonic", lambda: next(ticks)
    )
    with pytest.raises(DistributedAsyncRunnerError, match="artifact_dir"):
        run_scenario(
            manifest,
            "worker_exit",
            42,
            output_dir=tmp_path / "output",
            root=tmp_path,
            harness_commit="f" * 40,
            timeout_seconds=30,
        )


def test_dry_run_rotates_scenario_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert main([str(path), "--output-dir", str(tmp_path / "out"), "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:5] == [
        "seed=42 scenario=steady_state_soak",
        "seed=42 scenario=worker_exit",
        "seed=42 scenario=controller_restart",
        "seed=42 scenario=network_interruption",
        "seed=1337 scenario=worker_exit",
    ]
