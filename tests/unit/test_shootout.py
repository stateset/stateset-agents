"""Tests for the framework-neutral measured shootout orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location("shootout", BENCHMARKS / "shootout.py")
assert SPEC is not None and SPEC.loader is not None
shootout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shootout
SPEC.loader.exec_module(shootout)

ShootoutError = shootout.ShootoutError


def _manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "agent-rl-shootout-v1",
        "cache_policy": "prewarmed-v1",
        "algorithm": "grpo",
        "algorithm_revision": "objective-v1",
        "model": "Qwen/Qwen3.5-0.8B",
        "model_revision": "a" * 40,
        "task": "customer-support-v1",
        "dataset_revision": "b" * 40,
        "config": {"steps": 10, "global_batch_size": 8},
        "seeds": [42, 1337, 2026],
        "hardware": {"gpu": "NVIDIA H100", "gpu_count": 1},
        "implementations": [
            {
                "name": "stateset-agents",
                "version": "0.42.2",
                "command": [
                    "one",
                    "{seed}",
                    "{adapter_output}",
                    "{artifact_dir}",
                    "{model}",
                    "{model_revision}",
                    "{dataset_revision}",
                    "{task}",
                    "{config_json}",
                ],
            },
            {
                "name": "trl",
                "version": "1.7.0",
                "command": [
                    "two",
                    "{seed}",
                    "{adapter_output}",
                    "{artifact_dir}",
                    "{model}",
                    "{model_revision}",
                    "{dataset_revision}",
                    "{task}",
                    "{config_json}",
                ],
            },
        ],
    }
    manifest.update(overrides)
    return manifest


def test_manifest_requires_matched_three_seed_matrix(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    loaded = shootout.load_manifest(path)
    assert loaded["seeds"] == [42, 1337, 2026]

    path.write_text(json.dumps(_manifest(seeds=[42])), encoding="utf-8")
    with pytest.raises(ShootoutError, match="at least three unique"):
        shootout.load_manifest(path)


def test_manifest_requires_commands_to_receive_neutral_protocol(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["implementations"][0]["command"].remove("{config_json}")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ShootoutError, match="missing protocol placeholders"):
        shootout.load_manifest(path)


def test_execution_order_rotates_to_reduce_bias() -> None:
    implementations = _manifest()["implementations"]
    assert shootout.execution_order(implementations, 0)[0]["name"] == "stateset-agents"
    assert shootout.execution_order(implementations, 1)[0]["name"] == "trl"


def test_hash_artifact_covers_names_and_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"weights")
    first = shootout.hash_artifact(artifact)
    (artifact / "weights.bin").write_bytes(b"changed")
    second = shootout.hash_artifact(artifact)
    assert len(first) == 64
    assert first != second


def test_adapter_result_requires_measured_matching_hardware(tmp_path: Path) -> None:
    raw = {
        "status": "completed",
        "measured": True,
        "config_sha256": shootout.canonical_digest(_manifest()["config"]),
        "framework_version": "0.42.2",
        "artifact_path": "/tmp/artifact",
        "hardware": {"gpu": "NVIDIA H100", "gpu_count": 1, "cuda": "12.8"},
        "metrics": {
            "samples_processed": 10,
            "peak_vram_mb": 100,
            "eval_score_baseline": -0.2,
            "eval_score_final": 0.1,
        },
    }
    assert shootout.validate_adapter_result(
        raw,
        {"gpu": "NVIDIA H100", "gpu_count": 1},
        shootout.canonical_digest(_manifest()["config"]),
        "0.42.2",
        tmp_path / "result.json",
    )
    raw["measured"] = False
    with pytest.raises(ShootoutError, match="measured completion"):
        shootout.validate_adapter_result(
            raw,
            {"gpu": "NVIDIA H100", "gpu_count": 1},
            shootout.canonical_digest(_manifest()["config"]),
            "0.42.2",
            tmp_path / "result.json",
        )


def test_run_implementation_emits_valid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "output"
    adapter_code = (
        "import json,pathlib,sys; out=pathlib.Path(sys.argv[1]); "
        "artifact=pathlib.Path(sys.argv[2]); (artifact/'weights').write_bytes(b'x'); "
        "out.write_text(json.dumps({'status':'completed','measured':True,"
        "'artifact_path':str(artifact),'hardware':{'gpu':'NVIDIA H100',"
        "'gpu_count':1,'cuda':'12.8'},'metrics':{'samples_processed':10,"
        "'peak_vram_mb':100,'eval_score_baseline':0.2,'eval_score_final':0.3},"
        "'config_sha256':sys.argv[3],'framework_version':'0.42.2'}))"
    )
    implementation = {
        "name": "stateset-agents",
        "version": "0.42.2",
        "command": [
            sys.executable,
            "-c",
            adapter_code,
            "{adapter_output}",
            "{artifact_dir}",
            shootout.canonical_digest(_manifest()["config"]),
        ],
    }
    manifest = _manifest(
        implementations=[implementation, _manifest()["implementations"][1]]
    )
    monkeypatch.setattr(shootout, "git_commit", lambda _root: "c" * 40)

    evidence_path = shootout.run_implementation(
        manifest, implementation, 42, output, root, timeout_seconds=10
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["measured"] is True
    assert evidence["framework"] == "stateset-agents"
    assert evidence["metrics"]["samples_per_second"] > 0
    assert len(evidence["artifact_sha256"]) == 64


def test_unknown_placeholder_fails_closed() -> None:
    with pytest.raises(ShootoutError, match="unknown command placeholder"):
        shootout._format_command(["{secret}"], {"seed": 42})
