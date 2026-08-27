"""Tests for the measured StateSet algorithm shootout orchestrator."""

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
    "algorithm_shootout", BENCHMARKS / "algorithm_shootout.py"
)
assert SPEC is not None and SPEC.loader is not None
algorithm_shootout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = algorithm_shootout
SPEC.loader.exec_module(algorithm_shootout)

ShootoutError = algorithm_shootout.ShootoutError


def _algorithm(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "revision": f"stateset-{name}-objective-v1",
        "config": {
            "objective": f"{name}-objective",
            "max_steps": 2,
            "num_generations": 2,
        },
        "command": [
            "adapter",
            "{algorithm}",
            "{seed}",
            "{adapter_output}",
            "{artifact_dir}",
            "{phase0_output}",
            "{model}",
            "{model_revision}",
            "{dataset_revision}",
            "{task}",
            "{config_json}",
            "{num_train_examples}",
            "{num_eval_examples}",
        ],
    }


def _manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "stateset-five-algorithm-v1",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "a" * 40,
        "task": "gsm8k",
        "dataset_revision": "b" * 40,
        "cache_policy": "prewarmed-v1",
        "framework": {"name": "stateset-agents", "version": "0.42.4"},
        "config": {
            "num_train_examples": 8,
            "num_eval_examples": 4,
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_generations": 2,
            "num_iterations": 1,
        },
        "seeds": [42, 1337, 2026],
        "hardware": {"gpu": "NVIDIA H100", "gpu_count": 1, "cuda": "12.8"},
        "algorithms": [_algorithm("grpo"), _algorithm("gspo")],
    }
    manifest.update(overrides)
    return manifest


def _write_manifest(tmp_path: Path, manifest: dict[str, Any]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_requires_three_seeds_and_exact_cuda(tmp_path: Path) -> None:
    loaded = algorithm_shootout.load_manifest(_write_manifest(tmp_path, _manifest()))
    assert loaded["seeds"] == [42, 1337, 2026]

    with pytest.raises(ShootoutError, match="at least three unique"):
        algorithm_shootout.load_manifest(
            _write_manifest(tmp_path, _manifest(seeds=[42]))
        )
    bad_hardware = {"gpu": "NVIDIA H100", "gpu_count": 1}
    with pytest.raises(ShootoutError, match="hardware.cuda"):
        algorithm_shootout.load_manifest(
            _write_manifest(tmp_path, _manifest(hardware=bad_hardware))
        )


def test_manifest_forbids_false_accumulation_equivalence(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["config"]["gradient_accumulation_steps"] = 2
    with pytest.raises(ShootoutError, match="gradient_accumulation_steps=1"):
        algorithm_shootout.load_manifest(_write_manifest(tmp_path, manifest))


def test_manifest_requires_complete_adapter_contract(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["algorithms"][0]["command"].remove("{algorithm}")
    with pytest.raises(ShootoutError, match="missing placeholders"):
        algorithm_shootout.load_manifest(_write_manifest(tmp_path, manifest))


def test_execution_order_rotates_algorithms() -> None:
    algorithms = _manifest()["algorithms"]
    assert algorithm_shootout.execution_order(algorithms, 0)[0]["name"] == "grpo"
    assert algorithm_shootout.execution_order(algorithms, 1)[0]["name"] == "gspo"


def test_adapter_rejects_algorithm_config_mismatch(tmp_path: Path) -> None:
    manifest = _manifest()
    algorithm = manifest["algorithms"][0]
    raw = {
        "status": "completed",
        "measured": True,
        "framework_version": "0.42.4",
        "config_sha256": algorithm_shootout.canonical_digest(manifest["config"]),
        "algorithm_config": {"objective": "wrong"},
        "algorithm_config_sha256": algorithm_shootout.canonical_digest(
            {"objective": "wrong"}
        ),
        "artifact_path": "/tmp/artifact",
        "hardware": manifest["hardware"],
        "metrics": {
            "samples_processed": 4,
            "peak_vram_mb": 100,
            "eval_score_baseline": 0.1,
            "eval_score_final": 0.2,
        },
    }
    with pytest.raises(ShootoutError, match="algorithm config"):
        algorithm_shootout.validate_adapter_result(
            raw, manifest, algorithm, tmp_path / "adapter.json"
        )


def test_run_algorithm_emits_hashed_valid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "output"
    manifest = _manifest()
    algorithm = manifest["algorithms"][0]
    adapter_code = (
        "import json,pathlib,sys; out=pathlib.Path(sys.argv[1]); "
        "artifact=pathlib.Path(sys.argv[2])/'final_model'; artifact.mkdir(); "
        "(artifact/'weights').write_bytes(b'x'); cfg=json.loads(sys.argv[4]); "
        "out.write_text(json.dumps({'status':'completed','measured':True,"
        "'framework_version':'0.42.4','config_sha256':sys.argv[3],"
        "'algorithm_config':cfg,'algorithm_config_sha256':sys.argv[5],"
        "'artifact_path':str(artifact),'hardware':{'gpu':'NVIDIA H100',"
        "'gpu_count':1,'cuda':'12.8'},'metrics':{'samples_processed':4,"
        "'peak_vram_mb':100,'eval_score_baseline':0.1,'eval_score_final':0.2}}))"
    )
    algorithm["command"] = [
        sys.executable,
        "-c",
        adapter_code,
        "{adapter_output}",
        "{artifact_dir}",
        algorithm_shootout.canonical_digest(manifest["config"]),
        json.dumps(algorithm["config"]),
        algorithm_shootout.canonical_digest(algorithm["config"]),
        "{algorithm}",
        "{seed}",
        "{phase0_output}",
        "{model}",
        "{model_revision}",
        "{dataset_revision}",
        "{task}",
        "{config_json}",
        "{num_train_examples}",
        "{num_eval_examples}",
    ]
    monkeypatch.setattr(algorithm_shootout, "git_commit", lambda _root: "c" * 40)
    evidence_path = algorithm_shootout.run_algorithm(
        manifest, algorithm, 42, output, root, timeout_seconds=10
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["algorithm"] == "grpo"
    assert evidence["config"]["algorithm"] == algorithm["config"]
    assert evidence["metrics"]["samples_per_second"] > 0
    assert len(evidence["artifact_sha256"]) == 64


def test_required_roster_fails_closed(tmp_path: Path, capsys: Any) -> None:
    result = algorithm_shootout.main(
        [
            str(_write_manifest(tmp_path, _manifest())),
            "--output-dir",
            str(tmp_path / "out"),
            "--required-algorithm",
            "vapo",
            "--dry-run",
        ]
    )
    assert result == 2
    assert "missing required algorithms: vapo" in capsys.readouterr().err
