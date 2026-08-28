"""Tests for strict external-backend conformance evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.training.backends import BackendResult

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
SPEC = importlib.util.spec_from_file_location(
    "backend_conformance", BENCHMARKS / "backend_conformance.py"
)
assert SPEC is not None and SPEC.loader is not None
backend_conformance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = backend_conformance
SPEC.loader.exec_module(backend_conformance)

ConformanceError = backend_conformance.ConformanceError


def _manifest(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "backend": "nemo-rl",
        "backend_version": "0.6.0+abcdef0",
        "harness_revision": "a" * 40,
        "experiment": {
            "algorithm": "grpo",
            "model": "Qwen/example",
            "model_revision": "b" * 40,
            "dataset_uri": "/workspace/train.jsonl",
            "dataset_sha256": "c" * 64,
            "seed": 42,
            "config": {
                "generation_backend": "vllm",
                "max_num_steps": 1,
                "num_generations_per_prompt": 2,
            },
            "environment": {"type": "single_turn", "name": "math"},
            "reward": {
                "type": "nemo_builtin",
                "name": "math",
                "implementation": "hf_math_verify",
            },
            "requirements": ["distributed"],
        },
    }
    value.update(updates)
    return value


def _write_manifest(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_manifest_requires_known_backend_pins_and_exact_schema(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _manifest())
    assert backend_conformance.load_manifest(path)["backend"] == "nemo-rl"

    path.write_text(json.dumps(_manifest(backend="unknown")), encoding="utf-8")
    with pytest.raises(ConformanceError, match="backend must be"):
        backend_conformance.load_manifest(path)

    path.write_text(json.dumps(_manifest(harness_revision="main")), encoding="utf-8")
    with pytest.raises(ConformanceError, match="full lowercase git commit"):
        backend_conformance.load_manifest(path)

    invalid = _manifest(extra="drift")
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ConformanceError, match="unknown manifest fields"):
        backend_conformance.load_manifest(path)


def test_manifest_rejects_unknown_and_missing_experiment_fields(tmp_path: Path) -> None:
    unknown = _manifest()
    unknown["experiment"]["api_key"] = "secret"
    path = _write_manifest(tmp_path, unknown)
    with pytest.raises(ConformanceError, match="unknown experiment fields"):
        backend_conformance.load_manifest(path)

    missing = _manifest()
    del missing["experiment"]["dataset_sha256"]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ConformanceError, match="missing experiment fields"):
        backend_conformance.load_manifest(path)


def test_build_experiment_keeps_output_out_of_semantic_digest(tmp_path: Path) -> None:
    manifest = _manifest()
    first = backend_conformance.build_experiment(manifest, tmp_path / "one")
    second = backend_conformance.build_experiment(manifest, tmp_path / "two")
    assert first.sha256 == second.sha256
    assert first.output_dir != second.output_dir


def test_artifact_hash_covers_names_and_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    weights = artifact / "weights.bin"
    weights.write_bytes(b"first")
    first = backend_conformance.hash_artifact(artifact)
    weights.write_bytes(b"second")
    assert first != backend_conformance.hash_artifact(artifact)
    weights.unlink()
    with pytest.raises(ConformanceError, match="empty"):
        backend_conformance.hash_artifact(artifact)


def test_hardware_probe_records_every_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    output = (
        "NVIDIA H100 80GB HBM3, GPU-one, 81559, 580.65.06\n"
        "NVIDIA H100 80GB HBM3, GPU-two, 81559, 580.65.06\n"
    )
    monkeypatch.setattr(
        backend_conformance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )
    monkeypatch.setattr(
        backend_conformance, "_required_executable", lambda name: f"/bin/{name}"
    )
    hardware = backend_conformance.collect_nvidia_hardware()
    assert hardware["gpu_count"] == 2
    assert hardware["gpus"][0]["uuid"] == "GPU-one"
    assert hardware["gpus"][1]["memory_total_mb"] == 81559


def test_hardware_probe_fails_closed_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_conformance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=9, stdout="", stderr="driver unavailable"
        ),
    )
    monkeypatch.setattr(
        backend_conformance, "_required_executable", lambda name: f"/bin/{name}"
    )
    with pytest.raises(ConformanceError, match="driver unavailable"):
        backend_conformance.collect_nvidia_hardware()


def test_harness_revision_requires_clean_matching_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr=""),
        ]
    )
    monkeypatch.setattr(
        backend_conformance.subprocess, "run", lambda *args, **kwargs: next(responses)
    )
    backend_conformance.verify_harness_revision("a" * 40, tmp_path)

    monkeypatch.setattr(
        backend_conformance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="modified.py\n", stderr=""
        ),
    )
    with pytest.raises(ConformanceError, match="worktree must be clean"):
        backend_conformance.verify_harness_revision("a" * 40, tmp_path)


def test_run_conformance_binds_hardware_experiment_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hardware = {
        "gpu_count": 1,
        "gpus": [
            {
                "name": "NVIDIA H100 80GB HBM3",
                "uuid": "GPU-one",
                "memory_total_mb": 81559,
                "driver_version": "580.65.06",
            }
        ],
    }
    monkeypatch.setattr(
        backend_conformance, "collect_nvidia_hardware", lambda: hardware
    )
    monkeypatch.setattr(backend_conformance, "verify_harness_revision", lambda *_: None)
    captured: dict[str, Any] = {}

    class FakeBackend:
        def run(self, experiment: Any) -> BackendResult:
            captured["experiment"] = experiment
            artifact = experiment.output_dir / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "weights.bin").write_bytes(b"weights")
            return BackendResult(
                backend="nemo-rl",
                backend_version="0.6.0+abcdef0",
                experiment_sha256=experiment.sha256,
                artifact_uri=str(artifact),
                metrics={"completed": 1.0, "wall_time_seconds": 0.5},
            )

    def factory(*, version: str, timeout_seconds: int) -> FakeBackend:
        assert version == "0.6.0+abcdef0"
        assert timeout_seconds == 60
        return FakeBackend()

    monkeypatch.setitem(backend_conformance._BACKEND_FACTORIES, "nemo-rl", factory)
    monkeypatch.setattr(
        backend_conformance.importlib.metadata, "version", lambda _: "0.42.6"
    )
    evidence = backend_conformance.run_conformance(
        _manifest(), tmp_path / "evidence", timeout_seconds=60, root=tmp_path
    )
    assert evidence["status"] == "completed"
    assert evidence["hardware"] == hardware
    assert evidence["experiment_sha256"] == captured["experiment"].sha256
    assert len(evidence["artifact_sha256"]) == 64
    assert evidence["backend_metrics"]["completed"] == 1.0
    evidence_path = tmp_path / "conformance.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert backend_conformance.load_evidence(evidence_path)["status"] == "completed"
    Path(evidence["artifact_uri"]).joinpath("weights.bin").write_bytes(b"changed")
    with pytest.raises(ConformanceError, match="artifact digest"):
        backend_conformance.load_evidence(evidence_path)


def test_evidence_validator_rejects_digest_gpu_and_completion_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    hardware = {
        "gpu_count": 1,
        "gpus": [
            {
                "name": "NVIDIA H100",
                "uuid": "GPU-one",
                "memory_total_mb": 81559,
                "driver_version": "580.65.06",
            }
        ],
    }
    monkeypatch.setattr(
        backend_conformance, "collect_nvidia_hardware", lambda: hardware
    )
    monkeypatch.setattr(backend_conformance, "verify_harness_revision", lambda *_: None)

    class FakeBackend:
        def run(self, experiment: Any) -> BackendResult:
            artifact = experiment.output_dir / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "weights").write_bytes(b"x")
            return BackendResult(
                backend="nemo-rl",
                backend_version="0.6.0+abcdef0",
                experiment_sha256=experiment.sha256,
                artifact_uri=str(artifact),
                metrics={"completed": 1.0, "wall_time_seconds": 0.1},
            )

    monkeypatch.setitem(
        backend_conformance._BACKEND_FACTORIES,
        "nemo-rl",
        lambda **_: FakeBackend(),
    )
    monkeypatch.setattr(
        backend_conformance.importlib.metadata, "version", lambda _: "0.42.6"
    )
    evidence = backend_conformance.run_conformance(manifest, tmp_path, 60, tmp_path)
    changed = dict(evidence)
    changed["manifest_sha256"] = "0" * 64
    with pytest.raises(ConformanceError, match="manifest digest"):
        backend_conformance.validate_evidence(changed, manifest)
    changed = dict(evidence)
    changed["manifest"] = {**manifest, "backend_version": "different"}
    with pytest.raises(ConformanceError, match="embedded evidence manifest"):
        backend_conformance.validate_evidence(changed, manifest)
    changed = dict(evidence)
    changed["hardware"] = {"gpu_count": 2, "gpus": hardware["gpus"]}
    with pytest.raises(ConformanceError, match="every visible GPU"):
        backend_conformance.validate_evidence(changed, manifest)
    changed = dict(evidence)
    changed["backend_metrics"] = {"completed": 0.0, "wall_time_seconds": 0.1}
    with pytest.raises(ConformanceError, match="completed=1.0"):
        backend_conformance.validate_evidence(changed, manifest)


def test_run_conformance_uses_clock_resolution_for_a_zero_tick_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        backend_conformance,
        "collect_nvidia_hardware",
        lambda: {
            "gpu_count": 1,
            "gpus": [
                {
                    "name": "NVIDIA H100",
                    "uuid": "GPU-one",
                    "memory_total_mb": 81559,
                    "driver_version": "580.65.06",
                }
            ],
        },
    )
    monkeypatch.setattr(backend_conformance, "verify_harness_revision", lambda *_: None)
    ticks = iter((42.0, 42.0))
    monkeypatch.setattr(backend_conformance.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        backend_conformance.importlib.metadata, "version", lambda _: "0.42.6"
    )

    class FakeBackend:
        def run(self, experiment: Any) -> BackendResult:
            artifact = experiment.output_dir / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "weights").write_bytes(b"x")
            return BackendResult(
                backend="nemo-rl",
                backend_version="0.6.0+abcdef0",
                experiment_sha256=experiment.sha256,
                artifact_uri=str(artifact),
                metrics={"completed": 1.0, "wall_time_seconds": 0.0},
            )

    monkeypatch.setitem(
        backend_conformance._BACKEND_FACTORIES,
        "nemo-rl",
        lambda **_: FakeBackend(),
    )
    evidence = backend_conformance.run_conformance(manifest, tmp_path, 60, tmp_path)
    assert (
        evidence["wall_time_seconds"]
        == backend_conformance.time.get_clock_info("monotonic").resolution
    )


def test_main_retains_failure_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(tmp_path, _manifest())
    output = tmp_path / "output"
    monkeypatch.setattr(
        backend_conformance,
        "run_conformance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ConformanceError("GPU unavailable")
        ),
    )
    assert (
        backend_conformance.main(
            [str(manifest), "--output-dir", str(output), "--timeout-seconds", "60"]
        )
        == 2
    )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["error_type"] == "ConformanceError"
    assert failure["error"] == "GPU unavailable"


def test_evidence_writer_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    backend_conformance.write_json_once(path, {"status": "first"})
    with pytest.raises(ConformanceError, match="refusing to overwrite"):
        backend_conformance.write_json_once(path, {"status": "second"})
