"""Tests for the complete external-backend conformance roster gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend_conformance = _load_module(
    "backend_conformance", BENCHMARKS / "backend_conformance.py"
)
suite = _load_module(
    "backend_conformance_suite", BENCHMARKS / "backend_conformance_suite.py"
)


def _manifest(
    backend: str,
    *,
    seed: int = 42,
    harness: str = "a" * 40,
    max_cost_usd: float = 1.0,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "backend": backend,
        "backend_version": f"{backend}-version",
        "harness_revision": harness,
        "execution": {
            "provider": "runpod",
            "provider_tier": "SECURE",
            "container_image": f"registry.example/{backend}@sha256:" + "d" * 64,
            "gpu_name": "NVIDIA H100",
            "gpu_count": 1,
            "timeout_seconds": 60,
            "max_cost_usd": max_cost_usd,
        },
        "experiment": {
            "algorithm": "grpo",
            "model": "Qwen/example",
            "model_revision": "b" * 40,
            "dataset_uri": "/workspace/train.jsonl",
            "dataset_sha256": "c" * 64,
            "seed": seed,
            "task": "math-conformance",
            "config": {"max_steps": 1},
        },
    }


def _write_evidence(
    root: Path,
    backend: str,
    *,
    label: str | None = None,
    seed: int = 42,
    harness: str = "a" * 40,
    max_cost_usd: float = 1.0,
) -> Path:
    directory = root / (label or backend)
    artifact = directory / "run" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "weights.bin").write_bytes(backend.encode())
    manifest = _manifest(backend, seed=seed, harness=harness, max_cost_usd=max_cost_usd)
    evidence = {
        "schema_version": 2,
        "kind": "stateset-external-backend-conformance",
        "status": "completed",
        "measured": True,
        "backend": backend,
        "backend_version": manifest["backend_version"],
        "stateset_agents_version": "0.42.6",
        "harness_revision": harness,
        "execution": manifest["execution"],
        "manifest": manifest,
        "manifest_sha256": backend_conformance.canonical_digest(manifest),
        "experiment_sha256": backend_conformance.build_experiment(
            manifest, directory / "run"
        ).sha256,
        "started_at": "2026-08-28T12:00:00+00:00",
        "completed_at": "2026-08-28T12:00:01+00:00",
        "wall_time_seconds": 1.0,
        "hardware": {
            "gpu_count": 1,
            "gpus": [
                {
                    "name": "NVIDIA H100",
                    "uuid": f"GPU-{backend}",
                    "memory_total_mb": 81559,
                    "driver_version": "580.65.06",
                }
            ],
        },
        "runtime": {"python": "3.12.0", "platform": "Linux"},
        "artifact_uri": "run/artifact",
        "artifact_sha256": backend_conformance.hash_artifact(artifact),
        "backend_metrics": {"completed": 1.0, "wall_time_seconds": 0.5},
        "backend_metadata": {},
    }
    path = directory / "conformance.json"
    path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    return path


def _complete_roster(root: Path) -> list[Path]:
    return [_write_evidence(root, backend) for backend in suite.REQUIRED_BACKENDS]


def test_complete_suite_revalidates_artifacts_and_writes_bound_report(
    tmp_path: Path,
) -> None:
    _complete_roster(tmp_path)
    records = suite.load_records([tmp_path])
    suite.validate_suite(records)
    report = suite.summarize(records)
    assert report["status"] == "completed"
    assert report["backend_count"] == 3
    assert report["required_backends"] == ["nemo-rl", "openrlhf", "verl"]
    assert len(report["suite_sha256"]) == 64
    assert {entry["backend"] for entry in report["backends"]} == set(
        suite.REQUIRED_BACKENDS
    )
    assert all(
        entry["execution"]["max_cost_usd"] == 1.0 for entry in report["backends"]
    )


def test_suite_rejects_missing_duplicate_and_unexpected_backends(
    tmp_path: Path,
) -> None:
    paths = _complete_roster(tmp_path)
    records = suite.load_records(paths[:2])
    with pytest.raises(suite.ConformanceSuiteError, match="roster mismatch"):
        suite.validate_suite(records)

    duplicate = _write_evidence(tmp_path, "nemo-rl", label="nemo-copy")
    with pytest.raises(suite.ConformanceSuiteError, match="duplicate backend"):
        suite.validate_suite(suite.load_records([*paths, duplicate]))

    with pytest.raises(suite.ConformanceSuiteError, match="unsupported required"):
        suite.validate_suite(suite.load_records(paths), ["unknown"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 7, "experiment.seed"),
        ("harness", "d" * 40, "harness_revision"),
        ("max_cost_usd", 2.0, "execution.max_cost_usd"),
    ],
)
def test_suite_rejects_cross_backend_semantic_drift(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    paths = [
        _write_evidence(tmp_path, "nemo-rl"),
        _write_evidence(tmp_path, "openrlhf"),
        _write_evidence(tmp_path, "verl", **{field: value}),
    ]
    with pytest.raises(suite.ConformanceSuiteError, match=message):
        suite.validate_suite(suite.load_records(paths))


def test_suite_rejects_artifact_tampering(tmp_path: Path) -> None:
    paths = _complete_roster(tmp_path)
    (paths[1].parent / "run" / "artifact" / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(suite.ConformanceSuiteError, match="artifact digest"):
        suite.load_records(paths)


def test_suite_report_never_overwrites(tmp_path: Path) -> None:
    records = suite.load_records(_complete_roster(tmp_path / "runs"))
    output = tmp_path / "suite.json"
    backend_conformance.write_json_once(output, suite.summarize(records))
    assert suite.main([str(tmp_path / "runs"), "--output", str(output)]) == 2
