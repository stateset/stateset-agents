"""Replay evidence and reject altered circuits even after artifact re-hashing."""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

from benchmarks.weight_space import selective_relu_job as job  # noqa: E402
from benchmarks.weight_space.verify_selective_relu import verify  # noqa: E402


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "ARCHITECTURES", ((8,),))
    monkeypatch.setattr(job, "SEEDS", (0,))
    monkeypatch.setattr(job, "TRAIN_STEPS", 3)
    monkeypatch.setattr(job, "TRAIN_CASES", 16)
    directory = tmp_path / "evidence"
    job.run(directory)
    return directory


def rehash(directory, name: str) -> None:
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_replay_and_both_kinds_of_certificate(evidence) -> None:
    result = verify(evidence)
    assert result["passed"]
    assert result["variants"] == 3
    assert result["target_predictions"] == 3 * (33**2 + 4096)
    assert result["protected_values_equal"] == 2 * result["target_predictions"]
    assert all(
        c["arithmetic"]["real_arithmetic_identity"] for c in result["certificates"]
    )
    assert all(
        c["preservation"]["real_arithmetic_preservation"]
        for c in result["certificates"]
    )
    with pytest.raises(ValueError, match="new output directory"):
        job.run(evidence)


@pytest.mark.parametrize(
    "damage",
    [
        "protected",
        "target",
        "missing_condition",
        "ledger",
        "training",
        "overlap",
        "hash",
    ],
)
def test_tampered_evidence_rejected(evidence, damage: str) -> None:
    if damage in ("protected", "target"):
        name = "checkpoints.safetensors"
        tensors = load_file(str(evidence / name))
        tensors["8__0.maximum.2.bias"][0 if damage == "protected" else 1] += 0.5
        save_file(tensors, str(evidence / name))
    elif damage == "overlap":
        name = "inputs.safetensors"
        tensors = load_file(str(evidence / name))
        tensors["training"][0] = tensors["evaluation"][0]
        save_file(tensors, str(evidence / name))
    else:
        name = {
            "missing_condition": "summary.json",
            "hash": "summary.json",
            "ledger": "edit_ledgers.json",
            "training": "training.json",
        }[damage]
        data = json.loads((evidence / name).read_text())
        if damage in ("missing_condition", "hash"):
            data["variants"].pop()
        elif damage == "ledger":
            data["8__0.maximum"][0]["before"] += 1
        elif damage == "training":
            data[0]["history"][-1]["mse_by_output"][0] += 1
        (evidence / name).write_text(json.dumps(data))
    if damage != "hash":
        rehash(evidence, name)
    with pytest.raises(ValueError):
        verify(evidence)
