"""Replay locality evidence and reject forged metrics, objectives and splits."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

from benchmarks.weight_space.constrained_edit_job import run as run_base  # noqa: E402
from benchmarks.weight_space.locality_edit_job import run  # noqa: E402
from benchmarks.weight_space.verify_locality_edit import verify  # noqa: E402


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("locality")
    run_base(root / "base")
    run(root / "base", root / "weighted")
    return root / "weighted"


def rehash(root, name):
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_independent_replay_checks_original_controls_and_new_objective(evidence):
    result = verify(evidence)
    assert result["passed"]
    assert result["independent_kkt_checks"] == 30
    assert result["original_control_weight_equalities"] == 20
    assert result["fresh_confirmation_inputs"] == 4096
    with pytest.raises(ValueError, match="fresh output directory"):
        run(evidence / "base", evidence)


@pytest.mark.parametrize(
    "damage", ["weight", "coverage", "strength", "metric", "split", "objective"]
)
def test_rehashed_evidence_tampering_is_rejected(evidence, tmp_path, damage):
    out = tmp_path / "tampered"
    shutil.copytree(evidence, out)
    if damage in ("weight", "split"):
        name = "states.safetensors" if damage == "weight" else "inputs.safetensors"
        tensors = load_file(str(out / name))
        if damage == "weight":
            tensors["s0.r0.locality_weighted.weight"][0, 0] += 1
        else:
            tensors["confirmation"][0] = tensors["locality"][0]
        save_file(tensors, str(out / name))
    else:
        name = "protocol.json" if damage == "strength" else "trials.json"
        data = json.loads((out / name).read_text())
        if damage == "strength":
            data["locality_weight"] = 0
        elif damage == "coverage":
            data.pop()
        elif damage == "objective":
            data[0]["diagnostics"]["objective_value"] = 0
        else:
            data[0]["confirmation_mse_drift"] = 0
        (out / name).write_text(json.dumps(data))
    rehash(out, name)
    with pytest.raises(ValueError):
        verify(out)


def test_feasible_unweighted_edit_with_recomputed_metrics_fails_optimality(
    evidence, tmp_path
):
    """A feasible edit plus truthful metrics is insufficient to prove the objective."""
    out = tmp_path / "wrong-objective"
    shutil.copytree(evidence, out)
    states = load_file(str(out / "states.safetensors"))
    for part in ("weight", "delta"):
        states[f"s0.r0.locality_weighted.{part}"] = states[
            f"s0.r0.minimum_norm.{part}"
        ].clone()
    save_file(states, str(out / "states.safetensors"))
    trials = json.loads((out / "trials.json").read_text())
    plain = next(
        t for t in trials if (t["seed"], t["round"], t["arm"]) == (0, 0, "minimum_norm")
    )
    weighted = next(
        t
        for t in trials
        if (t["seed"], t["round"], t["arm"]) == (0, 0, "locality_weighted")
    )
    for name in plain:
        if name not in ("arm", "diagnostics"):
            weighted[name] = plain[name]
    # The outputs and preservation checks truthfully match the substituted
    # edit; only an independent weighted optimality check catches it.
    (out / "trials.json").write_text(json.dumps(trials))
    rehash(out, "states.safetensors")
    rehash(out, "trials.json")
    with pytest.raises(ValueError, match="independent KKT solution"):
        verify(out)
