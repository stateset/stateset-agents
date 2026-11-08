"""The independent audit must reject altered evidence even after re-hashing."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

from benchmarks.weight_space.constrained_edit_job import run  # noqa: E402
from benchmarks.weight_space.verify_constrained_edit import verify  # noqa: E402


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    out = tmp_path_factory.mktemp("constrained") / "evidence"
    run(out)
    return out


def test_full_replay_checks_all_seeds_rounds_and_controls(original):
    result = verify(original)
    assert result["passed"]
    assert result["trials"] == 30
    assert result["independent_minimum_norm_checks"] == 10
    assert result["protected_span_checks"] == 10
    with pytest.raises(ValueError, match="fresh output"):
        run(original)


@pytest.mark.parametrize(
    "damage", ["weight", "coverage", "tolerance", "metric", "overlap"]
)
def test_rehashed_tampering_is_rejected(original, tmp_path, damage):
    out = tmp_path / "changed"
    shutil.copytree(original, out)
    if damage in ("weight", "overlap"):
        name = "states.safetensors" if damage == "weight" else "inputs.safetensors"
        tensors = load_file(str(out / name))
        if damage == "weight":
            tensors["s0.r0.constrained.weight"][0, 0] += 1
        else:
            tensors["reference"][0] = tensors["train"][0]
        save_file(tensors, str(out / name))
    else:
        name = "protocol.json" if damage == "tolerance" else "trials.json"
        data = json.loads((out / name).read_text())
        if damage == "coverage":
            data.pop()
        elif damage == "tolerance":
            data["target_atol"] = 1
        else:
            data[0]["heldout_max_drift"] = 0
        (out / name).write_text(json.dumps(data))
    manifest = json.loads((out / "artifact_manifest.json").read_text())
    manifest[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    (out / "artifact_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(out)
