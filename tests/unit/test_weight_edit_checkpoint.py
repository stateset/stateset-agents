"""Checkpoint-level weight-edit tests on tiny synthetic safetensors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from stateset_agents.training.weight_edit import (
    MANIFEST_NAME,
    apply_safetensors_delta,
    inspect_checkpoint,
    merge_lora_adapter,
)


def _write_base(root: Path) -> Path:
    root.mkdir(parents=True)
    tensors = {
        "layers.0.q_proj.weight": torch.eye(3),
        "layers.0.mlp.weight": torch.ones(2, 2),
    }
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text('{"model_type": "tiny"}')
    return root


def _write_adapter(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    save_file(
        {
            "base_model.model.layers.0.q_proj.lora_A.weight": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            "base_model.model.layers.0.q_proj.lora_B.weight": torch.tensor(
                [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]
            ),
        },
        str(root / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    return root


def test_inspect_checkpoint_reports_shapes_and_shards(tmp_path: Path) -> None:
    base = _write_base(tmp_path / "base")

    info = inspect_checkpoint(base)

    assert info["layers.0.q_proj.weight"].shape == (3, 3)
    assert info["layers.0.q_proj.weight"].shard == "model.safetensors"
    assert info["layers.0.mlp.weight"].dtype == "F32"


def test_merge_lora_adapter_folds_scaled_update(tmp_path: Path) -> None:
    base = _write_base(tmp_path / "base")
    adapter = _write_adapter(tmp_path / "adapter")
    out = tmp_path / "merged"

    manifest = merge_lora_adapter(base, adapter, out)

    merged = load_file(str(out / "model.safetensors"))
    # scale = alpha / r = 2; W + 2 * (B @ A) adds diag(2, 4, 0).
    expected = torch.eye(3) + torch.diag(torch.tensor([2.0, 4.0, 0.0]))
    assert torch.equal(merged["layers.0.q_proj.weight"], expected)
    assert torch.equal(merged["layers.0.mlp.weight"], torch.ones(2, 2))
    assert (out / "config.json").exists()
    written = json.loads((out / MANIFEST_NAME).read_text())
    assert written == manifest
    assert manifest["operation"] == "merge_lora_adapter"
    assert [e["name"] for e in manifest["changed_tensors"]] == [
        "layers.0.q_proj.weight"
    ]
    entry = manifest["changed_tensors"][0]
    assert entry["delta_max_abs"] == pytest.approx(4.0)
    assert len(entry["sha256"]) == 64


def test_apply_safetensors_delta_scales_update(tmp_path: Path) -> None:
    base = _write_base(tmp_path / "base")
    delta_path = tmp_path / "delta.safetensors"
    save_file(
        {"layers.0.mlp.weight": torch.full((2, 2), 0.5)},
        str(delta_path),
        metadata={"format": "pt"},
    )

    manifest = apply_safetensors_delta(base, delta_path, tmp_path / "out", alpha=2.0)

    edited = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert torch.equal(edited["layers.0.mlp.weight"], torch.full((2, 2), 2.0))
    assert manifest["operation"] == "safetensors_delta"


def test_edits_reject_in_place_and_unknown_tensors(tmp_path: Path) -> None:
    base = _write_base(tmp_path / "base")
    adapter = _write_adapter(tmp_path / "adapter")
    with pytest.raises(ValueError, match="in-place"):
        merge_lora_adapter(base, adapter, base)

    delta_path = tmp_path / "delta.safetensors"
    save_file(
        {"missing.weight": torch.ones(2, 2)},
        str(delta_path),
        metadata={"format": "pt"},
    )
    with pytest.raises(ValueError, match="missing.weight"):
        apply_safetensors_delta(base, delta_path, tmp_path / "out")


def test_merge_rejects_shape_mismatch(tmp_path: Path) -> None:
    base = _write_base(tmp_path / "base")
    adapter = _write_adapter(tmp_path / "adapter")
    save_file(
        {
            "base_model.model.layers.0.q_proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.layers.0.q_proj.lora_B.weight": torch.ones(3, 2),
        },
        str(adapter / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )

    with pytest.raises(ValueError, match="shape"):
        merge_lora_adapter(base, adapter, tmp_path / "out")
