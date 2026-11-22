"""Unit and API tests for the StateSet 3D Weight Studio."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from safetensors.numpy import load_file, save_file

from stateset_agents.weight_studio.editor import WeightEditEngine
from stateset_agents.weight_studio.finetune_engine import FinetuneStudioEngine
from stateset_agents.weight_studio.inference_engine import InferenceStudioEngine
from stateset_agents.weight_studio.inspector import (
    WeightInspector,
    compute_svd_spectrum,
    compute_tensor_stats,
    downsample_matrix,
)
from stateset_agents.weight_studio.router import StudioRuntime, create_studio_app


def _make_sample_workspace(tmp_path: Path) -> Path:
    """Create a realistic local workspace with a multi-checkpoint LoRA trajectory."""
    run_dir = tmp_path / "outputs" / "demo_lora_run"
    rng = np.random.default_rng(1234)
    adapter_cfg = {
        "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj", "gate_proj"],
    }
    for step_idx, sub in enumerate(["checkpoint-10", "checkpoint-20", ""]):
        target_dir = run_dir / sub if sub else run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "adapter_config.json").write_text(
            json.dumps(adapter_cfg), encoding="utf-8"
        )
        scale = 0.05 * (step_idx + 1)
        tensors: dict[str, np.ndarray] = {}
        for layer in range(3):
            for proj in ("self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj"):
                prefix = f"base_model.model.model.layers.{layer}.{proj}"
                tensors[f"{prefix}.lora_A.weight"] = rng.normal(
                    0.0, scale, size=(8, 32)
                ).astype(np.float32)
                tensors[f"{prefix}.lora_B.weight"] = rng.normal(
                    0.0, scale, size=(32, 8)
                ).astype(np.float32)
        save_file(tensors, str(target_dir / "adapter_model.safetensors"))
    return tmp_path


def test_inspector_discovers_local_trajectory_and_presets(tmp_path: Path) -> None:
    ws = _make_sample_workspace(tmp_path)
    inspector = WeightInspector(workspace_root=ws)
    models = inspector.list_models()

    local_models = [m for m in models if m["kind"] == "local_adapter"]
    preset_models = [m for m in models if m["kind"] == "preset_twin"]
    assert len(local_models) == 1
    assert len(preset_models) >= 4
    assert local_models[0]["num_layers"] == 3
    assert local_models[0]["lora_r"] == 8
    assert len(local_models[0]["checkpoints"]) == 3

    struct = inspector.get_model_structure(local_models[0]["model_id"])
    assert struct["summary"]["num_layers"] == 3
    assert struct["summary"]["num_tensors"] == 18

    traj = inspector.compute_trajectory_evolution(local_models[0]["model_id"])
    assert len(traj["steps"]) == 3
    assert traj["steps"][-1]["total_norm"] > traj["steps"][0]["total_norm"]


def test_tensor_stats_downsample_and_exact_lora_qr_svd() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 0.2, size=(6, 48)).astype(np.float32)
    b = rng.normal(0.0, 0.2, size=(64, 6)).astype(np.float32)
    full_delta = 2.0 * (b @ a)

    stats = compute_tensor_stats(full_delta)
    assert stats["count"] == 64 * 48
    assert stats["frobenius_norm"] > 0

    grid = downsample_matrix(full_delta, max_rows=16, max_cols=16)
    assert grid["rows"] == 16
    assert grid["cols"] == 16
    assert len(grid["values"]) == 256

    svd_qr = compute_svd_spectrum(b, lora_pair=(a, b, 2.0))
    exact_s = np.linalg.svd(full_delta, compute_uv=False)[:6]
    assert svd_qr["mode"] == "lora_exact_qr"
    assert svd_qr["rank"] == 6
    assert np.allclose(svd_qr["singular_values"], exact_s, atol=1e-4)


def test_surgical_editor_ops_undo_redo_and_verified_export(tmp_path: Path) -> None:
    ws = _make_sample_workspace(tmp_path)
    inspector = WeightInspector(workspace_root=ws)
    editor = WeightEditEngine(inspector)
    model_id = "local:outputs/demo_lora_run"
    tensor_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"

    orig = inspector.load_tensor_array(model_id, tensor_name).copy()

    # 1. Scale
    res = editor.apply_edit(model_id, "scale", tensor_name, {"factor": 0.5})
    assert res["edit"]["changed_weights"] == orig.size
    edited = editor.get_edited_tensor(model_id, tensor_name)
    assert edited is not None
    assert np.allclose(edited, orig * 0.5, atol=1e-6)

    # 2. Region brush zero
    editor.apply_edit(
        model_id,
        "region_edit",
        tensor_name,
        {"row_start": 0, "row_end": 4, "col_start": 0, "col_end": 4, "action": "zero"},
    )
    edited2 = editor.get_edited_tensor(model_id, tensor_name)
    assert edited2 is not None
    assert np.all(edited2[:4, :4] == 0.0)

    # 3. Prune & SVD truncate
    editor.apply_edit(
        model_id, "prune", "layer:1", {"mode": "percentile", "sparsity": 0.5}
    )
    editor.apply_edit(model_id, "svd_truncate", tensor_name, {"rank": 3})

    # 4. Undo / Redo
    h_undo = editor.undo(model_id)
    assert h_undo["can_redo"] is True
    h_redo = editor.redo(model_id)
    assert h_redo["can_redo"] is False

    # 5. Verified export to .safetensors
    manifest = editor.export_verified_checkpoint(model_id, "test_export_v1")
    assert manifest["reload_verified"] is True
    assert len(manifest["sha256"]) == 64
    out_file = ws / manifest["export_file"]
    assert out_file.is_file()
    reloaded = load_file(str(out_file))
    expected_final = editor.get_edited_tensor(model_id, tensor_name)
    assert expected_final is not None
    assert np.allclose(reloaded[tensor_name], expected_final, atol=1e-6)


@pytest.mark.asyncio
async def test_live_pytorch_micro_finetune_updates_weights(tmp_path: Path) -> None:
    ws = _make_sample_workspace(tmp_path)
    inspector = WeightInspector(workspace_root=ws)
    editor = WeightEditEngine(inspector)
    finetuner = FinetuneStudioEngine(inspector, editor)
    model_id = "local:outputs/demo_lora_run"

    plan = finetuner.estimate(
        model_id,
        {
            "algorithm": "gspo",
            "lora_r": 16,
            "precision": "bf16",
            "provider": "runpod_a40",
        },
    )
    assert plan["vram_gb"]["total"] > 0
    assert "train-remote" in plan["cli_command"]

    job = await finetuner.start_job(
        model_id,
        {
            "algorithm": "gspo",
            "max_steps": 4,
            "learning_rate": 1e-3,
            "target_modules": ["q_proj"],
        },
    )
    job_id = job["job_id"]
    for _ in range(40):
        st = finetuner.get_job(job_id)
        if st["status"] in {"completed", "failed", "stopped"}:
            break
        await asyncio.sleep(0.1)

    final_state = finetuner.get_job(job_id)
    assert final_state["status"] == "completed"
    assert final_state["current_step"] == 4
    assert len(final_state["history"]) == 4
    assert len(editor.get_overrides(model_id)) > 0


def test_inference_ab_comparison_responds_to_surgical_edits(tmp_path: Path) -> None:
    ws = _make_sample_workspace(tmp_path)
    inspector = WeightInspector(workspace_root=ws)
    editor = WeightEditEngine(inspector)
    infer = InferenceStudioEngine(inspector, editor)
    model_id = "local:outputs/demo_lora_run"

    base_run = infer.run_inference(model_id, "My order #77701 arrived damaged.")
    assert base_run["has_edits"] is False
    assert len(base_run["active"]["tokens"]) > 5
    assert len(base_run["active"]["layer_activations"]) == 3

    # Apply a strong surgical prune across all layers
    editor.apply_edit(
        model_id, "prune", "all", {"mode": "percentile", "sparsity": 0.85}
    )
    edited_run = infer.run_inference(
        model_id, "My order #77701 arrived damaged.", compare_base=True
    )
    assert edited_run["has_edits"] is True
    assert edited_run["comparison"]["hidden_l2_shift"] > 0.0
    assert edited_run["comparison"]["hidden_cosine_similarity"] < 0.9999


def test_studio_fastapi_routes_and_static_ui(tmp_path: Path) -> None:
    ws = _make_sample_workspace(tmp_path)
    app = create_studio_app(workspace_root=ws)
    client = TestClient(app)

    r_html = client.get("/studio")
    assert r_html.status_code == 200
    assert "STATESET" in r_html.text

    r_models = client.get("/api/weight-studio/models")
    assert r_models.status_code == 200
    payload = r_models.json()
    assert payload["count"] >= 5

    mid = "local:outputs/demo_lora_run"
    r_struct = client.get(
        "/api/weight-studio/models/structure", params={"model_id": mid}
    )
    assert r_struct.status_code == 200
    first_tensor = r_struct.json()["layers"][0]["tensors"][0]["name"]

    r_tensor = client.get(
        "/api/weight-studio/models/tensor",
        params={"model_id": mid, "tensor_name": first_tensor, "resolution": 16},
    )
    assert r_tensor.status_code == 200
    assert r_tensor.json()["svd"]["mode"] == "lora_exact_qr"

    r_edit = client.post(
        "/api/weight-studio/edits/apply",
        json={
            "model_id": mid,
            "op": "scale",
            "target": first_tensor,
            "params": {"factor": 1.5},
        },
    )
    assert r_edit.status_code == 200
    assert r_edit.json()["edit"]["changed_weights"] > 0

    r_inf = client.post(
        "/api/weight-studio/inference/run",
        json={"model_id": mid, "prompt": "Check inventory for SKU-4092"},
    )
    assert r_inf.status_code == 200
    assert r_inf.json()["has_edits"] is True
    assert StudioRuntime(workspace_root=ws).inspector.workspace_root == ws.resolve()
