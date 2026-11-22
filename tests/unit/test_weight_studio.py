"""Unit and API tests for the StateSet 3D Weight Studio (v2)."""

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
    compute_wanda_fisher_importance,
    downsample_matrix,
)
from stateset_agents.weight_studio.router import StudioRuntime, create_studio_app


def _make_sample_workspace(tmp_path: Path) -> Path:
    """Create a realistic local workspace with multi-checkpoint LoRA, sharded model, real GPT-2, and JSONL dataset."""
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

    # Second adapter for TIES/DARE/SLERP merging & Sign Conflict tests
    run_dir2 = tmp_path / "outputs" / "demo_lora_alt"
    run_dir2.mkdir(parents=True, exist_ok=True)
    (run_dir2 / "adapter_config.json").write_text(
        json.dumps(adapter_cfg), encoding="utf-8"
    )
    rng2 = np.random.default_rng(9876)
    tensors2: dict[str, np.ndarray] = {}
    for layer in range(3):
        for proj in ("self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj"):
            prefix = f"base_model.model.model.layers.{layer}.{proj}"
            tensors2[f"{prefix}.lora_A.weight"] = rng2.normal(
                0.0, 0.12, size=(8, 32)
            ).astype(np.float32)
            tensors2[f"{prefix}.lora_B.weight"] = rng2.normal(
                0.0, 0.12, size=(32, 8)
            ).astype(np.float32)
    save_file(tensors2, str(run_dir2 / "adapter_model.safetensors"))

    # Multi-shard .safetensors.index.json model
    shard_dir = tmp_path / "outputs" / "demo_sharded"
    shard_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": rng.normal(
                0, 0.1, (16, 16)
            ).astype(np.float32)
        },
        str(shard_dir / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {
            "model.layers.1.mlp.down_proj.weight": rng.normal(0, 0.1, (16, 16)).astype(
                np.float32
            )
        },
        str(shard_dir / "model-00002-of-00002.safetensors"),
    )
    (shard_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 2048},
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.1.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    # Curated JSONL dataset with rewards
    ds_file = tmp_path / "outputs" / "curated_sft.jsonl"
    ds_rows = [
        {
            "messages": [
                {"role": "user", "content": "Refund order #77701"},
                {"role": "assistant", "content": "Verified and refunded."},
            ],
            "reward": 0.94,
        },
        {
            "messages": [
                {"role": "user", "content": "Check SKU-4092"},
                {"role": "assistant", "content": '{"tool":"inventory_check"}'},
            ],
            "reward": 0.88,
        },
    ]
    ds_file.write_text(
        "\n".join(json.dumps(r) for r in ds_rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_inspector_discovers_local_trajectory_sharded_and_presets(
    tmp_path: Path,
) -> None:
    ws = _make_sample_workspace(tmp_path)
    inspector = WeightInspector(workspace_root=ws)
    models = inspector.list_models()

    local_adapters = [m for m in models if m["kind"] == "local_adapter"]
    sharded_models = [m for m in models if m["kind"] == "local_sharded"]
    preset_models = [m for m in models if m["kind"] == "preset_twin"]
    assert len(local_adapters) == 2
    assert len(sharded_models) == 1
    assert len(preset_models) >= 4

    # Verify lazy sharded loading works cleanly
    sharded_id = sharded_models[0]["model_id"]
    sh_struct = inspector.get_model_structure(sharded_id)
    assert sh_struct["summary"]["num_layers"] == 2
    arr = inspector.load_tensor_array(sharded_id, "model.layers.1.mlp.down_proj.weight")
    assert arr.shape == (16, 16)

    # Verify multi-checkpoint trajectory
    traj = inspector.compute_trajectory_evolution("local:outputs/demo_lora_run")
    assert len(traj["steps"]) == 3
    assert traj["steps"][-1]["total_norm"] > traj["steps"][0]["total_norm"]


def test_tensor_stats_wanda_binary_payload_and_exact_lora_qr_svd(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 0.2, size=(6, 48)).astype(np.float32)
    b = rng.normal(0.0, 0.2, size=(64, 6)).astype(np.float32)
    full_delta = 2.0 * (b @ a)

    stats = compute_tensor_stats(full_delta)
    assert stats["count"] == 64 * 48

    imp = compute_wanda_fisher_importance(full_delta, seed_key="test_tensor")
    assert imp.shape == full_delta.shape
    assert float(np.max(imp)) > 0.0

    grid = downsample_matrix(full_delta, max_rows=16, max_cols=16)
    assert grid["rows"] == 16 and grid["cols"] == 16

    svd_qr = compute_svd_spectrum(b, lora_pair=(a, b, 2.0))
    exact_s = np.linalg.svd(full_delta, compute_uv=False)[:6]
    assert svd_qr["mode"] == "lora_exact_qr"
    assert np.allclose(svd_qr["singular_values"], exact_s, atol=1e-4)

    ws = _make_sample_workspace(tmp_path)
    inspector = WeightInspector(workspace_root=ws)
    raw_bytes, meta = inspector.build_binary_tensor_payload(
        "local:outputs/demo_lora_run",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        resolution=32,
        secondary_model_id="local:outputs/demo_lora_alt",
    )
    assert "values" in meta["channels"]
    assert "importance" in meta["channels"]
    assert "conflict" in meta["channels"]
    expected_floats = meta["rows"] * meta["cols"] * len(meta["channels"])
    assert len(raw_bytes) == expected_floats * 4


def test_advanced_surgical_ops_head_rome_ties_dare_slerp_and_merge_probe(
    tmp_path: Path,
) -> None:
    ws = _make_sample_workspace(tmp_path)
    inspector = WeightInspector(workspace_root=ws)
    editor = WeightEditEngine(inspector)
    model_id = "local:outputs/demo_lora_run"
    alt_id = "local:outputs/demo_lora_alt"
    tensor_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"

    # 1. Per-Head Attention Ablation
    r_head = editor.apply_edit(
        model_id,
        "head_edit",
        tensor_name,
        {"head_index": 0, "num_heads": 4, "axis": 0, "action": "ablate"},
    )
    assert r_head["edit"]["diagnostics"]["head_index"] == 0
    edited_h = editor.get_edited_tensor(model_id, tensor_name)
    assert edited_h is not None
    assert np.all(edited_h[:8, :] == 0.0)

    # 2. ROME Rank-1 Associative Edit with Protected-Prompt Drift Verification
    r_rome = editor.apply_edit(
        model_id,
        "rank1_edit",
        tensor_name,
        {"subject": "Order #77701", "target_fact": "Approved refund", "strength": 0.45},
    )
    assert "protected_mean_cosine" in r_rome["edit"]["diagnostics"]
    assert r_rome["edit"]["changed_weights"] > 0

    # 3. Wanda/Fisher Pruning + TIES, DARE, SLERP Merging
    editor.apply_edit(model_id, "prune", "layer:1", {"mode": "wanda", "sparsity": 0.4})
    r_ties = editor.apply_edit(
        model_id,
        "blend",
        tensor_name,
        {"secondary_model_id": alt_id, "method": "ties", "alpha": 0.5, "density": 0.7},
    )
    assert "conflict_rate" in r_ties["edit"]["diagnostics"]
    editor.apply_edit(
        model_id,
        "blend",
        tensor_name,
        {
            "secondary_model_id": alt_id,
            "method": "dare",
            "alpha": 0.5,
            "drop_rate": 0.3,
        },
    )
    editor.apply_edit(
        model_id,
        "blend",
        tensor_name,
        {"secondary_model_id": alt_id, "method": "slerp", "alpha": 0.5},
    )

    # 4. Undo / Redo
    assert editor.undo(model_id)["can_redo"] is True
    assert editor.redo(model_id)["can_redo"] is False

    # 5. Verified Export + merge_adapter Effect Probe
    manifest = editor.export_verified_checkpoint(
        model_id, "test_export_v2", run_merge_probe=True
    )
    assert manifest["reload_verified"] is True
    assert manifest["merge_probe"] is not None
    assert manifest["merge_probe"]["effect_verified"] is True
    assert manifest["merge_probe"]["composite_keys_remapped"] > 0
    out_file = ws / manifest["export_file"]
    reloaded = load_file(str(out_file))
    expected_final = editor.get_edited_tensor(model_id, tensor_name)
    assert expected_final is not None
    assert np.allclose(reloaded[tensor_name], expected_final, atol=1e-6)


@pytest.mark.asyncio
async def test_live_pytorch_micro_finetune_and_dataset_inspection(
    tmp_path: Path,
) -> None:
    ws = _make_sample_workspace(tmp_path)
    inspector = WeightInspector(workspace_root=ws)
    editor = WeightEditEngine(inspector)
    finetuner = FinetuneStudioEngine(inspector, editor)
    model_id = "local:outputs/demo_lora_run"

    datasets = finetuner.list_datasets()
    assert len(datasets) == 1
    ds_info = finetuner.inspect_dataset(datasets[0]["path"])
    assert ds_info["total_rows"] == 2
    assert ds_info["mean_reward"] == pytest.approx(0.91, abs=1e-3)

    plan = finetuner.estimate(
        model_id,
        {
            "algorithm": "gspo",
            "lora_r": 16,
            "precision": "bf16",
            "provider": "runpod_a40",
            "dataset_path": datasets[0]["path"],
        },
    )
    assert plan["vram_gb"]["total"] > 0
    assert datasets[0]["path"] in plan["cli_command"]

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
    assert len(editor.get_overrides(model_id)) > 0


def test_inference_real_pytorch_hooks_logit_lens_and_causal_patching(
    tmp_path: Path,
) -> None:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    ws = _make_sample_workspace(tmp_path)
    # Create a real tiny GPT-2 checkpoint to test genuine PyTorch register_forward_hook execution!
    tiny_dir = ws / "outputs" / "tiny_gpt2"
    torch.manual_seed(42)
    cfg = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=2)
    GPT2LMHeadModel(cfg).save_pretrained(str(tiny_dir), safe_serialization=True)

    inspector = WeightInspector(workspace_root=ws)
    editor = WeightEditEngine(inspector)
    infer = InferenceStudioEngine(inspector, editor)
    gpt2_id = "local:outputs/tiny_gpt2"

    # Run with real PyTorch forward hooks + causal ablation on Layer 0!
    res = infer.run_inference(
        gpt2_id,
        "Check order #77701 refund",
        causal_patches=[{"layer": 0, "action": "ablate"}],
        compare_base=True,
    )
    assert res["execution_backend"] == "pytorch_hooks"
    assert len(res["active"]["logit_lens"]) == 2
    assert res["active"]["logit_lens"][0]["patched"] is True
    assert res["comparison"]["hidden_l2_shift"] > 0.0


def test_studio_fastapi_v2_routes_and_binary_stream(tmp_path: Path) -> None:
    ws = _make_sample_workspace(tmp_path)
    app = create_studio_app(workspace_root=ws)
    client = TestClient(app)

    assert client.get("/studio").status_code == 200
    assert client.get("/api/weight-studio/finetune/datasets").json()["count"] == 1

    mid = "local:outputs/demo_lora_run"
    r_struct = client.get(
        "/api/weight-studio/models/structure", params={"model_id": mid}
    )
    first_tensor = r_struct.json()["layers"][0]["tensors"][0]["name"]

    r_bin = client.get(
        "/api/weight-studio/models/tensor.bin",
        params={"model_id": mid, "tensor_name": first_tensor, "resolution": 32},
    )
    assert r_bin.status_code == 200
    meta = json.loads(r_bin.headers["X-Tensor-Meta"])
    assert len(r_bin.content) == meta["rows"] * meta["cols"] * len(meta["channels"]) * 4

    r_rome = client.post(
        "/api/weight-studio/edits/apply",
        json={
            "model_id": mid,
            "op": "rank1_edit",
            "target": first_tensor,
            "params": {"subject": "SKU-4092", "target_fact": "In stock"},
        },
    )
    assert r_rome.status_code == 200
    assert "protected_mean_cosine" in r_rome.json()["edit"]["diagnostics"]

    r_inf = client.post(
        "/api/weight-studio/inference/run",
        json={
            "model_id": mid,
            "prompt": "Check inventory for SKU-4092",
            "causal_patches": [{"layer": 1, "action": "ablate"}],
        },
    )
    assert r_inf.status_code == 200
    assert len(r_inf.json()["active"]["logit_lens"]) == 3
    assert StudioRuntime(workspace_root=ws).inspector.workspace_root == ws.resolve()


def test_studio_v3_circuit_steering_eval_gate_and_recipes(tmp_path: Path) -> None:
    ws = _make_sample_workspace(tmp_path)
    app = create_studio_app(workspace_root=ws)
    client = TestClient(app)
    mid = "local:outputs/demo_lora_run"

    # 1. Causal Circuit Discovery & 3D Attention-Flow Arcs
    r_circ = client.post(
        "/api/weight-studio/inference/circuit",
        json={
            "model_id": mid,
            "prompt": "Customer says: My order #77701 arrived damaged. Can you refund me?",
            "top_k": 8,
        },
    )
    assert r_circ.status_code == 200
    circ = r_circ.json()
    assert len(circ["top_nodes"]) == 8
    assert len(circ["arcs"]) > 4
    assert len(circ["tokens"]) >= 4

    # 2. Contrastive Activation Steering (CAA / RepE) extraction, live inference & weight baking
    r_presets = client.get("/api/weight-studio/steering/presets")
    assert r_presets.status_code == 200
    presets = r_presets.json()["presets"]
    assert len(presets) >= 4

    r_ext = client.post(
        "/api/weight-studio/steering/extract",
        json={
            "model_id": mid,
            "name": presets[0]["name"],
            "positive_prompt": presets[0]["positive"],
            "negative_prompt": presets[0]["negative"],
        },
    )
    assert r_ext.status_code == 200
    vec = r_ext.json()
    assert vec["vector_id"].startswith("sv_")
    assert vec["peak_separation"] > 0.0

    r_steered = client.post(
        "/api/weight-studio/inference/run",
        json={
            "model_id": mid,
            "prompt": "My order #77701 arrived damaged.",
            "steering": {
                "vector_id": vec["vector_id"],
                "strength": 1.2,
                "layer": vec["recommended_layer"],
            },
        },
    )
    assert r_steered.status_code == 200
    assert r_steered.json()["comparison"]["hidden_l2_shift"] > 0.0

    r_bake = client.post(
        "/api/weight-studio/steering/bake",
        json={
            "model_id": mid,
            "vector_id": vec["vector_id"],
            "strength": 0.75,
        },
    )
    assert r_bake.status_code == 200
    assert r_bake.json()["edit"]["changed_weights"] > 0

    # 3. Portable Edit Recipe Export & Deterministic Replay
    r_rec = client.get("/api/weight-studio/edits/recipe", params={"model_id": mid})
    assert r_rec.status_code == 200
    recipe = r_rec.json()
    assert recipe["format"] == "stateset-weight-studio-recipe"
    assert len(recipe["operations"]) == 1

    alt_id = "local:outputs/demo_lora_alt"
    r_replay = client.post(
        "/api/weight-studio/edits/recipe/apply",
        json={"model_id": alt_id, "recipe": recipe, "reset_first": True},
    )
    assert r_replay.status_code == 200
    assert r_replay.json()["applied_count"] == 1

    # 4. Pre-Export Batch Eval Gate Scorecard
    r_eval = client.post(
        "/api/weight-studio/eval/run",
        json={"model_id": mid, "max_cases": 6},
    )
    assert r_eval.status_code == 200
    ev = r_eval.json()
    assert ev["verdict"] in {"GO", "WARN", "BLOCK"}
    assert ev["case_count"] == 6
    assert len(ev["cases"]) == 6
