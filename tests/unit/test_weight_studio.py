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


def test_weight_studio_quickstart_example_and_cli_help(tmp_path: Path) -> None:
    import importlib.util

    from typer.testing import CliRunner

    from stateset_agents.cli import app as cli_app

    ws = _make_sample_workspace(tmp_path)
    recipe_file = tmp_path / "quickstart.studio-recipe.json"

    example_path = (
        Path(__file__).resolve().parents[2] / "examples" / "weight_studio_quickstart.py"
    )
    spec = importlib.util.spec_from_file_location("ws_quickstart", example_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    summary = mod.run_quickstart(ws, export_name="qs-verified", recipe_out=recipe_file)
    assert summary["eval_verdict"] in {"GO", "WARN", "BLOCK"}
    assert summary["recipe_ops"] == 2
    assert summary["circuit_nodes"] == 8
    assert summary["manifest"]["reload_verified"] is True
    assert recipe_file.is_file()

    cli_result = CliRunner().invoke(cli_app, ["studio", "--help"])
    assert cli_result.exit_code == 0
    assert "--workspace" in cli_result.output
    assert "--port" in cli_result.output


def test_studio_v4_cross_checkpoint_sae_dispatch_and_mcp_tools(tmp_path: Path) -> None:
    from stateset_agents.mcp_server import (
        studio_apply_edit,
        studio_discover_circuit,
        studio_eval_and_export,
        studio_extract_steering,
        studio_inspect_model,
        studio_list_models,
    )

    ws = _make_sample_workspace(tmp_path)
    app = create_studio_app(workspace_root=ws)
    client = TestClient(app)
    mid = "local:outputs/demo_lora_run"
    alt_id = "local:outputs/demo_lora_alt"

    # 1. Cross-Checkpoint / Cross-Model secondary_grid + binary channel for Split A|B
    r_struct = client.get(
        "/api/weight-studio/models/structure", params={"model_id": mid}
    )
    first_tensor = r_struct.json()["layers"][0]["tensors"][0]["name"]
    r_diff = client.get(
        "/api/weight-studio/models/tensor",
        params={
            "model_id": mid,
            "tensor_name": first_tensor,
            "secondary_model_id": mid,
            "secondary_checkpoint_id": "checkpoint-100",
            "resolution": 16,
        },
    )
    assert r_diff.status_code == 200
    assert r_diff.json()["secondary_grid"] is not None

    r_bin = client.get(
        "/api/weight-studio/models/tensor.bin",
        params={
            "model_id": mid,
            "tensor_name": first_tensor,
            "secondary_model_id": alt_id,
            "resolution": 16,
        },
    )
    assert r_bin.status_code == 200
    meta = json.loads(r_bin.headers["X-Tensor-Meta"])
    assert "secondary" in meta["channels"]

    # 2. SAE Monosemantic Feature Explorer & Live Feature Clamping
    r_sae = client.post(
        "/api/weight-studio/sae/analyze",
        json={
            "model_id": mid,
            "prompt": "My order #77701 arrived damaged and I was billed twice.",
            "top_k": 8,
            "clamps": [{"feature_id": "F00", "value": 2.5}],
        },
    )
    assert r_sae.status_code == 200
    sae = r_sae.json()
    assert sae["dictionary_size"] == 24
    assert len(sae["features"]) == 8
    f00 = [f for f in sae["features"] if f["feature_id"] == "F00"]
    assert f00 and f00[0]["clamped"] is True and f00[0]["activation"] == 2.5

    # 3. Managed Training Dispatch (Dry-Run Plan Validation + Hot-Reload)
    r_disp = client.post(
        "/api/weight-studio/finetune/dispatch",
        json={
            "model_id": mid,
            "mode": "dry_run",
            "config": {"algorithm": "gspo", "provider": "runpod_a40", "lora_r": 16},
        },
    )
    assert r_disp.status_code == 200
    disp = r_disp.json()
    assert disp["status"] == "completed"
    assert disp["catalog_refreshed"] is True
    assert (ws / disp["plan_path"]).is_file()

    # 4. Weight Studio MCP Server Tools (plain functions testable without optional mcp extra)
    m_list = studio_list_models(workspace=str(ws))
    assert m_list["count"] >= 2
    m_insp = studio_inspect_model(mid, workspace=str(ws))
    assert m_insp["summary"]["num_layers"] == 3
    m_steer = studio_extract_steering(
        mid,
        "I will help verify your refund right away.",
        "Request denied.",
        workspace=str(ws),
    )
    assert m_steer["vector_id"].startswith("sv_")
    m_edit = studio_apply_edit(
        mid, "scale", first_tensor, {"factor": 1.05}, workspace=str(ws)
    )
    assert m_edit["edit"]["changed_weights"] > 0
    m_circ = studio_discover_circuit(mid, "Check SKU-4092 stock", workspace=str(ws))
    assert len(m_circ["top_nodes"]) == 10
    m_exp = studio_eval_and_export(
        mid, export_name="mcp-verified-v4", require_go=False, workspace=str(ws)
    )
    assert m_exp["exported"] is True
    assert m_exp["manifest"]["reload_verified"] is True


def test_studio_v5_hf_range_quantize_sensitivity_telemetry_stream_and_html_report(
    tmp_path: Path,
) -> None:
    from safetensors.numpy import save_file

    ws = _make_sample_workspace(tmp_path)
    app = create_studio_app(workspace_root=ws)
    client = TestClient(app)
    mid = "local:outputs/demo_lora_run"

    # 1. Zero-Download Hugging Face HTTP-Range safetensors inspection (mocked byte range fetcher)
    remote_sf = tmp_path / "remote_mock.safetensors"
    rng = np.random.default_rng(7)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": rng.normal(
                0, 0.05, size=(16, 16)
            ).astype(np.float32),
            "model.layers.1.mlp.down_proj.weight": rng.normal(
                0, 0.05, size=(16, 16)
            ).astype(np.float32),
        },
        str(remote_sf),
    )
    remote_bytes = remote_sf.read_bytes()
    cfg_bytes = json.dumps({"model_type": "qwen2", "num_attention_heads": 4}).encode()

    def _mock_range_fetcher(
        url: str, start: int | None = None, end: int | None = None
    ) -> bytes:
        if url.endswith("config.json"):
            return cfg_bytes
        if url.endswith("model.safetensors"):
            if start is not None and end is not None:
                return remote_bytes[start : end + 1]
            return remote_bytes
        raise FileNotFoundError(url)

    inspector = WeightInspector(workspace_root=ws)
    hf_summary = inspector.register_remote_hf_model(
        "stateset/mock-70b-twin", range_fetcher=_mock_range_fetcher
    )
    assert hf_summary["kind"] == "remote_hf"
    assert hf_summary["num_layers"] == 2
    hf_arr = inspector.load_tensor_array(
        hf_summary["model_id"], "model.layers.0.self_attn.q_proj.weight"
    )
    assert hf_arr.shape == (16, 16)

    # 2. Quantization Lab (NF4, INT4 AWQ, FP8, INT8 + SmoothQuant SQNR diagnostics)
    r_struct = client.get(
        "/api/weight-studio/models/structure", params={"model_id": mid}
    )
    first_tensor = r_struct.json()["layers"][0]["tensors"][0]["name"]
    for scheme in ("nf4", "int4_awq", "fp8", "int8"):
        r_q = client.post(
            "/api/weight-studio/edits/apply",
            json={
                "model_id": mid,
                "op": "quantize",
                "target": first_tensor,
                "params": {"scheme": scheme, "smooth_outliers": True},
            },
        )
        assert r_q.status_code == 200
        diag = r_q.json()["edit"]["diagnostics"]
        assert diag["scheme"] == scheme
        assert diag["sqnr_db"] > 5.0

    # 3. Auto-Sensitivity Profiler & Adaptive Rank/Sparsity Recipe
    r_sens = client.post(
        "/api/weight-studio/models/sensitivity",
        json={"model_id": mid, "target_compression": 0.35},
    )
    assert r_sens.status_code == 200
    sens = r_sens.json()
    assert len(sens["layers"]) == 3
    assert sens["recommended_recipe"]["format"] == "stateset-weight-studio-recipe"

    # 4. Live Token-by-Token SSE Inference Stream
    r_stream = client.post(
        "/api/weight-studio/inference/stream",
        json={"model_id": mid, "prompt": "Check order #77701 refund", "max_tokens": 12},
    )
    assert r_stream.status_code == 200
    assert '"type": "token"' in r_stream.text
    assert '"type": "done"' in r_stream.text

    # 5. External Trainer Telemetry Sync (trainer_state.json)
    ts_file = ws / "outputs" / "demo_lora_run" / "trainer_state.json"
    ts_file.write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 10, "loss": 1.42, "reward": 0.45, "learning_rate": 5e-6},
                    {"step": 20, "loss": 0.91, "reward": 0.78, "learning_rate": 3e-6},
                ]
            }
        ),
        encoding="utf-8",
    )
    r_srcs = client.get("/api/weight-studio/finetune/telemetry-sources")
    assert r_srcs.status_code == 200
    assert r_srcs.json()["count"] >= 1
    r_sync = client.post(
        "/api/weight-studio/finetune/sync-telemetry",
        json={"path": r_srcs.json()["sources"][0]["path"]},
    )
    assert r_sync.status_code == 200
    assert r_sync.json()["points"] == 2

    # 6. Shareable Standalone Single-File Interactive HTML PR Report
    r_html = client.post(
        "/api/weight-studio/report/html",
        json={
            "model_id": mid,
            "tensor_name": first_tensor,
            "include_eval": True,
            "include_sensitivity": True,
        },
    )
    assert r_html.status_code == 200
    assert "text/html" in r_html.headers["content-type"]
    assert "STATESET 3D WEIGHT STUDIO" in r_html.text
    assert 'canvas id="c3d"' in r_html.text


def test_v6_diff_as_lora_and_gguf_export(tmp_path: Path) -> None:
    """Verify Studio v6 Diff-as-LoRA SVD extraction and GGUF v3 binary + Ollama Modelfile export."""
    import struct

    ws = _make_sample_workspace(tmp_path)
    insp = WeightInspector(workspace_root=ws)
    ed = WeightEditEngine(insp)
    mid = "local:outputs/demo_lora_run"

    # Apply a surgical edit so ΔW is non-zero
    ed.apply_edit(mid, "scale", "layer:0", {"factor": 1.25}, label="Scale L0")

    # 1. Diff-as-LoRA PEFT SVD Extraction (low-rank compression + exact full-rank)
    lora_manifest = ed.export_diff_as_lora(mid, "surgical-lora-test", rank=4, alpha=8)
    assert lora_manifest["format"] == "peft-lora-diff"
    assert lora_manifest["reload_verified"] is True
    assert lora_manifest["adapter_tensor_count"] >= 2
    assert 0.0 <= lora_manifest["mean_relative_error"] < 0.75
    assert (ws / lora_manifest["export_file"]).is_file()
    assert (ws / lora_manifest["export_dir"] / "adapter_config.json").is_file()

    exact_lora = ed.export_diff_as_lora(mid, "surgical-lora-exact", rank=16, alpha=32)
    assert exact_lora["reload_verified"] is True
    assert exact_lora["mean_relative_error"] < 1e-4

    # 2. GGUF v3 Binary Export + Ollama Modelfile
    for q in ("Q8_0", "F16"):
        gguf = ed.export_gguf_package(mid, f"gguf-test-{q.lower()}", quant_type=q)
        assert gguf["format"] == "gguf-v3"
        assert gguf["header_verified"] is True
        assert gguf["tensor_count"] >= 4
        gguf_file = ws / gguf["gguf_file"]
        assert gguf_file.is_file()
        raw = gguf_file.read_bytes()
        assert raw[:4] == b"GGUF"
        ver, t_cnt, kv_cnt = struct.unpack("<IQQ", raw[4:24])
        assert ver == 3
        assert t_cnt == gguf["tensor_count"]
        assert kv_cnt == gguf["kv_count"]
        assert (ws / gguf["modelfile"]).is_file()
        assert "ollama create" in gguf["ollama_command"]


def test_v6_custom_eval_arena_sae_attention_and_mcp(tmp_path: Path) -> None:
    """Verify Studio v6 Custom JSONL Eval Gate, Multi-Edit Arena, SAE Atlas Loader, Attention 3D & MCP tools."""
    from fastapi.testclient import TestClient
    from safetensors.numpy import save_file

    from stateset_agents.mcp_server import (
        studio_arena_compare,
        studio_export_lora,
    )

    ws = _make_sample_workspace(tmp_path)
    app = create_studio_app(workspace_root=ws)
    client = TestClient(app)
    mid = "local:outputs/demo_lora_run"

    # 1. Eval Rewards catalog + Custom JSONL Eval Gate with 95% CI & W/T/L
    r_rew = client.get("/api/weight-studio/eval/rewards")
    assert r_rew.status_code == 200
    assert any(r["id"] == "customer_support" for r in r_rew.json()["rewards"])

    custom_jsonl = (
        '{"prompt": "Customer says: order #99102 arrived cracked", "category": "Custom Support"}\n'
        "Can you check inventory for SKU-8841?\n"
    )
    r_eval = client.post(
        "/api/weight-studio/eval/run",
        json={
            "model_id": mid,
            "reward_fn": "customer_support",
            "custom_jsonl": custom_jsonl,
            "max_cases": 4,
        },
    )
    assert r_eval.status_code == 200
    ev = r_eval.json()
    assert ev["reward_fn"] == "customer_support"
    assert ev["verdict"] in {"GO", "WARN", "BLOCK"}
    assert "ci_95_low" in ev["paired_stats"]
    assert ev["cases"][0]["category"] == "Custom Support"

    # 2. Multi-Edit A/B Arena (must not mutate session edit history)
    r_arena = client.post(
        "/api/weight-studio/arena/compare",
        json={
            "model_id": mid,
            "prompt": "My order #77701 arrived damaged. Can you refund me?",
            "reward_fn": "composite",
        },
    )
    assert r_arena.status_code == 200
    arena = r_arena.json()
    assert len(arena["candidates"]) >= 4
    assert arena["winner"]
    # Confirm session history remained untouched (0 edits)
    r_struct = client.get(f"/api/weight-studio/models/structure?model_id={mid}")
    assert r_struct.status_code == 200
    assert r_struct.json()["edit_history"]["edits"] == []

    # 3. Pretrained SAE Dictionary Atlas + Local .safetensors SAE Decoder Loader
    r_dicts = client.get("/api/weight-studio/sae/dictionaries")
    assert r_dicts.status_code == 200
    assert any(d["id"] == "gemma_scope_2b" for d in r_dicts.json()["dictionaries"])

    r_load_atlas = client.post(
        "/api/weight-studio/sae/load", json={"source_id": "gemma_scope_2b"}
    )
    assert r_load_atlas.status_code == 200
    assert r_load_atlas.json()["feature_count"] == 24

    r_sae = client.post(
        "/api/weight-studio/sae/analyze",
        json={"model_id": mid, "prompt": "Check refund policy for damaged item"},
    )
    assert r_sae.status_code == 200
    assert "Gemma Scope" in r_sae.json()["sae_source"]

    # Local .safetensors SAE decoder test
    sae_file = ws / "outputs" / "tiny_sae.safetensors"
    rng = np.random.default_rng(7)
    save_file(
        {"W_dec": rng.normal(0.0, 1.0, size=(20, 32)).astype(np.float32)},
        str(sae_file),
    )
    r_load_local = client.post(
        "/api/weight-studio/sae/load",
        json={"source_id": "local", "local_path": "outputs/tiny_sae.safetensors"},
    )
    assert r_load_local.status_code == 200
    assert r_load_local.json()["source"] == "local_file"
    assert r_load_local.json()["feature_count"] == 20

    # 4. Multi-Head Causal Attention 3D Matrix API
    r_attn = client.post(
        "/api/weight-studio/inference/attention",
        json={
            "model_id": mid,
            "prompt": "My order #77701 arrived damaged after 9 days",
            "layer_index": 0,
            "num_heads": 8,
            "max_tokens": 8,
        },
    )
    assert r_attn.status_code == 200
    attn = r_attn.json()
    assert attn["num_heads"] == 8
    assert len(attn["heads"]) == 8
    mat0 = np.asarray(attn["heads"][0]["matrix"], dtype=np.float32)
    assert mat0.shape == (len(attn["tokens"]), len(attn["tokens"]))
    # Strict causal mask: upper triangle must be 0.0 and each row sums to ~1.0
    assert np.allclose(np.triu(mat0, k=1), 0.0, atol=1e-5)
    assert np.allclose(np.sum(mat0, axis=1), 1.0, atol=1e-3)

    # 5. Export endpoints (/export/lora & /export/gguf)
    r_lora = client.post(
        "/api/weight-studio/export/lora",
        json={"model_id": mid, "export_name": "api-lora-v6", "rank": 4, "alpha": 8},
    )
    assert r_lora.status_code == 200
    assert r_lora.json()["reload_verified"] is True

    r_gguf = client.post(
        "/api/weight-studio/export/gguf",
        json={"model_id": mid, "export_name": "api-gguf-v6", "quant_type": "Q8_0"},
    )
    assert r_gguf.status_code == 200
    assert r_gguf.json()["header_verified"] is True

    # 6. New v6 MCP Tools
    mcp_lora = studio_export_lora(
        mid, export_name="mcp-lora-v6", rank=4, alpha=8, workspace=str(ws)
    )
    assert mcp_lora["reload_verified"] is True

    mcp_arena = studio_arena_compare(
        mid,
        prompt="Refund order #77701 due to shipping damage",
        workspace=str(ws),
    )
    assert mcp_arena["winner"]
    assert len(mcp_arena["candidates"]) >= 4


def test_v7_residual_stream_auto_repair_hf_card_sessions_and_mcp(
    tmp_path: Path,
) -> None:
    """Verify Studio v7 Residual 3D stream, Auto-Repair, HF Model Card, Session Persistence & MCP tools."""
    from fastapi.testclient import TestClient

    from stateset_agents.mcp_server import (
        studio_residual_stream,
        studio_save_session,
    )

    ws = _make_sample_workspace(tmp_path)
    app = create_studio_app(workspace_root=ws)
    client = TestClient(app)
    mid = "local:outputs/demo_lora_run"

    # 1. Apply an aggressive destructive prune (95% all) so Eval Gate degrades, then test Residual 3D & Auto-Repair
    r_prune = client.post(
        "/api/weight-studio/edits/apply",
        json={
            "model_id": mid,
            "op": "prune",
            "target": "all",
            "params": {"sparsity": 0.95},
            "label": "Extreme Prune 95%",
        },
    )
    assert r_prune.status_code == 200

    # 2. Residual Stream 3D API (Base vs. Active Edited trajectory + Attention/MLP split + Logit Lens)
    r_res = client.post(
        "/api/weight-studio/inference/residual",
        json={
            "model_id": mid,
            "prompt": "My order #77701 arrived damaged after 9 days. Can you refund me?",
        },
    )
    assert r_res.status_code == 200
    res_data = r_res.json()
    assert res_data["has_edits"] is True
    assert res_data["layer_count"] >= 2
    assert len(res_data["stations"]) == res_data["layer_count"]
    st0 = res_data["stations"][0]
    assert len(st0["base_coord"]) == 2
    assert len(st0["edited_coord"]) == 2
    assert st0["edited_top_token"] and st0["edited_top_token"] != "—"
    assert st0["edited_top_prob"] > 0.0
    assert abs((st0["attn_ratio"] + st0["mlp_ratio"]) - 1.0) < 0.02

    # 3. Surgical Auto-Repair -> dials back destructive edit until Eval Gate hits GO
    r_rep = client.post(
        "/api/weight-studio/edits/auto-repair",
        json={"model_id": mid, "reward_fn": "composite", "max_steps": 4},
    )
    assert r_rep.status_code == 200
    rep = r_rep.json()
    assert rep["repaired"] is True
    assert rep["final_verdict"] == "GO"

    # 4. HF Model Card & Reproducible Recipe Export
    r_card = client.post(
        "/api/weight-studio/export/hf-card",
        json={
            "model_id": mid,
            "export_name": "hf-card-v7-test",
            "repo_id": "stateset/test-surgical-v7",
            "include_eval": True,
            "push_to_hub": False,
        },
    )
    assert r_card.status_code == 200
    card = r_card.json()
    assert card["repo_id"] == "stateset/test-surgical-v7"
    assert (ws / card["model_card_file"]).is_file()
    assert (ws / card["recipe_file"]).is_file()
    assert "base_model:" in card["card_preview"]
    assert "Pre-Export Batch Eval Gate" in card["card_preview"]

    # 5. Workspace Session Save, List & Restore
    r_scale = client.post(
        "/api/weight-studio/edits/apply",
        json={
            "model_id": mid,
            "op": "scale",
            "target": "layer:0",
            "params": {"factor": 1.15},
            "label": "Session Scale L0",
        },
    )
    assert r_scale.status_code == 200

    r_save = client.post(
        "/api/weight-studio/sessions/save",
        json={
            "session_name": "refund-v7-check",
            "model_id": mid,
            "metadata": {"prompt": "Check order #77701"},
        },
    )
    assert r_save.status_code == 200
    assert r_save.json()["operation_count"] >= 1

    r_list = client.get("/api/weight-studio/sessions")
    assert r_list.status_code == 200
    assert any(
        s["session_name"] == "refund-v7-check" for s in r_list.json()["sessions"]
    )

    # Reset edits then restore session and confirm recipe replayed cleanly
    client.post("/api/weight-studio/edits/reset", json={"model_id": mid})
    r_load = client.post(
        "/api/weight-studio/sessions/load",
        json={"session_name": "refund-v7-check"},
    )
    assert r_load.status_code == 200
    assert r_load.json()["replay"]["applied_count"] >= 1
    assert r_load.json()["metadata"]["prompt"] == "Check order #77701"

    # 6. New v7 MCP Tools (10 Studio MCP tools total)
    mcp_res = studio_residual_stream(
        mid,
        prompt="Refund order #77701 due to shipping damage",
        workspace=str(ws),
    )
    assert mcp_res["layer_count"] >= 2

    mcp_sess = studio_save_session(
        "mcp-session-v7",
        mid,
        metadata={"source": "mcp"},
        workspace=str(ws),
    )
    assert mcp_sess["session_name"] == "mcp-session-v7"
