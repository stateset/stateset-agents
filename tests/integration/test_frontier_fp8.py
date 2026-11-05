"""Serialized FP8 attention/expert loads and trainable BF16 adapter contracts.

All weights are generated locally. No published checkpoint, GPU, optimized
kernel or distributed execution is certified by this contract.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
from packaging.version import Version

if os.environ.get("STATESET_REQUIRE_FRONTIER_RUNTIME") == "1":
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
else:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
if Version(transformers.__version__) < Version("5.20.0"):
    if os.environ.get("STATESET_REQUIRE_FRONTIER_RUNTIME") == "1":
        raise RuntimeError("FP8 training contracts require Transformers 5.20+")
    pytest.skip("requires optional [frontier] runtime", allow_module_level=True)

from stateset_agents.core.frontier_models import get_frontier_model  # noqa: E402
from tests.integration.test_frontier_architectures import _tiny_model  # noqa: E402


def _write_fp8_checkpoint(base, path, *, flat_forget_gate):
    """Use released per-expert names and scales, rather than native fused tensors."""
    from safetensors.torch import save_file
    from transformers import FineGrainedFP8Config

    state = {}
    expected = {}
    quantized = []

    def pack(name, weight):
        scale = torch.full((weight.shape[0], 1), 0.125, dtype=torch.float32)
        packed = (weight.float() / scale).to(torch.float8_e4m3fn)
        state[name] = packed
        state[name.removesuffix(".weight") + ".weight_scale_inv"] = scale
        quantized.append(name)
        return (packed.float() * scale).to(weight.dtype)

    for name, value in base.state_dict().items():
        value = value.detach().clone()
        if name.endswith(".mlp.experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            gates, ups = [], []
            for index, expert in enumerate(value):
                gate, up = expert.chunk(2, dim=0)
                gates.append(pack(f"{prefix}{index}.gate_proj.weight", gate))
                ups.append(pack(f"{prefix}{index}.up_proj.weight", up))
            expected[name] = torch.cat([torch.stack(gates), torch.stack(ups)], dim=1)
        elif name.endswith(".mlp.experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            expected[name] = torch.stack(
                [
                    pack(f"{prefix}{index}.down_proj.weight", expert)
                    for index, expert in enumerate(value)
                ]
            )
        elif ".self_attn." in name and name.endswith(".weight") and value.ndim == 2:
            expected[name] = pack(name, value)
        else:
            state[name] = value
            expected[name] = value
    if flat_forget_gate:
        state = {
            name.replace(".self_attn.forget_gate.", ".self_attn."): value
            for name, value in state.items()
        }
    path.mkdir()
    base.config.quantization_config = FineGrainedFP8Config(
        weight_block_size=(1, 16)
    ).to_dict()
    base.config.save_pretrained(path)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    return expected, quantized


@pytest.mark.parametrize("name", ["glm5.3", "glm5.3-flash"])
def test_real_serialized_fp8_experts_dequantize_then_train_reload(name, tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import FineGrainedFP8Config

    spec = get_frontier_model(name)
    torch.manual_seed(42)
    base = _tiny_model(spec).to(torch.bfloat16)
    folder = tmp_path / "checkpoint"
    expected, packed = _write_fp8_checkpoint(
        base, folder, flat_forget_gate=name.endswith("flash")
    )
    loaded, loading_info = type(base).from_pretrained(
        folder,
        dtype=torch.bfloat16,
        quantization_config=FineGrainedFP8Config(dequantize=True),
        local_files_only=True,
        output_loading_info=True,
    )
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]
    assert loaded.is_quantized is False
    assert getattr(loaded, "hf_quantizer", None) is None
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[key].to(value.dtype), rtol=0, atol=0)
        assert value.dtype != torch.float8_e4m3fn
    assert any("experts.0." in key for key in packed)
    model = get_peft_model(
        loaded,
        LoraConfig(r=2, lora_alpha=4, target_modules=spec.lora_regex, lora_dropout=0),
    )
    before = {
        key: p.detach().clone()
        for key, p in model.named_parameters()
        if p.requires_grad
    }
    frozen = {
        key: p.detach().clone()
        for key, p in model.named_parameters()
        if not p.requires_grad
    }
    ids = torch.tensor([[3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, 5, 6]])
    model.train()
    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.any(grad != 0) for grad in gradients)
    torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    ).step()
    assert any(
        not torch.equal(before[key], p)
        for key, p in model.named_parameters()
        if key in before
    )
    assert all(
        torch.equal(frozen[key], p)
        for key, p in model.named_parameters()
        if key in frozen
    )
    model.eval()
    with torch.no_grad():
        expected_logits = model(input_ids=ids, use_cache=False).logits
    model.save_pretrained(tmp_path / "adapter")
    restored_base = type(base).from_pretrained(
        folder,
        dtype=torch.bfloat16,
        quantization_config=FineGrainedFP8Config(dequantize=True),
        local_files_only=True,
    )
    restored = PeftModel.from_pretrained(restored_base, tmp_path / "adapter").eval()
    with torch.no_grad():
        actual_logits = restored(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    output = os.environ.get("STATESET_FRONTIER_CONTRACT_REPORT_DIR")
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        (path / (spec.key + "-fp8.json")).write_text(
            json.dumps(
                {
                    "model": spec.model_id,
                    "architecture": spec.architecture,
                    "quantized_tensor_count": len(packed),
                    "loss": float(loss.detach()),
                    "all_weights_matched_dequantized_reference": True,
                    "frozen_parameters_unchanged": True,
                    "adapter_reload_equal": True,
                    "optimizer_updates": 1,
                    "scope": "local random tiny FP8 attention/per-expert tensors; CPU BF16; no published weights, GPU or distributed execution",
                },
                indent=2,
            )
            + "\n"
        )
