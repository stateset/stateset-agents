"""Real PyTorch Forward-Hook CAA Surgical Extraction & Step-0 RL Warm-Start.

Bridges 3D Weight Studio's Contrastive Activation Addition (CAA) surgery
directly onto real PyTorch ``nn.Module`` causal language models (HuggingFace
Qwen3, DeepSeek, Llama, Gemma, Mistral, GPT-2, PEFT LoRA models, or custom
PyTorch Transformers):

1. ``find_transformer_layers(model)`` discovers the decoder block list across
   standard HuggingFace and PEFT model wrappers.
2. ``extract_torch_caa_vector(model, pos_inputs, neg_inputs)`` registers real
   PyTorch forward hooks to compute mean residual activation contrasts across
   every layer and picks the peak-separation layer.
3. ``bake_torch_caa_steering(model, direction, layer_index, alpha)`` directly
   bakes the steering vector into the target layer's residual output projection
   (bias or rank-1 weight delta) as a **Step-0 Surgical Warm-Start** before
   GRPO/GSPO/DAPO training — returning an undoable rollback handle.

Torch is imported strictly lazily (zero eager ``torch`` import at module load).
"""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for GPU surgical warm-start. "
            "Install with `pip install 'stateset-agents[training]'`."
        ) from exc


def find_transformer_layers(model: Any) -> list[Any]:
    """Discover the ordered list of transformer decoder blocks on a PyTorch LM."""
    candidates = [
        ("model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "transformer", "h"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
        ("layers",),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj)
    raise ValueError(
        f"Could not locate transformer layer list on {type(model).__name__}. "
        "Expected model.model.layers, model.transformer.h, or model.layers."
    )


def _extract_hidden_tensor(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def extract_torch_caa_vector(
    model: Any,
    positive_input_ids: Any,
    negative_input_ids: Any,
    *,
    layer_index: int | None = None,
) -> dict[str, Any]:
    """Extract a Contrastive Activation Addition (CAA) vector via live PyTorch forward hooks.

    Parameters
    ----------
    model:
        Any PyTorch causal LM exposing transformer blocks discoverable by
        :func:`find_transformer_layers`.
    positive_input_ids:
        Token ID tensor ``[B, T]`` representing desired target behavior.
    negative_input_ids:
        Token ID tensor ``[B, T]`` representing undesired baseline/failing behavior.
    layer_index:
        Optional fixed layer index; if omitted, selects the layer with peak
        L2 separation between positive and negative mean residual activations.
    """
    torch = _require_torch()
    layers = find_transformer_layers(model)
    num_layers = len(layers)
    if layer_index is not None and not (0 <= layer_index < num_layers):
        raise ValueError(
            f"layer_index {layer_index} out of range for model with {num_layers} layers."
        )

    def _capture(input_ids: Any) -> dict[int, Any]:
        captured: dict[int, Any] = {}
        handles = []

        def _make_hook(idx: int) -> Any:
            def _hook(_mod: Any, _inp: Any, out: Any) -> None:
                h = _extract_hidden_tensor(out)
                if torch.is_tensor(h) and h.ndim >= 2:
                    # Mean-pool over batch & sequence -> [H] in fp32
                    dims = tuple(range(h.ndim - 1))
                    captured[idx] = h.detach().float().mean(dim=dims).cpu()

            return _hook

        for idx, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(_make_hook(idx)))
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    pos_acts = _capture(positive_input_ids)
    neg_acts = _capture(negative_input_ids)

    separations: list[dict[str, Any]] = []
    diffs: dict[int, Any] = {}
    for idx in range(num_layers):
        if idx not in pos_acts or idx not in neg_acts:
            continue
        diff = pos_acts[idx] - neg_acts[idx]
        sep = float(torch.linalg.vector_norm(diff).item())
        diffs[idx] = diff
        separations.append({"layer_index": idx, "separation_l2": round(sep, 6)})

    if not diffs:
        raise RuntimeError(
            "No layer residual activations were captured during forward pass."
        )

    if layer_index is not None:
        chosen = layer_index
    else:
        chosen = max(
            diffs.keys(), key=lambda i: float(torch.linalg.vector_norm(diffs[i]).item())
        )

    vec = diffs[chosen]
    norm = float(torch.linalg.vector_norm(vec).item())
    unit_vec = vec / max(norm, 1e-8)

    return {
        "recommended_layer": int(chosen),
        "num_layers": num_layers,
        "hidden_dim": int(unit_vec.numel()),
        "peak_separation": round(norm, 6),
        "direction_tensor": unit_vec,
        "direction": [round(float(x), 6) for x in unit_vec[:32].tolist()],
        "layer_separations": separations,
    }


def _find_residual_output_linear(layer: Any) -> Any:
    """Locate the residual output projection Linear layer inside a transformer block."""
    candidates = [
        ("mlp", "down_proj"),
        ("mlp", "c_proj"),
        ("mlp", "fc2"),
        ("feed_forward", "w2"),
        ("self_attn", "o_proj"),
        ("attn", "c_proj"),
        ("out_proj",),
        ("linear2",),
    ]
    for path in candidates:
        obj = layer
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and hasattr(obj, "weight") and obj.weight is not None:
            return obj

    # Fallback: last Linear-like submodule whose out_features matches hidden_dim
    last_linear = None
    for mod in layer.modules():
        if hasattr(mod, "weight") and getattr(mod.weight, "ndim", 0) == 2:
            last_linear = mod
    if last_linear is not None:
        return last_linear
    raise ValueError(
        f"Could not locate output Linear projection in {type(layer).__name__}."
    )


def bake_torch_caa_steering(
    model: Any,
    direction_tensor: Any,
    *,
    layer_index: int,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Bake a CAA steering vector into a real PyTorch layer as Step-0 RL Warm-Start.

    If the target residual output projection has a ``bias``, adds
    ``alpha * direction`` directly to ``bias`` (exact constant residual shift).
    Otherwise applies a bounded rank-1 directional update to ``weight`` so
    bias-free architectures (Qwen/Llama/DeepSeek RMSNorm MLPs) are supported
    out of the box. Returns a snapshot dict with a ``rollback()`` callable.
    """
    torch = _require_torch()
    layers = find_transformer_layers(model)
    if not (0 <= layer_index < len(layers)):
        raise ValueError(
            f"layer_index {layer_index} out of range for {len(layers)} layers."
        )

    proj = _find_residual_output_linear(layers[layer_index])
    weight = proj.weight
    out_dim = int(weight.shape[0])

    vec = (
        direction_tensor.detach().clone()
        if torch.is_tensor(direction_tensor)
        else torch.tensor(direction_tensor, dtype=torch.float32)
    ).reshape(-1)
    if vec.numel() != out_dim:
        raise ValueError(
            f"Steering direction length {vec.numel()} does not match projection out_features {out_dim}."
        )
    vec = vec / torch.linalg.vector_norm(vec).clamp_min(1e-8)

    with torch.no_grad():
        if getattr(proj, "bias", None) is not None:
            orig_bias = proj.bias.detach().clone()
            delta = (float(alpha) * vec).to(
                device=proj.bias.device, dtype=proj.bias.dtype
            )
            proj.bias.add_(delta)
            delta_norm = float(torch.linalg.vector_norm(delta.float()).item())
            mode = "bias_shift"

            def _rollback() -> None:
                with torch.no_grad():
                    proj.bias.copy_(orig_bias)

        else:
            orig_weight = weight.detach().clone()
            in_dim = int(weight.shape[1])
            # Align rank-1 update with mean input sensitivity direction of the projection
            col_dir = weight.detach().float().mean(dim=0)
            col_norm = float(torch.linalg.vector_norm(col_dir).item())
            if col_norm < 1e-8:
                col_dir = torch.ones(in_dim, dtype=torch.float32, device=weight.device)
                col_norm = float(torch.linalg.vector_norm(col_dir).item())
            col_unit = col_dir / col_norm
            rank1 = (float(alpha) * 0.05) * torch.outer(
                vec.to(device=weight.device, dtype=torch.float32), col_unit
            )
            delta_w = rank1.to(dtype=weight.dtype)
            weight.add_(delta_w)
            delta_norm = float(torch.linalg.vector_norm(delta_w.float()).item())
            mode = "rank1_weight_bake"

            def _rollback() -> None:
                with torch.no_grad():
                    weight.copy_(orig_weight)

    return {
        "baked": True,
        "mode": mode,
        "layer_index": int(layer_index),
        "alpha": float(alpha),
        "delta_l2_norm": round(delta_norm, 6),
        "target_module": type(proj).__name__,
        "rollback": _rollback,
    }


__all__ = [
    "bake_torch_caa_steering",
    "extract_torch_caa_vector",
    "find_transformer_layers",
]
