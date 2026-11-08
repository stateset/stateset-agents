"""Independent all-vocabulary primal/dual and Qwen activation evidence checks.

Imports no target designer, layer solver, session or experiment runner.
"""

from __future__ import annotations

import math

import torch

from benchmarks.weight_space.verify_behavior_trial import require


@torch.inference_mode()
def check_captures(model, encode, protocol, factors, base_factors):
    """Recapture keys and the pre-normalization residual on the fresh model."""
    require(model.get_output_embeddings().bias is None, "head must be bias-free")
    require(type(model.model.norm).__name__ == "Qwen3RMSNorm", "wrong norm")
    layer = model.get_submodule(protocol["module"])
    keys, residuals = [], []

    def key_hook(_m, args):
        keys.append(args[0][0, -1].clone())

    def residual_hook(_m, args):
        residuals.append(args[0][0, -1].clone())

    hooks = [
        layer.register_forward_pre_hook(key_hook),
        model.model.norm.register_forward_pre_hook(residual_hook),
    ]
    try:
        for prompt in [protocol["target"][0], *protocol["post_answer"]]:
            ids = torch.tensor([encode(prompt)])
            model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        for hook in hooks:
            hook.remove()
    require(len(keys) == len(protocol["post_answer"]) + 1, "missing captures")
    require(torch.equal(keys[0][None], base_factors["key"]), "edit key changed")
    require(
        torch.equal(torch.stack(keys[1:]), factors["post_answer"]),
        "post-answer keys changed",
    )
    require(torch.equal(residuals[0], factors["residual"]), "residual changed")


@torch.inference_mode()
def check_design(
    model, encode, protocol, factors, base_factors, trial, margin, protected
):
    """Check global feasibility, a dual lower bound, and the deployed mapping."""
    arm = trial["arm"]
    change = factors[f"{arm}.change"].double()
    original_output = base_factors["baseline_output"].double()
    actual = base_factors["key"] @ model.get_submodule(protocol["module"]).weight.T
    require(
        torch.allclose(
            actual.double(), original_output + change, atol=0.002, rtol=1e-5
        ),
        "retargeted layer output mismatch",
    )
    if arm in ("sham", "weighted_128"):
        require(
            torch.equal(change, (margin * base_factors["direction"]).double()),
            "embedding control changed",
        )
        return
    residual = factors["residual"].double()
    target = residual + change
    head = model.get_output_embeddings().weight
    target_id = encode(protocol["target"][1])[0]
    scale = model.model.norm.weight.double()
    chosen = head[target_id].double()
    worst, min_margin = -math.inf, math.inf
    for start in range(0, len(head), 4096):
        stop = min(start + 4096, len(head))
        contrast = (chosen - head[start:stop].double()) * scale
        keep = torch.arange(start, stop) != target_id
        values = contrast[keep] @ target
        lengths = contrast[keep].norm(dim=1)
        require(bool((lengths > 0).all()), "duplicate target contrast")
        worst = max(worst, float(((margin - values) / lengths).max()))
        min_margin = min(min_margin, float(values.min()))
    require(worst <= 1e-8, "margin not feasible over the full vocabulary")
    design = trial["target_design"]
    indices = design["token_ids"]
    require(
        len(indices) == len(set(indices))
        and all(0 <= i < len(head) and i != target_id for i in indices),
        "invalid certificate indices",
    )
    normals = (chosen - head[indices].double()) * scale
    lengths = normals.norm(dim=1)
    normals /= lengths[:, None]
    rhs = margin / lengths - normals @ residual
    multipliers = factors[f"{arm}.multipliers"]
    require(
        bool(torch.isfinite(multipliers).all() and (multipliers >= 0).all()),
        "invalid dual witness",
    )
    witness = normals.T @ multipliers
    require(
        torch.allclose(witness, change, atol=1e-8, rtol=1e-9), "stationarity mismatch"
    )
    primal = float(change.square().sum() / 2)
    dual = float(rhs @ multipliers - witness.square().sum() / 2)
    require(abs(primal - dual) <= 1e-6, "minimum-change certificate gap too large")
    for field, value in (
        ("update_norm", float(change.norm())),
        ("minimum_margin", min_margin),
        ("max_constraint_violation", max(0.0, worst)),
        ("primal_objective", primal),
        ("dual_objective", dual),
        ("duality_gap", primal - dual),
    ):
        require(
            math.isclose(design[field], value, abs_tol=1e-7, rel_tol=1e-6),
            f"certificate diagnostic mismatch: {field}",
        )
    require(protected == arm.endswith("_protected"), "protection condition changed")
