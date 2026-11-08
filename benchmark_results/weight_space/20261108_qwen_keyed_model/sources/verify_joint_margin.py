"""Independent joint-edit witness checks using a Woodbury inverse metric.

No editing, target-design, metric-whitening or experiment-runner imports.
"""

from __future__ import annotations

import math

import torch

from benchmarks.weight_space.verify_behavior_trial import require


@torch.inference_mode()
def check_captures(model, encode, protocol, factors, base):
    require(
        model.lm_head.bias is None
        and type(model.model.norm).__name__ == "Qwen3RMSNorm",
        "unsupported architecture",
    )
    keys, residuals = [], []

    def capture_key(_m, args):
        keys.append(args[0][0].clone())

    def capture_residual(_m, args):
        residuals.append(args[0][0, -1].clone())

    handles = [
        model.get_submodule(protocol["module"]).register_forward_pre_hook(capture_key),
        model.model.norm.register_forward_pre_hook(capture_residual),
    ]
    prompts = [
        *protocol["design_prompts"],
        *protocol["extra_calibration"],
        *protocol["post_answer"],
    ]
    try:
        for prompt in prompts:
            ids = torch.tensor([encode(prompt)])
            model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()
    n, h = len(protocol["design_prompts"]), len(protocol["extra_calibration"])
    require(
        torch.equal(torch.stack([k[-1] for k in keys[:n]]), factors["keys"]),
        "design keys changed",
    )
    require(
        torch.equal(torch.stack(residuals[:n]), factors["residuals"]),
        "design residuals changed",
    )
    require(
        torch.equal(torch.cat(keys[n : n + h]), factors["extra_calibration"]),
        "calibration keys changed",
    )
    require(
        torch.equal(
            torch.stack([k[-1] for k in keys[n + h :]]), factors["post_answer"]
        ),
        "post-answer keys changed",
    )
    require(torch.equal(factors["keys"][:1], base["key"]), "original edit key changed")


@torch.inference_mode()
def check_design(model, encode, protocol, factors, base, trial, n, coverage, original):
    arm = trial["arm"]
    if arm == "sham":
        return
    keys, residuals = factors["keys"][:n].double(), factors["residuals"][:n].double()
    h = base["calibration"].double()
    if coverage != "narrow":
        h = torch.cat(
            (h, factors["extra_calibration"].double(), factors["post_answer"].double())
        )
    p = base["protected"].double()
    if coverage == "hard":
        p = torch.cat((p, h))
    pn = p / p.norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    _, singular, vt = torch.linalg.svd(pn, full_matrices=False)
    q = vt[singular > singular[0] * 1e-10]
    kp = keys - keys @ q.T @ q
    hp = h - h @ q.T @ q
    # Solve (I + alpha H.T H)^-1 K.T via Woodbury, without inverse square roots.
    alpha = protocol["locality_weight"] / len(h)
    reduced = torch.eye(len(h), dtype=torch.float64) / alpha + hp @ hp.T
    precision_keys = kp.T - hp.T @ torch.linalg.solve(reduced, hp @ kp.T)
    key_products = keys @ precision_keys
    key_norms = key_products.diag().clamp_min(0).sqrt()
    layer = model.get_submodule(protocol["module"])
    realized = layer.weight.double() - original.double()
    span_error = float((realized @ q.T).norm())
    require(
        span_error <= protocol["joint_solver"]["preservation_atol"],
        "deployed protected span changed",
    )
    pd = p.float() @ layer.weight.T - p.float() @ original.T
    require(
        bool(
            (
                pd.abs().double() <= 0.0005 * p.norm(dim=1, keepdim=True).clamp_min(1)
            ).all()
        ),
        "protected activation changed",
    )
    deployed = (
        residuals + (keys.float() @ layer.weight.T - keys.float() @ original.T).double()
    )
    design = trial["design"]
    if arm == "separate_four":
        ideal_points = residuals + factors[f"{arm}.changes"]
        require(
            torch.allclose(deployed, ideal_points, atol=0.002, rtol=1e-5),
            "independent target mapping changed",
        )
    else:
        ideal = factors[f"{arm}.ideal_B"] @ factors[f"{arm}.ideal_A"]
        ideal_points = residuals + keys @ ideal.T
        contexts, tokens = design["context_ids"], design["token_ids"]
        require(
            len(contexts) == len(tokens)
            and len(set(zip(contexts, tokens, strict=True))) == len(tokens),
            "invalid certificate indices",
        )
        target_id = encode(protocol["target"][1])[0]
        require(
            all(0 <= i < n for i in contexts)
            and all(
                0 <= j < len(model.lm_head.weight) and j != target_id for j in tokens
            ),
            "certificate index out of range",
        )
        contrast = (
            model.lm_head.weight[target_id].double()
            - model.lm_head.weight[tokens].double()
        ) * model.model.norm.weight.double()
        norms = contrast.norm(dim=1)
        normal = contrast / norms[:, None]
        lam = factors[f"{arm}.multipliers"]
        require(
            bool(torch.isfinite(lam).all() and (lam >= 0).all()), "invalid dual witness"
        )
        rhs = (
            protocol["margin"] / norms - (normal * residuals[contexts]).sum(1)
        ) / key_norms[contexts]
        witness = (normal.T * lam) @ (
            precision_keys[:, contexts].T / key_norms[contexts, None]
        )
        require(
            torch.allclose(ideal, witness, atol=1e-8, rtol=1e-7),
            "joint stationarity mismatch",
        )
        require(float((ideal @ q.T).norm()) <= 1e-7, "ideal protection mismatch")
        gram = (
            (normal @ normal.T)
            * key_products[contexts][:, contexts]
            / (key_norms[contexts, None] * key_norms[contexts][None])
        )
        primal = float(
            (ideal.square().sum() + alpha * (h @ ideal.T).square().sum()) / 2
        )
        dual = float(rhs @ lam - lam @ gram @ lam / 2)
        require(abs(primal - dual) <= 1e-8, "joint objective certificate gap")
        for field, value in (
            ("primal_objective", primal),
            ("dual_objective", dual),
            ("duality_gap", primal - dual),
        ):
            require(
                math.isclose(design[field], value, rel_tol=1e-6, abs_tol=1e-8),
                f"joint diagnostic mismatch: {field}",
            )
    target_id = encode(protocol["target"][1])[0]
    head = model.lm_head.weight
    min_ideal = torch.full((n,), math.inf, dtype=torch.float64)
    min_actual = min_ideal.clone()
    worst = -math.inf
    for start in range(0, len(head), 4096):
        stop = min(start + 4096, len(head))
        keep = torch.arange(start, stop) != target_id
        c = (head[target_id].double() - head[start:stop].double())[
            keep
        ] * model.model.norm.weight.double()
        values = ideal_points @ c.T
        min_ideal = torch.minimum(min_ideal, values.min(1).values)
        min_actual = torch.minimum(min_actual, (deployed @ c.T).min(1).values)
        worst = max(
            worst,
            float(
                (
                    (protocol["margin"] - values)
                    / c.norm(dim=1)[None]
                    / key_norms[:, None]
                ).max()
            ),
        )
    require(
        float(min_actual.min()) >= protocol["margin"] - 0.002,
        "deployed margin violation",
    )
    if arm != "separate_four":
        require(worst <= 1e-9, "joint vocabulary constraint violation")
        require(
            torch.allclose(
                min_ideal,
                torch.tensor(design["ideal_minimum_margins"], dtype=torch.float64),
                atol=1e-7,
                rtol=1e-7,
            ),
            "ideal margins changed",
        )
        require(
            torch.allclose(
                min_actual,
                torch.tensor(design["deployed_minimum_margins"], dtype=torch.float64),
                atol=1e-7,
                rtol=1e-7,
            ),
            "deployed margins changed",
        )
