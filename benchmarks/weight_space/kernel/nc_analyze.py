# Kernel code — ANALYZE (free, no forward passes): design a NEW thresholded neuron in layer L
# from stored captures. New neuron v: gate row = c @ W_gate (so its pre-activation is c . gate_out,
# exactly computable from the captures), up row = s * W_up[k] for a near-constant up neuron k,
# down column = beta * B (</think> logit direction). act = SiLU(c.gate_out) * (s * up_out[k]).
import json

import torch

L = 23
assert "outs" in globals(), "captures from circuit_read.py are gone"
Zg = outs[f"model.layers.{L}.mlp.gate_proj"].float()  # (n, 3584) pre-activations
Zu = outs[f"model.layers.{L}.mlp.up_proj"].float()
H = CIRCUIT["H"][L]
G = CIRCUIT["G"]
n, m = Zg.shape
y = torch.full((n,), -1.0)
y[G["ans"]] = 1.0
idx = torch.arange(n)
tr, te = idx[::2], idx[1::2]


def auc(pos, neg):
    v = torch.cat([pos, neg])
    r = v.argsort().argsort().float() + 1
    return float(
        (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


# 1. near-constant up neurons (act as a bias / multiplier); prefer large |mean|/std
mu, sd = Zu.mean(0), Zu.std(0) + 1e-6
cv_rank = (mu.abs() / sd).topk(5)
k = int(cv_rank.indices[0])
u = Zu[:, k] * torch.sign(mu[k]) / mu[k].abs()  # ~1 everywhere
print(
    json.dumps(
        {
            "near_constant_up": [
                (int(j), round(float(mu[j]), 3), round(float(sd[j]), 3))
                for j in cv_rank.indices
            ],
            "u_stats": {
                g: [round(float(u[v].mean()), 3), round(float(u[v].std()), 3)]
                for g, v in G.items()
            },
        }
    )
)

# Same for gate: a near-constant gate pre-activation can supply the threshold (bias).
gmu, gsd = Zg.mean(0), Zg.std(0) + 1e-6
gk = int((gmu.abs() / gsd).argmax())
bias_feat = Zg[:, gk] / gmu[gk]  # ~1 everywhere
print(
    json.dumps(
        {
            "bias_gate_neuron": gk,
            "mean": round(float(gmu[gk]), 3),
            "sd": round(float(gsd[gk]), 3),
            "bias_feat_by_ctx": {
                g: [
                    round(float(bias_feat[v].mean()), 3),
                    round(float(bias_feat[v].std()), 3),
                ]
                for g, v in G.items()
            },
        }
    )
)

# 2. ridge fit z = c . gate_out (no explicit intercept: bias comes from the near-constant feature)
X = Zg
results = {}
for lam in (1.0, 10.0, 100.0, 1000.0):
    Xt, yt = X[tr], y[tr]
    c = torch.linalg.solve(Xt.T @ Xt + lam * torch.eye(m), Xt.T @ yt)
    z = X @ c
    zt = {g: z[v[v % 2 == 1]] for g, v in G.items()}  # held-out half only
    act = torch.nn.functional.silu(4 * z) * u  # sharpen: gain 4 inside SiLU
    at = {g: act[v[v % 2 == 1]] for g, v in G.items()}
    results[lam] = {
        "auc_ans_vs_work": round(auc(zt["ans"], zt["work"]), 3),
        "auc_ans_vs_post": round(auc(zt["ans"], zt["post"]), 3),
        "z_mean": {g: round(float(v.mean()), 3) for g, v in zt.items()},
        "act_mean": {g: round(float(v.mean()), 3) for g, v in at.items()},
        "act_p90": {g: round(float(v.quantile(0.9)), 3) for g, v in at.items()},
        "frac_act_gt_0.5": {
            g: round(float((v > 0.5).float().mean()), 3) for g, v in at.items()
        },
        "c_norm": round(float(c.norm()), 3),
    }
    globals().setdefault("NC", {})[lam] = c
print(json.dumps(results, indent=1))

# 3. victim neuron: least used (mean |act| * column norm) in layer L
Wd = weight(f"model.layers.{L}.mlp.down_proj")
use = H.abs().mean(0) * Wd.norm(dim=0)
victims = use.topk(5, largest=False)
print(
    json.dumps(
        {
            "victims": [
                (int(j), round(float(v), 5))
                for v, j in zip(victims.values, victims.indices, strict=True)
            ],
            "median_use": round(float(use.median()), 4),
        }
    )
)
NC_META = {
    "L": L,
    "k_up": k,
    "u_scale": float(torch.sign(mu[k]) / mu[k].abs()),
    "victim": int(victims.indices[0]),
}
print(json.dumps(NC_META))
