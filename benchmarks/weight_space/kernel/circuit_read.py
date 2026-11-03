# Kernel code — READ (1 forward pass): find "answer computed (while still thinking)" detector neurons.
import json
import random

import torch

c = json.load(open("/workspace/answer_contrast.json"))
rng = random.Random(0)


def short(xs):
    """Prefixes of at most 3,000 tokens (activation memory on a 44 GB L40S)."""
    return [p for p in xs if len(p) <= 3000]


ans = short(c["answer"])
work = rng.sample(short(c["working"]), k=min(150, len(short(c["working"]))))
post = rng.sample(short(c["post"]), k=min(150, len(short(c["post"]))))
prefixes = ans + work + post
LAYERS = list(range(16, 24))
mods = [f"model.layers.{L}.mlp.{p}" for L in LAYERS for p in ("gate_proj", "up_proj")]
# Memory-safe capture: reduce each batch to last-token rows INSIDE the hook and move them
# to CPU (capture_many retained every batch's full activations -> OOM). Budget still spent.
torch.cuda.empty_cache()
_fv = dict(
    zip(
        capture.__code__.co_freevars,
        (c.cell_contents for c in capture.__closure__),
        strict=True,
    )
)
_session, _spend = _fv["session"], _fv["spend"]
_model = _session.model


def capture_rows(modules, prefixes, reason, batch=4):
    _spend(reason)
    rows, cur, hooks = {m: [] for m in modules}, {}, []
    for m in modules:

        def _h(_mod, _inp, out, m=m):
            o = out.detach()
            idx = torch.tensor(cur["last"], device=o.device)
            rows[m].append(
                o[torch.arange(o.shape[0], device=o.device), idx].float().cpu()
            )

        hooks.append(_model.get_submodule(m).register_forward_hook(_h))
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    try:
        for s0 in range(0, len(order), batch):
            chunk = [prefixes[i] for i in order[s0 : s0 + batch]]
            ids, mask = _session._pad(chunk)
            cur["last"] = [len(c) - 1 for c in chunk]
            with torch.no_grad():
                _model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
    finally:
        for hk in hooks:
            hk.remove()
    inv = torch.tensor(order).argsort()
    return {m: torch.cat(v)[inv] for m, v in rows.items()}


outs = capture_rows(
    mods,
    prefixes,
    reason=f"circuit read: {len(mods)} modules x {len(prefixes)} prefixes",
)
na, nw, npo = len(ans), len(work), len(post)
G = {
    "ans": torch.arange(0, na),
    "work": torch.arange(na, na + nw),
    "post": torch.arange(na + nw, na + nw + npo),
}
H = {
    L: torch.nn.functional.silu(outs[f"model.layers.{L}.mlp.gate_proj"])
    * outs[f"model.layers.{L}.mlp.up_proj"]
    for L in LAYERS
}


def auc(pos, neg):
    v = torch.cat([pos, neg])
    r = v.argsort().argsort().float() + 1
    return float(
        (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


# split: neurons chosen on even indices, selectivity measured on odd indices
sel = {k: v[::2] for k, v in G.items()}
tst = {k: v[1::2] for k, v in G.items()}
CIRCUIT = {"H": H, "G": G, "A": {}, "report": {}}
for L in LAYERS:
    h = H[L]
    a, neg = h[sel["ans"]], torch.cat([h[sel["work"]], h[sel["post"]]])
    sd = torch.sqrt((a.var(0) + neg.var(0)) / 2) + 1e-4
    d = (a.mean(0) - neg.mean(0)) / sd
    top = d.topk(16).indices
    w = d[top].clamp_min(0)
    A = torch.zeros(h.shape[1])
    A[top] = w / w.norm()
    pa, pw, pp = h[tst["ans"]] @ A, h[tst["work"]] @ A, h[tst["post"]] @ A
    CIRCUIT["A"][L] = A
    CIRCUIT["report"][L] = {
        "auc_vs_working": round(auc(pa, pw), 3),
        "auc_vs_postthink": round(auc(pa, pp), 3),
        "top_neurons": top[:8].tolist(),
        "best_effect_size": round(float(d[top[0]]), 2),
        "mean_proj": {
            "ans": round(float(pa.mean()), 3),
            "work": round(float(pw.mean()), 3),
            "post": round(float(pp.mean()), 3),
        },
    }
print(f"prefixes: answer {na}, working {nw}, post-think {npo}")
for L, r in CIRCUIT["report"].items():
    print(L, json.dumps(r))
