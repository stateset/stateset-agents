# Kernel code — READ (1 forward pass): newline-anchored "answer computed -> conclude" detector.
# The model's own conclusions look like "...answer\n</think>", so the conclude unit should fire at the
# LINE BREAK after the answer is computed, not on the number token (which put </think> mid-sentence).
import json
import random

import torch

L = 23
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    MODEL_ID, revision="2fc06364715b967f1860aea9cf38778875588b17"
)
c = json.load(open("/workspace/answer_contrast.json"))
by_id = {r["id"]: r for r in traces}
NL = {i for i in range(len(tok)) if "\n" in tok.decode([i])}
rng = random.Random(1)
MAXLEN = 3000

pos, neg_work = [], []
for m in c["meta"]:
    r = by_id.get(m["id"])
    if r is None:
        continue
    P, g, k = r["prompt_ids"], r["gen_ids"], m["answer_token"]
    after = [j for j in range(k, len(g)) if g[j] in NL][
        :2
    ]  # first two line breaks after the answer
    before = [j for j in range(int(0.1 * k), int(0.9 * k)) if g[j] in NL]
    for j in after:
        if len(P) + j + 1 <= MAXLEN:
            pos.append(P + g[: j + 1])
    for j in rng.sample(before, k=min(2, len(before))):
        if len(P) + j + 1 <= MAXLEN:
            neg_work.append(P + g[: j + 1])
post = [p for p in c["post"] if len(p) <= MAXLEN]
post = rng.sample(post, k=min(150, len(post)))
prefixes = pos + neg_work + post
mods = [f"model.layers.{L}.mlp.gate_proj", f"model.layers.{L}.mlp.up_proj"]
outs_nl = capture_rows(
    mods,
    prefixes,
    reason=f"newline read: {len(mods)} modules x {len(prefixes)} prefixes",
)
a, b = len(pos), len(neg_work)
G_NL = {
    "ans": torch.arange(0, a),
    "work": torch.arange(a, a + b),
    "post": torch.arange(a + b, len(prefixes)),
}
print(json.dumps({"answer_newlines": a, "working_newlines": b, "post": len(post)}))

Zg, Zu = outs_nl[mods[0]].float(), outs_nl[mods[1]].float()
y = -torch.ones(len(prefixes))
y[G_NL["ans"]] = 1.0
idx = torch.arange(len(prefixes))
tr = idx[idx % 2 == 0]


def auc(p, q):
    v = torch.cat([p, q])
    rk = v.argsort().argsort().float() + 1
    return float((rk[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))


k_up, u_scale = NC_META["k_up"], NC_META["u_scale"]
u = u_scale * Zu[:, k_up]
NL_FIT = {}
for lam in (10.0, 100.0, 1000.0):
    cvec = torch.linalg.solve(
        Zg[tr].T @ Zg[tr] + lam * torch.eye(Zg.shape[1]), Zg[tr].T @ y[tr]
    )
    z = Zg @ cvec
    act = torch.nn.functional.silu(4 * z) * u
    te = {g: v[v % 2 == 1] for g, v in G_NL.items()}
    NL_FIT[lam] = cvec
    print(
        json.dumps(
            {
                "lam": lam,
                "auc_vs_work": round(auc(z[te["ans"]], z[te["work"]]), 3),
                "auc_vs_post": round(auc(z[te["ans"]], z[te["post"]]), 3),
                "act_mean": {g: round(float(act[v].mean()), 3) for g, v in te.items()},
                "fire_rate_act_gt_0.5": {
                    g: round(float((act[v] > 0.5).float().mean()), 3)
                    for g, v in te.items()
                },
            }
        )
    )
