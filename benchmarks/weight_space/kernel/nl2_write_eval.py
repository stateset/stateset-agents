# Kernel code — REFIT (free) + WRITE (free) + EVALUATE (1 pass).
# Newline-anchored conclude neuron v2: positives = line breaks after the answer; negatives = working
# line breaks, post-</think> prefixes, AND the answer token itself (from the earlier capture) — so the
# unit stays quiet mid-line and fires at the line break, where the model's own conclusions put </think>.
import json

import torch

GAIN, WRITE, LAM = 4.0, 3.0, 100.0
L, v, k_up, u_scale = (
    NC_META["L"],
    NC_META["victim"],
    NC_META["k_up"],
    NC_META["u_scale"],
)
gk, uk = f"model.layers.{L}.mlp.gate_proj", f"model.layers.{L}.mlp.up_proj"

Zg = torch.cat([outs_nl[gk].float(), outs[gk].float()[CIRCUIT["G"]["ans"]]])
Zu = torch.cat([outs_nl[uk].float(), outs[uk].float()[CIRCUIT["G"]["ans"]]])
n_nl, n_tok = len(outs_nl[gk]), len(CIRCUIT["G"]["ans"])
groups = {
    **G_NL,
    "answer_token": torch.arange(n_nl, n_nl + n_tok),
}
y = -torch.ones(n_nl + n_tok)
y[G_NL["ans"]] = 1.0
idx = torch.arange(n_nl + n_tok)
tr = idx[idx % 2 == 0]
cvec = torch.linalg.solve(
    Zg[tr].T @ Zg[tr] + LAM * torch.eye(Zg.shape[1]), Zg[tr].T @ y[tr]
)
act = torch.nn.functional.silu(GAIN * (Zg @ cvec)) * (u_scale * Zu[:, k_up])
te = {g: ix[ix % 2 == 1] for g, ix in groups.items()}
print(
    json.dumps(
        {
            "heldout_half_fire_rate": {
                g: round(float((act[ix] > 0.5).float().mean()), 3)
                for g, ix in te.items()
            },
            "heldout_half_act_mean": {
                g: round(float(act[ix].mean()), 3) for g, ix in te.items()
            },
        }
    )
)
NL2_C = cvec

while edits():
    revert()
clear_token_rows()
scale_token_rows([13784, 13428], 0.5, note="Wait x0.5")
Wg = weight(gk)
Wu = weight(uk)
Wd = weight(f"model.layers.{L}.mlp.down_proj")
e_v = torch.zeros(Wg.shape[0])
e_v[v] = 1.0
g_new, u_new = GAIN * (cvec @ Wg), u_scale * Wu[k_up]
act_hi = float(act[G_NL["ans"]][act[G_NL["ans"]] > 0.5].mean())
d_new = (WRITE / act_hi) * WRITE_DIR
apply_edit(gk, e_v[:, None], (g_new - Wg[v])[None, :], note="NL2 neuron gate")
apply_edit(uk, e_v[:, None], (u_new - Wu[v])[None, :], note="NL2 neuron up")
apply_edit(
    f"model.layers.{L}.mlp.down_proj",
    (d_new - Wd[:, v])[:, None],
    e_v[None, :],
    note="NL2 neuron down",
)

rep = evaluate(n=200)
rows = evaluate.__globals__["STATE"]["last_eval"]
clean = [r["finish"] == "stop" and r["text"].count("</think>") == 1 for r in rows]
print(
    json.dumps(
        {
            "lenient": rep["accuracy"],
            "hit_limit": rep["hit_limit"],
            "strict": round(
                sum(r["correct"] and c for r, c in zip(rows, clean, strict=True))
                / len(rows),
                3,
            ),
            "clean_finish": round(sum(clean) / len(rows), 3),
            "multi_think_close": round(
                sum(r["text"].count("</think>") > 1 for r in rows) / len(rows), 3
            ),
        }
    )
)
for ex in rep["examples"]:
    print("--", ex["kind"], ex["tokens"], repr(ex["tail"][-200:]))
