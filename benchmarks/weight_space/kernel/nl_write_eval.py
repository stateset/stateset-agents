# Kernel code — WRITE (free) + EVALUATE (1 pass): newline-anchored conclude neuron L23#victim, on Wait x0.5.
# Reports lenient AND strict accuracy (strict = correct, finished, exactly one </think>).
import json

import torch

GAIN, WRITE, LAM = 4.0, 3.0, 100.0
L, v, k_up, u_scale = (
    NC_META["L"],
    NC_META["victim"],
    NC_META["k_up"],
    NC_META["u_scale"],
)
cvec = NL_FIT[LAM]

while edits():
    revert()
clear_token_rows()
scale_token_rows([13784, 13428], 0.5, note="Wait x0.5")

Wg = weight(f"model.layers.{L}.mlp.gate_proj")
Wu = weight(f"model.layers.{L}.mlp.up_proj")
Wd = weight(f"model.layers.{L}.mlp.down_proj")
e_v = torch.zeros(Wg.shape[0])
e_v[v] = 1.0
g_new, u_new = GAIN * (cvec @ Wg), u_scale * Wu[k_up]
Zg = outs_nl[f"model.layers.{L}.mlp.gate_proj"].float()
Zu = outs_nl[f"model.layers.{L}.mlp.up_proj"].float()
act = torch.nn.functional.silu(GAIN * (Zg @ cvec)) * (u_scale * Zu[:, k_up])
act_hi = float(
    act[G_NL["ans"]][act[G_NL["ans"]] > 0.5].mean()
)  # typical activation when it fires
d_new = (WRITE / act_hi) * WRITE_DIR
apply_edit(
    f"model.layers.{L}.mlp.gate_proj",
    e_v[:, None],
    (g_new - Wg[v])[None, :],
    note="NL neuron gate",
)
apply_edit(
    f"model.layers.{L}.mlp.up_proj",
    e_v[:, None],
    (u_new - Wu[v])[None, :],
    note="NL neuron up",
)
apply_edit(
    f"model.layers.{L}.mlp.down_proj",
    (d_new - Wd[:, v])[:, None],
    e_v[None, :],
    note="NL neuron down",
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
