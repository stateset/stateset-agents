# Kernel code — WRITE (free): repurpose one MLP neuron as a thresholded "answer computed -> conclude" unit.
#   gate row  <- GAIN * (c @ W_gate)        pre-activation = GAIN * z,  z = c . gate_out  (ridge-fit detector)
#   up row    <- u_scale * W_up[k_up]        ~1.0 in every context (near-constant up neuron) -> multiplier
#   down col  <- (WRITE / act_ans) * B       B = unit </think> logit direction through the final norm
# act = SiLU(GAIN*z) * ~1  ->  ~3.4 when the answer is computed, ~-0.03 otherwise (held-out half).
import json

import torch

GAIN = 4.0
WRITE = __WRITE__  # target residual write norm at answer-computed contexts
EXTRA_THINK = __THINK__  # global </think> row scale on top (1.0 = none)
L, v, k_up, u_scale = (
    NC_META["L"],
    NC_META["victim"],
    NC_META["k_up"],
    NC_META["u_scale"],
)
c = NC[100.0]

while edits():
    revert()
clear_token_rows()
scale_token_rows([13784, 13428], 0.5, note="Wait x0.5")
if EXTRA_THINK != 1.0:
    scale_token_rows([248069], EXTRA_THINK, note=f"</think> x{EXTRA_THINK}")

Wg = weight(f"model.layers.{L}.mlp.gate_proj")  # (3584, 1024)
Wu = weight(f"model.layers.{L}.mlp.up_proj")
Wd = weight(f"model.layers.{L}.mlp.down_proj")  # (1024, 3584)
n_int = Wg.shape[0]
e_v = torch.zeros(n_int)
e_v[v] = 1.0

g_new = GAIN * (c @ Wg)
u_new = u_scale * Wu[k_up]
act = torch.nn.functional.silu(
    GAIN * (outs[f"model.layers.{L}.mlp.gate_proj"].float() @ c)
) * (u_scale * outs[f"model.layers.{L}.mlp.up_proj"].float()[:, k_up])
act_ans = float(act[CIRCUIT["G"]["ans"]].mean())
d_new = (WRITE / act_ans) * WRITE_DIR

ids = [
    apply_edit(
        f"model.layers.{L}.mlp.gate_proj",
        e_v[:, None],
        (g_new - Wg[v])[None, :],
        note=f"neuron L{L}#{v} gate <- answer-computed detector x{GAIN}",
    ),
    apply_edit(
        f"model.layers.{L}.mlp.up_proj",
        e_v[:, None],
        (u_new - Wu[v])[None, :],
        note=f"neuron L{L}#{v} up <- near-constant up#{k_up}",
    ),
    apply_edit(
        f"model.layers.{L}.mlp.down_proj",
        (d_new - Wd[:, v])[:, None],
        e_v[None, :],
        note=f"neuron L{L}#{v} down <- </think> direction, write {WRITE} at answer",
    ),
]
# Read back the edited rows from the live weights to confirm the neuron is what we designed.
chk = {
    "gate_row_err": float(
        (weight(f"model.layers.{L}.mlp.gate_proj")[v] - g_new).abs().max()
    ),
    "up_row_err": float(
        (weight(f"model.layers.{L}.mlp.up_proj")[v] - u_new).abs().max()
    ),
    "down_col_err": float(
        (weight(f"model.layers.{L}.mlp.down_proj")[:, v] - d_new).abs().max()
    ),
}
G = CIRCUIT["G"]
print(
    json.dumps(
        {
            "edits": ids,
            "neuron": f"L{L}#{v}",
            "write_at_answer": WRITE,
            "extra_think": EXTRA_THINK,
            "predicted_write_by_ctx": {
                g: round(float(act[ix].mean()) * WRITE / act_ans, 3)
                for g, ix in G.items()
            },
            "readback_max_abs_err": chk,
        }
    )
)
