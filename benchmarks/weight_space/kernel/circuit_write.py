# Kernel code — WRITE (free): rank-1 conditional circuit "answer computed -> conclude", on top of Wait x0.5.
#   down_proj[L] += BETA * B (x) A    with A = detector neurons (read), B = </think> logit direction (write)
LAYER = __LAYER__
BETA = __BETA__
import json

from huggingface_hub import snapshot_download
from safetensors import safe_open

if "WRITE_DIR" not in globals():
    snap = snapshot_download(
        MODEL_ID, revision="2fc06364715b967f1860aea9cf38778875588b17"
    )
    shard = __import__("glob").glob(snap + "/*.safetensors")[0]
    with safe_open(shard, "pt") as f:
        gain = (
            1.0 + f.get_tensor("model.language_model.norm.weight").float()
        )  # Qwen3_5RMSNorm: x * (1 + w)
        e_think = f.get_tensor("model.language_model.embed_tokens.weight")[
            248069
        ].float()
    WRITE_DIR = gain * e_think
    WRITE_DIR = WRITE_DIR / WRITE_DIR.norm()

while edits():
    revert()
clear_token_rows()
scale_token_rows([13784, 13428], 0.5, note="Wait x0.5")
A = CIRCUIT["A"][LAYER]
eid = apply_edit(
    f"model.layers.{LAYER}.mlp.down_proj",
    BETA * WRITE_DIR[:, None],
    A[None, :],
    note=f"circuit L{LAYER} beta={BETA}: answer-computed detectors -> </think>",
)
h = CIRCUIT["H"][LAYER]
G = CIRCUIT["G"]
push = {k: round(float((h[v] @ A).mean() * BETA), 3) for k, v in G.items()}
print(
    json.dumps(
        {
            "edit": eid,
            "layer": LAYER,
            "beta": BETA,
            "mean_write_norm_by_context": push,
            "edits": edits(),
        }
    )
)
