# Kernel code — SCALE (free): how big is layer-23's normal MLP write vs the circuit's write?
import json

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

L = 23
Wd = weight(f"model.layers.{L}.mlp.down_proj")  # (1024, 3584) float32 CPU
h = CIRCUIT["H"][L]
G = CIRCUIT["G"]
mlp_out_norm = (h @ Wd.T).norm(dim=1)  # norm of the MLP's write per prefix
snap = snapshot_download(MODEL_ID, revision="2fc06364715b967f1860aea9cf38778875588b17")
shard = __import__("glob").glob(snap + "/*.safetensors")[0]
with safe_open(shard, "pt") as f:
    gain = 1.0 + f.get_tensor("model.language_model.norm.weight").float()
    e_think = f.get_tensor("model.language_model.embed_tokens.weight")[248069].float()
    e_wait = f.get_tensor("model.language_model.embed_tokens.weight")[13784].float()
wdir = gain * e_think
WRITE_DIR = wdir / wdir.norm()
A = CIRCUIT["A"][L]
proj = {k: float((h[v] @ A).median()) for k, v in G.items()}
print(
    json.dumps(
        {
            "mlp_write_norm_median": {
                k: round(float(mlp_out_norm[v].median()), 3) for k, v in G.items()
            },
            "detector_proj_median": {k: round(p, 3) for k, p in proj.items()},
            "|gain*e_think|": round(float(wdir.norm()), 3),
            "cos(write_dir, gain*e_wait)": round(
                float(
                    torch.nn.functional.cosine_similarity(
                        WRITE_DIR, gain * e_wait, dim=0
                    )
                ),
                3,
            ),
        }
    )
)
