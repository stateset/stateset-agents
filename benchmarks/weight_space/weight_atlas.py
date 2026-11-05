"""Weight atlas for Qwen3.5-0.8B: which weights correspond to which behaviors (no forward passes).

Part A — token rows: nearest neighbours of anchor tokens in the (tied) embedding.
Part B — MLP neurons: direct logit attribution of every down_proj column,
         (gamma * w_j) . E[token], i.e. how strongly neuron j's output writes
         toward an anchor token through the final RMSNorm and unembedding.
         For the top neurons, list the tokens they push up the most.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

torch.set_num_threads(4)
SNAP = Path(
    glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/"
        )
    )[0]
)
SHARD = next(SNAP.glob("*.safetensors"))
P = "model.language_model."
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17"
)

with safe_open(str(SHARD), "pt") as f:
    E = f.get_tensor(P + "embed_tokens.weight").float()  # [V, d]
    gamma = f.get_tensor(P + "norm.weight").float()  # [d]
    n_layers = 24
    W = {
        L: f.get_tensor(f"{P}layers.{L}.mlp.down_proj.weight").float()
        for L in range(n_layers)
    }  # [d, 3584]

# Qwen3.5 RMSNorm uses (1 + weight) scaling ("zero-centered") if the norm weights are ~0.
gain = gamma + 1.0  # Qwen3_5RMSNorm: output * (1 + weight)
V = E.shape[0]


def ids(word: str) -> list[int]:
    out = []
    for w in (word, " " + word):
        t = tok(w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            out.append(t[0])
    return sorted(set(out))


def show(i: int) -> str:
    s = tok.decode([i])
    return repr(s if s.strip() else tok.convert_ids_to_tokens(i))


ANCHORS = {
    "Wait": ids("Wait"),
    "</think>": [248069],
    "Therefore": ids("Therefore"),
    "Yes": ids("Yes"),
    "Hmm": ids("Hmm"),
    "sorry": ids("sorry"),
}
report: dict = {"gain_mode": "1+gamma (forced; Qwen3_5RMSNorm)", "anchors": {}}

En = E / E.norm(dim=1, keepdim=True).clamp_min(1e-6)
print("## A. Token rows most similar to each anchor (cosine, tied embedding)")
for name, tids in ANCHORS.items():
    sims = En @ En[tids].mean(0)
    sims[tids] = -1
    top = sims.topk(12).indices.tolist()
    print(f"{name:10s} -> " + ", ".join(f"{show(i)}" for i in top))
    report["anchors"][name] = {
        "ids": tids,
        "neighbours": [tok.decode([i]) for i in top],
    }

print("\n## B. MLP neurons whose output column writes most strongly toward each anchor")
for name, tids in ANCHORS.items():
    target = (E[tids] * gain).mean(0)  # [d]
    best = []
    for L in range(n_layers):
        cols = W[L]  # [d, 3584]
        score = (target @ cols) / cols.norm(dim=0).clamp_min(
            1e-6
        )  # cosine-like, per neuron
        v, j = score.topk(3)
        best += [(float(v[k]), L, int(j[k])) for k in range(3)]
    best.sort(reverse=True)
    print(f"\n{name}:")
    rows = []
    for v, L, j in best[:5]:
        col = W[L][:, j] * gain
        promoted = (E @ col).topk(6).indices.tolist()
        line = f"  layer {L:2d} neuron {j:4d}  align {v:+.3f}  pushes up: " + ", ".join(
            show(i) for i in promoted
        )
        print(line)
        rows.append(
            {
                "layer": L,
                "neuron": j,
                "align": round(v, 4),
                "pushes_up": [tok.decode([i]) for i in promoted],
            }
        )
    report["anchors"][name]["neurons"] = rows

out = Path(__file__).parent / "weight_atlas.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nwrote {out}")
