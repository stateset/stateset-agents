"""Finish the kernel's held-out run from the pod shell (the kernel refuses code after submit).

1. Recover the submitted edits' factors from the edited checkpoint the evaluator already wrote
   (/workspace/ckpt/b_claude_edit): for gate/up the edit replaced row v (B = e_v, A = row delta);
   for down it replaced column v (B = column delta, A = e_v).
2. Build the evaluator's control exactly as ws_server.heldout() does: for each module in edit order,
   R = randn(B.shape, generator seeded 7), rescaled to B's column norm; delta = R @ A. Token rows
   (Wait x0.5) stay, as in the evaluator.
3. Write the control checkpoint, then generate edit and control held-out with pod_job.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

W = Path("/workspace")
EDIT = W / "ckpt" / "b_claude_edit"
CTRL = W / "ckpt" / "b_random_control"
PFX = "model.language_model.layers.23.mlp."
MODS = [
    "gate_proj",
    "up_proj",
    "down_proj",
]  # edit order in the kernel (gate, up, down)


def main() -> None:
    from huggingface_hub import snapshot_download

    base_dir = Path(
        snapshot_download(
            "Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17"
        )
    )
    base_shard = next(base_dir.glob("*.safetensors"))
    edit_shard = next(EDIT.glob("*.safetensors"))
    base, edit = load_file(str(base_shard)), load_file(str(edit_shard))

    factors = {}
    for m in MODS:
        k = PFX + m + ".weight"
        d = edit[k].float() - base[k].float()
        rows = torch.nonzero(d.abs().sum(1)).flatten().tolist()
        cols = torch.nonzero(d.abs().sum(0)).flatten().tolist()
        if m == "down_proj":
            assert len(cols) == 1, (m, cols)
            v = cols[0]
            B, A = d[:, v : v + 1].clone(), torch.zeros(1, d.shape[1])
            A[0, v] = 1.0
        else:
            assert len(rows) == 1, (m, rows)
            v = rows[0]
            B, A = torch.zeros(d.shape[0], 1), d[v : v + 1].clone()
            B[v, 0] = 1.0
        factors[m] = (B, A, v)
        print(
            json.dumps(
                {
                    "module": m,
                    "neuron": v,
                    "B_norm": round(float(B.norm()), 4),
                    "A_norm": round(float(A.norm()), 4),
                }
            )
        )

    g = torch.Generator().manual_seed(7)
    if CTRL.exists():
        shutil.rmtree(CTRL)
    shutil.copytree(EDIT, CTRL, symlinks=False)
    tensors = load_file(str(edit_shard))
    for m in MODS:
        k = PFX + m + ".weight"
        B, A, _ = factors[m]
        R = torch.randn(B.shape, generator=g)
        R = R / R.norm(dim=0, keepdim=True) * B.norm(dim=0, keepdim=True)
        # control = (edited - edit delta) + control delta == base + token rows + control delta
        orig = base[k].float()
        tensors[k] = (orig + R @ A).to(base[k].dtype).contiguous()
    ctrl_shard = CTRL / edit_shard.name
    ctrl_shard.unlink()
    save_file(tensors, str(ctrl_shard), metadata={"format": "pt"})
    print(json.dumps({"event": "control_checkpoint", "path": str(CTRL)}))

    for name, model in (("claude_edit", EDIT), ("random_control", CTRL)):
        label = f"b_heldout_{name}"
        data = (
            W / "data" / "b_heldout_claude_edit.jsonl"
        )  # identical held-out rows for both
        out = W / "out" / f"gen_{label}.jsonl.gz"
        cmd = [
            sys.executable,
            str(W / "pod_job.py"),
            "generate",
            "--model",
            str(model),
            "--data",
            str(data),
            "--out",
            str(out),
            "--max-tokens",
            "8192",
            "--label",
            label,
        ]
        print(json.dumps({"event": "generate_start", "label": label}), flush=True)
        subprocess.run(cmd, check=True)
    print(json.dumps({"event": "finish_heldout_done"}), flush=True)


if __name__ == "__main__":
    main()
