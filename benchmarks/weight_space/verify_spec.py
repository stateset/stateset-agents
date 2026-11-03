"""Verify every edit in a token spec against the real checkpoint: exact tensors, rows/cols, scales.

python verify_spec.py token_spec_runN.json
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WS_WORK"] = tempfile.mkdtemp()
H = Path(__file__).parent
spec_ = importlib.util.spec_from_file_location("pod_job", H / "pod_job.py")
pj = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(pj)

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

spec = json.loads(Path(sys.argv[1]).read_text())
P = "model.language_model."
shard = "model.safetensors-00001-of-00001.safetensors"
base = safe_open(
    str(Path(snapshot_download(pj.MODEL, revision=pj.REVISIONS[pj.MODEL])) / shard),
    "pt",
)
for name, entries in spec["edits"].items():
    ck = Path(pj.write_checkpoint(f"v_{name}", {}, token_edit=entries))
    expect: dict[str, dict] = {}
    for e in entries:
        if e.get("kind") == "down_cols":
            expect.setdefault(
                f"{P}layers.{e['layer']}.mlp.down_proj.weight", {}
            ).update({("col", j): e["scale"] for j in e["neurons"]})
        else:
            expect.setdefault(P + "embed_tokens.weight", {}).update(
                {("row", i): e["scale"] for i in e["ids"]}
            )
    with safe_open(str(ck / shard), "pt") as f:
        changed = [
            k for k in f.keys() if not torch.equal(base.get_tensor(k), f.get_tensor(k))
        ]
        want_changed = [
            k for k, v in expect.items() if any(s != 1.0 for s in v.values())
        ]
        assert sorted(changed) == sorted(want_changed), (name, changed, want_changed)
        for k in changed:
            A, B = base.get_tensor(k).float(), f.get_tensor(k).float()
            diff_rows = set(torch.nonzero((A != B).any(1)).flatten().tolist())
            diff_cols = set(torch.nonzero((A != B).any(0)).flatten().tolist())
            for (axis, idx), scale in expect[k].items():
                got = (B[idx] / A[idx]) if axis == "row" else (B[:, idx] / A[:, idx])
                g = float(
                    got[A[idx].abs() > 1e-3].median()
                    if axis == "row"
                    else got[A[:, idx].abs() > 1e-3].median()
                )
                assert abs(g - scale) < 0.01, (name, k, axis, idx, g, scale)
            rows_exp = {i for (ax, i) in expect[k] if ax == "row"}
            cols_exp = {j for (ax, j) in expect[k] if ax == "col"}
            if rows_exp:
                assert diff_rows <= rows_exp, (name, k, diff_rows - rows_exp)
            if cols_exp:
                assert diff_cols <= cols_exp, (name, k, diff_cols - cols_exp)
    shutil.rmtree(ck)
    print(f"ok  {name:22s} tensors changed: {len(changed)}")
print("SPEC VERIFIED:", sys.argv[1])
shutil.rmtree(os.environ["WS_WORK"])
