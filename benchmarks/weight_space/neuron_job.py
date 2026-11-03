"""Regenerate the 'answer computed -> conclude' neuron with the packaged recipe.

Runs on a GPU pod with this repository's wheel installed:

    python neuron_job.py --out /workspace/out --contrast answer_contrast.json \\
        --trace gen_b_base_trace.jsonl.gz [--neuron 1998 --up-unit 44]

1. Rebuild the newline-anchored contrast exactly as the interactive kernel did
   (kernel/nl_read.py + kernel/nl2_write_eval.py): positives = the first two
   line breaks after the gold answer; negatives = working line breaks, post-
   ``</think>`` prefixes, and the answer token itself.
2. Capture layer 23's ``gate_proj`` / ``up_proj`` outputs at those prefixes.
3. ``design_conditional_neuron`` -> a ``NeuronRewrite``; save
   ``conclude_neuron_spec.json`` (with Wait rows x0.5) and its control spec
   (same read vectors, random write direction of equal norm, same Wait rows).
4. Apply both specs to the pinned base checkpoint, verify every tensor, and
   generate practice (200) and held-out (1,000: base, edit, control) with the
   same decoding as every other run; report lenient and strict accuracy and
   strict paired statistics.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import pod_job as pj  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LAYER = 23
THINK_END = 248069
WAIT_IDS = (13784, 13428)
MAXLEN = 3000


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f]


def contrast_sets(tok, contrast: dict, traces: list[dict]) -> tuple[list, list[float]]:
    """Prefixes and +1/-1 labels, in the kernel's order (newline set, then answer tokens)."""
    by_id = {r["id"]: r for r in traces}
    newline = {i for i in range(len(tok)) if "\n" in tok.decode([i])}
    rng = random.Random(1)
    pos, work = [], []
    for m in contrast["meta"]:
        r = by_id.get(m["id"])
        if r is None:
            continue
        p, g, k = r["prompt_ids"], r["gen_ids"], m["answer_token"]
        after = [j for j in range(k, len(g)) if g[j] in newline][:2]
        before = [j for j in range(int(0.1 * k), int(0.9 * k)) if g[j] in newline]
        for j in after:
            if len(p) + j + 1 <= MAXLEN:
                pos.append(p + g[: j + 1])
        for j in rng.sample(before, k=min(2, len(before))):
            if len(p) + j + 1 <= MAXLEN:
                work.append(p + g[: j + 1])
    post = [x for x in contrast["post"] if len(x) <= MAXLEN]
    post = rng.sample(post, k=min(150, len(post)))
    answer_tokens = [x for x in contrast["answer"] if len(x) <= MAXLEN]
    prefixes = pos + work + post + answer_tokens
    labels = [1.0] * len(pos) + [-1.0] * (len(work) + len(post) + len(answer_tokens))
    print(
        json.dumps(
            {
                "event": "contrast",
                "answer_newlines": len(pos),
                "working_newlines": len(work),
                "post": len(post),
                "answer_tokens": len(answer_tokens),
            }
        ),
        flush=True,
    )
    return prefixes, labels


def capture_last(
    model, modules: list[str], prefixes: list[list[int]], batch: int = 4
) -> dict:
    """Each module's output at each prefix's last token (right padding), on CPU float32."""
    import torch

    rows: dict[str, list] = {m: [] for m in modules}
    last: list[int] = []
    hooks = []
    for name in modules:

        def hook(_m, _i, out, name=name):
            o = out.detach()
            idx = torch.tensor(last, device=o.device)
            rows[name].append(
                o[torch.arange(o.shape[0], device=o.device), idx].float().cpu()
            )

        hooks.append(model.get_submodule(name).register_forward_hook(hook))
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    device = next(model.parameters()).device
    try:
        for s in range(0, len(order), batch):
            chunk = [prefixes[i] for i in order[s : s + batch]]
            width = max(len(c) for c in chunk)
            ids = torch.zeros(len(chunk), width, dtype=torch.long)
            mask = torch.zeros(len(chunk), width, dtype=torch.long)
            for i, c in enumerate(chunk):
                ids[i, : len(c)] = torch.tensor(c)
                mask[i, : len(c)] = 1
            last[:] = [len(c) - 1 for c in chunk]
            with torch.no_grad():
                model(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    logits_to_keep=1,
                )
    finally:
        for h in hooks:
            h.remove()
    inverse = torch.tensor(order).argsort()
    return {m: torch.cat(v)[inverse] for m, v in rows.items()}


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    clean = [r["finish"] == "stop" and r["text"].count("</think>") == 1 for r in rows]
    return {
        "n": n,
        "lenient": round(sum(r["correct"] for r in rows) / n, 4),
        "strict": round(
            sum(bool(r["correct"]) and c for r, c in zip(rows, clean, strict=True)) / n,
            4,
        ),
        "hit_limit": round(sum(r["finish"] == "length" for r in rows) / n, 4),
        "multi_think_close": round(
            sum(r["text"].count("</think>") > 1 for r in rows) / n, 4
        ),
    }


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stateset_agents.experimental.weight_space import (
        EditSpec,
        NeuronRewrite,
        TokenRowScale,
        design_conditional_neuron,
        logit_direction,
        matched_control,
        paired_compare,
        strict_rows,
    )
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
        resolve_checkpoint,
        verify_checkpoint,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--contrast", default="/workspace/answer_contrast.json")
    ap.add_argument("--trace", default="/workspace/gen_b_base_trace.jsonl.gz")
    ap.add_argument("--neuron", type=int, default=None)
    ap.add_argument("--up-unit", type=int, default=None)
    ap.add_argument("--write-norm", type=float, default=3.0)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--gain", type=float, default=4.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)  # nosec B615 - pinned
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16
    )  # nosec B615
    model = model.to("cuda").eval()
    prefixes, labels = contrast_sets(
        tok, json.load(open(a.contrast)), read_jsonl(Path(a.trace))
    )
    mlp = f"model.layers.{LAYER}.mlp"
    caps = capture_last(model, [f"{mlp}.gate_proj", f"{mlp}.up_proj"], prefixes)
    w = {
        n: model.get_submodule(f"{mlp}.{n}").weight.detach().float().cpu()
        for n in ("gate_proj", "up_proj", "down_proj")
    }
    embed = model.get_input_embeddings().weight.detach().float().cpu()
    norm_w = model.get_submodule("model.norm").weight.detach().float().cpu()
    direction = logit_direction(embed, [THINK_END], norm_w, zero_centered=True)
    del model
    torch.cuda.empty_cache()

    design = design_conditional_neuron(
        layer=LAYER,
        gate_weight=w["gate_proj"],
        up_weight=w["up_proj"],
        down_weight=w["down_proj"],
        gate_acts=caps[f"{mlp}.gate_proj"],
        up_acts=caps[f"{mlp}.up_proj"],
        labels=torch.tensor(labels),
        write_direction=direction,
        write_norm=a.write_norm,
        gain=a.gain,
        lam=a.lam,
        neuron=a.neuron,
        up_unit=a.up_unit,
    )
    print(json.dumps({"event": "designed", **design.report}), flush=True)
    wait = TokenRowScale(ids=WAIT_IDS, scale=0.5)
    spec = EditSpec(name="conclude-neuron-L23", edits=(wait, design.rewrite))
    ctrl_rw = matched_control(
        EditSpec(name="conclude-neuron-L23", edits=(design.rewrite,)),
        rng=random.Random(0),
        vocab_size=embed.shape[0],
        n_neurons=w["down_proj"].shape[1],
    ).edits[0]
    assert isinstance(ctrl_rw, NeuronRewrite)
    control = EditSpec(name="control:conclude-neuron-L23", edits=(wait, ctrl_rw))
    (out / "conclude_neuron_spec.json").write_text(spec.to_json())
    (out / "conclude_neuron_control_spec.json").write_text(control.to_json())

    base = resolve_checkpoint(MODEL, REVISION)
    ckpts = {}
    for name, s in (("neuron", spec), ("control", control)):
        dst = pj.WORK / "ckpt" / f"spec_{name}"
        manifest = apply_spec_to_checkpoint(s, base, dst)
        verify_checkpoint(s, base, dst)
        ckpts[name] = str(dst)
        print(
            json.dumps(
                {
                    "event": "checkpoint_verified",
                    "name": name,
                    "hash": manifest["spec_hash"],
                    "changed_weights": manifest["changed_weights"],
                }
            ),
            flush=True,
        )

    probs = pj.load_problems(256, 200, 500)
    result: dict = {
        "design": design.report,
        "spec_hash": spec.content_hash(),
        "control_hash": control.content_hash(),
    }
    practice = pj.generate(
        "spec_neuron_select", ckpts["neuron"], probs["select"][:200], 8192, out
    )
    result["practice"] = stats(practice)
    print(json.dumps({"event": "practice", **result["practice"]}), flush=True)
    held = {
        "base": pj.generate("spec_base_heldout", pj.MODEL, probs["heldout"], 8192, out),
        "neuron": pj.generate(
            "spec_neuron_heldout", ckpts["neuron"], probs["heldout"], 8192, out
        ),
        "control": pj.generate(
            "spec_control_heldout", ckpts["control"], probs["heldout"], 8192, out
        ),
    }
    result["heldout"] = {k: stats(v) for k, v in held.items()}
    strict = {k: strict_rows(v) for k, v in held.items()}
    result["paired_strict"] = {
        "neuron_vs_base": paired_compare(strict["base"], strict["neuron"]),
        "neuron_vs_control": paired_compare(strict["control"], strict["neuron"]),
        "control_vs_base": paired_compare(strict["base"], strict["control"]),
    }
    (out / "neuron_summary.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({"event": "neuron_job_done", "heldout": result["heldout"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
