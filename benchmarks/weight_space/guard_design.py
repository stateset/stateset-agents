"""Design a "</think> already seen" guard head offline (CPU) and stack it on a spec.

    python guard_design.py --think gen_op_r0_think.jsonl.gz \\
        --nothink gen_op_r0_nothink.jsonl.gz \\
        --start-spec specs/conclude_neuron_v3t_spec.json [more specs] --out designs/ \\
        --strength 16 32

The conclude neuron also fires after reasoning has closed: a second
``</think>`` fails a thinking-on reply under strict scoring, and with thinking
off (whose template already ends ``<think>\\n\\n</think>\\n\\n``) it writes stray
closes. One rule covers both: once ``</think>`` is in context, never write it
again. No existing head looks back at an earlier ``</think>`` (at most 7% of
any full-attention head's weight), so this builds one, with no fitting; every
quantity is a mean over positions labelled exactly by the token sequence:

1. Run the base model over edited-model replies (``--think`` replies that
   closed, ``--nothink`` replies), capturing at ``--layer``: the attention
   input, and keys (after the key norm) and values of the query head's KV group.
2. Query: :func:`circuits.lookback_query` on the keys at ``</think>``
   (position-free dims only). Query and gate read the mean attention input,
   which must be positive at every captured position (checked).
3. Simulate the head's causal attention; ``out_read`` is the mean head output
   after a ``</think>`` minus before, made orthogonal to the before-mean and
   scaled to read 1 after.
4. Write: ``-strength`` times :func:`atlas.protected_write` for ``</think>``,
   with every token whose readout overlaps more than ``--protect-overlap``
   protected (``<|im_end|>`` must stay reachable after a close).

Writes ``<start>+guard_g<strength>_spec.json`` per strength, its random-write
control, and ``guard_design.json`` (attention and read diagnostics).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
END = 248069
MAXLEN = 2400


def read_gz(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def sequences(think: list[dict], nothink: list[dict], n: int) -> list[list[int]]:
    """``n`` thinking-on replies that closed, then ``n`` thinking-off replies."""
    out = [
        r["prompt_ids"] + r["gen_ids"]
        for r in think
        if END in r["gen_ids"] and len(r["prompt_ids"]) + len(r["gen_ids"]) <= MAXLEN
    ][:n]
    return out + [(r["prompt_ids"] + r["gen_ids"])[:MAXLEN] for r in nothink[:n]]


def main(argv: list[str] | None = None) -> None:
    import random

    import torch
    from transformers import AutoModelForCausalLM

    from stateset_agents.experimental.weight_space import (
        EditSpec,
        HeadRewrite,
        lookback_query,
        matched_control,
        protected_write,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--think", required=True)
    ap.add_argument("--nothink", required=True)
    ap.add_argument("--start-spec", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--head", type=int, default=1)
    ap.add_argument("--n", type=int, default=6, help="replies per kind")
    ap.add_argument("--strength", type=float, nargs="+", default=[16.0, 32.0])
    ap.add_argument("--protect-overlap", type=float, default=0.2)
    ap.add_argument("--gate", type=float, default=0.1)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32
    ).eval()  # nosec B615 - pinned revision
    cfg = model.config
    hd = cfg.head_dim
    rot = int(hd * cfg.rope_parameters["partial_rotary_factor"])
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    kv = a.head // group
    att = model.model.layers[a.layer].self_attn
    cap: dict = {}
    model.model.layers[a.layer].input_layernorm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("x", o[0].detach())
    )
    att.k_norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("k", o[0, :, kv].detach())
    )
    att.v_proj.register_forward_hook(
        lambda m, i, o: cap.__setitem__(
            "v", o[0].view(o.shape[1], -1, hd)[:, kv].detach()
        )
    )
    seqs = sequences(read_gz(a.think), read_gz(a.nothink), a.n)
    data = []
    for ids in seqs:
        with torch.no_grad():
            model(torch.tensor([ids]))
        data.append({"ids": ids, **{k: v.clone() for k, v in cap.items()}})

    x_all = torch.cat([d["x"] for d in data])
    read_dir = x_all.mean(0)
    reads = x_all @ read_dir
    if float(reads.min()) <= 0:
        raise SystemExit("the mean attention input is not a positive read everywhere")
    is_end = torch.cat([torch.tensor([t == END for t in d["ids"]]) for d in data])
    keys = torch.cat([d["k"] for d in data])
    gain = 1.0 + att.q_norm.weight.detach()
    query_out = lookback_query(keys[is_end], gain, rotary_dims=rot)
    q = query_out / query_out.pow(2).mean().sqrt() * gain  # the normalised query

    after_o, before_o, weight_on_end = [], [], []
    for d in data:
        ids = d["ids"]
        s = (d["k"] @ q) / hd**0.5
        first = ids.index(END) if END in ids else None
        for t in range(len(ids)):
            w = torch.softmax(s[: t + 1], 0)
            o = w @ d["v"][: t + 1]
            if first is not None and t > first:
                after_o.append(o)
                weight_on_end.append(
                    float(sum(w[j] for j in range(t + 1) if ids[j] == END))
                )
            elif first is None or t < first:
                before_o.append(o)
    after, before = torch.stack(after_o), torch.stack(before_o)
    mb, ma = before.mean(0), after.mean(0)
    delta = ma - mb
    out_read = delta - (delta @ mb) / (mb @ mb) * mb
    out_read = out_read / (out_read @ ma)

    sd = model.state_dict()
    embed, norm = sd["model.embed_tokens.weight"], sd["model.norm.weight"]
    readout = embed.double() * (1.0 + norm.double())
    rel = (readout @ readout[END]) / (readout[END] @ readout[END])
    protect = [
        int(i)
        for i in torch.nonzero(rel.abs() > a.protect_overlap).flatten()
        if int(i) != END
    ]
    write = protected_write(embed, END, norm, protect=protect)

    starts = [EditSpec.from_json(Path(p).read_text()) for p in a.start_spec]
    w_end = torch.tensor(weight_on_end)
    ra, rb = after @ out_read, before @ out_read
    report: dict = {
        "layer": a.layer,
        "head": a.head,
        "kv_head": kv,
        "sequences": len(seqs),
        "query_read_min": round(float(reads.min()), 3),
        "query_norm": round(float(q.norm()), 3),
        "weight_on_think_end_after": {
            "min": round(float(w_end.min()), 4),
            "median": round(float(w_end.median()), 4),
        },
        "value_read_before": {
            "p99": round(float(torch.quantile(rb, 0.99)), 4),
            "max": round(float(rb.max()), 4),
        },
        "value_read_after": {
            "min": round(float(ra.min()), 4),
            "median": round(float(ra.median()), 4),
        },
        "protected_tokens": protect,
        "write_logit_per_unit": round(float(readout[END].float() @ write), 4),
        "specs": [],
    }
    for g in a.strength:
        head = HeadRewrite(
            layer=a.layer,
            head=a.head,
            query_out=tuple(float(v) for v in query_out),
            query_in=tuple(float(v) for v in read_dir),
            gate_out=tuple(a.gate for _ in range(hd)),
            gate_in=tuple(float(v) for v in read_dir),
            out_write=tuple(float(-g * v) for v in write),
            out_read=tuple(float(v) for v in out_read),
        )
        # Control: the start's edits unchanged, the guard's write random (same norm).
        (random_head,) = matched_control(
            EditSpec(name="guard", edits=(head,)),
            rng=random.Random(0),
            vocab_size=embed.shape[0],
            n_neurons=1,
        ).edits
        for start in starts:
            spec = EditSpec(
                name=f"{start.name}+guard_g{g:g}", edits=(*start.edits, head)
            )
            control = EditSpec(
                name=f"control:{spec.name}", edits=(*start.edits, random_head)
            )
            (out / f"{spec.name}_spec.json").write_text(spec.to_json())
            (out / f"{spec.name}_control_spec.json").write_text(control.to_json())
            report["specs"].append(
                {
                    "strength": g,
                    "start": start.name,
                    "name": spec.name,
                    "spec_hash": spec.content_hash(),
                }
            )
    (out / "guard_design.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
