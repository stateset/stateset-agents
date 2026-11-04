"""Design a "</think> already seen" guard for the conclude neuron, offline (CPU).

    python guard_design.py --think gen_op_r0_think.jsonl.gz \\
        --nothink gen_op_r0_nothink.jsonl.gz \\
        --start-spec specs/conclude_neuron_v3t_spec.json [more specs] --out designs/

The conclude neuron also fires after reasoning has closed: a second
``</think>`` fails a thinking-on reply under strict scoring, and with thinking
off (whose template already ends ``<think>\\n\\n</think>\\n\\n``) it writes stray
closes. The base model leaks none, so the fix is an AND gate on the neuron:
fire only while no ``</think>`` is in context. No existing head looks back at
an earlier ``</think>`` (at most 7% of any full-attention head's weight), so a
query head of the neuron's layer is rewritten into one that does, and it
writes a private flag that silences the neuron. Every quantity is a mean over
positions labelled exactly by the token sequence; nothing is fitted.

1. Run the base model over edited-model replies (``--think`` replies that
   closed, ``--nothink`` replies), capturing at ``--layer``: the attention input,
   keys (after the key norm) and values of the head's KV group, and the
   residual entering the MLP norm.
2. Query: :func:`circuits.lookback_query` on the keys at ``</think>``
   (position-free dims). Query and gate read the mean attention input, which
   must be positive at every captured position (checked).
3. Simulate the head's causal attention; ``out_read`` is the mean head output
   after a ``</think>`` minus before, orthogonal to the before-mean and scaled
   to read 1 after.
4. Flag: :func:`atlas.quiet_direction` (``--quiet-k`` logit-least-visible
   directions, least used by the MLP's input). The head writes
   ``flag_read * median rms`` along it, so the MLP's normalised input carries
   the flag at about ``flag_read``.
5. Each start spec's conclude neuron gets :func:`circuits.silence_on_flag`:
   its near-constant multiplier drops by 1 and its gate by
   ``gate_weight * flag_read`` once the flag is up.

Pushing the ``</think>`` logit down directly from the head (an earlier design)
needed write strengths that swamp the small final residual (rms ~0.3): it
blocked every second close but also every reply's end and ten-fold raised
loss on ordinary text. Which head to rewrite matters: in Qwen3.5-0.8B layer 23
heads 1 and 3 hold back ``</think>`` at line breaks and head 2 promotes it;
head 0 (the default) is the most neutral.

Writes ``<start>+guard_spec.json`` per start spec, a control whose flag is
written in a random direction of the same norm (the neuron hears nothing),
and ``guard_design.json`` with the diagnostics.
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

    import torch
    from transformers import AutoModelForCausalLM

    from stateset_agents.experimental.weight_space import (
        EditSpec,
        HeadRewrite,
        NeuronRewrite,
        lookback_query,
        quiet_direction,
        silence_on_flag,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--think", required=True)
    ap.add_argument("--nothink", required=True)
    ap.add_argument("--start-spec", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--n", type=int, default=6, help="replies per kind")
    ap.add_argument("--quiet-k", type=int, default=512)
    ap.add_argument("--flag-read", type=float, default=4.2)
    ap.add_argument("--gate-weight", type=float, default=1.5)
    ap.add_argument("--gate", type=float, default=0.1, help="head output gate weight")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32
    ).eval()  # nosec B615 - pinned revision
    cfg = model.config
    hd = cfg.head_dim
    rot = int(hd * cfg.rope_parameters["partial_rotary_factor"])
    kv = a.head // (cfg.num_attention_heads // cfg.num_key_value_heads)
    block = model.model.layers[a.layer]
    att = block.self_attn
    cap: dict = {}
    block.input_layernorm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("x", o[0].detach())
    )
    block.post_attention_layernorm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("res", i[0][0].detach())
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
            model.model(torch.tensor([ids]))
        data.append({"ids": ids, **{k: v.clone() for k, v in cap.items()}})

    x_all = torch.cat([d["x"] for d in data])
    read_dir = x_all.mean(0)
    reads = x_all @ read_dir
    if float(reads.min()) <= 0:
        raise SystemExit("the mean attention input is not a positive read everywhere")
    is_end = torch.cat([torch.tensor([t == END for t in d["ids"]]) for d in data])
    keys = torch.cat([d["k"] for d in data])
    q_gain = 1.0 + att.q_norm.weight.detach()
    query_out = lookback_query(keys[is_end], q_gain, rotary_dims=rot)
    q = query_out / query_out.pow(2).mean().sqrt() * q_gain  # the normalised query

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
    readout = sd["model.embed_tokens.weight"].float() * (
        1.0 + sd["model.norm.weight"].float()
    )
    res = torch.cat([d["res"] for d in data])
    rms = res.pow(2).mean(1).sqrt()
    flag = quiet_direction(readout, res, k=a.quiet_k)
    write_norm = a.flag_read * float(rms.median())
    mlp_gain = 1.0 + block.post_attention_layernorm.weight.detach().float()
    use = (res / rms.unsqueeze(1)) @ flag
    visible = (readout @ flag).abs().max() / readout[END].norm()

    def head_writing(direction: torch.Tensor) -> HeadRewrite:
        return HeadRewrite(
            layer=a.layer,
            head=a.head,
            query_out=tuple(float(v) for v in query_out),
            query_in=tuple(float(v) for v in read_dir),
            gate_out=tuple(a.gate for _ in range(hd)),
            gate_in=tuple(float(v) for v in read_dir),
            out_write=tuple(float(write_norm * v) for v in direction),
            out_read=tuple(float(v) for v in out_read),
        )

    g = torch.Generator().manual_seed(0)
    noise = torch.randn(flag.shape[0], generator=g)
    guard, control_head = head_writing(flag), head_writing(noise / noise.norm())
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
        "residual_rms_median": round(float(rms.median()), 4),
        "flag_write_norm": round(write_norm, 4),
        "flag_residual_use_std": round(float(use.std()), 4),
        "flag_logit_visibility": round(float(visible), 4),
        "flag_read": a.flag_read,
        "gate_weight": a.gate_weight,
        "specs": [],
    }
    for path in a.start_spec:
        start = EditSpec.from_json(Path(path).read_text())
        (neuron,) = [e for e in start.edits if isinstance(e, NeuronRewrite)]
        gated = silence_on_flag(
            neuron,
            flag,
            norm_gain=mlp_gain,
            flag_read=a.flag_read,
            gate_weight=a.gate_weight,
        )
        edits = tuple(gated if e is neuron else e for e in start.edits)
        spec = EditSpec(name=f"{start.name}+guard", edits=(*edits, guard))
        control = EditSpec(name=f"control:{spec.name}", edits=(*edits, control_head))
        (out / f"{spec.name}_spec.json").write_text(spec.to_json())
        (out / f"{spec.name}_control_spec.json").write_text(control.to_json())
        report["specs"].append(
            {"start": start.name, "name": spec.name, "spec_hash": spec.content_hash()}
        )
    (out / "guard_design.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
