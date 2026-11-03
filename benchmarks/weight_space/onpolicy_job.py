"""Practice-gated, on-policy redesign of the conclude neuron (runs on a GPU pod).

    python onpolicy_job.py --out /workspace/out \\
        --start-spec /workspace/conclude_neuron_v3t_spec.json --rounds 3 --heldout

v4 and v5 improved fire rates on contexts curated from the *base* model's
traces and lost 5 and 19 points in generation: the neuron misfires in the
edited model's own trajectories, which curated contexts never cover. This job
judges every design by generation and learns from the edit's own replies:

Round 0 is the starting spec (v3t). Each round:

1. build the round's checkpoint (verified tensor by tensor);
2. generate thinking on and off (cap ``--practice-tokens``) on two disjoint
   TRAINING sets: MINE (GSM8K-train ``select[:N]`` + N MATH-train) and SELECT
   (GSM8K-train ``select[N:2N]`` + N other MATH-train), held-out never touched;
3. score SELECT strictly (thinking on: accuracy; thinking off: stray rate);
4. mine labelled contexts from the MINE replies (``misfires.mine``: extra,
   premature and stray closes as negatives; clean correct closes and
   unclosed answers as positives), capture the base layer at them;
5. refit on the start design's training contexts (v3t's: GSM8K contrast,
   thinking-off and prose line breaks; even indices, as v3t was fit) plus
   every context mined so far.

``misfires.choose_round`` keeps the round with the best SELECT strict score
whose thinking-off stray rate rose at most ``--max-stray-rise``. With
``--heldout`` the kept spec then runs the held-out 1,000 (cap 8,192) on the
same pod with the start spec and base thinking off, paired as in
``thinking_budget_job``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import neuron_job as nj  # noqa: E402
import pod_job as pj  # noqa: E402
from misfires import LABELS, choose_round, mine  # noqa: E402

MINED_MAXLEN = 5000


def practice_sets(n: int, probs: dict) -> tuple[list[dict], list[dict]]:
    """Two disjoint training sets (MINE, SELECT), GSM8K-train + MATH-train."""
    gsm = probs["select"]
    math = nj.math_train_rows(2 * n, probs["heldout"], seed=11)
    return gsm[:n] + math[:n], gsm[n : 2 * n] + math[n : 2 * n]


def select_scores(think: list[dict], nothink: list[dict]) -> dict:
    from stateset_agents.experimental.weight_space import strict_rows

    def frac(xs: list) -> float:
        return round(sum(xs) / max(1, len(xs)), 4)

    st = strict_rows(think, thinking=True)
    return {
        "select_think_strict": frac([r["correct"] for r in st]),
        "select_think_lenient": frac([bool(r["correct"]) for r in think]),
        "select_think_multi_close": frac(
            [r["text"].count("</think>") > 1 for r in think]
        ),
        "select_think_no_close": frac(["</think>" not in r["text"] for r in think]),
        "select_nothink_strict": frac(
            [r["correct"] for r in strict_rows(nothink, thinking=False)]
        ),
        "select_nothink_stray": frac(["</think>" in r["text"] for r in nothink]),
        "by_kind": {
            k: frac([r["correct"] for r in st if r["kind"] == k])
            for k in ("gsm8k", "math")
        },
    }


def mine_replies(
    tok, newline: set[int], think: list[dict], nothink: list[dict], cap: int, seed: int
) -> list[tuple[str, list[int]]]:
    """Labelled prefixes from one round's MINE replies, at most ``cap`` per group."""
    from answer_positions import answer_end

    found: dict[str, list[list[int]]] = {g: [] for g in LABELS}
    for rows, thinking in ((think, True), (nothink, False)):
        for r in rows:
            k = None
            if thinking and len(r["gold"].strip().strip("$")) >= 2:
                end = answer_end(r["text"], r["gold"])
                if end is not None:
                    k = nj.token_index_for_char(tok, r["gen_ids"], end)
            for group, i in mine(
                r["gen_ids"],
                newline,
                nj.THINK_END,
                thinking=thinking,
                answer_token=k,
                correct=bool(r["correct"]),
                finish=r["finish"],
            ):
                prefix = r["prompt_ids"] + r["gen_ids"][: i + 1]
                if len(prefix) <= MINED_MAXLEN:
                    found[group].append(prefix)
    rng = random.Random(seed)
    out = []
    for group, ps in found.items():
        out += [(group, p) for p in rng.sample(ps, k=min(cap, len(ps)))]
    return out


def capture(prefixes: list[list[int]]) -> tuple:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        nj.MODEL, revision=nj.REVISION, dtype=torch.bfloat16
    )  # nosec B615 - pinned revision
    model = model.to("cuda").eval()
    mlp = f"model.layers.{nj.LAYER}.mlp"
    caps = nj.capture_last(model, [f"{mlp}.gate_proj", f"{mlp}.up_proj"], prefixes, 2)
    w = {
        n: model.get_submodule(f"{mlp}.{n}").weight.detach().float().cpu()
        for n in ("gate_proj", "up_proj", "down_proj")
    }
    w["embed"] = model.get_input_embeddings().weight.detach().float().cpu()
    w["norm"] = model.get_submodule("model.norm").weight.detach().float().cpu()
    del model
    torch.cuda.empty_cache()
    return caps[f"{mlp}.gate_proj"], caps[f"{mlp}.up_proj"], w


def main(argv: list[str] | None = None) -> None:
    import torch
    from transformers import AutoTokenizer

    from stateset_agents.experimental.weight_space import (
        EditSpec,
        TokenRowScale,
        design_conditional_neuron,
        logit_direction,
    )
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
        resolve_checkpoint,
        verify_checkpoint,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--start-spec", required=True)
    ap.add_argument("--contrast", default="/workspace/answer_contrast.json")
    ap.add_argument("--trace", default="/workspace/gen_b_base_trace.jsonl.gz")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--practice-n", type=int, default=100, help="per kind, per set")
    ap.add_argument("--practice-tokens", type=int, default=4096)
    ap.add_argument("--mine-cap", type=int, default=150, help="per group per round")
    ap.add_argument("--max-stray-rise", type=float, default=0.02)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--write-norm", type=float, default=3.0)
    ap.add_argument("--heldout", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (pj.WORK / "data").mkdir(parents=True, exist_ok=True)
    (pj.WORK / "ckpt").mkdir(parents=True, exist_ok=True)

    probs = pj.load_problems(256, 200, 500)
    mine_set, select_set = practice_sets(a.practice_n, probs)
    # v3t's thinking-off negatives (GSM8K-train trace split), as neuron_job builds them.
    pj.generate("op_nothink_train", pj.MODEL, probs["trace"][:256], 2048, out, False)
    tok = AutoTokenizer.from_pretrained(nj.MODEL, revision=nj.REVISION)  # nosec B615
    newline = nj.newline_ids(tok)
    prefixes, labels, groups = nj.contrast_sets(
        tok, json.load(open(a.contrast)), nj.read_jsonl(Path(a.trace))
    )
    for p in nj.nothink_negatives(newline, out / "gen_op_nothink_train.jsonl.gz", 300):
        prefixes.append(p)
        labels.append(-1.0)
        groups.append("nothink_newline")
    for p in nj.prose_negatives(tok, newline, 150):
        prefixes.append(p)
        labels.append(-1.0)
        groups.append("prose_newline")
    g0, u0, w = capture(prefixes)
    direction = logit_direction(
        w["embed"], [nj.THINK_END], w["norm"], zero_centered=True
    )
    train0 = torch.arange(len(prefixes)) % 2 == 0  # how v3t was fit
    mined_g, mined_u, mined_labels, mined_groups = [], [], [], []

    base = resolve_checkpoint(nj.MODEL, nj.REVISION)
    wait = TokenRowScale(ids=nj.WAIT_IDS, scale=0.5)
    spec = EditSpec.from_json(Path(a.start_spec).read_text())
    rounds: list[dict] = []
    specs: list = []
    summary_path = out / "onpolicy_summary.json"
    for r in range(a.rounds + 1):
        if r > 0:
            design = design_conditional_neuron(
                layer=nj.LAYER,
                gate_weight=w["gate_proj"],
                up_weight=w["up_proj"],
                down_weight=w["down_proj"],
                gate_acts=torch.cat([g0, *mined_g]),
                up_acts=torch.cat([u0, *mined_u]),
                labels=torch.tensor(labels + mined_labels),
                write_direction=direction,
                write_norm=a.write_norm,
                gain=a.gain,
                lam=a.lam,
                neuron=1998,
                up_unit=44,
                train=torch.cat(
                    [train0, torch.ones(len(mined_labels), dtype=torch.bool)]
                ),
            )
            spec = EditSpec(
                name=f"conclude-neuron-L23_op{r}", edits=(wait, design.rewrite)
            )
        specs.append(spec)
        (out / f"onpolicy_r{r}_spec.json").write_text(spec.to_json())
        ckpt = pj.WORK / "ckpt" / f"op{r}"
        manifest = apply_spec_to_checkpoint(spec, base, ckpt)
        verify_checkpoint(spec, base, ckpt)
        both = mine_set + select_set
        think = pj.generate(f"op_r{r}_think", str(ckpt), both, a.practice_tokens, out)
        nothink = pj.generate(
            f"op_r{r}_nothink", str(ckpt), both, a.practice_tokens, out, thinking=False
        )
        sel_ids = {x["id"] for x in select_set}
        scores = select_scores(
            [x for x in think if x["id"] in sel_ids],
            [x for x in nothink if x["id"] in sel_ids],
        )
        row = {
            "round": r,
            "spec": spec.name,
            "spec_hash": manifest["spec_hash"],
            **scores,
        }
        if r < a.rounds:
            mined = mine_replies(
                tok,
                newline,
                [x for x in think if x["id"] not in sel_ids],
                [x for x in nothink if x["id"] not in sel_ids],
                a.mine_cap,
                seed=100 + r,
            )
            if mined:
                gm, um, _ = capture([p for _, p in mined])
                mined_g.append(gm)
                mined_u.append(um)
                mined_labels += [LABELS[g] for g, _ in mined]
                mined_groups += [g for g, _ in mined]
            row["mined_this_round"] = {g: sum(x == g for x, _ in mined) for g in LABELS}
        rounds.append(row)
        pj.log(event="round", **{k: v for k, v in row.items() if k != "by_kind"})
        summary_path.write_text(json.dumps({"rounds": rounds}, indent=2))

    keep = choose_round(rounds, max_stray_rise=a.max_stray_rise)
    result: dict = {"rounds": rounds, "kept_round": keep}
    pj.log(event="kept", round=keep, spec_hash=rounds[keep]["spec_hash"])
    if a.heldout and keep > 0:
        import thinking_budget_job as tb

        result.update(
            tb.heldout_check(
                probs["heldout"],
                str(pj.WORK / "ckpt" / "op0"),
                str(pj.WORK / "ckpt" / f"op{keep}"),
                out,
                "op",
                lambda part: summary_path.write_text(
                    json.dumps({**result, **part}, indent=2)
                ),
            )
        )
    summary_path.write_text(json.dumps(result, indent=2))
    pj.log(event="onpolicy_done", kept=keep)


if __name__ == "__main__":
    main()
