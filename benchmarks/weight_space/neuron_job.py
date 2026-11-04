"""Regenerate the 'answer computed -> conclude' neuron with the packaged recipe.

Runs on a GPU pod with this repository's wheel installed:

    python neuron_job.py --out /workspace/out --contrast answer_contrast.json \\
        --trace gen_b_base_trace.jsonl.gz [--neuron 1998 --up-unit 44]

For a new 2B design, generate its own training traces and stop before evaluation:

    python neuron_job.py --model Qwen/Qwen3.5-2B --generate-traces \\
        --nothink-train 256 --prose-negatives 200 --save-captures \\
        --design-only --tag _2b --out /workspace/out

Select candidates with ``practice_job.py --model`` before any held-out run.
The recipe supports the two pinned Qwen3.5 sizes; it does not transfer the
0.8B neuron's vectors or slot choices to 2B. Captures and summaries identify
the model and revision, which ``offline_design.py`` uses for further designs.

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
import re
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
MATH_TRAIN = "DigitalLearningGmbH/MATH-lighteval"
MATH_TRAIN_FILE = "data/train-00000-of-00001.parquet"
MATH_TRAIN_REVISION = "0530c78699ea5e8eb5530600900e1f328b48acad"


def validate_design_source(records: list[dict], model: str) -> None:
    """Reject another model's traces/captures; accept legacy files only for 0.8B."""
    revision = pj.REVISIONS[model]
    for record in records:
        if model == MODEL and "model" not in record and "revision" not in record:
            continue
        if record.get("model") != model or record.get("revision") != revision:
            raise ValueError(
                f"design input must identify {model} at revision {revision}"
            )


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f]


def newline_ids(tok) -> set[int]:
    return {i for i in range(len(tok)) if "\n" in tok.decode([i])}


def nothink_negatives(newline: set[int], path: Path, limit: int, seed: int = 2) -> list:
    """Line breaks in the answer part of thinking-off replies (prompt + reply prefix).

    These are contexts where an answer has just been written but the model is
    *not* thinking, so a conclude neuron must stay quiet there.
    """
    rng = random.Random(seed)
    candidates = []
    for r in read_jsonl(path):
        p, g = r["prompt_ids"], r["gen_ids"]
        lines = [j for j in range(int(0.3 * len(g)), len(g)) if g[j] in newline]
        for j in rng.sample(lines, k=min(2, len(lines))):
            if len(p) + j + 1 <= MAXLEN:
                candidates.append(p + g[: j + 1])
    return rng.sample(candidates, k=min(limit, len(candidates)))


def post_close_negatives(
    newline: set[int], path: Path, limit: int, per_reply: int = 6, seed: int = 8
) -> list:
    """Line breaks after the first ``</think>`` in an edited model's run-on replies.

    v4 showed the conclude neuron firing again inside answer sections where the
    model kept reasoning after an early close (a second ``</think>``). The
    ``post`` negatives come from clean, formatted answers; these come from
    run-on ones (two or more closes, or one close and then the token cap),
    starting with the line break right before each extra ``</think>`` (where
    the neuron fired), then other line breaks after the first close. ``path``
    must hold replies to *training* problems.
    """
    rng = random.Random(seed)
    out = []
    for r in read_jsonl(path):
        p, g = r["prompt_ids"], r["gen_ids"]
        if THINK_END not in g:
            continue
        first = g.index(THINK_END)
        extra = [j for j in range(first + 1, len(g)) if g[j] == THINK_END]
        if not extra and r["finish"] != "length":
            continue
        before_extra = [j - 1 for j in extra if g[j - 1] in newline]
        others = [
            j
            for j in range(first + 1, len(g))
            if g[j] in newline and j not in before_extra
        ]
        picks = before_extra + rng.sample(others, k=min(per_reply, len(others)))
        for j in picks[:per_reply]:
            if len(p) + j + 1 <= MAXLEN:
                out.append(p + g[: j + 1])
    return rng.sample(out, k=min(limit, len(out)))


def prose_negatives(tok, newline: set[int], n: int, seed: int = 3) -> list:
    """Line breaks in plain prose (WikiText-2 *train*; the side-effect check uses test)."""
    from datasets import load_dataset
    from side_effect_job import WIKITEXT_REVISION

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        revision=WIKITEXT_REVISION,
    )  # nosec B615 - pinned
    ids = tok("\n\n".join(ds["text"][:4000]), add_special_tokens=False).input_ids
    rng = random.Random(seed)
    lines = [j for j in range(512, len(ids)) if ids[j] in newline]
    return [ids[j - 511 : j + 1] for j in rng.sample(lines, k=min(n, len(lines)))]


RESULT_LINE = re.compile(
    r"=\s*\\?\$?\s*-?\d[\d,]*(?:\.\d+)?(?:\s*\\?%)?\s*\\?\$?"
    r"(?:\s*(?:\\text\{[^}]*\}|[A-Za-z][A-Za-z/ ]{0,24}))?\s*\$?\s*\.?\**\s*$"
)


def intermediate_negatives(
    tok,
    contrast: dict,
    traces: list[dict],
    newline: set[int],
    per_trace: int = 3,
    seed: int = 4,
) -> list:
    """Line breaks right after an intermediate result, before the gold answer.

    v3t's premature closes (the conclude neuron firing on "Total = 126 x 6 =
    756" when the question asks for something further) come from lines that
    end in a computed number. The ``working`` negatives are random line breaks
    mid-trace, mostly not after a result; these are the hard ones: a line that
    ends "= <number> [units]" strictly before the model first writes the gold
    answer as a result (``answer_token``), so the number is not the answer.
    """
    by_id = {r["id"]: r for r in traces}
    rng = random.Random(seed)
    out = []
    for m in contrast["meta"]:
        r = by_id.get(m["id"])
        if r is None:
            continue
        p, g, k = r["prompt_ids"], r["gen_ids"], m["answer_token"]
        breaks = [j for j in range(k) if g[j] in newline]
        hits = []
        starts = [-1, *breaks][: len(breaks)]  # the break before each break
        for prev, j in zip(starts, breaks, strict=True):
            line = tok.decode(g[prev + 1 : j]).strip()
            if RESULT_LINE.search(line) and len(p) + j + 1 <= MAXLEN:
                hits.append(j)
        for j in rng.sample(hits, k=min(per_trace, len(hits))):
            out.append(p + g[: j + 1])
    return out


def math_train_rows(n: int, heldout: list[dict], seed: int = 0) -> list[dict]:
    """``n`` MATH *train* problems (pinned), none matching a MATH-500 question.

    Problems whose question already states the answer are skipped, as for GSM8K.
    """
    import pyarrow.parquet as pq
    from answer_positions import gold_in_question
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        MATH_TRAIN, MATH_TRAIN_FILE, repo_type="dataset", revision=MATH_TRAIN_REVISION
    )  # nosec B615 - pinned revision
    held = {" ".join(r["question"].split()) for r in heldout if r["kind"] == "math"}
    rows, overlap = [], 0
    for i, r in enumerate(pq.read_table(path).to_pylist()):
        gold = pj.last_boxed(r["solution"])
        if gold is None or gold_in_question(r["problem"], gold):
            continue
        if " ".join(r["problem"].split()) in held:
            overlap += 1
            continue
        rows.append(
            {
                "id": f"mathtrain-{i}",
                "kind": "math",
                "question": r["problem"],
                "gold": gold,
            }
        )
    print(
        json.dumps(
            {
                "event": "math_train",
                "usable": len(rows),
                "dropped_heldout_overlap": overlap,
            }
        ),
        flush=True,
    )
    return random.Random(seed).sample(rows, k=min(n, len(rows)))


def token_index_for_char(tok, gen_ids: list[int], char_end: int) -> int:
    """Fewest generated tokens whose decoded text reaches ``char_end`` characters."""
    lo, hi = 1, len(gen_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(gen_ids[:mid])) >= char_end:
            hi = mid
        else:
            lo = mid + 1
    return lo


def math_contrast(
    tok, traces: list[dict], newline: set[int], seed: int = 6
) -> tuple[list, list, list[dict]]:
    """Positives (line breaks right after the answer is first written as a result
    in thinking) and mid-trace working negatives from base MATH-train traces."""
    from answer_positions import answer_end

    rng = random.Random(seed)
    pos, work, meta = [], [], []
    for r in traces:
        if len(r["gold"].strip().strip("$")) < 2:
            continue  # "x = 2" mid-solution would read as the answer
        end = answer_end(r["text"], r["gold"])
        if end is None:
            continue
        p, g = r["prompt_ids"], r["gen_ids"]
        k = token_index_for_char(tok, g, end)
        if k < 24:
            continue
        stop = g.index(THINK_END) if THINK_END in g else len(g)
        after = [j for j in range(k, stop) if g[j] in newline][:2]
        before = [j for j in range(int(0.1 * k), int(0.9 * k)) if g[j] in newline]
        for j in after:
            if len(p) + j + 1 <= MAXLEN:
                pos.append(p + g[: j + 1])
        for j in rng.sample(before, k=min(2, len(before))):
            if len(p) + j + 1 <= MAXLEN:
                work.append(p + g[: j + 1])
        meta.append({"id": r["id"], "answer_token": k})
    return pos, work, meta


def contrast_sets(
    tok, contrast: dict, traces: list[dict]
) -> tuple[list, list[float], list[str]]:
    """Prefixes, +1/-1 labels and group names, in the kernel's order
    (newline set, then answer tokens)."""
    by_id = {r["id"]: r for r in traces}
    newline = {i for i in range(len(tok)) if "\n" in tok.decode([i])}
    rng = random.Random(1)
    pos, work = [], []
    for m in contrast["meta"]:
        r = by_id.get(m["id"])
        if r is None:
            continue
        p, g, k = r["prompt_ids"], r["gen_ids"], m["answer_token"]
        stop = g.index(THINK_END) if THINK_END in g else len(g)
        if k > stop:
            continue  # old contrasts could match an answer after thinking closed
        after = [j for j in range(k, stop) if g[j] in newline][:2]
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
    groups = (
        ["answer_newline"] * len(pos)
        + ["working_newline"] * len(work)
        + ["post_think"] * len(post)
        + ["answer_token"] * len(answer_tokens)
    )
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
    return prefixes, labels, groups


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


def main(argv: list[str] | None = None) -> None:
    """Design and optionally evaluate an edit for a pinned Qwen3.5 model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stateset_agents.experimental.weight_space import (
        EditSpec,
        NeuronRewrite,
        TokenRowScale,
        auc,
        design_conditional_neuron,
        logit_direction,
        matched_control,
        paired_compare,
        rewrite_activation,
        strict_rows,
    )
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
        resolve_checkpoint,
        verify_checkpoint,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=pj.CONCLUDE_MODELS, default=MODEL)
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--contrast", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument(
        "--generate-traces",
        action="store_true",
        help="generate the 256 GSM8K-train design traces and build their contrast "
        "for --model (exclusive with --trace/--contrast)",
    )
    ap.add_argument("--trace-tokens", type=int, default=3072)
    ap.add_argument("--neuron", type=int, default=None)
    ap.add_argument("--up-unit", type=int, default=None)
    ap.add_argument("--write-norm", type=float, default=3.0)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument(
        "--nothink-negatives",
        default=None,
        help="thinking-off replies (gen_*.jsonl.gz) whose line breaks become negatives",
    )
    ap.add_argument("--nothink-limit", type=int, default=300)
    ap.add_argument(
        "--nothink-train",
        type=int,
        default=0,
        help="generate thinking-off base replies to this many GSM8K-train (trace) "
        "problems and use their line breaks as negatives (never held-out questions)",
    )
    ap.add_argument(
        "--prose-negatives", type=int, default=0, help="WikiText-train line breaks"
    )
    ap.add_argument(
        "--intermediate-negatives",
        type=int,
        default=0,
        help="up to this many line breaks right after an intermediate '= number' "
        "result in the base traces (before the gold answer) as negatives",
    )
    ap.add_argument("--tag", default="", help="suffix for spec/output names, e.g. _v3")
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="skip base and control held-out (compare to --compare-spec instead)",
    )
    ap.add_argument(
        "--compare-spec", default=None, help="another spec to run on held-out, paired"
    )
    ap.add_argument(
        "--math-traces",
        type=int,
        default=0,
        help="generate base thinking-on traces for this many MATH-train problems "
        "and add their answer line breaks as positives (MATH-500 never used)",
    )
    ap.add_argument("--math-trace-tokens", type=int, default=3072)
    ap.add_argument(
        "--post-close-negatives",
        default=None,
        help="an edited model's replies to TRAINING problems (gen_*.jsonl.gz); "
        "line breaks after the first </think> in run-on replies become negatives",
    )
    ap.add_argument("--post-close-limit", type=int, default=300)
    ap.add_argument(
        "--reference-spec",
        default=None,
        help="an earlier spec whose layer-23 neuron is scored on these contexts "
        "(fire rate per group), for comparison with the new design",
    )
    ap.add_argument(
        "--save-captures",
        action="store_true",
        help="save the layer's gate/up captures, labels and groups (captures.pt) "
        "so detector settings can be swept offline",
    )
    ap.add_argument(
        "--design-only",
        action="store_true",
        help="stop after the design and its diagnostics (no candidate evaluation; "
        "requested training traces are still generated)",
    )
    ap.add_argument(
        "--side-effects",
        action="store_true",
        help="then run side_effect_job for base and this neuron",
    )
    a = ap.parse_args(argv)
    if a.generate_traces and (a.trace or a.contrast):
        ap.error("--generate-traces cannot be combined with --trace or --contrast")
    if a.trace_tokens <= 0:
        ap.error("--trace-tokens must be positive")
    if not 0 <= a.nothink_train <= 256:
        ap.error("--nothink-train must be between 0 and 256")
    revision = pj.REVISIONS[a.model]
    identity = {"model": a.model, "revision": revision}
    pj.MODEL = a.model
    tag = a.tag
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (pj.WORK / "data").mkdir(
        parents=True, exist_ok=True
    )  # pj.generate writes rows here
    (pj.WORK / "ckpt").mkdir(parents=True, exist_ok=True)

    probs = pj.load_problems(256, 200, 500)
    tok = AutoTokenizer.from_pretrained(
        a.model, revision=revision
    )  # nosec B615 - pinned
    if a.generate_traces:
        from build_answer_contrast import build_contrast

        traces = pj.generate(
            f"base_trace{tag}", a.model, probs["trace"], a.trace_tokens, out
        )
        contrast = {
            **build_contrast(tok, traces, probs["trace"], post_rows=traces),
            **identity,
        }
        (out / f"answer_contrast{tag}.json").write_text(json.dumps(contrast))
    else:
        trace_path = Path(a.trace or "/workspace/gen_b_base_trace.jsonl.gz")
        contrast_path = Path(a.contrast or "/workspace/answer_contrast.json")
        traces = read_jsonl(trace_path)
        contrast = json.loads(contrast_path.read_text())
    validate_design_source([contrast, *traces], a.model)
    if not contrast["meta"]:
        raise ValueError("no answer-before-close contexts found in the design traces")
    nothink_path = Path(a.nothink_negatives) if a.nothink_negatives else None
    if nothink_path is not None:
        validate_design_source(read_jsonl(nothink_path), a.model)
    if a.nothink_train:
        # Thinking-off base replies to GSM8K-*train* problems (the trace split,
        # which never overlaps held-out), generated before the HF model loads.
        pj.generate(
            f"nothink_train{tag}",
            pj.MODEL,
            probs["trace"][: a.nothink_train],
            2048,
            out,
            thinking=False,
        )
        nothink_path = out / f"gen_nothink_train{tag}.jsonl.gz"
    math_traces: list[dict] = []
    if a.math_traces:
        math_traces = pj.generate(
            f"math_trace{tag}",
            pj.MODEL,
            math_train_rows(a.math_traces, probs["heldout"]),
            a.math_trace_tokens,
            out,
        )

    prefixes, labels, groups = contrast_sets(tok, contrast, traces)
    newline = newline_ids(tok)
    extra: list[tuple[str, list]] = []
    if a.intermediate_negatives:
        hard = intermediate_negatives(tok, contrast, traces, newline)
        hard = random.Random(5).sample(hard, k=min(a.intermediate_negatives, len(hard)))
        extra += [("intermediate_newline", p) for p in hard]
    if math_traces:
        mpos, mwork, mmeta = math_contrast(tok, math_traces, newline)
        for group, label, ps in (
            ("math_answer_newline", 1.0, mpos),
            ("math_working_newline", -1.0, mwork),
        ):
            prefixes.extend(ps)
            labels.extend([label] * len(ps))
            groups.extend([group] * len(ps))
        if a.intermediate_negatives:
            mhard = intermediate_negatives(tok, {"meta": mmeta}, math_traces, newline)
            mhard = random.Random(7).sample(
                mhard, k=min(a.intermediate_negatives, len(mhard))
            )
            extra += [("math_intermediate_newline", p) for p in mhard]
        print(
            json.dumps(
                {
                    "event": "math_contrast",
                    "traces": len(math_traces),
                    "with_answer": len(mmeta),
                    "positives": len(mpos),
                    "working": len(mwork),
                }
            ),
            flush=True,
        )
    if nothink_path is not None:
        extra += [
            ("nothink_newline", p)
            for p in nothink_negatives(newline, nothink_path, a.nothink_limit)
        ]
    if a.post_close_negatives:
        extra += [
            ("post_close_newline", p)
            for p in post_close_negatives(
                newline, Path(a.post_close_negatives), a.post_close_limit
            )
        ]
    if a.prose_negatives:
        extra += [
            ("prose_newline", p)
            for p in prose_negatives(tok, newline, a.prose_negatives)
        ]
    for group, p in extra:
        prefixes.append(p)
        labels.append(-1.0)
        groups.append(group)
    if not any(label > 0 for label in labels):
        raise ValueError(
            "no answer line breaks before thinking closed in the design inputs"
        )
    print(
        json.dumps(
            {
                "event": "contrast_total",
                **{g: groups.count(g) for g in dict.fromkeys(groups)},
            }
        ),
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=revision, dtype=torch.bfloat16
    )  # nosec B615
    model = model.to("cuda").eval()
    mlp = f"model.layers.{LAYER}.mlp"
    caps = capture_last(model, [f"{mlp}.gate_proj", f"{mlp}.up_proj"], prefixes)
    w = {
        n: model.get_submodule(f"{mlp}.{n}").weight.detach().float().cpu()
        for n in ("gate_proj", "up_proj", "down_proj")
    }
    if a.save_captures:
        torch.save(
            {
                **identity,
                "gate_acts": caps[f"{mlp}.gate_proj"].to(torch.bfloat16),
                "up_acts": caps[f"{mlp}.up_proj"].to(torch.bfloat16),
                "labels": labels,
                "groups": groups,
                "layer": LAYER,
            },
            out / f"captures{tag}.pt",
        )
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
    act = design.activation(caps[f"{mlp}.gate_proj"], caps[f"{mlp}.up_proj"])
    held_idx = [i for i in range(len(groups)) if i % 2 == 1]
    fire_by_group = {
        g: round(
            sum(float(act[i]) > 0.5 for i in held_idx if groups[i] == g)
            / max(1, sum(1 for i in held_idx if groups[i] == g)),
            4,
        )
        for g in dict.fromkeys(groups)
    }
    design.report["heldout_fire_rate_by_group"] = fire_by_group
    gate_acts, up_acts = caps[f"{mlp}.gate_proj"], caps[f"{mlp}.up_proj"]
    held = torch.arange(len(groups)) % 2 == 1
    in_group = {
        g: torch.tensor([x == g for x in groups]) for g in dict.fromkeys(groups)
    }
    score = gate_acts.float() @ design.detector
    positive = torch.tensor(labels) > 0
    target = held & positive
    # Held-out separation of the target from each negative group on its own.
    design.report["heldout_detector_auc_by_group"] = {
        g: round(auc(score[target], score[held & m]), 4)
        for g, m in in_group.items()
        if not bool((m & positive).any()) and bool((held & m).any())
    }
    if a.reference_spec:
        ref = EditSpec.from_json(Path(a.reference_spec).read_text())
        rws = [
            e for e in ref.edits if isinstance(e, NeuronRewrite) and e.layer == LAYER
        ]
        if rws:
            ref_act = rewrite_activation(
                rws[0],
                gate_weight=w["gate_proj"],
                up_weight=w["up_proj"],
                gate_acts=gate_acts,
                up_acts=up_acts,
            )
            # Same held-out contexts as the new design's fire rates.
            design.report["reference"] = {
                "spec": ref.name,
                "heldout_fire_rate_by_group": {
                    g: round(float((ref_act[held & m] > 0.5).float().mean()), 4)
                    for g, m in in_group.items()
                    if bool((held & m).any())
                },
            }
    print(json.dumps({"event": "designed", **design.report}), flush=True)
    wait = TokenRowScale(ids=WAIT_IDS, scale=0.5)
    name = f"conclude-neuron-L23{tag}"
    spec = EditSpec(name=name, edits=(wait, design.rewrite))
    ctrl_rw = matched_control(
        EditSpec(name=name, edits=(design.rewrite,)),
        rng=random.Random(0),
        vocab_size=embed.shape[0],
        n_neurons=w["down_proj"].shape[1],
    ).edits[0]
    assert isinstance(ctrl_rw, NeuronRewrite)
    control = EditSpec(name=f"control:{name}", edits=(wait, ctrl_rw))
    spec_path = out / f"conclude_neuron{tag}_spec.json"
    spec_path.write_text(spec.to_json())
    (out / f"conclude_neuron{tag}_control_spec.json").write_text(control.to_json())
    if a.design_only:
        (out / f"neuron{tag}_summary.json").write_text(
            json.dumps(
                {
                    **identity,
                    "design": design.report,
                    "spec_hash": spec.content_hash(),
                    "control_hash": control.content_hash(),
                    "design_only": True,
                },
                indent=2,
            )
        )
        print(json.dumps({"event": "neuron_job_done", "design_only": True}), flush=True)
        return

    base = resolve_checkpoint(a.model, revision)
    ckpts = {}
    to_build = [("neuron", spec), ("control", control)]
    if a.compare_spec:
        to_build.append(
            ("compare", EditSpec.from_json(Path(a.compare_spec).read_text()))
        )
    for name, s in to_build:
        dst = pj.WORK / "ckpt" / f"spec_{name}{tag}"
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

    result: dict = {
        **identity,
        "design": design.report,
        "spec_hash": spec.content_hash(),
        "control_hash": control.content_hash(),
    }
    practice = pj.generate(
        f"spec_neuron{tag}_select", ckpts["neuron"], probs["select"][:200], 8192, out
    )
    result["practice"] = stats(practice)
    print(json.dumps({"event": "practice", **result["practice"]}), flush=True)
    runs = ["neuron"] + ([] if a.skip_baseline else ["base", "control"])
    if a.compare_spec:
        runs.append("compare")
    held = {}
    for name in runs:
        model_dir = pj.MODEL if name == "base" else ckpts[name]
        held[name] = pj.generate(
            f"spec_{name}{tag}_heldout", model_dir, probs["heldout"], 8192, out
        )
        print(
            json.dumps({"event": "heldout", "model": name, **stats(held[name])}),
            flush=True,
        )
    result["heldout"] = {k: stats(v) for k, v in held.items()}
    strict = {k: strict_rows(v) for k, v in held.items()}
    pairs = [
        ("neuron_vs_base", "base"),
        ("neuron_vs_control", "control"),
        ("neuron_vs_compare", "compare"),
    ]
    result["paired_strict"] = {
        label: paired_compare(strict[other], strict["neuron"])
        for label, other in pairs
        if other in strict
    }
    (out / f"neuron{tag}_summary.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({"event": "neuron_job_done", "heldout": result["heldout"]}),
        flush=True,
    )
    if a.side_effects:
        import side_effect_job

        side_effect_job.main(
            [
                "--model",
                a.model,
                "--out",
                str(out),
                "--neuron-spec",
                str(spec_path),
                "--models",
                "base,neuron",
                "--tag",
                tag,
            ]
        )


if __name__ == "__main__":
    main()
