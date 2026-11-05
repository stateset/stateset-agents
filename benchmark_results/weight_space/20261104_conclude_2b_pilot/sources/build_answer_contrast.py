"""Contrast prefixes for the "answer computed -> conclude" circuit, from real base traces.

    python build_answer_contrast.py --trace gen_base_trace.jsonl.gz \\
        [--post-from gen_W5T15_select.jsonl.gz] --out answer_contrast.json

answer:  prefix ending right after the model first writes the gold answer as a RESULT
         (= / is / answer ... / boxed{); problems whose gold number appears in the
         question are excluded.
working: prefixes ending earlier in the same trace (25% and 60% of the way to the
         answer) — mid-computation, same problem, same context.
post:    prefixes ending inside the answer section (after </think>) of concluded
         traces — negatives, so the detector does not fire once thinking has ended.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pod_job as pj
from answer_positions import answer_end, gold_in_question

THINK_END = 248069  # "</think>" in the Qwen3.5 tokenizer


def build_contrast(
    tok,
    rows: list[dict],
    training_rows: list[dict],
    post_rows: list[dict] | None = None,
) -> dict:
    """Build prefixes from identified training problems, never held-out replies.

    ``training_rows`` supplies the pinned questions and gold answers. Validate
    every trace before extracting prefixes, including post-close negatives.
    Only answers written before the first close can become positive examples.
    """
    training = {r["id"]: r for r in training_rows}
    for source in (rows, post_rows or []):
        seen: set[str] = set()
        for r in source:
            expected = training.get(r["id"])
            if expected is None or not r["id"].startswith("gsm8k-train-"):
                raise ValueError(f"trace is not in the training split: {r['id']}")
            if r["id"] in seen:
                raise ValueError(f"duplicate training trace: {r['id']}")
            seen.add(r["id"])
            if r["gold"] != expected["gold"]:
                raise ValueError(f"trace gold differs from the pinned data: {r['id']}")

    def token_index_for_char(gen_ids: list[int], char_end: int) -> int:
        lo, hi = 1, len(gen_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tok.decode(gen_ids[:mid])) >= char_end:
                hi = mid
            else:
                lo = mid + 1
        return lo

    answer, working, meta = [], [], []
    for r in rows:
        if r.get("thinking") is False or THINK_END in r["prompt_ids"]:
            raise ValueError(f"positive trace must have thinking enabled: {r['id']}")
        gold = r["gold"]
        if gold_in_question(training[r["id"]]["question"], gold):
            continue
        # Decode the exact tokens: text offsets must address this token sequence.
        end = answer_end(tok.decode(r["gen_ids"]), gold)
        if end is None:
            continue
        k = token_index_for_char(r["gen_ids"], end)
        if k < 24:
            continue
        prompt = r["prompt_ids"]
        answer.append(prompt + r["gen_ids"][:k])
        for f in (0.25, 0.6):
            working.append(prompt + r["gen_ids"][: max(8, int(k * f))])
        meta.append({"id": r["id"], "answer_token": k})

    post = []
    for r in post_rows or []:
        g = r["gen_ids"]
        if THINK_END not in g:
            continue
        k = g.index(THINK_END)
        tail = len(g) - k - 1
        if tail < 12:
            continue
        for f in (0.3, 0.8):
            post.append(r["prompt_ids"] + g[: k + 1 + max(2, int(tail * f))])
    return {"answer": answer, "working": working, "post": post[:300], "meta": meta}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace", required=True, help="base traces (gen_base_trace.jsonl.gz)"
    )
    ap.add_argument(
        "--post-from", default=None, help="concluded traces for post-</think> negatives"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--model", choices=pj.CONCLUDE_MODELS, default=pj.CONCLUDE_MODELS[0]
    )
    a = ap.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, revision=pj.REVISIONS[a.model])
    train = load_dataset(
        "openai/gsm8k", "main", split="train", revision=pj.GSM8K_REVISION
    )
    training = [
        {
            "id": f"gsm8k-train-{i}",
            "question": r["question"],
            "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for i, r in enumerate(train)
    ]
    rows = pj.read_jsonl(Path(a.trace))
    post_rows = pj.read_jsonl(Path(a.post_from)) if a.post_from else []
    # Legacy 0.8B files predate provenance; new-model inputs must identify their source.
    from neuron_job import validate_design_source

    validate_design_source(rows, a.model)
    result = build_contrast(tok, rows, training, post_rows)
    result.update(model=a.model, revision=pj.REVISIONS[a.model])
    Path(a.out).write_text(json.dumps(result))
    print(
        f"answer {len(result['answer'])}, working {len(result['working'])}, "
        f"post {len(result['post'])}"
    )


if __name__ == "__main__":
    main()
