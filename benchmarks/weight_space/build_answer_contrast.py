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
import glob
import gzip
import json
import os
import re
from pathlib import Path

THINK_END = 248069  # "</think>" in the Qwen3.5 tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace", required=True, help="base traces (gen_base_trace.jsonl.gz)"
    )
    ap.add_argument(
        "--post-from", default=None, help="concluded traces for post-</think> negatives"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17"
    )
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
    )
    train = pq.read_table(glob.glob(pattern)[0]).to_pylist()
    rows = [json.loads(line) for line in gzip.open(a.trace, "rt")]

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
        i = int(r["id"].rsplit("-", 1)[1])
        gold = r["gold"]
        if re.search(
            r"(?<![\d.])" + re.escape(gold) + r"(?![\d])", train[i]["question"]
        ):
            continue
        pat = (
            r"(?:=|\bis|answer[^\n]{0,15}?|boxed\{)\s*\\?\$?\**\s*"
            + re.escape(gold)
            + r"(?![\d])"
        )
        m = re.search(pat, r["text"])
        if not m:
            continue
        k = token_index_for_char(r["gen_ids"], m.end())
        if k < 24:
            continue
        P = r["prompt_ids"]
        answer.append(P + r["gen_ids"][:k])
        for f in (0.25, 0.6):
            working.append(P + r["gen_ids"][: max(8, int(k * f))])
        meta.append({"id": r["id"], "answer_token": k})

    post = []
    if a.post_from:
        for r in (json.loads(line) for line in gzip.open(a.post_from, "rt")):
            g = r["gen_ids"]
            if THINK_END not in g:
                continue
            k = g.index(THINK_END)
            tail = len(g) - k - 1
            if tail < 12:
                continue
            for f in (0.3, 0.8):
                post.append(r["prompt_ids"] + g[: k + 1 + max(2, int(tail * f))])
    Path(a.out).write_text(
        json.dumps(
            {"answer": answer, "working": working, "post": post[:300], "meta": meta}
        )
    )
    print(f"answer {len(answer)}, working {len(working)}, post {len(post[:300])}")


if __name__ == "__main__":
    main()
