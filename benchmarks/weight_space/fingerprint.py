"""Behavioral fingerprint per condition: did suppressing "Wait" stop re-checking, or move it?

    python fingerprint.py <results_dir> [split=select]

For each gen_<cond>_<split>.jsonl.gz: accuracy, loop rate, </think> reached,
mean tokens, and the share of lines opening with each re-check word.
"""

from __future__ import annotations

import collections
import gzip
import json
import re
import sys
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
WATCH = (
    "Wait",
    "But",
    "Yes",
    "Hmm",
    "Actually",
    "Let",
    "So",
    "Therefore",
    "Alternatively",
    "No",
)


def openers(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        m = re.match(r"\s*(?:[*\-]\s+)?([A-Za-z]+)", line)
        if m:
            out.append(m.group(1))
    return out


def main(results: Path, split: str) -> None:
    files = sorted(results.glob(f"gen_*_{split}.jsonl.gz"))
    header = (
        f"{'condition':16s} {'n':>4s} {'acc':>6s} {'loop':>6s} {'</think>':>8s} {'tokens':>7s} {'cjk/10k':>7s}  "
        + " ".join(f"{w[:6]:>6s}" for w in WATCH)
    )
    print(header)
    for f in files:
        cond = f.name[len("gen_") : -len(f"_{split}.jsonl.gz")]
        rows = [json.loads(line) for line in gzip.open(f, "rt")]
        n = len(rows)
        words = collections.Counter()
        for r in rows:
            words.update(openers(r["text"]))
        total = max(1, sum(words.values()))
        cjk = (
            sum(len(CJK.findall(r["text"])) for r in rows)
            / max(1, sum(len(r["text"]) for r in rows))
            * 1e4
        )
        shares = " ".join(f"{words[w] / total:6.3f}" for w in WATCH)
        print(
            f"{cond:16s} {n:4d} {sum(r['correct'] for r in rows) / n:6.3f} "
            f"{sum(r['loop'] is not None for r in rows) / n:6.3f} "
            f"{sum('</think>' in r['text'] for r in rows) / n:8.3f} "
            f"{sum(r['tokens'] for r in rows) / n:7.0f} {cjk:7.2f}  {shares}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "select")
