"""Turn a Phase-1 results dir into a verdict: paired, held-out, control-checked.

    python analyze.py <results_dir>

Reads summary.json and gen_*_heldout.jsonl.gz. For edit-vs-base and
control-vs-base (same held-out problems, same decoding seed) it reports
accuracy / loop / hit-limit deltas per dataset, an exact McNemar test on
correctness flips, and a bootstrap 95% CI on the accuracy difference.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import sys
from pathlib import Path

ARTICLE = {  # "Sparks of Weight-Space Programming", Exp. 8 (thinking on, 32,768 tokens)
    "math": {"base": 0.079, "edited_50fp": 0.481},
    "gsm8k": {"base": 0.138, "edited_50fp": 0.671},
}


def read(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (base-only right), c (edit-only right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(
    diffs: list[int], iters: int = 10000, seed: int = 0
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    return stats[int(0.025 * iters)], stats[int(0.975 * iters)]


def rates(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "loop": sum(r["loop"] is not None for r in rows) / n,
        "hit_limit": sum(r["finish"] == "length" for r in rows) / n,
        "mean_tokens": sum(r["tokens"] for r in rows) / n,
    }


def compare(base: list[dict], other: list[dict]) -> dict:
    by_id = {r["id"]: r for r in base}
    pairs = [(by_id[r["id"]], r) for r in other if r["id"] in by_id]
    b = sum(x["correct"] and not y["correct"] for x, y in pairs)
    c = sum(y["correct"] and not x["correct"] for x, y in pairs)
    diffs = [int(y["correct"]) - int(x["correct"]) for x, y in pairs]
    lo, hi = bootstrap_ci(diffs)
    return {
        "n": len(pairs),
        "acc_delta": sum(diffs) / len(pairs),
        "acc_delta_95ci": (lo, hi),
        "base_only_correct": b,
        "other_only_correct": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "loops_fixed": sum(
            x["loop"] is not None and y["loop"] is None for x, y in pairs
        ),
        "loops_created": sum(
            x["loop"] is None and y["loop"] is not None for x, y in pairs
        ),
    }


def main(results: Path) -> None:
    summary = json.loads((results / "summary.json").read_text())
    kept, control = summary.get("kept"), summary.get("control")
    print(f"# Phase 1 verdict — {results}\n")
    print(
        f"kept edit: {kept}   control: {control}   practice budget: {summary.get('budget')}\n"
    )
    if "select" in summary:
        print("## Practice selection (GSM8K train, 8k tokens)")
        for name, s in summary["select"].items():
            a = s["all"]
            print(
                f"- {name:10s} acc {a['accuracy']:.3f}  loop {a['loop']:.3f}  hit_limit {a['hit_limit']:.3f}  tokens {a['mean_tokens']:.0f}"
            )
        print()
    conds = {"base": "base", "edit": kept, "control": control}
    rows = {}
    for label, name in conds.items():
        path = results / f"gen_{name}_heldout.jsonl.gz"
        if path.exists():
            rows[label] = read(path)
    if "base" not in rows:
        print("no held-out results yet")
        return
    print("## Held-out (GSM8K test + MATH-500, 16k tokens)")
    for kind in ("gsm8k", "math", None):
        title = kind or "all"
        print(f"\n### {title}")
        for label, rs in rows.items():
            sub = [r for r in rs if kind is None or r["kind"] == kind]
            if not sub:
                continue
            r = rates(sub)
            print(
                f"- {label:8s} n={r['n']:4d}  acc {r['accuracy']:.3f}  loop {r['loop']:.3f}  hit_limit {r['hit_limit']:.3f}  tokens {r['mean_tokens']:.0f}"
            )
        for label in ("edit", "control"):
            if label in rows:
                base_sub = [
                    r for r in rows["base"] if kind is None or r["kind"] == kind
                ]
                other_sub = [
                    r for r in rows[label] if kind is None or r["kind"] == kind
                ]
                c = compare(base_sub, other_sub)
                lo, hi = c["acc_delta_95ci"]
                print(
                    f"  {label} vs base: Δacc {c['acc_delta']:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]  "
                    f"flips +{c['other_only_correct']}/-{c['base_only_correct']}  McNemar p={c['mcnemar_p']:.2g}  "
                    f"loops fixed {c['loops_fixed']} / created {c['loops_created']}"
                )
        if kind in ARTICLE:
            art = ARTICLE[kind]
            print(
                f"  article (32k tokens): base {art['base']:.3f} -> edited {art['edited_50fp']:.3f}"
            )
    if "small_model" in summary:
        sm = summary["small_model"]
        print(
            "\n## Phase-0 small-model edit, real generation (Qwen3-0.6B, GSM8K test, 8k tokens)"
        )
        for k, v in sm.items():
            if isinstance(v, dict) and "all" in v:
                a = v["all"]
                print(
                    f"- {k:18s} acc {a['accuracy']:.3f}  loop {a['loop']:.3f}  hit_limit {a['hit_limit']:.3f}"
                )
            elif k.endswith("paired_vs_base"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
