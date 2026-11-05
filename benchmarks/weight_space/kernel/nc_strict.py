# Kernel code — ANALYZE (free): lenient vs strict accuracy for every kernel evaluation so far.
import json

_g = evaluate.__globals__
pj, OUT = _g["pj"], _g["OUT"]
for i in range(1, 10):
    p = OUT / f"gen_b_eval{i}.jsonl.gz"
    if not p.exists():
        continue
    rows = pj.read_jsonl(p)
    n = len(rows)
    clean = [r["finish"] == "stop" and r["text"].count("</think>") == 1 for r in rows]
    print(
        json.dumps(
            {
                "eval": i,
                "n": n,
                "lenient": round(sum(r["correct"] for r in rows) / n, 3),
                "strict": round(
                    sum(r["correct"] and c for r, c in zip(rows, clean, strict=True))
                    / n,
                    3,
                ),
                "clean_finish": round(sum(clean) / n, 3),
                "multi_think_close": round(
                    sum(r["text"].count("</think>") > 1 for r in rows) / n, 3
                ),
            }
        )
    )
