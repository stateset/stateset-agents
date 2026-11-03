"""Does an edited model that thinks beat the base model that does not?

Runs on a GPU pod with this repository's wheel installed:

    python thinking_budget_job.py --out /workspace/out \\
        --spec v3t=/workspace/conclude_neuron_v3t_spec.json \\
        --spec v4=/workspace/conclude_neuron_v4_spec.json

The edits repair Qwen3.5-0.8B's thinking mode (base: about 1% held-out, since
it almost never closes its reasoning within 8,192 tokens). The fair bar is
the base model with thinking switched off. On the held-out set every
weight-edit run used (GSM8K-test 500 + MATH-500), with pod_job's sampling,
seed and an 8,192-token cap for every condition, this job generates:

  base, thinking off;  then for each edit: thinking on, thinking off.

(Base with thinking on at this cap is on record from the edit runs.)

Scoring is strict: a clean finish with exactly one ``</think>`` (thinking on)
or none (thinking off). Each condition is also scored at token budgets 512 to
8,192 by truncation (``protocol.at_budget``: with a fixed per-request seed the
run at a smaller cap is this run cut short). Comparisons are paired per
problem, overall and per subset: edit-thinking-on vs base-thinking-off (the
question), edit-thinking-off vs base-thinking-off (side effect), and each later
edit against the first with thinking on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import pod_job as pj  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CAP = 8192
BUDGETS = (512, 1024, 2048, 4096, 6144, 8192)
KINDS = ("gsm8k", "math")


def describe(rows: list[dict], thinking: bool) -> dict:
    from stateset_agents.experimental.weight_space import (
        budget_curve,
        clean_finish,
        strict_rows,
    )

    out: dict = {}
    for kind in ("all", *KINDS):
        rs = rows if kind == "all" else [r for r in rows if r["kind"] == kind]
        n = len(rs)
        out[kind] = {
            "n": n,
            "strict": round(
                sum(r["correct"] for r in strict_rows(rs, thinking=thinking)) / n, 4
            ),
            "lenient": round(sum(bool(r["correct"]) for r in rs) / n, 4),
            "clean_finish": round(
                sum(clean_finish(r, thinking=thinking) for r in rs) / n, 4
            ),
            "hit_cap": round(sum(r["finish"] == "length" for r in rs) / n, 4),
            "mean_tokens": round(sum(r["tokens"] for r in rs) / n, 1),
            "strict_by_budget": {
                str(b): round(a, 4)
                for b, a in budget_curve(
                    rs, BUDGETS, cap=CAP, thinking=thinking
                ).items()
            },
        }
    return out


def compare(
    base: list[dict], other: list[dict], base_thinking: bool, other_thinking: bool
) -> dict:
    from stateset_agents.experimental.weight_space import paired_compare, strict_rows

    out: dict = {}
    for kind in ("all", *KINDS):
        b = base if kind == "all" else [r for r in base if r["kind"] == kind]
        o = other if kind == "all" else [r for r in other if r["kind"] == kind]
        out[kind] = {
            "strict": paired_compare(
                strict_rows(b, thinking=base_thinking),
                strict_rows(o, thinking=other_thinking),
            ),
            "lenient": paired_compare(b, o),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    from stateset_agents.experimental.weight_space import EditSpec
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
        resolve_checkpoint,
        verify_checkpoint,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument(
        "--spec",
        action="append",
        required=True,
        help="NAME=PATH of an EditSpec JSON; repeat to compare edits on one pod "
        "(each later edit is also paired against the first, thinking on)",
    )
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _v3t")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (pj.WORK / "data").mkdir(parents=True, exist_ok=True)

    base = resolve_checkpoint(MODEL, REVISION)
    specs: dict[str, tuple[EditSpec, str, str]] = {}
    for item in a.spec:
        name, _, path = item.partition("=")
        if not name or not path or name in specs or name == "base":
            raise SystemExit(f"--spec wants a unique NAME=PATH (not 'base'): {item}")
        spec = EditSpec.from_json(Path(path).read_text())
        edit_dir = pj.WORK / "ckpt" / f"edit_{name}{a.tag}"
        manifest = apply_spec_to_checkpoint(spec, base, edit_dir)
        verify_checkpoint(spec, base, edit_dir)
        specs[name] = (spec, manifest["spec_hash"], str(edit_dir))
        pj.log(
            event="checkpoint_verified",
            name=name,
            spec=spec.name,
            hash=manifest["spec_hash"],
        )

    rows = pj.load_problems(256, 200, 500)["heldout"]
    runs: dict[str, tuple[str, bool]] = {"base_nothink": (pj.MODEL, False)}
    for name, (_spec, _h, edit_dir) in specs.items():
        runs[f"{name}_think"] = (edit_dir, True)
        runs[f"{name}_nothink"] = (edit_dir, False)
    result: dict = {
        "specs": {n: {"name": s.name, "hash": h} for n, (s, h, _d) in specs.items()},
        "cap": CAP,
        "budgets": list(BUDGETS),
        "runs": {},
        "paired": {},
    }
    summary = out / f"thinking_budget_summary{a.tag}.json"
    gens: dict[str, list[dict]] = {}
    first = next(iter(specs))
    for label, (model_dir, thinking) in runs.items():
        gens[label] = pj.generate(
            f"tb_{label}{a.tag}", model_dir, rows, CAP, out, thinking=thinking
        )
        result["runs"][label] = {
            "thinking": thinking,
            **describe(gens[label], thinking),
        }
        pj.log(event="described", run=label, all=result["runs"][label]["all"])
        if label != "base_nothink":
            result["paired"][f"{label}_vs_base_nothink"] = compare(
                gens["base_nothink"], gens[label], False, thinking
            )
        if label.endswith("_think") and label != f"{first}_think":
            result["paired"][f"{label}_vs_{first}_think"] = compare(
                gens[f"{first}_think"], gens[label], True, True
            )
        summary.write_text(json.dumps(result, indent=2))  # partial results survive
    pj.log(event="thinking_budget_done", runs=list(result["runs"]))


if __name__ == "__main__":
    main()
