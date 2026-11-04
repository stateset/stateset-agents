"""Judge named edit specs by generation on practice problems, then held-out (GPU pod).

    python practice_job.py --out /workspace/out \\
        --spec v3t=/workspace/conclude_neuron_v3t_spec.json \\
        --spec guard=/workspace/v3t+guard_spec.json --heldout

The first ``--spec`` is the start. Every spec generates thinking on and off
(cap ``--practice-tokens``) on the practice problems
(``onpolicy_job.practice_sets``: MINE + SELECT, GSM8K-train and MATH-train;
held-out never touched) and is scored strictly on all of them.
``misfires.choose_round`` keeps the best strict thinking-on score whose
thinking-off stray ``</think>`` rate rose at most ``--max-stray-rise`` over the
start's; with ``--heldout`` a kept spec other than the start runs
``thinking_budget_job.heldout_check`` on the same pod. Same scoring and
selection as ``threshold_job.py``, for any list of specs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import pod_job as pj  # noqa: E402
from misfires import choose_round  # noqa: E402


def parse_specs(items: list[str]) -> list[tuple[str, Path]]:
    """``NAME=PATH`` pairs, names unique, in order (the first is the start)."""
    out: list[tuple[str, Path]] = []
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--spec wants NAME=PATH, got {item!r}")
        if any(name == n for n, _ in out):
            raise SystemExit(f"duplicate --spec name {name!r}")
        out.append((name, Path(path)))
    if len(out) < 2:
        raise SystemExit("give a start spec and at least one candidate")
    return out


def main(argv: list[str] | None = None) -> None:
    import neuron_job as nj
    import onpolicy_job as op
    import thinking_budget_job as tb

    from stateset_agents.experimental.weight_space import EditSpec
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
        resolve_checkpoint,
        verify_checkpoint,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--spec", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--practice-n", type=int, default=100, help="per kind, per set")
    ap.add_argument("--practice-tokens", type=int, default=4096)
    ap.add_argument("--max-stray-rise", type=float, default=0.02)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument(
        "--model",
        default=nj.MODEL,
        help="a model pinned in pod_job.REVISIONS (specs must match its shapes)",
    )
    a = ap.parse_args(argv)
    specs = parse_specs(a.spec)
    if a.model not in pj.REVISIONS:
        raise SystemExit(f"--model must be pinned in pod_job.REVISIONS: {a.model}")
    pj.MODEL = a.model  # the held-out base run uses pj.MODEL
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (pj.WORK / "data").mkdir(parents=True, exist_ok=True)
    (pj.WORK / "ckpt").mkdir(parents=True, exist_ok=True)

    probs = pj.load_problems(256, 200, 500)
    mine_set, select_set = op.practice_sets(a.practice_n, probs)
    practice = mine_set + select_set
    base = resolve_checkpoint(a.model, pj.REVISIONS[a.model])
    rows: list[dict] = []
    start_gens: dict[str, list[dict]] = {}
    summary_path = out / "practice_summary.json"

    def checkpoint(name: str, path: Path) -> tuple[Path, str]:
        spec = EditSpec.from_json(path.read_text())
        ckpt = pj.WORK / "ckpt" / name
        manifest = apply_spec_to_checkpoint(spec, base, ckpt)
        verify_checkpoint(spec, base, ckpt)
        return ckpt, manifest["spec_hash"]

    for i, (name, path) in enumerate(specs):
        ckpt, spec_hash = checkpoint(name, path)
        think = pj.generate(
            f"pr_{name}_think", str(ckpt), practice, a.practice_tokens, out
        )
        nothink = pj.generate(
            f"pr_{name}_nothink",
            str(ckpt),
            practice,
            a.practice_tokens,
            out,
            thinking=False,
        )
        row = {"name": name, "spec_hash": spec_hash, **op.select_scores(think, nothink)}
        if i == 0:
            start_gens.update(think=think, nothink=nothink)
        else:
            row["vs_start_think"] = tb.compare(start_gens["think"], think, True, True)
            row["vs_start_nothink"] = tb.compare(
                start_gens["nothink"], nothink, False, False
            )
            shutil.rmtree(ckpt)  # rebuilt (and re-verified) if kept
        rows.append(row)
        pj.log(
            event="candidate",
            **{k: v for k, v in row.items() if not k.startswith(("vs_", "by_"))},
        )
        summary_path.write_text(json.dumps({"rows": rows}, indent=2))

    keep = choose_round(rows, max_stray_rise=a.max_stray_rise)
    result: dict = {"rows": rows, "kept": keep, "kept_name": rows[keep]["name"]}
    pj.log(event="kept", index=keep, name=rows[keep]["name"])
    summary_path.write_text(json.dumps(result, indent=2))
    if a.heldout and keep > 0:
        kept_ckpt, _ = checkpoint(*specs[keep])
        result.update(
            tb.heldout_check(
                probs["heldout"],
                str(pj.WORK / "ckpt" / specs[0][0]),
                str(kept_ckpt),
                out,
                "pr",
                lambda part: summary_path.write_text(
                    json.dumps({**result, **part}, indent=2)
                ),
            )
        )
    summary_path.write_text(json.dumps(result, indent=2))
    pj.log(event="practice_done", kept=keep)


if __name__ == "__main__":
    main()
