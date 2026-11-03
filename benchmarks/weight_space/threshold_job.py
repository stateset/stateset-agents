"""Judge explicit-threshold variants of the conclude neuron by generation (GPU pod).

    python threshold_job.py --out /workspace/out \\
        --grid /workspace/threshold_grid.json --heldout

``threshold_design.py`` builds, offline, the start spec (v3t) with an
explicit firing threshold (``bias``) and write strength (``write_norm``) on a
grid, keeping v3t's detector fixed: refitting the detector (v4, v5, the
on-policy rounds) moved its implicit threshold too, and every refit lost.

Stage 1 runs the start spec and every grid point on its two axes (each bias at
the start write norm, each write norm at bias 0). Stage 2 runs the
combination of the axis winners when both moved away from the start. Every
candidate generates thinking on and off (cap ``--practice-tokens``) on the
practice problems (``onpolicy_job.practice_sets``: MINE + SELECT, GSM8K-train
and MATH-train; held-out never touched) and is scored strictly on all of them
(the ``select_*`` fields). ``misfires.choose_round`` keeps the best strict
thinking-on score whose thinking-off stray ``</think>`` rate rose at most
``--max-stray-rise``; with ``--heldout`` a kept candidate other than the
start runs ``thinking_budget_job.heldout_check`` on the same pod.
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

START = (0.0, 3.0)


def axes(points: list[tuple[float, float]]) -> tuple[list, list]:
    """Grid points on the start's bias axis and write-norm axis (start excluded)."""
    bias_axis = sorted(p for p in points if p[1] == START[1] and p != START)
    write_axis = sorted(p for p in points if p[0] == START[0] and p != START)
    return bias_axis, write_axis


def combination(
    rows: list[dict], *, max_stray_rise: float
) -> tuple[float, float] | None:
    """The (bias, write_norm) pairing both axis winners, if both left the start.

    ``rows[0]`` is the start; each axis winner is chosen as the kept round is.
    """
    winners = []
    for on_axis in (
        lambda r: r["write_norm"] == START[1],
        lambda r: r["bias"] == START[0],
    ):
        sub = [rows[0]] + [r for r in rows[1:] if on_axis(r)]
        winners.append(sub[choose_round(sub, max_stray_rise=max_stray_rise)])
    b, w = winners[0]["bias"], winners[1]["write_norm"]
    if b == START[0] or w == START[1]:
        return None
    return (b, w)


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
    ap.add_argument("--grid", default="/workspace/threshold_grid.json")
    ap.add_argument("--practice-n", type=int, default=100, help="per kind, per set")
    ap.add_argument("--practice-tokens", type=int, default=4096)
    ap.add_argument("--max-stray-rise", type=float, default=0.02)
    ap.add_argument("--heldout", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (pj.WORK / "data").mkdir(parents=True, exist_ok=True)
    (pj.WORK / "ckpt").mkdir(parents=True, exist_ok=True)

    grid_path = Path(a.grid)
    grid = json.loads(grid_path.read_text())
    points = {(g["bias"], g["write_norm"]): g for g in grid["grid"]}
    if points[START]["spec_hash"] != grid["start_spec_hash"]:
        raise SystemExit("grid point (0, 3) is not the start spec")

    probs = pj.load_problems(256, 200, 500)
    mine_set, select_set = op.practice_sets(a.practice_n, probs)
    practice = mine_set + select_set
    base = resolve_checkpoint(nj.MODEL, nj.REVISION)
    rows: list[dict] = []
    start_gens: dict[str, list[dict]] = {}
    summary_path = out / "threshold_summary.json"

    def checkpoint(key: tuple[float, float]) -> tuple[Path, str]:
        g = points[key]
        spec = EditSpec.from_json((grid_path.parent / g["spec"]).read_text())
        ckpt = pj.WORK / "ckpt" / f"b{key[0]:+.1f}_w{key[1]:.1f}"
        manifest = apply_spec_to_checkpoint(spec, base, ckpt)
        verify_checkpoint(spec, base, ckpt)
        if manifest["spec_hash"] != g["spec_hash"]:
            raise SystemExit(f"{g['name']}: built {manifest['spec_hash']}")
        return ckpt, spec.name

    def run(key: tuple[float, float]) -> None:
        ckpt, name = checkpoint(key)
        tag = f"th_b{key[0]:+.1f}_w{key[1]:.1f}"
        think = pj.generate(f"{tag}_think", str(ckpt), practice, a.practice_tokens, out)
        nothink = pj.generate(
            f"{tag}_nothink",
            str(ckpt),
            practice,
            a.practice_tokens,
            out,
            thinking=False,
        )
        row = {
            "bias": key[0],
            "write_norm": key[1],
            "spec": name,
            "spec_hash": points[key]["spec_hash"],
            **op.select_scores(think, nothink),
        }
        if key == START:
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

    bias_axis, write_axis = axes(list(points))
    for key in (START, *bias_axis, *write_axis):
        run(key)
    combo = combination(rows, max_stray_rise=a.max_stray_rise)
    if combo is not None and combo in points:
        run(combo)

    keep = choose_round(rows, max_stray_rise=a.max_stray_rise)
    result: dict = {"rows": rows, "kept": keep, "kept_spec": rows[keep]["spec"]}
    pj.log(event="kept", index=keep, spec_hash=rows[keep]["spec_hash"])
    summary_path.write_text(json.dumps(result, indent=2))
    if a.heldout and keep > 0:
        kept_ckpt, _ = checkpoint((rows[keep]["bias"], rows[keep]["write_norm"]))
        start_ckpt = pj.WORK / "ckpt" / f"b{START[0]:+.1f}_w{START[1]:.1f}"
        result.update(
            tb.heldout_check(
                probs["heldout"],
                str(start_ckpt),
                str(kept_ckpt),
                out,
                "th",
                lambda part: summary_path.write_text(
                    json.dumps({**result, **part}, indent=2)
                ),
            )
        )
    summary_path.write_text(json.dumps(result, indent=2))
    pj.log(event="threshold_done", kept=keep)


if __name__ == "__main__":
    main()
