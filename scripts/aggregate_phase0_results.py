"""
Aggregate Phase 0 benchmark JSON files into a publication-ready summary.

Reads every ``*.json`` file under ``benchmark_results/whitepaper_v1/`` (or a
directory passed via ``--results-dir``), validates that each file conforms to
the schema in ``benchmark_results/SCHEMA.md``, groups by (trainer, model), and
emits three artifacts:

1. ``summary.md`` — a Markdown table with mean ± std per group, ready to paste
   into §11.7 of the whitepaper.
2. ``summary.csv`` — one row per individual run, for downstream plotting.
3. ``passes_gates.json`` — for each (trainer, model) group, a pass/fail report
   against the publication gates in SCHEMA.md (3 seeds, σ < 0.1, +0.03 gain).

Usage:

    python scripts/aggregate_phase0_results.py
    python scripts/aggregate_phase0_results.py --results-dir benchmark_results/whitepaper_v1
    python scripts/aggregate_phase0_results.py --results-dir RESULTS --output-dir OUT

The exit code is 0 if every (trainer, model) group passes its gates, 1 if any
group fails — making this script suitable as a CI gate before publishing the
whitepaper.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("aggregate_phase0")


REQUIRED_TOP_FIELDS = ["trainer", "model", "seed", "commit", "timestamp", "metrics"]
REQUIRED_METRIC_FIELDS = ["eval_pass_at_1"]

# Publication gates from benchmark_results/SCHEMA.md
MIN_SEEDS_PER_GROUP = 3
MAX_STD_PASS_AT_1 = 0.10
MIN_IMPROVEMENT = 0.03


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    """Load and minimally validate every JSON file in ``results_dir``."""
    if not results_dir.exists():
        logger.warning("Results directory %s does not exist", results_dir)
        return []

    runs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("Skipping %s: invalid JSON (%s)", path.name, e)
            continue

        missing = [f for f in REQUIRED_TOP_FIELDS if f not in data]
        if missing:
            logger.warning("Skipping %s: missing fields %s", path.name, missing)
            continue

        missing_metrics = [
            f for f in REQUIRED_METRIC_FIELDS if f not in data.get("metrics", {})
        ]
        if missing_metrics:
            logger.warning(
                "Skipping %s: missing metric fields %s", path.name, missing_metrics
            )
            continue

        data["__source__"] = path.name
        runs.append(data)

    logger.info("Loaded %d valid runs from %s", len(runs), results_dir)
    return runs


def group_runs(
    runs: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group runs by ``(trainer, model)``."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (run["trainer"], run["model"])
        grouped[key].append(run)
    return grouped


def summarize_group(group_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean / std / n for the metrics that matter."""
    pass_at_1 = [r["metrics"]["eval_pass_at_1"] for r in group_runs]
    baseline = [
        r["metrics"]["eval_pass_at_1_baseline"]
        for r in group_runs
        if "eval_pass_at_1_baseline" in r["metrics"]
    ]
    wall = [
        r["metrics"]["wall_clock_seconds"]
        for r in group_runs
        if "wall_clock_seconds" in r["metrics"]
    ]

    def _stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        if len(xs) == 1:
            return {"mean": xs[0], "std": 0.0, "n": 1}
        return {
            "mean": statistics.mean(xs),
            "std": statistics.stdev(xs),
            "n": len(xs),
        }

    summary = {
        "pass_at_1": _stats(pass_at_1),
        "baseline": _stats(baseline),
        "wall_clock_seconds": _stats(wall),
        "seeds": sorted({r["seed"] for r in group_runs}),
        "commits": sorted({r["commit"] for r in group_runs}),
    }

    pass_stats = summary["pass_at_1"]
    base_stats = summary["baseline"]
    if base_stats["n"] > 0 and not math.isnan(base_stats["mean"]):
        summary["improvement"] = pass_stats["mean"] - base_stats["mean"]
    else:
        summary["improvement"] = float("nan")
    return summary


def check_gates(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply the publication gates from SCHEMA.md."""
    failures: list[str] = []

    n_seeds = summary["pass_at_1"]["n"]
    if n_seeds < MIN_SEEDS_PER_GROUP:
        failures.append(f"Only {n_seeds} seeds, need ≥{MIN_SEEDS_PER_GROUP}")

    std = summary["pass_at_1"]["std"]
    if not math.isnan(std) and std > MAX_STD_PASS_AT_1:
        failures.append(f"std={std:.3f} exceeds gate of {MAX_STD_PASS_AT_1}")

    improvement = summary["improvement"]
    if not math.isnan(improvement) and improvement < MIN_IMPROVEMENT:
        failures.append(
            f"improvement={improvement:+.3f} below gate of {MIN_IMPROVEMENT:+.3f}"
        )

    n_commits = len(summary["commits"])
    if n_commits > 1:
        failures.append(
            f"runs span {n_commits} commits ({summary['commits']}); should be 1"
        )

    return (len(failures) == 0), failures


def render_markdown(
    grouped_summary: dict[tuple[str, str], dict[str, Any]],
    gate_results: dict[tuple[str, str], tuple[bool, list[str]]],
) -> str:
    """Render the headline whitepaper table."""
    lines: list[str] = []
    lines.append("# Phase 0 Benchmark Results — Aggregated")
    lines.append("")
    lines.append(
        "Generated by `scripts/aggregate_phase0_results.py`. Each row is the"
        " mean ± std across all seeds for a (trainer, model) group."
    )
    lines.append("")
    lines.append(
        "| Trainer | Model | Baseline pass@1 | Final pass@1 | Improvement |"
        " Seeds (n) | Wall-clock (s) | Status |"
    )
    lines.append(
        "|---------|-------|-----------------|--------------|-------------|"
        "-----------|----------------|--------|"
    )

    if not grouped_summary:
        lines.append("| _no results_ | | | | | | | |")
    else:
        for (trainer, model), summary in sorted(grouped_summary.items()):
            pass_stats = summary["pass_at_1"]
            base_stats = summary["baseline"]
            wall_stats = summary["wall_clock_seconds"]
            improvement = summary["improvement"]

            def _fmt(stats: dict[str, float], digits: int = 3) -> str:
                if stats["n"] == 0:
                    return "—"
                if stats["n"] == 1:
                    return f"{stats['mean']:.{digits}f}"
                return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"

            passed, failures = gate_results[(trainer, model)]
            status = "✅ pass" if passed else "❌ " + "; ".join(failures)
            improvement_str = "—" if math.isnan(improvement) else f"{improvement:+.3f}"

            lines.append(
                f"| {trainer.upper()} | `{model}` | {_fmt(base_stats)} |"
                f" {_fmt(pass_stats)} | {improvement_str} |"
                f" {pass_stats['n']} | {_fmt(wall_stats, digits=0)} | {status} |"
            )

    lines.append("")
    lines.append("## Publication gates")
    lines.append("")
    lines.append(f"- Minimum {MIN_SEEDS_PER_GROUP} seeds per (trainer, model) group")
    lines.append(f"- Standard deviation of `eval_pass_at_1` ≤ {MAX_STD_PASS_AT_1}")
    lines.append(f"- Improvement over baseline ≥ {MIN_IMPROVEMENT:+.3f}")
    lines.append("- All runs in a group must share a single git commit")
    lines.append("")
    lines.append("See `benchmark_results/SCHEMA.md` for the full schema.")
    return "\n".join(lines)


def render_csv(runs: list[dict[str, Any]], path: Path) -> None:
    """One row per individual run, for downstream plotting."""
    fieldnames = [
        "source",
        "trainer",
        "model",
        "seed",
        "commit",
        "timestamp",
        "eval_pass_at_1",
        "eval_pass_at_1_baseline",
        "improvement",
        "wall_clock_seconds",
        "peak_vram_mb",
        "wandb_run_url",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            metrics = run.get("metrics", {})
            base = metrics.get("eval_pass_at_1_baseline")
            final = metrics.get("eval_pass_at_1")
            improvement = (
                (final - base) if (base is not None and final is not None) else None
            )
            writer.writerow(
                {
                    "source": run.get("__source__"),
                    "trainer": run.get("trainer"),
                    "model": run.get("model"),
                    "seed": run.get("seed"),
                    "commit": run.get("commit"),
                    "timestamp": run.get("timestamp"),
                    "eval_pass_at_1": final,
                    "eval_pass_at_1_baseline": base,
                    "improvement": improvement,
                    "wall_clock_seconds": metrics.get("wall_clock_seconds"),
                    "peak_vram_mb": metrics.get("peak_vram_mb"),
                    "wandb_run_url": run.get("wandb_run_url"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark_results/whitepaper_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --results-dir.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any group fails its gates.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    grouped = group_runs(runs)
    grouped_summary = {key: summarize_group(group) for key, group in grouped.items()}
    gate_results = {
        key: check_gates(summary) for key, summary in grouped_summary.items()
    }

    md = render_markdown(grouped_summary, gate_results)
    md_path = output_dir / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("Wrote %s", md_path)

    csv_path = output_dir / "summary.csv"
    render_csv(runs, csv_path)
    logger.info("Wrote %s", csv_path)

    gates_path = output_dir / "passes_gates.json"
    gates_path.write_text(
        json.dumps(
            {
                f"{trainer}|{model}": {
                    "passed": passed,
                    "failures": failures,
                    "summary": grouped_summary[(trainer, model)],
                }
                for (trainer, model), (passed, failures) in gate_results.items()
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", gates_path)

    all_passed = all(passed for passed, _ in gate_results.values())
    if not all_passed:
        logger.warning("Some (trainer, model) groups failed their publication gates.")
        if args.strict:
            return 1
    else:
        logger.info("All groups passed publication gates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
