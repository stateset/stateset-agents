"""
Plot Phase 0 benchmark results into publication-ready figures.

Reads ``summary.csv`` produced by ``aggregate_phase0_results.py`` and emits
two PNGs:

1. ``fig_pass_at_1_per_trainer.png`` — bar chart of mean pass@1 (with error
   bars from seed variance) per trainer, plus a horizontal baseline line.
2. ``fig_improvement_per_trainer.png`` — improvement over baseline per
   trainer, ranked descending.

These are the figures that go alongside §11.7 of the whitepaper.

Falls back gracefully if matplotlib isn't available — emits a tab-separated
plain-text "ascii plot" instead so the script still produces something useful
in headless / minimal environments. This keeps the figure pipeline runnable
in the same CI that runs the smoke test.

Usage:

    python scripts/plot_phase0_results.py
    python scripts/plot_phase0_results.py --results-dir benchmark_results/whitepaper_v1
    python scripts/plot_phase0_results.py --no-matplotlib    # force text fallback
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("plot_phase0")


def load_csv(path: Path) -> list[dict[str, Any]]:
    """Load summary.csv, coercing numeric columns to floats."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `scripts/aggregate_phase0_results.py` first."
        )
    numeric_cols = {
        "eval_pass_at_1",
        "eval_pass_at_1_baseline",
        "improvement",
        "wall_clock_seconds",
        "peak_vram_mb",
        "seed",
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in numeric_cols:
                if col in row and row[col] not in (None, "", "None"):
                    try:
                        row[col] = float(row[col])
                    except (TypeError, ValueError):
                        row[col] = None
                else:
                    row[col] = None
            rows.append(row)
    return rows


def group_by_trainer(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("trainer"):
            grouped[r["trainer"]].append(r)
    return grouped


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def render_text_plots(
    grouped: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    """Plain-text fallback when matplotlib is unavailable."""
    lines = ["# Phase 0 results — text plots"]
    lines.append("")
    lines.append("## pass@1 per trainer (mean ± std)")
    lines.append("")
    lines.append("| Trainer | Baseline | Final | Improvement | n |")
    lines.append("|---------|----------|-------|-------------|---|")
    for trainer, runs in sorted(grouped.items()):
        finals = [r["eval_pass_at_1"] for r in runs if r["eval_pass_at_1"] is not None]
        bases = [
            r["eval_pass_at_1_baseline"]
            for r in runs
            if r["eval_pass_at_1_baseline"] is not None
        ]
        f_mean, f_std = _mean_std(finals)
        b_mean, b_std = _mean_std(bases)
        improvement = (
            f_mean - b_mean
            if (not math.isnan(f_mean) and not math.isnan(b_mean))
            else float("nan")
        )
        lines.append(
            f"| {trainer.upper()} |"
            f" {b_mean:.3f} ± {b_std:.3f} |"
            f" {f_mean:.3f} ± {f_std:.3f} |"
            f" {improvement:+.3f} |"
            f" {len(finals)} |"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "text_plots.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out)


def render_matplotlib(
    grouped: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    """Generate the two publication figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    trainers = sorted(grouped.keys())
    final_means: list[float] = []
    final_stds: list[float] = []
    baseline_means: list[float] = []
    improvements: list[float] = []
    n_seeds: list[int] = []

    for trainer in trainers:
        runs = grouped[trainer]
        finals = [r["eval_pass_at_1"] for r in runs if r["eval_pass_at_1"] is not None]
        bases = [
            r["eval_pass_at_1_baseline"]
            for r in runs
            if r["eval_pass_at_1_baseline"] is not None
        ]
        f_mean, f_std = _mean_std(finals)
        b_mean, _ = _mean_std(bases)
        final_means.append(f_mean)
        final_stds.append(f_std)
        baseline_means.append(b_mean)
        improvements.append(
            f_mean - b_mean if not (math.isnan(f_mean) or math.isnan(b_mean)) else 0.0
        )
        n_seeds.append(len(finals))

    # Figure 1: pass@1 per trainer (with baseline reference line).
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = list(range(len(trainers)))
    bars = ax.bar(
        x,
        final_means,
        yerr=final_stds,
        capsize=4,
        color="#3b82f6",
        edgecolor="#1e3a8a",
        label="Post-training (mean ± std)",
    )
    # Baseline averaged across all groups — drawn as a dashed horizontal line.
    baseline_overall = (
        statistics.mean([b for b in baseline_means if not math.isnan(b)])
        if any(not math.isnan(b) for b in baseline_means)
        else 0.0
    )
    ax.axhline(
        baseline_overall,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Un-tuned baseline ({baseline_overall:.3f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in trainers])
    ax.set_ylabel("GSM8K pass@1")
    ax.set_title("GSM8K pass@1 by trainer (Qwen 3.5 0.8B)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    for bar, n in zip(bars, n_seeds, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={n}",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    f1 = output_dir / "fig_pass_at_1_per_trainer.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", f1)

    # Figure 2: improvement over baseline, ranked.
    order = sorted(range(len(trainers)), key=lambda i: improvements[i], reverse=True)
    trainers_sorted = [trainers[i].upper() for i in order]
    improvements_sorted = [improvements[i] for i in order]
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in improvements_sorted]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(trainers_sorted, improvements_sorted, color=colors, edgecolor="#374151")
    ax.axvline(0, color="#374151", linewidth=1)
    ax.axvline(
        0.03,
        color="#16a34a",
        linestyle=":",
        linewidth=1,
        label="Publication gate (+0.03)",
    )
    ax.set_xlabel("Improvement over baseline (Δ pass@1)")
    ax.set_title("Trainer improvement over un-tuned baseline")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    f2 = output_dir / "fig_improvement_per_trainer.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", f2)


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
        "--no-matplotlib",
        action="store_true",
        help="Skip the PNG figures, emit only text_plots.md.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output_dir = args.output_dir or args.results_dir
    rows = load_csv(args.results_dir / "summary.csv")
    if not rows:
        logger.warning("No rows in summary.csv — nothing to plot.")
        return 0

    grouped = group_by_trainer(rows)
    if not grouped:
        logger.warning("No trainer rows found — nothing to plot.")
        return 0

    render_text_plots(grouped, output_dir)

    if args.no_matplotlib:
        logger.info("--no-matplotlib set; skipping PNG figures.")
        return 0

    try:
        render_matplotlib(grouped, output_dir)
    except ImportError:
        logger.warning("matplotlib not installed; emitted only text plots.")
        logger.warning("Install with `pip install matplotlib` to get PNG figures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
