"""
Post-hoc analysis of auto-research experiment history.

After running 50-100+ experiments, this module helps users understand:
- Which hyperparameters are most predictive of good performance
- The convergence trajectory
- Diminishing returns and where to focus next
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _ascii_chart(
    values: list[float],
    width: int = 50,
    height: int = 8,
) -> list[str] | None:
    """Render a simple ASCII line chart.

    Returns a list of strings (one per row), or None if not enough data.
    """
    if len(values) < 2:
        return None

    lo = min(values)
    hi = max(values)
    val_range = hi - lo
    if val_range == 0:
        # Flat line
        chart_rows = [" " * width for _ in range(height)]
        chart_rows[height // 2] = "─" * width
        return chart_rows

    # Resample values to width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = list(values)
        # Pad to width by repeating last value
        while len(sampled) < width:
            sampled.append(sampled[-1])

    # Quantize to height levels
    levels = []
    for v in sampled:
        level = int((v - lo) / val_range * (height - 1))
        levels.append(min(height - 1, max(0, level)))

    # Build rows (top = highest value)
    rows: list[str] = []
    for row_idx in range(height - 1, -1, -1):
        chars = []
        for col_idx in range(width):
            if levels[col_idx] == row_idx:
                chars.append("█")
            elif levels[col_idx] > row_idx:
                # Below the line — fill for area chart effect
                chars.append("░")
            else:
                chars.append(" ")
        rows.append("".join(chars))

    return rows


def compare_runs(
    *run_dirs: str,
    objective_metric: str = "eval_reward",
    direction: str = "maximize",
) -> str:
    """Compare multiple auto-research runs side-by-side.

    Args:
        *run_dirs: Paths to auto-research output directories.
        objective_metric: Which metric to compare.
        direction: "maximize" or "minimize".

    Returns:
        A formatted comparison report string.

    Example:
        report = compare_runs(
            "./run_perturbation",
            "./run_smart",
            "./run_bayesian",
        )
        print(report)
    """
    from pathlib import Path

    from .experiment_tracker import ExperimentTracker

    trackers = []
    names = []
    for d in run_dirs:
        p = Path(d)
        tracker = ExperimentTracker.load(p, objective_metric, direction)
        trackers.append(tracker)
        names.append(p.name)

    if not trackers:
        return "No runs to compare."

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("RUN COMPARISON")
    lines.append("=" * 60)
    lines.append("")

    # Header
    name_width = max(20, max(len(n) for n in names) + 2)
    header = f"{'Run':<{name_width}} {'Best':>10} {'Exps':>6} {'Kept':>6} {'Rate':>6} {'Time':>8}"
    lines.append(header)
    lines.append("─" * len(header))

    # Rows
    best_value = None
    best_idx = 0
    for i, (name, tracker) in enumerate(zip(names, trackers, strict=True)):
        bv = tracker.best_value
        n_exp = tracker.num_experiments
        n_kept = tracker.num_kept
        rate = f"{n_kept / n_exp * 100:.0f}%" if n_exp > 0 else "N/A"

        times = [r.training_time for r in tracker.records if r.training_time > 0]
        total_time = f"{sum(times) / 60:.1f}m" if times else "N/A"

        bv_str = f"{bv:.6f}" if bv is not None else "N/A"
        lines.append(
            f"{name:<{name_width}} {bv_str:>10} {n_exp:>6} {n_kept:>6} {rate:>6} {total_time:>8}"
        )

        if bv is not None and (
            best_value is None
            or (direction == "maximize" and bv > best_value)
            or (direction == "minimize" and bv < best_value)
        ):
            best_value = bv
            best_idx = i

    lines.append("")
    lines.append(f"Winner: {names[best_idx]} (best {objective_metric}={best_value:.6f})")

    # Parameter comparison — show best params side-by-side
    lines.append("")
    lines.append("Best params per run:")
    all_params: set[str] = set()
    for tracker in trackers:
        if tracker.best_record:
            all_params.update(tracker.best_record.params.keys())

    if all_params:
        param_header = f"{'Param':<25}"
        for name in names:
            param_header += f" {name:>15}"
        lines.append(param_header)
        lines.append("─" * len(param_header))

        for param in sorted(all_params):
            row = f"{param:<25}"
            for tracker in trackers:
                val = (
                    tracker.best_record.params.get(param, "—")
                    if tracker.best_record
                    else "—"
                )
                if isinstance(val, float):
                    row += f" {val:>15.6g}"
                else:
                    row += f" {str(val):>15}"
            lines.append(row)

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def compute_parameter_importance(
    records: list[Any],
    top_fraction: float = 0.3,
) -> dict[str, float]:
    """Estimate which parameters are most important for objective.

    For numeric parameters: compares the mean value in top-performing
    experiments vs. bottom-performing experiments. Larger divergence
    = more important.

    For categorical parameters (str/bool): compares the distribution of
    categories between top and bottom experiments. Higher divergence in
    category frequencies = more important.

    Args:
        records: List of ExperimentRecord objects.
        top_fraction: Fraction of experiments to consider "top" (default 30%).

    Returns:
        Dict of {param_name: importance_score} where score is in [0, 1].
        Higher score = parameter varies more between good and bad experiments.
    """
    successful = [r for r in records if r.status != "crash"]
    if len(successful) < 6:
        return {}

    sorted_records = sorted(
        successful, key=lambda r: r.objective_value, reverse=True
    )

    n_top = max(2, int(len(sorted_records) * top_fraction))
    top = sorted_records[:n_top]
    bottom = sorted_records[-n_top:]

    # Classify parameters as numeric or categorical
    numeric_params: set[str] = set()
    categorical_params: set[str] = set()
    for r in successful:
        for k, v in r.params.items():
            if isinstance(v, (int, float)):
                numeric_params.add(k)
            elif isinstance(v, (str, bool)):
                categorical_params.add(k)

    importance: dict[str, float] = {}

    # Numeric importance: normalized mean divergence
    for param in numeric_params:
        top_vals = [r.params[param] for r in top if param in r.params and isinstance(r.params[param], (int, float))]
        bottom_vals = [r.params[param] for r in bottom if param in r.params and isinstance(r.params[param], (int, float))]

        if not top_vals or not bottom_vals:
            continue

        top_mean = sum(top_vals) / len(top_vals)
        bottom_mean = sum(bottom_vals) / len(bottom_vals)

        all_vals = [r.params[param] for r in successful if param in r.params and isinstance(r.params[param], (int, float))]
        val_range = max(all_vals) - min(all_vals)

        if val_range == 0:
            importance[param] = 0.0
        else:
            importance[param] = abs(top_mean - bottom_mean) / val_range

    # Categorical importance: distribution divergence (Jensen-Shannon-like)
    for param in categorical_params:
        top_vals = [r.params[param] for r in top if param in r.params]
        bottom_vals = [r.params[param] for r in bottom if param in r.params]

        if not top_vals or not bottom_vals:
            continue

        # Compute frequency distributions
        all_categories = set(top_vals) | set(bottom_vals)
        if len(all_categories) < 2:
            importance[param] = 0.0
            continue

        top_freq: dict[str, float] = {}
        bottom_freq: dict[str, float] = {}
        for cat in all_categories:
            top_freq[str(cat)] = top_vals.count(cat) / len(top_vals)
            bottom_freq[str(cat)] = bottom_vals.count(cat) / len(bottom_vals)

        # Total variation distance: 0.5 * sum(|p - q|)
        tvd = 0.5 * sum(
            abs(top_freq.get(c, 0) - bottom_freq.get(c, 0))
            for c in all_categories
        )
        importance[param] = tvd

    # Normalize to [0, 1]
    if importance:
        max_imp = max(importance.values())
        if max_imp > 0:
            importance = {k: v / max_imp for k, v in importance.items()}

    return dict(sorted(importance.items(), key=lambda x: -x[1]))



def compute_convergence_curve(
    records: list[Any],
    direction: str = "maximize",
) -> list[tuple[int, float]]:
    """Compute the running accepted-best objective over experiment index.

    Crashed experiments are omitted. Discarded experiments remain on the
    timeline but do not advance the running best.
    """
    successful = [r for r in records if r.status != "crash"]
    if not successful:
        return []

    curve: list[tuple[int, float]] = []
    running_best = None

    for i, r in enumerate(successful):
        if running_best is None:
            running_best = r.objective_value
        elif r.status == "keep":
            if direction == "maximize" and r.objective_value > running_best:
                running_best = r.objective_value
            elif direction == "minimize" and r.objective_value < running_best:
                running_best = r.objective_value
        curve.append((i, running_best))

    return curve


def compute_diminishing_returns(
    records: list[Any],
    direction: str = "maximize",
    window: int = 10,
) -> float | None:
    """Estimate how much the last N experiments improved vs. earlier ones.

    Returns a ratio: 1.0 = still improving at same rate, 0.0 = no improvement.
    Returns None if not enough data.
    """
    curve = compute_convergence_curve(records, direction)
    if len(curve) < window * 2:
        return None

    early_improvement = abs(curve[window][1] - curve[0][1])
    recent_improvement = abs(curve[-1][1] - curve[-window][1])

    if early_improvement == 0:
        return 0.0 if recent_improvement == 0 else 1.0

    return min(1.0, recent_improvement / early_improvement)


def generate_report(
    records: list[Any],
    objective_metric: str = "eval_reward",
    direction: str = "maximize",
) -> str:
    """Generate a human-readable analysis report.

    Designed to be printed after a long auto-research run.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("HYPERPARAMETER IMPORTANCE ANALYSIS")
    lines.append("=" * 60)

    successful = [r for r in records if r.status != "crash"]
    if len(successful) < 6:
        lines.append(f"Not enough data for analysis ({len(successful)} experiments).")
        lines.append("Need at least 6 non-crashed experiments.")
        lines.append("=" * 60)
        return "\n".join(lines)

    lines.append(f"Experiments analyzed: {len(successful)}")

    # Best and worst
    if direction == "maximize":
        best = max(successful, key=lambda r: r.objective_value)
        worst = min(successful, key=lambda r: r.objective_value)
    else:
        best = min(successful, key=lambda r: r.objective_value)
        worst = max(successful, key=lambda r: r.objective_value)

    lines.append(f"Best:  {best.experiment_id} ({best.objective_value:.6f}) — {best.description}")
    lines.append(f"Worst: {worst.experiment_id} ({worst.objective_value:.6f}) — {worst.description}")

    # Parameter importance
    importance = compute_parameter_importance(successful)
    if importance:
        lines.append("")
        lines.append("Parameter importance (higher = more predictive of success):")
        for param, score in importance.items():
            bar = "#" * int(score * 20)
            lines.append(f"  {param:>25s}  {score:.3f}  {bar}")

        # Highlight most/least important
        most = list(importance.keys())[:3]
        least = [k for k, v in importance.items() if v < 0.1]
        if most:
            lines.append(f"\n  Focus on: {', '.join(most)}")
        if least:
            lines.append(f"  Probably don't matter: {', '.join(least)}")

    # Convergence analysis with ASCII chart
    curve = compute_convergence_curve(successful, direction)
    if len(curve) >= 10:
        lines.append("")
        lines.append("Convergence:")
        lines.append(f"  Start:  {curve[0][1]:.6f}")
        lines.append(f"  End:    {curve[-1][1]:.6f}")
        total_gain = abs(curve[-1][1] - curve[0][1])
        lines.append(f"  Gain:   {total_gain:.6f}")

        dr = compute_diminishing_returns(successful, direction)
        if dr is not None:
            if dr < 0.1:
                lines.append(f"  Trend:  Diminishing returns (ratio={dr:.2f}) — consider stopping")
            elif dr < 0.5:
                lines.append(f"  Trend:  Slowing improvement (ratio={dr:.2f})")
            else:
                lines.append(f"  Trend:  Still improving well (ratio={dr:.2f})")

        # ASCII convergence chart
        chart = _ascii_chart([v for _, v in curve], width=50, height=8)
        if chart:
            lines.append("")
            lines.append("  " + "─" * 52)
            for row in chart:
                lines.append(f"  │{row}│")
            lines.append("  " + "─" * 52)
            lines.append(f"  {'experiment 1':>1s}{' ' * 36}{'experiment ' + str(len(curve)):>14s}")

    # Timing
    times = [r.training_time for r in records if r.training_time > 0]
    if times:
        lines.append("")
        lines.append(f"Total wall time: {sum(times)/60:.1f} min ({len(records)} experiments)")
        lines.append(f"Avg per experiment: {sum(times)/len(times):.1f}s")

    # Improvement rate
    kept = sum(1 for r in records if r.status == "keep")
    crashed = sum(1 for r in records if r.status == "crash")
    if len(records) > 1:
        lines.append("")
        lines.append(f"Improvement rate: {kept}/{len(records)} ({kept/len(records)*100:.0f}%)")
        if crashed:
            lines.append(f"Crash rate: {crashed}/{len(records)} ({crashed/len(records)*100:.0f}%)")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
