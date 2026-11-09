"""Roll a fan-out's runs up into per-arm statistics and an honest verdict.

The verdict names a winner only when the evidence separates the best arm from
every other: all arms complete, each with at least two seeds, and non-overlapping
95% intervals. Non-overlap is deliberately conservative. A failed seed makes an arm
incomplete rather than quietly dropping out of its mean, because a seed that
crashed is evidence about the arm, not missing data.

Intervals use Student's t: fan-outs usually have few seeds, and a normal
approximation would be badly overconfident at n=3.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypeGuard

__all__ = ["REPORT_SCHEMA_VERSION", "build_report", "render_report"]

REPORT_SCHEMA_VERSION = 1

#: Two-sided 95% critical values of Student's t, by degrees of freedom.
_T_95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)
_T_95_LARGE = 1.96


def _usable(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _strict_json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in metrics.items()
        if not isinstance(value, float) or math.isfinite(value)
    }


def _summarize_arm(
    arm_id: str, runs: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda run: run["seed"])
    samples: list[tuple[int, float]] = []
    failed: list[int] = []
    missing: list[int] = []
    for run in ordered:
        if run["status"] != "succeeded":
            failed.append(run["seed"])
            continue
        value = (run.get("metrics") or {}).get(metric)
        if _usable(value):
            samples.append((run["seed"], float(value)))
        else:
            missing.append(run["seed"])

    values = [value for _, value in samples]
    count = len(values)
    mean = math.fsum(values) / count if count else None
    std = ci95 = None
    if count >= 2 and mean is not None:
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (count - 1))
        degrees = count - 1
        critical = _T_95[degrees - 1] if degrees <= len(_T_95) else _T_95_LARGE
        half_width = critical * std / math.sqrt(count)
        ci95 = [mean - half_width, mean + half_width]
    return {
        "arm_id": arm_id,
        "params": ordered[0].get("params") or {},
        "n": count,
        "seeds": [seed for seed, _ in samples],
        "failed_seeds": failed,
        "missing_metric_seeds": missing,
        "complete": not failed and not missing,
        "mean": mean,
        "std": std,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "ci95": ci95,
        "values": values,
    }


def _verdict(
    ranked: list[dict[str, Any]], arms: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    def decided(status: str, reason: str, best: str | None = None) -> dict[str, Any]:
        return {"status": status, "best_arm": best, "reason": reason}

    if not ranked:
        return decided("no_data", "no run produced a usable value for the metric")
    if len(arms) == 1:
        return decided("single_arm", "one arm: there is nothing to compare it against")
    # Every arm has to count, not just the top two: an arm with a wide interval or
    # missing seeds can rank third and still turn out to be the best.
    for arm in arms:
        if not arm["complete"]:
            return decided(
                "unresolved",
                f"{arm['arm_id']} has failed, unfinished or metric-less seeds, so "
                "its mean may reflect only the seeds that survived",
            )
    if any(arm["ci95"] is None for arm in ranked):
        return decided("unresolved", "separating arms needs at least 2 seeds per arm")
    best = ranked[0]
    for other in ranked[1:]:
        separated = (
            best["ci95"][0] > other["ci95"][1]
            if mode == "max"
            else best["ci95"][1] < other["ci95"][0]
        )
        if not separated:
            return decided(
                "unresolved",
                f"the 95% intervals of {best['arm_id']} and {other['arm_id']} overlap",
            )
    return decided(
        "resolved",
        f"{best['arm_id']}'s 95% interval is clear of every other arm's",
        best=best["arm_id"],
    )


def build_report(
    *,
    name: str,
    digest: str,
    metric: str,
    mode: str,
    runs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build the JSON-safe report for a set of run records."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_arm.setdefault(run["arm_id"], []).append(run)
    arms = [
        _summarize_arm(arm_id, members, metric) for arm_id, members in by_arm.items()
    ]
    ranked = sorted(
        (arm for arm in arms if arm["n"] > 0),
        key=lambda arm: arm["mean"],
        reverse=mode == "max",
    )
    statuses = [run["status"] for run in runs]
    known_costs = [run["cost_cents"] for run in runs if _usable(run.get("cost_cents"))]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "name": name,
        "manifest_digest": digest,
        "metric": metric,
        "mode": mode,
        "runs": {
            "total": len(runs),
            "succeeded": statuses.count("succeeded"),
            "failed": statuses.count("failed"),
            "cancelled": statuses.count("cancelled"),
            "incomplete": sum(
                status not in ("succeeded", "failed", "cancelled")
                for status in statuses
            ),
        },
        "arms": sorted(arms, key=lambda arm: arm["arm_id"]),
        "ranking": [arm["arm_id"] for arm in ranked],
        "verdict": _verdict(ranked, arms, mode),
        "cost": {
            "known_cents": round(math.fsum(known_costs), 4),
            "runs_with_unknown_cost": len(runs) - len(known_costs),
        },
        "artifacts_missing": [
            {"run_id": run["run_id"], "errors": run["collect_errors"]}
            for run in runs
            if run.get("collect_errors")
        ],
        "run_results": [
            {
                "run_id": run["run_id"],
                "arm_id": run["arm_id"],
                "seed": run["seed"],
                "status": run["status"],
                "metrics": _strict_json_metrics(run.get("metrics") or {}),
            }
            for run in runs
        ],
    }


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _params_label(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items())) or "(no parameters)"


def render_report(report: dict[str, Any]) -> str:
    """Render a report for a terminal."""
    runs = report["runs"]
    verdict = report["verdict"]
    lines = [
        f"{report['name']}: {report['mode']} {report['metric']}",
        f"Runs: {runs['succeeded']}/{runs['total']} succeeded, "
        f"{runs['failed']} failed, {runs['cancelled']} cancelled, "
        f"{runs['incomplete']} incomplete",
        "",
    ]
    by_id = {arm["arm_id"]: arm for arm in report["arms"]}
    ranked = report["ranking"]
    unranked = [arm_id for arm_id in sorted(by_id) if arm_id not in ranked]
    for arm_id in [*ranked, *unranked]:
        arm = by_id[arm_id]
        interval = (
            f" 95% CI [{arm['ci95'][0]:.4g}, {arm['ci95'][1]:.4g}]"
            if arm["ci95"]
            else ""
        )
        notes = []
        if arm["failed_seeds"]:
            notes.append(f"failed seeds {arm['failed_seeds']}")
        if arm["missing_metric_seeds"]:
            notes.append(f"no metric on seeds {arm['missing_metric_seeds']}")
        suffix = f"  ({'; '.join(notes)})" if notes else ""
        lines.append(
            f"  {arm_id}  mean {_number(arm['mean'])} n={arm['n']}{interval}"
            f"  {_params_label(arm['params'])}{suffix}"
        )
    lines += ["", f"Verdict: {verdict['status']}: {verdict['reason']}"]
    for missing in report.get("artifacts_missing", []):
        for error in missing["errors"]:
            lines.append(f"Not collected from {missing['run_id']}: {error}")
    cost = report["cost"]
    cost_line = f"Cost: ${cost['known_cents'] / 100:.4f}"
    if cost["runs_with_unknown_cost"]:
        cost_line += f" known; unknown for {cost['runs_with_unknown_cost']} run(s)"
    lines.append(cost_line)
    return "\n".join(lines)
