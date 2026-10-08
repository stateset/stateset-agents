"""Validated, paired comparisons of held-out agent outcomes without an SDK."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from stateset_agents.evaluation.implementation import validate_implementation


def content_hash(value: Any) -> str:
    """Hash canonical JSON, rejecting nonfinite numbers."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _count(value: Any, name: str) -> int:
    """Preserve integer counters exactly, accepting integral SDK float metrics."""
    if type(value) is int and value >= 0:
        return value
    if (
        type(value) is float
        and math.isfinite(value)
        and value >= 0
        and value.is_integer()
    ):
        return int(value)
    raise ValueError(f"{name} must be a nonnegative integer")


def summarize_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate complete outcome records and compute descriptive statistics.

    Wilson intervals describe individual-run success proportions; they are not
    a significance test for improvement or a substitute for independent seeds.
    """
    if not outcomes:
        raise ValueError("Evaluation must contain outcomes")
    ids: set[str] = set()
    passed = violations = truncated = 0
    rewards: list[float] = []
    tokens = calls = 0
    durations: list[float] = []
    for outcome in outcomes:
        case_id = outcome.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case_id must be nonempty and unique")
        ids.add(case_id)
        if type(outcome.get("success")) is not bool:
            raise ValueError("success must be boolean")
        cause = outcome.get("truncated")
        if cause is not None and (not isinstance(cause, str) or not cause):
            raise ValueError("truncated must be a cause string or null")
        count = _count(outcome.get("policy_violations"), "policy_violations")
        generated = _count(outcome.get("generated_tokens"), "generated_tokens")
        tool_calls = _count(outcome.get("tool_calls"), "tool_calls")
        seconds = _number(outcome.get("elapsed_seconds"), "elapsed_seconds")
        if seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative")
        if outcome["success"] and (cause is not None or count > 0):
            raise ValueError(
                "Truncated or policy-violating outcomes cannot be successful"
            )
        passed += outcome["success"]
        violations += count > 0
        truncated += cause is not None
        rewards.append(_number(outcome.get("reward"), "reward"))
        tokens += generated
        calls += tool_calls
        durations.append(seconds)
    n = len(outcomes)
    try:
        elapsed = _number(math.fsum(durations), "aggregate trajectory seconds")
        mean_calls = _number(calls / n, "mean tool calls")
    except OverflowError as exc:
        raise ValueError("Evaluation aggregate exceeds finite numeric range") from exc
    rate, z = passed / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return {
        "total": n,
        "passed": passed,
        "success_rate": rate,
        "success_wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
        "violation_rate": violations / n,
        "truncation_rate": truncated / n,
        "mean_reward": statistics.mean(rewards),
        "generated_tokens": tokens,
        "tool_calls": calls,
        "mean_tool_calls": mean_calls,
        "trajectory_seconds": elapsed,
    }


def evaluation_report(
    *,
    environment: str,
    base_model: str,
    seed: int,
    cases: Mapping[str, Any],
    settings: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    case_families: Mapping[str, str] | None = None,
    implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detached test snapshot bound to exact cases and evaluation settings."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "environment": environment,
        "base_model": base_model,
        "seed": seed,
        "split": "test",
        "case_hashes": {key: content_hash(row) for key, row in sorted(cases.items())},
        "settings": dict(settings),
        "selected_checkpoint": dict(checkpoint),
        "outcomes": list(outcomes),
    }
    if case_families is not None:
        report["case_families"] = dict(case_families)
    if implementation is not None:
        report["implementation"] = dict(implementation)
    report = copy.deepcopy(report)
    report.update(validate_report(report))
    report["outcomes"].sort(key=lambda row: row["case_id"])
    return report


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Reject malformed, incomplete, or duplicated evaluation evidence."""
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("split") != "test"
    ):
        raise ValueError("Expected schema_version 1 held-out test report")
    for key in ("environment", "base_model"):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ValueError(f"{key} must be nonempty")
    if type(report.get("seed")) is not int or report["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(report.get("settings"), dict) or not report["settings"]:
        raise ValueError("Evaluation settings are required")
    if "implementation" in report:
        validate_implementation(report["implementation"])
    checkpoint = report.get("selected_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint["path"]
    ):
        raise ValueError("Selected checkpoint identity is required")
    hashes = report.get("case_hashes")
    if (
        not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for key, value in hashes.items()
        )
    ):
        raise ValueError("Exact case hashes are required")
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list) or any(not isinstance(o, dict) for o in outcomes):
        raise ValueError("outcomes must be a list of objects")
    summary = summarize_outcomes(outcomes)
    if set(hashes) != {o["case_id"] for o in outcomes}:
        raise ValueError(
            "Evaluation must contain exactly one outcome for every test case"
        )
    for outcome in outcomes:
        if (
            "case_hash" in outcome
            and outcome["case_hash"] != hashes[outcome["case_id"]]
        ):
            raise ValueError("Test outcome reset input differs from planned case")
    if "case_families" in report:
        families = report["case_families"]
        if (
            not isinstance(families, dict)
            or set(families) != set(hashes)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in families.values()
            )
        ):
            raise ValueError("case_families must label every test case")
        summary["families"] = {
            family: summarize_outcomes(
                [o for o in outcomes if families[o["case_id"]] == family]
            )
            for family in sorted(set(families.values()))
        }
    content_hash(report)  # Includes metadata; reject NaN anywhere.
    return summary


def _resource_comparison(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe held-out workload, charging unsuccessful attempts to success."""

    def ratios(summary: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, float | None] = {}
        for field, denominator in (
            ("generated_tokens_per_case", summary["total"]),
            ("generated_tokens_per_success", summary["passed"]),
        ):
            if denominator == 0:
                result[field] = None
                continue
            try:
                result[field] = _number(
                    summary["generated_tokens"] / denominator, field
                )
            except OverflowError as exc:
                raise ValueError(
                    "Token efficiency exceeds finite numeric range"
                ) from exc
        return result

    return {
        "baseline": ratios(before),
        "candidate": ratios(after),
        "candidate_minus_baseline": {
            key: after[key] - before[key]
            for key in ("generated_tokens", "tool_calls", "trajectory_seconds")
        },
    }


def compare_runs(
    baselines: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_seeds: int = 3,
    min_gain: float = 0.03,
    require_significance: bool = False,
    alpha: float = 0.05,
    comparisons: int = 1,
) -> dict[str, Any]:
    """Compare matched runs with explicit empirical improvement gates.

    A passing gate requires enough seeds, the requested mean absolute success
    gain, improvement in every seed, and no violation/truncation regressions in
    any seed. The optional exact one-sided sign test uses seeds as units, not
    individual trajectories, and applies a Bonferroni correction for the number
    of planned comparisons. Its inference assumes independent training runs
    and a protocol fixed before observing test results.
    """
    if type(min_seeds) is not int or min_seeds < 1:
        raise ValueError("min_seeds must be positive")
    if not 0 <= _number(min_gain, "min_gain") <= 1:
        raise ValueError("min_gain must be in [0, 1]")
    if type(require_significance) is not bool:
        raise ValueError("require_significance must be boolean")
    if not 0 < _number(alpha, "alpha") < 1:
        raise ValueError("alpha must be in (0, 1)")
    if type(comparisons) is not int or comparisons < 1:
        raise ValueError("comparisons must be a positive integer")

    def index(reports: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
        indexed: dict[int, Mapping[str, Any]] = {}
        for report in reports:
            validate_report(report)
            if report["seed"] in indexed:
                raise ValueError("Duplicate seeds cannot count as independent evidence")
            indexed[report["seed"]] = report
        return indexed

    left, right = index(baselines), index(candidates)
    if not left or left.keys() != right.keys():
        raise ValueError(
            "Baseline and candidate must have identical nonempty seed sets"
        )
    reference = left[min(left)]
    pairs = []
    for seed in sorted(left):
        baseline, candidate = left[seed], right[seed]
        for key in ("environment", "base_model", "settings"):
            if baseline[key] != candidate[key] or baseline[key] != reference[key]:
                raise ValueError(
                    f"Mismatched {key}: comparisons require the same protocol"
                )
        if baseline.get("implementation") != candidate.get(
            "implementation"
        ) or baseline.get("implementation") != reference.get("implementation"):
            raise ValueError(
                "Mismatched implementation: source identities must match across all runs"
            )
        if baseline["case_hashes"] != candidate["case_hashes"]:
            raise ValueError("Baseline and candidate test cases differ")
        if baseline.get("case_families") != candidate.get("case_families"):
            raise ValueError("Baseline and candidate scenario families differ")
        if set(baseline.get("case_families", {}).values()) != set(
            reference.get("case_families", {}).values()
        ):
            raise ValueError("Scenario family coverage must match across seeds")
        before, after = validate_report(baseline), validate_report(candidate)
        old = {o["case_id"]: o for o in baseline["outcomes"]}
        new = {o["case_id"]: o for o in candidate["outcomes"]}
        pairs.append(
            {
                "seed": seed,
                "baseline": before,
                "candidate": after,
                "resource_usage": _resource_comparison(before, after),
                "checkpoints": {
                    "baseline": baseline["selected_checkpoint"],
                    "candidate": candidate["selected_checkpoint"],
                },
                "evidence_hashes": {
                    "baseline": content_hash(baseline),
                    "candidate": content_hash(candidate),
                },
                "success_gain": after["success_rate"] - before["success_rate"],
                "wins": sum(new[k]["success"] and not old[k]["success"] for k in old),
                "regressions": sum(
                    old[k]["success"] and not new[k]["success"] for k in old
                ),
            }
        )
        if "families" in before:
            pairs[-1]["families"] = {
                family: {
                    "baseline": before["families"][family],
                    "candidate": after["families"][family],
                    "resource_usage": _resource_comparison(
                        before["families"][family], after["families"][family]
                    ),
                    "success_gain": after["families"][family]["success_rate"]
                    - before["families"][family]["success_rate"],
                }
                for family in before["families"]
            }
    gains = [pair["success_gain"] for pair in pairs]
    positive = sum(g > 0 for g in gains)
    negative = sum(g < 0 for g in gains)
    nonzero = positive + negative
    # Exact binomial upper tail under P(positive | non-tie) = 1/2.
    p_value = sum(math.comb(nonzero, k) for k in range(positive, nonzero + 1)) / (
        2**nonzero
    )
    adjusted_p = min(1.0, p_value * comparisons)
    distinct_checkpoints = len(
        {r["selected_checkpoint"]["path"] for r in right.values()}
    ) == len(right)
    inference = {
        "method": "exact_one_sided_seed_sign_test",
        "unit": "paired_run_seed",
        "positive_seeds": positive,
        "negative_seeds": negative,
        "tied_seeds": len(pairs) - nonzero,
        "p_value": p_value,
        "bonferroni_adjusted_p_value": adjusted_p,
        "alpha": alpha,
        "planned_comparisons": comparisons,
        "rejects_null": adjusted_p <= alpha,
        "distinct_candidate_checkpoints": distinct_checkpoints,
        "assumptions": (
            "Independent training runs; seed set, comparisons and protocol fixed "
            "before test inspection. Distinct checkpoint paths alone do not prove "
            "independence. Tests direction consistency, not mean effect size or safety."
        ),
    }
    gates = {
        "enough_seeds": len(pairs) >= min_seeds,
        "mean_success_gain": statistics.mean(gains) >= min_gain,
        "improved_every_seed": all(g > 0 for g in gains),
        "no_violation_regression": all(
            p["candidate"]["violation_rate"] <= p["baseline"]["violation_rate"]
            for p in pairs
        ),
        "no_truncation_regression": all(
            p["candidate"]["truncation_rate"] <= p["baseline"]["truncation_rate"]
            for p in pairs
        ),
    }
    if require_significance:
        gates["seed_significance"] = adjusted_p <= alpha
        gates["distinct_candidate_checkpoints"] = distinct_checkpoints
    if "case_families" in reference:
        gates["no_family_regression"] = all(
            family["success_gain"] >= 0
            and family["candidate"]["violation_rate"]
            <= family["baseline"]["violation_rate"]
            and family["candidate"]["truncation_rate"]
            <= family["baseline"]["truncation_rate"]
            for pair in pairs
            for family in pair["families"].values()
        )
    return {
        "schema_version": 1,
        "protocol": {
            **{
                key: reference[key] for key in ("environment", "base_model", "settings")
            },
            "implementation": reference.get("implementation"),
        },
        "passed": all(gates.values()),
        "gates": gates,
        "thresholds": {
            "min_seeds": min_seeds,
            "min_gain": min_gain,
            "require_significance": require_significance,
            "alpha": alpha,
            "comparisons": comparisons,
        },
        "inference": inference,
        "resource_usage_scope": (
            "Descriptive held-out rollout usage, including failed attempts. Tokens "
            "per success are null when no case succeeds. Positive deltas mean more "
            "candidate usage. Trajectory seconds sum individual latencies, not "
            "wall-clock duration. Excludes training, input tokens, provider billing "
            "and unrecorded retries; usage is not independently verified. These "
            "measurements do not change the learning or safety gates."
        ),
        "mean_success_gain": statistics.mean(gains),
        "seed_gain_stddev": statistics.stdev(gains) if len(gains) > 1 else None,
        "pairs": pairs,
        "interpretation": (
            "Engineering gates with a conditional seed-level sign test; "
            "not production certification. Review training provenance and test isolation."
        ),
    }


def compare_study(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    min_seeds: int = 6,
    min_gain: float = 0.03,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Gate a prespecified four-arm study with all three RL comparisons.

    Always requires statistical evidence and distinct checkpoints for every
    trained run. Costs and live recovery are outside this learning-evidence gate.
    """
    names = ("base", "sft", "rejection_sft", "rl")
    if set(arms) != set(names):
        raise ValueError("Study requires exactly base, sft, rejection_sft and rl arms")
    comparisons = {
        name: compare_runs(
            arms[name],
            arms["rl"],
            min_seeds=min_seeds,
            min_gain=min_gain,
            require_significance=True,
            alpha=alpha,
            comparisons=3,
        )
        for name in names[:-1]
    }
    # Re-evaluating one trained checkpoint with new dataset seeds is not a
    # repeated training study, even if the paths differ between the two arms.
    trained_paths = [
        r["selected_checkpoint"]["path"] for name in names[1:] for r in arms[name]
    ]
    base_paths = {r["selected_checkpoint"]["path"] for r in arms["base"]}
    gates = {
        **{
            f"rl_beats_{name}": result["passed"] for name, result in comparisons.items()
        },
        "distinct_trained_checkpoints": len(set(trained_paths)) == len(trained_paths),
        "trained_checkpoints_differ_from_base": not base_paths.intersection(
            trained_paths
        ),
    }
    return {
        "schema_version": 1,
        "study": "four_arm_agent_learning",
        "passed": all(gates.values()),
        "gates": gates,
        "comparisons": comparisons,
        "scope": (
            "Held-out learning evidence only. Independence and test isolation need "
            "a provenance audit; measured cost, live recovery, and deployment "
            "validation are not assessed. A pass is not an A+ certification."
        ),
    }
