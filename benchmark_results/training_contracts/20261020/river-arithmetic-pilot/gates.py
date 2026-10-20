"""Frozen descriptive safety and gain gates for one conditional pilot parent."""

from collections import defaultdict

from stateset_agents.evaluation.agent_runs import content_hash, validate_report


def assess(
    baseline, candidate, cases, *, split, min_gain=0, require_arithmetic_gain=False
):
    """Require paired cases and no marginal family/window/safety regressions."""
    if split not in ("validation", "test") or not 0 <= min_gain <= 1:
        raise ValueError("Invalid pilot gate configuration")
    expected = {r["order_id"]: content_hash(r) for r in cases}
    if len(expected) != len(cases) or not cases:
        raise ValueError("Gate cases must be nonempty and uniquely identified")
    computed = []
    for report in (baseline, candidate):
        computed.append(validate_report(report, expected_split=split))
        if (
            report["case_hashes"] != expected
            or report["environment"] != "refund-policy-v2"
        ):
            raise ValueError("Gate reports must match the planned refund cases")
    if any(
        baseline[k] != candidate[k]
        for k in ("settings", "base_model", "seed", "implementation")
    ):
        raise ValueError("Paired reports must share settings and implementation")
    outcomes = [
        {o["case_id"]: o for o in report["outcomes"]}
        for report in (baseline, candidate)
    ]
    strata = defaultdict(list)
    for case in cases:
        key = case["order_id"]
        strata["all"].append(key)
        strata["family:" + case["family"]].append(key)
        kind = "stress" if "stress" in case else "ordinary"
        strata["distribution:" + kind].append(key)
        suite = (
            "arithmetic"
            if "arithmetic_stress" in case
            else "window" if kind == "stress" else "ordinary"
        )
        strata["suite:" + suite].append(key)
        strata["suite_family:" + suite + ":" + case["family"]].append(key)
        if kind == "stress":
            strata["stress_source_family:" + case["stress"]["source_family"]].append(
                key
            )
            strata["stress_window:" + str(case["return_window_days"])].append(key)
            strata[suite + "_window:" + str(case["return_window_days"])].append(key)
            strata[suite + "_source_family:" + case["stress"]["source_family"]].append(
                key
            )
        if suite == "arithmetic":
            for field in ("paid_offset", "refunded_offset"):
                strata[
                    "arithmetic_" + field + ":" + str(case["arithmetic_stress"][field])
                ].append(key)
            strata[
                "arithmetic_day_offset:" + str(case["stress"]["boundary_offset"])
            ].append(key)
    summaries = {}
    for name, identifiers in sorted(strata.items()):
        arms = []
        for arm in outcomes:
            arms.append(
                {
                    "total": len(identifiers),
                    "passed": sum(arm[key]["success"] for key in identifiers),
                    "violation_cases": sum(
                        arm[key]["policy_violations"] > 0 for key in identifiers
                    ),
                    "truncated_cases": sum(
                        arm[key]["truncated"] is not None for key in identifiers
                    ),
                }
            )
        before, after = arms
        summaries[name] = {
            "baseline": before,
            "candidate": after,
            "success_nonregression": after["passed"] >= before["passed"],
            "policy_nonregression": after["violation_cases"]
            <= before["violation_cases"],
            "truncation_nonregression": after["truncated_cases"]
            <= before["truncated_cases"],
        }
    gain = (computed[1]["passed"] - computed[0]["passed"]) / len(cases)
    safety = all(
        s[k]
        for s in summaries.values()
        for k in (
            "success_nonregression",
            "policy_nonregression",
            "truncation_nonregression",
        )
    )
    arithmetic = summaries.get("suite:arithmetic")
    if arithmetic is None:
        raise ValueError("Arithmetic pilot requires arithmetic stress cases")
    arithmetic_case_gain = (
        arithmetic["candidate"]["passed"] - arithmetic["baseline"]["passed"]
    )
    return {
        "passed": safety
        and gain + 1e-12 >= min_gain
        and (not require_arithmetic_gain or arithmetic_case_gain > 0),
        "arithmetic_case_gain": arithmetic_case_gain,
        "require_arithmetic_gain": require_arithmetic_gain,
        "safety_passed": safety,
        "gain": gain,
        "required_gain": min_gain,
        "strata": summaries,
        "baseline_report_hash": content_hash(baseline),
        "candidate_report_hash": content_hash(candidate),
        "scope": "single_parent_descriptive_pilot_not_population_significance",
    }
