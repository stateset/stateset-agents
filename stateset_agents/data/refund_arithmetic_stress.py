"""Prospective subtraction stress cases for the synthetic refund policy."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.evaluation.agent_runs import content_hash

ARITHMETIC_STRESS_POLICY = "refund_arithmetic_grid_v1"


def refund_arithmetic_stress_cases(
    split: str,
    *,
    seed: int,
    windows: Sequence[int],
    quantum: int,
    repetitions: int = 1,
    excluded: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Draw independent partial orders for every window/day/amount offset cell.

    Use days -1/0 around each inclusive window and offsets -1/0/+1 around
    independently drawn payment and previous-refund decimal anchors. Return
    18 cases per window per repetition. No source order is reused between cells.
    Windows, decimal quantum, repetitions and exclusions must be declared before
    model evaluation; these are stress fixtures, not canonical training inputs.
    Overlap is rejected, never silently resampled after observing outcomes.
    """
    if split not in ("validation", "test"):
        raise ValueError("Arithmetic stress requires validation or test")
    if (
        not windows
        or any(type(window) is not int or window < 1 for window in windows)
        or len(set(windows)) != len(windows)
    ):
        raise ValueError("Stress windows must be distinct positive integers")
    if type(quantum) is not int or quantum < 10 or str(quantum).strip("0") != "1":
        raise ValueError("Arithmetic quantum must be a power of ten, at least ten")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Arithmetic repetitions must be a positive integer")
    count = 18 * len(windows) * repetitions
    sources = iter(
        sorted(
            (
                row
                for row in refund_policy_benchmark(split, count=8 * count, seed=seed)
                if row["family"] == "partial_refund"
            ),
            key=lambda row: row["order_id"],
        )
    )
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    used_ids = {row["order_id"] for row in excluded}
    used_facts = {fingerprint(row) for row in excluded}
    cases = []
    for window in windows:
        for day_offset in (-1, 0):
            for paid_offset in (-1, 0, 1):
                for refunded_offset in (-1, 0, 1):
                    for repeat in range(repetitions):
                        source = next(sources)
                        paid_anchor = (
                            (source["paid_cents"] + quantum - 1) // quantum
                        ) * quantum
                        refunded_anchor = min(
                            max(quantum, source["refunded_cents"] // quantum * quantum),
                            paid_anchor - 3,
                        )
                        case = copy.deepcopy(source)
                        case.update(
                            {
                                "paid_cents": paid_anchor + paid_offset,
                                "refunded_cents": refunded_anchor + refunded_offset,
                                "return_window_days": window,
                                "days_since_delivery": window + day_offset,
                                "family": (
                                    "partial_refund"
                                    if day_offset < 0
                                    else "window_boundary"
                                ),
                            }
                        )
                        case["stress"] = {
                            "policy": ARITHMETIC_STRESS_POLICY,
                            "source_family": "partial_refund",
                            "source_case_hash": content_hash(source),
                            "return_window_days": window,
                            "boundary_offset": day_offset,
                        }
                        case["arithmetic_stress"] = {
                            "quantum": quantum,
                            "paid_offset": paid_offset,
                            "refunded_offset": refunded_offset,
                            "repetition": repeat,
                        }
                        identity = content_hash(
                            {
                                "split": split,
                                "seed": seed,
                                "stress": case["stress"],
                                "arithmetic": case["arithmetic_stress"],
                            }
                        )
                        case["order_id"] = f"policy-{identity[:24]}"
                        facts = fingerprint(case)
                        if case["order_id"] in used_ids or facts in used_facts:
                            raise ValueError(
                                "Arithmetic stress cases overlap excluded or generated cases"
                            )
                        used_ids.add(case["order_id"])
                        used_facts.add(facts)
                        cases.append(case)
    return sorted(cases, key=lambda row: row["order_id"])
