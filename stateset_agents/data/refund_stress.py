"""Explicit synthetic policy stress cases, separate from canonical study splits."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.utils.provenance import content_hash

STRESS_POLICY = "policy_window_grid_v1"


def refund_policy_stress_cases(
    split: str,
    *,
    seed: int,
    windows: Sequence[int],
    excluded: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build independent cases for every source family, window and adjacent day.

    Return 24 cases per window: eight scenario families at offsets -1, 0, +1.
    Each cell has its own independently drawn order, rather than sharing a
    training parent across held-out splits. Monetary, priority and note facts
    are preserved from that draw. ``family`` describes the resulting facts;
    private ``stress`` metadata records the original stratum and provenance.

    This changes the evaluation distribution and must be declared before model
    execution. It is not a canonical reference-data bundle or a quality result.
    Callers must supply historical and current training/held-out exclusions;
    overlap fails rather than silently resampling after observing outcomes.
    """
    if split not in ("validation", "test"):
        raise ValueError("Stress cases require a validation or test split")
    if (
        not windows
        or any(type(window) is not int or window < 1 for window in windows)
        or len(set(windows)) != len(windows)
    ):
        raise ValueError("Stress windows must be distinct positive integers")
    sources = refund_policy_benchmark(split, count=24 * len(windows), seed=seed)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for source in sorted(sources, key=lambda row: row["order_id"]):
        by_family.setdefault(source["family"], []).append(source)
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    used_ids = {row["order_id"] for row in excluded}
    used_facts = {fingerprint(row) for row in excluded}
    cases = []
    for family, rows in sorted(by_family.items()):
        for index, source in enumerate(rows):
            window, offset = windows[index // 3], index % 3 - 1
            case = copy.deepcopy(source)
            case["return_window_days"] = window
            case["days_since_delivery"] = window + offset
            if family != "misleading_note":
                if case["chargeback_open"]:
                    case["family"] = "chargeback"
                elif case["refunded_cents"] == case["paid_cents"]:
                    case["family"] = "already_refunded"
                elif case["status"] != "delivered":
                    case["family"] = "undelivered"
                elif offset > 0:
                    case["family"] = "outside_window"
                elif offset == 0:
                    case["family"] = "window_boundary"
                else:
                    case["family"] = (
                        "partial_refund" if case["refunded_cents"] else "eligible"
                    )
            case["stress"] = {
                "policy": STRESS_POLICY,
                "source_family": family,
                "source_case_hash": content_hash(source),
                "return_window_days": window,
                "boundary_offset": offset,
            }
            identity = content_hash(
                {"split": split, "seed": seed, "stress": case["stress"]}
            )
            case["order_id"] = f"policy-{identity[:24]}"
            facts = fingerprint(case)
            if case["order_id"] in used_ids or facts in used_facts:
                raise ValueError("Stress cases overlap excluded or generated cases")
            used_ids.add(case["order_id"])
            used_facts.add(facts)
            cases.append(case)
    return sorted(cases, key=lambda row: row["order_id"])
