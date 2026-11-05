"""Stress evaluation coverage, provenance, and hold-out isolation."""

import copy
import json

import pytest

from stateset_agents.core.environments.refund_policy_environment import (
    FAMILIES,
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.data.refund_demonstrations import reference_refund_demonstration
from stateset_agents.data.refund_stress import refund_policy_stress_cases


@pytest.mark.asyncio
async def test_grid_has_independent_orders_private_strata_and_verified_decisions():
    cases = refund_policy_stress_cases("validation", seed=42, windows=(5, 18, 40))
    assert len(cases) == 72
    assert len({row["stress"]["source_case_hash"] for row in cases}) == 72
    assert {
        (
            row["stress"]["source_family"],
            row["return_window_days"],
            row["stress"]["boundary_offset"],
        )
        for row in cases
    } == {
        (family, window, offset)
        for family in FAMILIES
        for window in (5, 18, 40)
        for offset in (-1, 0, 1)
    }
    for case in cases:
        assert case["days_since_delivery"] == (
            case["return_window_days"] + case["stress"]["boundary_offset"]
        )
        demonstration = await reference_refund_demonstration(case)
        messages = demonstration["messages"]
        observed = json.loads(messages[2]["content"])
        assert "stress" not in observed and "family" not in observed
        assert "source_case_hash" not in messages[0]["content"]
    assert cases == refund_policy_stress_cases(
        "validation", seed=42, windows=(5, 18, 40)
    )


def test_split_and_seed_isolation_and_aliased_fact_exclusions():
    train = refund_policy_benchmark("train", count=128, seed=42)
    before = copy.deepcopy(train)
    validation = refund_policy_stress_cases(
        "validation", seed=42, windows=(5, 18, 40), excluded=train
    )
    test = refund_policy_stress_cases(
        "test", seed=42, windows=(8, 25, 75), excluded=train + validation
    )
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    assert len({fingerprint(r) for r in train + validation + test}) == 272
    assert len({r["order_id"] for r in train + validation + test}) == 272
    assert train == before
    assert validation != refund_policy_stress_cases(
        "validation", seed=43, windows=(5, 18, 40)
    )
    alias = copy.deepcopy(validation[0])
    alias["order_id"] = "changed-id"
    with pytest.raises(ValueError, match="overlap"):
        refund_policy_stress_cases(
            "validation", seed=42, windows=(5, 18, 40), excluded=[alias]
        )


@pytest.mark.parametrize("windows", [(), (0,), (-1,), (True,), (1.5,), (5, 5)])
def test_invalid_windows_fail(windows):
    with pytest.raises(ValueError, match="windows"):
        refund_policy_stress_cases("validation", seed=42, windows=windows)


@pytest.mark.parametrize("split", ["train", "unknown"])
def test_training_cannot_be_generated_as_stress_holdout(split):
    with pytest.raises(ValueError, match="split"):
        refund_policy_stress_cases(split, seed=42, windows=(5,))
