"""Offline control-data invariants checked before any provider resources."""

from collections import Counter
import copy
import importlib.util
from pathlib import Path

import pytest

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.data.refund_demonstrations import replay_refund_candidate
from stateset_agents.evaluation.agent_runs import content_hash

SOURCE = Path(__file__).with_name("control.py")
SPEC = importlib.util.spec_from_file_location("study_control", SOURCE)
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


@pytest.mark.asyncio
async def test_repetition_keeps_facts_balances_families_and_replays_actions():
    canonical = refund_policy_benchmark("train", 16, 9901)
    original = copy.deepcopy(canonical)
    cases, examples, lineage = await CONTROL.repetition_control(canonical, 31, 9902)
    assert canonical == original
    assert len(cases) == len(examples) == len(lineage) == 31
    assert len({case["order_id"] for case in cases}) == 31
    assert {RefundPolicyEnvironment.case_fingerprint(row) for row in cases} == {
        RefundPolicyEnvironment.case_fingerprint(row) for row in canonical
    }
    source = {row["order_id"]: row for row in canonical}
    families = Counter(row["family"] for row in cases)
    assert max(families.values()) - min(families.values()) <= 1
    for case, example, origin in zip(cases, examples, lineage, strict=True):
        parent = source[origin["source_case_id"]]
        assert {key: value for key, value in case.items() if key != "order_id"} == {
            key: value for key, value in parent.items() if key != "order_id"
        }
        assert example["metadata"]["split"] == "train"
        assert example["metadata"]["case_hash"] == content_hash(case)
        await replay_refund_candidate(
            case,
            {
                "case_id": case["order_id"],
                "case_hash": content_hash(case),
                "messages": example["messages"],
                "truncated": None,
            },
        )
        if origin["alias"]:
            assert case["order_id"].startswith("curriculum-")
            assert case["order_id"] not in source
            assert len(case["order_id"]) == len("curriculum-") + 24


@pytest.mark.asyncio
async def test_control_rebuild_is_seeded_without_changing_policy_facts():
    canonical = refund_policy_benchmark("train", 16, 9903)
    first = await CONTROL.repetition_control(canonical, 31, 9904)
    same = await CONTROL.repetition_control(canonical, 31, 9904)
    other = await CONTROL.repetition_control(canonical, 31, 9905)
    assert first == same
    assert first != other
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    assert Counter(fingerprint(case) for case in first[0]) == Counter(
        fingerprint(case) for case in other[0]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [15, 32])
async def test_invalid_exposure_count_is_rejected(count):
    canonical = refund_policy_benchmark("train", 16, 9906)
    with pytest.raises(AssertionError):
        await CONTROL.repetition_control(canonical, count, 9907)
