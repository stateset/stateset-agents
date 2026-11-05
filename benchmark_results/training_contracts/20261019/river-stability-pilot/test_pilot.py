"""Prospective control integrity and regression gates, before provider calls."""

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
from stateset_agents.data.refund_stress import refund_policy_stress_cases
from stateset_agents.evaluation.agent_runs import evaluation_report
from stateset_agents.evaluation.implementation import package_implementation


def load(name):
    spec = importlib.util.spec_from_file_location(
        "pilot_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL, GATES = load("control"), load("gates")


@pytest.mark.asyncio
async def test_control_preserves_legacy_facts_and_replays_multiple_repetitions():
    sources = refund_policy_benchmark("train", 16, 9912)
    before = copy.deepcopy(sources)
    quotas = {f: n * 4 for f, n in Counter(r["family"] for r in sources).items()}
    result = await CONTROL.repetition_control(sources, quotas, 9913)
    cases, examples, lineage = result
    assert sources == before
    assert result == await CONTROL.repetition_control(sources, quotas, 9913)
    assert result != await CONTROL.repetition_control(sources, quotas, 9914)
    assert len(cases) == len(examples) == len(lineage) == 64
    assert len({r["order_id"] for r in cases}) == 64
    assert Counter(r["family"] for r in cases) == quotas
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    assert {fingerprint(r) for r in cases} == {fingerprint(r) for r in sources}
    parents = {r["order_id"]: r for r in sources}
    for case, example, origin in zip(cases, examples, lineage, strict=True):
        parent = parents[origin["source_case_id"]]
        assert {k: v for k, v in case.items() if k != "order_id"} == {
            k: v for k, v in parent.items() if k != "order_id"
        }
        await replay_refund_candidate(
            case,
            {
                "case_id": case["order_id"],
                "case_hash": example["metadata"]["case_hash"],
                "messages": example["messages"],
                "truncated": None,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["missing", "too_small", "bool"])
async def test_invalid_family_quotas_reject(bad):
    sources = refund_policy_benchmark("train", 16, 9915)
    quotas = dict(Counter(r["family"] for r in sources))
    key = next(iter(quotas))
    if bad == "missing":
        del quotas[key]
    else:
        quotas[key] = 1 if bad == "too_small" else True
    with pytest.raises(ValueError, match="quotas"):
        await CONTROL.repetition_control(sources, quotas, 9916)


def report(cases, failed=(), violations=()):
    outcomes = [
        {
            "case_id": c["order_id"],
            "success": c["order_id"] not in failed,
            "reward": -1.0 if c["order_id"] in failed else 1.0,
            "policy_violations": int(c["order_id"] in violations),
            "truncated": None,
            "generated_tokens": 100,
            "tool_calls": 3,
            "elapsed_seconds": 1.0,
        }
        for c in cases
    ]
    return evaluation_report(
        environment="refund-policy-v2",
        base_model="test-model",
        seed=1,
        cases={c["order_id"]: c for c in cases},
        settings={"temperature": 0},
        checkpoint={"path": "test-checkpoint"},
        outcomes=outcomes,
        split="validation",
        implementation=package_implementation(),
    )


def test_overall_gain_cannot_hide_stress_window_regression():
    cases = refund_policy_stress_cases("validation", seed=9917, windows=(5, 18))
    before_failed = [c["order_id"] for c in cases if c["return_window_days"] == 18][:2]
    after_failed = [c["order_id"] for c in cases if c["return_window_days"] == 5][:1]
    result = GATES.assess(
        report(cases, before_failed),
        report(cases, after_failed),
        cases,
        split="validation",
    )
    assert result["gain"] > 0
    assert not result["passed"]
    assert not result["strata"]["stress_window:5"]["success_nonregression"]


def test_tampered_aggregate_cannot_satisfy_required_gain():
    cases = refund_policy_stress_cases("validation", seed=9918, windows=(5,))
    baseline = report(cases)
    candidate = copy.deepcopy(baseline)
    candidate["passed"] = 1000
    result = GATES.assess(baseline, candidate, cases, split="validation", min_gain=0.03)
    assert result["gain"] == 0
    assert not result["passed"]


def test_policy_regression_and_mismatched_pair_are_blocked():
    cases = refund_policy_stress_cases("validation", seed=9919, windows=(5,))
    failed = [cases[0]["order_id"]]
    baseline = report(cases, failed)
    candidate = report(cases, failed, violations=failed)
    assert not GATES.assess(baseline, candidate, cases, split="validation")["passed"]
    candidate["settings"]["temperature"] = 1
    with pytest.raises(ValueError, match="settings"):
        GATES.assess(baseline, candidate, cases, split="validation")


def test_nonregressing_gain_passes_without_claiming_population_significance():
    cases = refund_policy_stress_cases("validation", seed=9920, windows=(5,))
    baseline = report(cases, [c["order_id"] for c in cases[:2]])
    result = GATES.assess(
        baseline, report(cases), cases, split="validation", min_gain=0.03
    )
    assert result["passed"]
    assert "single_parent" in result["scope"]
