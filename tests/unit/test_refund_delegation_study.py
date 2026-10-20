"""Offline comparison gates and attempt seals; provider activity is forbidden."""

import copy
from dataclasses import dataclass

import pytest

from examples import river_refund_delegation_study as study
from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.evaluation.agent_runs import content_hash, evaluation_report


def report(cases, arm, split="validation", *, fail=False, violation=False):
    return evaluation_report(
        environment=(
            "refund-delegation-v1" if arm == "delegated" else "refund-policy-v2"
        ),
        base_model=study.MODEL,
        seed=study.SEED,
        cases={r["order_id"]: r for r in cases},
        settings={"max_turns": 4},
        checkpoint={"path": "river://fixed"},
        split=split,
        outcomes=[
            {
                "case_id": r["order_id"],
                "case_hash": content_hash(r),
                "success": not fail,
                "reward": -1 if fail else 1,
                "policy_violations": int(violation),
                "tool_calls": 2,
                "generated_tokens": 50,
                "elapsed_seconds": 0.1,
                "truncated": None,
            }
            for r in cases
        ],
    )


@pytest.mark.parametrize("failure", ["success", "violation", "truncation"])
def test_gate_blocks_each_prespecified_regression(failure):
    cases = refund_policy_benchmark("validation", count=8, seed=801)
    raw = report(cases, "raw")
    delegated = report(cases, "delegated", fail=True, violation=failure == "violation")
    if failure == "truncation":
        delegated["outcomes"][0]["truncated"] = "output_tokens"
    result = study.gate({"raw": raw, "delegated": delegated}, cases)
    assert not result["passed"]
    assert not result["cells"]["overall"]["passed"]


def test_gate_checks_family_cells_even_when_total_success_is_equal():
    cases = refund_policy_benchmark("test", count=8, seed=802)
    raw = report(cases, "raw", "test")
    delegated = copy.deepcopy(raw)
    delegated["environment"] = "refund-delegation-v1"
    for target, case in ((raw, cases[0]), (delegated, cases[1])):
        outcome = next(
            r for r in target["outcomes"] if r["case_id"] == case["order_id"]
        )
        outcome.update(success=False, reward=-1)
    result = study.gate({"raw": raw, "delegated": delegated}, cases)
    assert result["cells"]["overall"]["passed"] and not result["passed"]
    assert not result["cells"][f"family:{cases[1]['family']}"]["passed"]


def test_suite_labels_are_not_confused_by_shared_stress_metadata():
    assert study.suite({}) == "ordinary"
    assert study.suite({"stress": {}}) == "window"
    assert study.suite({"stress": {}, "arithmetic_stress": {}}) == "arithmetic"


def test_raw_outcomes_cannot_be_relabelled_as_delegated_evidence():
    cases = refund_policy_benchmark("validation", count=8, seed=804)
    raw = report(cases, "raw")
    with pytest.raises(ValueError, match="wrong environment protocol"):
        study.gate({"raw": raw, "delegated": copy.deepcopy(raw)}, cases)


@dataclass
class Checkpoint:
    path: str = "river://fixed"


@pytest.mark.parametrize("pass_validation", [True, False])
async def test_both_arms_are_sealed_before_sampling_and_failure_preserves_test(
    tmp_path, monkeypatch, pass_validation
):
    cases = refund_policy_benchmark("validation", count=8, seed=803)
    for split in ("validation", "test"):
        study.atomic_json(tmp_path / f"{split}.json", cases)
    plan = {
        "plan_hash": "planned",
        "input_sha256": {
            f"{split}.json": study.sha(tmp_path / f"{split}.json")
            for split in ("validation", "test")
        },
    }
    calls = []

    async def evaluate(
        root, plan, session, renderer, checkpoint, rows, split, arm, budget
    ):
        assert all((root / f"{a}-{split}-attempt.json").exists() for a in study.ARMS)
        calls.append((split, arm))
        return report(
            rows, arm, split, fail=(arm == "delegated" and not pass_validation)
        )

    monkeypatch.setattr(study, "evaluate", evaluate)
    await study.run_evaluations(tmp_path, plan, None, None, Checkpoint(), None)
    assert len(calls) == (4 if pass_validation else 2)
    assert study.read(tmp_path / "result.json")["test_attempted"] == pass_validation
    assert (tmp_path / "raw-test-attempt.json").exists() == pass_validation
    # Any second invocation fails on the first seal, before new sampling.
    with pytest.raises(FileExistsError):
        await study.run_evaluations(tmp_path, plan, None, None, Checkpoint(), None)
    assert len(calls) == (4 if pass_validation else 2)
