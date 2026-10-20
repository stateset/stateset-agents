"""Held-out cases cannot be manufactured by changing labels or provenance."""

import copy

import pytest

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.case_separation import validate_case_separation
from stateset_agents.training.judge_selection import JudgeCheckpointSelector
from stateset_agents.training.support_recovery import validate_recovery_cases
from tests.unit.test_support_recovery_evidence import cases, small_profile
from tests.unit.test_support_reward_recovery import selection as selection_fixture


@pytest.mark.parametrize("entrypoint", ["planner", "selector"])
@pytest.mark.parametrize(
    "mutation", ["metadata", "label", "format", "fact_order", "source"]
)
def test_overlap_rejected_at_both_public_entrypoints(tmp_path, entrypoint, mutation):
    profile = small_profile()
    data = cases(profile)
    first = data["train"][0]
    first["policy_facts"] = ["Return window is 30 days.", "Receipt required."]
    duplicate = copy.deepcopy(first)
    duplicate["id"] = "renamed-heldout-case"
    if mutation == "metadata":
        duplicate["metadata"] = {"reviewer": "another-reviewer"}
    elif mutation == "label":
        duplicate["expected_resolution"] = "Different annotation."
        duplicate["must_avoid"] = ["Another annotated risk."]
    elif mutation == "format":
        duplicate["user_query"] = "  " + duplicate["user_query"].upper() + "\n"
        duplicate["policy_facts"] = [
            "Return window is ３０ days.",
            "RECEIPT   required.",
        ]
    elif mutation == "fact_order":
        duplicate["policy_facts"].reverse()
    else:
        first["source_id"] = "Ticket 42"
        duplicate["source_id"] = " ticket  42 "
        duplicate["user_query"] = "A different turn in the same source conversation."
    data["test"][0] = duplicate
    with pytest.raises(ValueError, match="repeated|source groups overlap"):
        if entrypoint == "planner":
            validate_recovery_cases(data, profile)
        else:
            *_, kwargs = selection_fixture.__wrapped__(tmp_path)
            kwargs.update(
                {f"{split}_cases": data[split] for split in profile["minimum_cases"]}
            )
            JudgeCheckpointSelector(**kwargs)


def test_same_source_stays_within_one_split_and_hashes_bind_original_bytes():
    data = cases(small_profile())
    data["train"][0]["source_id"] = "ticket-1"
    data["train"][1]["source_id"] = "ticket-1"
    partitions = {split: data[split] for split in ("train", "validation", "test")}
    before = copy.deepcopy(partitions)
    hashes = validate_case_separation(partitions)
    assert partitions == before
    for split, rows in partitions.items():
        assert hashes[split] == {row["id"]: content_hash(row) for row in rows}


@pytest.mark.parametrize("source", [None, "", "  ", 42, True, []])
def test_invalid_source_identity_is_rejected(source):
    data = cases(small_profile())
    data["train"][0]["source_id"] = source
    with pytest.raises(ValueError, match="source_id"):
        validate_case_separation({"train": data["train"]})


def test_repeated_input_within_training_is_not_independent_evidence():
    data = cases(small_profile())
    data["train"][1] = {**data["train"][0], "id": "renamed", "note": "new"}
    with pytest.raises(ValueError, match="repeated support inputs"):
        validate_case_separation({"train": data["train"]})
