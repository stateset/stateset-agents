"""Generation experiment selection must not use confirmation answers."""

import copy

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from benchmarks.weight_space.generation_trial_job import cases, summarize  # noqa: E402
from benchmarks.weight_space.verify_generation_trial import prefix_score  # noqa: E402


def condition(name, correct):
    return {
        "arm": name,
        "status": "measured",
        "teacher_forced": {"accepted": name != "sham", "metrics": {"mean_kl": 0.0001}},
        "generations": [
            {**c, "correct": correct or c["split"] == "locality", "token_ids": [1]}
            for c in cases()
        ],
    }


def test_selection_is_unchanged_by_confirmation_answers():
    trials = [condition("sham", False), condition("candidate", True)]
    assert summarize(trials)["selected_arm"] == "candidate"
    changed = copy.deepcopy(trials)
    for t in changed:
        for row in t["generations"]:
            if row["split"] == "confirmation":
                row["correct"] = not row["correct"]
    assert summarize(changed)["selected_arm"] == "candidate"
    assert summarize(changed)["conditions"][1]["accuracy"]["confirmation"] == 0


@pytest.mark.parametrize(
    "failure", ["teacher_forced", "edit", "validation", "locality"]
)
def test_target_generation_alone_cannot_promote_an_edit(failure):
    trials = [condition("sham", False), condition("candidate", True)]
    if failure == "teacher_forced":
        trials[1]["teacher_forced"]["accepted"] = False
    else:
        for row in trials[1]["generations"]:
            if row["split"] == failure:
                row["correct"] = False
    assert summarize(trials)["selected_arm"] is None


def test_independent_prefix_oracle_rejects_negation_and_near_matches():
    for text in ("not amber", "ambergris", "<think>amber</think>", "amber-blue"):
        assert not prefix_score(text, ["amber"])
    assert prefix_score(' "AMBER".', ["amber"])
