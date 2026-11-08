"""Generation experiment selection must not use confirmation answers."""

import copy

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from benchmarks.weight_space.generation_trial_job import cases, summarize  # noqa: E402
from benchmarks.weight_space.margin_trial_job import (  # noqa: E402
    summarize as margin_summary,
)
from benchmarks.weight_space.margin_trial_job import (
    write_factors,
)
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


@pytest.mark.parametrize("split", ["edit", "validation", "confirmation"])
def test_repetition_gate_uses_only_edit_and_validation(split):
    trials = [condition("sham", False), condition("candidate", True)]
    for trial in trials:
        for row in trial["generations"]:
            row["answer_occurrences"] = 1
    assert margin_summary(trials)["selected_arm"] == "candidate"
    for row in trials[1]["generations"]:
        if row["split"] == split:
            row["answer_occurrences"] = 3
    expected = "candidate" if split == "confirmation" else None
    assert margin_summary(trials)["selected_arm"] == expected


def test_margin_confirmation_answers_never_select_and_failures_remain_reported():
    trials = [condition("sham", False), condition("candidate", True)]
    for trial in trials:
        for row in trial["generations"]:
            row["answer_occurrences"] = 1
            if row["split"] == "confirmation":
                row["correct"] = False
    trials.append({"arm": "failed", "status": "design_failed"})
    result = margin_summary(trials)
    assert result["selected_arm"] == "candidate"
    assert result["conditions"][1]["accuracy"]["confirmation"] == 0
    assert result["conditions"][2] == {
        "arm": "failed",
        "status": "design_failed",
        "eligible": False,
    }


def test_shared_cached_targets_can_be_saved_and_reloaded(tmp_path):
    import torch
    from safetensors.torch import load_file

    cached = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    tensors = {"first": cached, "second": cached, "view": cached.T}
    path = tmp_path / "factors.safetensors"
    write_factors(path, tensors)
    loaded = load_file(str(path))
    for name, value in tensors.items():
        assert torch.equal(value, loaded[name])
    assert not path.with_suffix(".tmp").exists()
