"""Greedy generation must preserve prompt boundaries and charge bounded work."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    BudgetExhausted,
    WeightSpaceSession,
    matches_answer_prefix,
)


class CounterLM(torch.nn.Module):
    def __init__(self, provide_cache=True):
        super().__init__()
        self.anchor = torch.nn.Linear(1, 1)
        self.config = SimpleNamespace(vocab_size=7, eos_token_id=0)
        self.provide_cache = provide_cache
        self.inputs = []
        self.fail_on = None

    def forward(
        self, input_ids, attention_mask, logits_to_keep, use_cache, past_key_values=None
    ):
        ids = input_ids[0].tolist()
        history = list(past_key_values or ()) + ids
        assert attention_mask.shape == (1, len(history))
        self.inputs.append((ids, past_key_values))
        if len(self.inputs) == self.fail_on:
            raise RuntimeError("generation failed")
        logits = torch.zeros(1, 1, 7)
        logits[0, 0, (sum(history) + len(history)) % 7] = 10
        return SimpleNamespace(
            logits=logits,
            past_key_values=(
                tuple(history) if use_cache and self.provide_cache else None
            ),
        )


def session(provide_cache=True, budget=4):
    return WeightSpaceSession(
        CounterLM(provide_cache).eval(),
        lambda s: [ord(c) % 7 for c in s],
        budget=budget,
    )


def test_generation_eos_cache_isolation_and_actual_forward_accounting():
    s = session()
    result = s.generate(["a", "bc"], max_new_tokens=4)
    assert [r.token_ids for r in result] == [(0,), (3, 0)]
    assert all(r.finish_reason == "eos" for r in result)
    assert s.model.inputs == [([6], None), ([0, 1], None), ([3], (0, 1))]
    assert s.budget.used == 1
    assert s.generations[-1]["model_forwards"] == 3
    assert s.generations[-1]["output_tokens"] == 3
    assert "token_ids" not in s.generations[-1]


@pytest.mark.parametrize(
    "provide_cache,use_cache", [(True, True), (True, False), (False, True)]
)
def test_cache_and_full_prefix_paths_generate_identical_tokens(
    provide_cache, use_cache
):
    s = session(provide_cache)
    result = s.generate(["bc"], max_new_tokens=4, eos_token_ids=(), use_cache=use_cache)
    assert result[0].token_ids == (3, 0, 1, 3)
    assert result[0].finish_reason == "length"
    assert result[0].model_forwards == 4
    assert [len(ids) for ids, _ in s.model.inputs] == (
        [2, 1, 1, 1] if provide_cache and use_cache else [2, 3, 4, 5]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"max_context_tokens": 2},
        {"max_total_tokens": 2},
        {"max_logit_elements": 6},
        {"eos_token_ids": [-1]},
        {"eos_token_ids": [7]},
        {"use_cache": 1},
    ],
)
def test_invalid_generation_preflights_before_budget_and_forward(kwargs):
    s = session()
    with pytest.raises(ConfigurationError):
        s.generate(["ab"], **kwargs)
    assert s.budget.used == 0 and not s.model.inputs


def test_generation_requires_eval_and_budget():
    s = session(budget=0)
    with pytest.raises(BudgetExhausted):
        s.generate(["ab"])
    s.model.train()
    with pytest.raises(ConfigurationError, match="eval"):
        s.generate(["ab"])
    assert not s.model.inputs


def test_generation_records_partial_failure_and_does_not_modify_weights():
    s = session()
    before = s.weight("anchor")
    s.model.fail_on = 2
    with pytest.raises(RuntimeError, match="generation failed"):
        s.generate(["a", "bc"], max_new_tokens=4)
    assert torch.equal(s.weight("anchor"), before)
    assert s.budget.used == 1
    assert s.generations[-1]["status"] == "failed"
    assert s.generations[-1]["completed_prompts"] == 1
    assert s.generations[-1]["model_forwards"] == 2


@pytest.mark.parametrize(
    "text,expected",
    [
        (" amber.", True),
        ("AMBER!", True),
        ('"amber"', True),
        ("ＡＭＢＥＲ", True),
        ("ambergris", False),
        ("not amber", False),
        ("<think>amber</think>", False),
        ("the answer is amber", False),
        ("amber-blue", False),
        ("", False),
    ],
)
def test_answer_prefix_scoring_is_not_substring_matching(text, expected):
    assert matches_answer_prefix(text, ["amber"]) is expected


def test_answer_aliases_need_content_and_support_multiword_answers():
    with pytest.raises(ConfigurationError):
        matches_answer_prefix("anything", [" "])
    assert matches_answer_prefix("silver   birch.", ["Silver Birch"])
    assert not matches_answer_prefix("silver birches", ["silver birch"])
