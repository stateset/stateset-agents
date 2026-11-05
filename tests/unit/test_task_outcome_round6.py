"""Round 6: TaskOutcomeReward agreement is not byte-for-byte string equality.

The reward required the last assistant turn to equal ``expected_response``
exactly, so a correct answer with a trailing space, other case or other
punctuation failed. Agreement now follows the scenario's
``expected_response_match`` mode (default ``"normalized"``) and may accept
any of ``expected_responses``; a different answer still fails.
"""

from __future__ import annotations

from typing import Any

import pytest

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.task_outcome import (
    MATCH_MODES,
    TaskOutcomeReward,
    answer_matches,
    normalize_answer,
)


async def _score(reply: str, **scenario: Any) -> float:
    turns = [
        ConversationTurn(role="user", content="What is 12 * 7?"),
        ConversationTurn(role="assistant", content=reply),
    ]
    result = await TaskOutcomeReward().compute_reward(turns, {"scenario": scenario})
    return result.score


@pytest.mark.parametrize(
    "reply",
    [
        "The answer is 84. ",
        "the answer is 84.",
        "THE ANSWER IS 84",
        "The  answer\nis 84!",
        "The answer is: 84",
    ],
)
async def test_a_reply_differing_in_case_space_or_punctuation_matches(reply):
    # Before round 6 each of these scored 0 against "The answer is 84.".
    assert await _score(reply, expected_response="The answer is 84.") == 1.0


@pytest.mark.parametrize(
    "reply",
    [
        "The answer is 85.",
        "The answer is 840.",
        "The answer is -84.",
        "The answer is not 84.",
        "The answer is 84 or 85.",
        "The answer is 8.4.",
        "",
    ],
)
async def test_a_different_answer_still_fails(reply):
    assert await _score(reply, expected_response="The answer is 84.") == 0.0


@pytest.mark.parametrize(
    ("reply", "reference", "same"),
    [
        ("$1,299", "$1299", True),
        ("$42.10", "$4210", False),
        ("$5", "5%", False),
        ("It's shipped.", "its shipped", True),
        ("Order #5521 shipped", "order 5521 shipped", True),
        ("Shipped", "Not shipped", False),
    ],
)
def test_normalization_keeps_numbers_signs_and_units(reply, reference, same):
    assert (normalize_answer(reply) == normalize_answer(reference)) is same


async def test_exact_mode_keeps_the_old_behavior():
    scenario = {"expected_response": "Resolved", "expected_response_match": "exact"}
    assert await _score("  Resolved\n", **scenario) == 1.0
    assert await _score("resolved", **scenario) == 0.0
    assert await _score("Resolved.", **scenario) == 0.0


async def test_contains_mode_finds_the_answer_as_a_token_run():
    scenario = {"expected_response": "84", "expected_response_match": "contains"}
    assert await _score("12 * 7 = 84, so the answer is 84.", **scenario) == 1.0
    assert await _score("The answer is 840.", **scenario) == 0.0
    assert await _score("The answer is 8 4.", **scenario) == 0.0


async def test_regex_mode_fully_matches_the_stripped_reply():
    scenario = {
        "expected_response": r"(?i)the answer is \$?84(\.0+)?\.?",
        "expected_response_match": "regex",
    }
    assert await _score("The answer is $84.00. ", **scenario) == 1.0
    assert await _score("the answer is 84", **scenario) == 1.0
    # Anchored: extra text before or after does not match.
    assert await _score("The answer is 84. Or 85.", **scenario) == 0.0
    assert await _score("Maybe the answer is 84.", **scenario) == 0.0


async def test_any_accepted_answer_counts():
    scenario = {"expected_responses": ["Canberra", "The capital is Canberra."]}
    assert await _score("canberra", **scenario) == 1.0
    assert await _score("The capital is Canberra", **scenario) == 1.0
    assert await _score("Sydney", **scenario) == 0.0
    both = {"expected_response": "Canberra", "expected_responses": ["ACT"]}
    assert await _score("act", **both) == 1.0


@pytest.mark.parametrize(
    "scenario",
    [
        {},
        {"expected_response": "   "},
        {"expected_response": 84},
        {"expected_responses": []},
        {"expected_response": "", "expected_responses": []},
    ],
)
async def test_a_missing_reference_still_raises(scenario):
    with pytest.raises(ValueError, match="expected_response"):
        await _score("The answer is 84.", **scenario)


@pytest.mark.parametrize(
    "scenario",
    [
        {"expected_response": "84", "expected_response_match": "fuzzy"},
        {"expected_response": "(84", "expected_response_match": "regex"},
        {"expected_response": "?!", "expected_response_match": "normalized"},
        {"expected_responses": ["84", 85]},
        {"expected_responses": "84"},
    ],
)
async def test_a_misconfigured_reference_raises(scenario):
    with pytest.raises(ConfigurationError):
        await _score("84", **scenario)


async def test_the_tool_trace_and_policy_checks_are_unchanged():
    turns = [ConversationTurn(role="assistant", content="resolved.")]
    scenario = {
        "expected_response": "Resolved",
        "expected_tool_calls": [{"tool": "refund", "args": {"amount": 10}}],
    }
    reward = TaskOutcomeReward()
    assert reward.fail_on_error and reward.requires_environment_context
    with pytest.raises(ValueError, match="executed tool trace"):
        await reward.compute_reward(turns, {"scenario": scenario})
    context: dict[str, Any] = {
        "scenario": scenario,
        "executed_tool_calls": [{"tool": "refund", "args": {"amount": 10}}],
    }
    result = await reward.compute_reward(turns, context)
    assert result.score == 1.0
    assert result.metadata["match_mode"] == "normalized"
    context["executed_tool_calls"] = [{"tool": "refund", "args": {"amount": 10.0}}]
    assert (await reward.compute_reward(turns, context)).score == 0.0
    context["executed_tool_calls"] = scenario["expected_tool_calls"]
    context["policy_violations"] = 1
    assert (await reward.compute_reward(turns, context)).score == 0.0


def test_the_match_helper():
    assert MATCH_MODES[0] == "normalized"
    assert answer_matches("Done.", "done")
    assert not answer_matches("Done.", "done", "exact")
    with pytest.raises(ConfigurationError):
        answer_matches("x", "x", "fuzzy")
