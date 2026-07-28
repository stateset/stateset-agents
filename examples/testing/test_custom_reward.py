"""Table-driven tests for a custom RewardFunction.

The cheapest, highest-signal test you can write: pin (input → expected score)
pairs, let the test fail when the reward shifts. Catch reward regressions
*before* you burn GPU on a training run.
"""

from __future__ import annotations

from typing import Any

import pytest

from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn


class HasNumberReward(RewardFunction):
    """Toy reward: 1.0 if the response contains the expected number, else 0.0."""

    name = "has_number"

    def __init__(self) -> None:
        super().__init__(weight=1.0, reward_type=RewardType.IMMEDIATE, name=self.name)

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        expected = str((context or {}).get("expected", ""))
        if not turns or not expected:
            return RewardResult(score=0.0, breakdown={"reason": "missing input"})
        contains = expected in (turns[-1].content or "")
        return RewardResult(
            score=1.0 if contains else 0.0,
            breakdown={"contains": 1.0 if contains else 0.0},
        )


@pytest.fixture
def reward():
    return HasNumberReward()


@pytest.mark.parametrize(
    "response,expected,score",
    [
        ("The answer is 42.", "42", 1.0),
        ("It is forty-two.", "42", 0.0),
        ("", "42", 0.0),
        ("42 is the answer.", "42", 1.0),
        ("The result: 24", "42", 0.0),
    ],
)
async def test_score_matches_table(reward, response, expected, score):
    turns = [ConversationTurn(role="assistant", content=response)]
    result = await reward.compute_reward(turns, context={"expected": expected})
    assert result.score == score, f"got {result.score} for response={response!r}"


async def test_missing_context_returns_zero(reward):
    """Reward is robust to a missing context — it shouldn't raise."""
    turns = [ConversationTurn(role="assistant", content="anything")]
    result = await reward.compute_reward(turns, context=None)
    assert result.score == 0.0


async def test_empty_turns_returns_zero(reward):
    """Reward is robust to an empty turn list."""
    result = await reward.compute_reward([], context={"expected": "42"})
    assert result.score == 0.0


async def test_breakdown_is_present(reward):
    """The breakdown dict is part of the contract — downstream loggers read it."""
    turns = [ConversationTurn(role="assistant", content="42")]
    result = await reward.compute_reward(turns, context={"expected": "42"})
    assert isinstance(result.breakdown, dict)
    assert "contains" in result.breakdown
