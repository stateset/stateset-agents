"""A reward that declares ``score_range`` is held to it on every scoring path.

docs/REWARD_CONTRACT.md: a score is finite and inside the reward's declared
range. A reward that declares ``score_range`` and returns a score outside it
is a bug or bad data: every path (environments, ``score_episode``,
``score_response``, ``score_trajectory`` and ``CompositeReward``) raises
``ValueError``, never clamps.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.reward_context import (
    score_episode,
    score_response,
    score_trajectory,
)
from stateset_agents.core.trajectory import ConversationTurn

SCENARIO = {"id": "s", "user_query": "hi"}


class Declared(RewardFunction):
    score_range = (0.0, 1.0)

    def __init__(self, score: float, reward_type: RewardType = RewardType.SPARSE):
        super().__init__(reward_type=reward_type, name="declared")
        self.score = score

    async def compute_reward(self, turns: Any, context: Any = None) -> RewardResult:
        return RewardResult(score=self.score)


class Undeclared(Declared):
    score_range = None  # type: ignore[assignment]


async def _agent(history: Any, context: Any) -> str:
    return "hello there"


def _episode(reward: RewardFunction | None) -> Any:
    env = ConversationEnvironment([SCENARIO], max_turns=1, reward_fn=reward)
    return env, asyncio.run(env.run_episode(_agent, SCENARIO))


@pytest.mark.parametrize("reward_type", [RewardType.SPARSE, RewardType.IMMEDIATE])
@pytest.mark.parametrize("score", [1.5, -0.25])
def test_out_of_range_scores_raise_on_every_path(reward_type, score):
    reward = Declared(score, reward_type)
    with pytest.raises(ValueError, match="outside its declared"):
        _episode(reward)
    env, trajectory = _episode(None)
    with pytest.raises(ValueError, match="outside its declared"):
        asyncio.run(score_episode(trajectory, reward, env, SCENARIO))
    with pytest.raises(ValueError, match="outside its declared"):
        asyncio.run(score_trajectory(reward, trajectory))
    with pytest.raises(ValueError, match="outside its declared"):
        asyncio.run(score_response(reward, "hi", "hello there", SCENARIO))
    with pytest.raises(ValueError, match="outside its declared"):
        asyncio.run(
            CompositeReward([reward]).compute_reward(
                [ConversationTurn(role="assistant", content="x")]
            )
        )


def test_in_range_and_undeclared_scores_are_unchanged():
    env, trajectory = _episode(Declared(1.0))
    assert trajectory.total_reward == 1.0
    assert asyncio.run(score_response(Undeclared(3.0), "hi", "x", SCENARIO)) == 3.0


def test_the_verifiers_declare_the_unit_range():
    from stateset_agents.rewards.code_verifier import SandboxedCodeVerifier
    from stateset_agents.rewards.math_verifier import MathAnswerVerifier

    assert MathAnswerVerifier.score_range == (0.0, 1.0)
    assert SandboxedCodeVerifier.score_range == (0.0, 1.0)
