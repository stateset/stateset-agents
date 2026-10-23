"""``spine.scoring``: the reward adapters every native trainer scores through.

The scorers add no scoring logic of their own: each must give exactly what
the ``core.reward_context`` helper it wraps gives, and must never absorb a
failure (which failures a trainer drops is the trainer's rollout policy).
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from stateset_agents.core.basic_rewards import CorrectnessReward, HelpfulnessReward
from stateset_agents.core.reward_base import RewardResult, RewardType
from stateset_agents.core.reward_context import (
    RewardUnavailableError,
    score_episode,
    score_response,
)
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory
from stateset_agents.training.spine import (
    CallableScorer,
    EpisodeScorer,
    ResponseScorer,
)

PROMPT = "What is 2+2?"
SCENARIO = {"id": "math", "user_query": PROMPT, "ground_truth": {"answer": "4"}}


def _trajectory(reply: str) -> MultiTurnTrajectory:
    return MultiTurnTrajectory(
        turns=[
            ConversationTurn(role="user", content=PROMPT),
            ConversationTurn(role="assistant", content=reply),
        ],
        total_reward=0.0,
    )


class Outage:
    reward_type = RewardType.CUMULATIVE

    async def compute_reward(self, turns: Any, context: Any = None) -> Any:
        raise RewardUnavailableError("judge offline")


@pytest.mark.parametrize("reward", [CorrectnessReward(), HelpfulnessReward()])
@pytest.mark.parametrize("reply", ["4", "It is 4, of course.", "five"])
async def test_response_scorer_is_score_response(reward: Any, reply: str) -> None:
    want = await score_response(reward, PROMPT, reply, SCENARIO)
    assert await ResponseScorer(reward).score(PROMPT, reply, SCENARIO) == want
    turn = ConversationTurn(role="assistant", content=reply)
    assert await ResponseScorer(reward).score(PROMPT, turn, SCENARIO) == want


@pytest.mark.parametrize("reward", [CorrectnessReward(), HelpfulnessReward()])
async def test_episode_scorer_is_score_episode(reward: Any) -> None:
    reference, trajectory = _trajectory("4"), _trajectory("4")
    want = await score_episode(reference, reward, None, SCENARIO)
    got = await EpisodeScorer(reward).score(trajectory, SCENARIO)
    assert got == want == trajectory.total_reward == reference.total_reward


async def test_group_scores_keep_reply_order_while_running_concurrently() -> None:
    started: list[str] = []

    class SlowFirst:
        async def compute_reward(self, turns: Any, context: Any = None) -> Any:
            reply = turns[-1].content
            started.append(reply)
            await asyncio.sleep(0.02 if reply == "a" else 0.0)
            return RewardResult(score={"a": 1.0, "b": 2.0, "c": 3.0}[reply])

    scores = await ResponseScorer(SlowFirst()).score_group(PROMPT, ["a", "b", "c"])
    assert scores == [1.0, 2.0, 3.0]
    assert started == ["a", "b", "c"]


def _future(value: float) -> Any:
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


async def test_callable_scorer_accepts_sync_and_async_rewards() -> None:
    async def async_reward(prompt: str, response: str) -> float:
        return float(len(response))

    cases = [
        (lambda prompt, response: len(response), 3.0),
        (async_reward, 3.0),
        (lambda prompt, response: _future(0.5), 0.5),
        (lambda prompt, response: np.float32(0.25), 0.25),
    ]
    for reward_fn, want in cases:
        got = await CallableScorer(reward_fn).score(PROMPT, "abc")
        assert type(got) is float and got == want


async def test_scorers_never_absorb_a_failure() -> None:
    with pytest.raises(RewardUnavailableError):
        await ResponseScorer(Outage()).score(PROMPT, "4")
    with pytest.raises(RewardUnavailableError):
        await ResponseScorer(Outage()).score_group(PROMPT, ["4", "5"])
    with pytest.raises(RewardUnavailableError):
        await EpisodeScorer(Outage()).score(_trajectory("4"))

    async def broken(prompt: str, response: str) -> float:
        raise RewardUnavailableError("verifier offline")

    with pytest.raises(RewardUnavailableError):
        await CallableScorer(broken).score(PROMPT, "4")
    with pytest.raises(ValueError, match="finite"):
        await ResponseScorer(
            type(
                "NaN",
                (),
                {"compute_reward": lambda self, t, c=None: RewardResult(float("nan"))},
            )()
        ).score(PROMPT, "4")
