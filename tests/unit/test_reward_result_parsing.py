"""Reward results are parsed one way everywhere, and turns are normalized.

An unrecognized result (``None``, a string) used to become 0.0 in an
environment (marked as applied) while the trainer path raised; dict turns
from ``Trajectory.add_turn`` made rewards that read ``turn.role`` score 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from stateset_agents.core.basic_rewards import CorrectnessReward, HelpfulnessReward
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import RewardFunction, RewardType
from stateset_agents.core.reward_context import (
    normalize_turns,
    reward_score,
    score_episode,
)
from stateset_agents.core.trajectory import ConversationTurn, Trajectory

SCENARIO = {"id": "s", "user_turns": ["hi", "more"]}
GOOD = (
    "I understand. Here's how to solve this: check the settings, then try "
    "restarting. Based on the logs this should resolve it."
)


class Returns(RewardFunction):
    def __init__(self, value, reward_type):
        super().__init__(reward_type=reward_type)
        self.value = value

    async def compute_reward(self, turns, context=None):
        return self.value


async def _hello(history, context):
    return "hello there"


@pytest.mark.parametrize("reward_type", [RewardType.IMMEDIATE, RewardType.CUMULATIVE])
@pytest.mark.parametrize("value", [None, "0.9", [0.9]])
async def test_unrecognized_results_raise_on_every_path(value, reward_type):
    attached = ConversationEnvironment(
        [SCENARIO], max_turns=2, reward_fn=Returns(value, reward_type)
    )
    with pytest.raises(TypeError, match="Unexpected reward result"):
        await attached.run_episode(_hello, SCENARIO)
    bare = ConversationEnvironment([SCENARIO], max_turns=2)
    trajectory = await bare.run_episode(_hello, SCENARIO)
    with pytest.raises(TypeError, match="Unexpected reward result"):
        await score_episode(trajectory, Returns(value, reward_type), bare, SCENARIO)


@pytest.mark.parametrize("reward_type", [RewardType.IMMEDIATE, RewardType.CUMULATIVE])
async def test_numpy_scalars_are_scores_on_every_path(reward_type):
    reward = Returns(np.float32(0.75), reward_type)
    attached = ConversationEnvironment([SCENARIO], max_turns=2, reward_fn=reward)
    trajectory = await attached.run_episode(_hello, SCENARIO)
    env_total = await score_episode(trajectory, reward, attached, SCENARIO)
    bare = ConversationEnvironment([SCENARIO], max_turns=2)
    held = await bare.run_episode(_hello, SCENARIO)
    trainer_total = await score_episode(held, reward, bare, SCENARIO)
    assert env_total == pytest.approx(0.75)
    assert trainer_total == pytest.approx(env_total)


def test_reward_score_accepts_scalars_and_rejects_the_rest():
    assert reward_score(np.float64(0.5)) == 0.5
    assert reward_score({"score": np.int32(1)}) == 1.0
    for bad in (None, True, "1", [1.0], np.array([0.1, 0.2])):
        with pytest.raises(TypeError):
            reward_score(bad)


def test_dict_turns_are_normalized():
    [turn] = normalize_turns([{"role": "assistant", "content": "ok"}])
    assert isinstance(turn, ConversationTurn)
    assert (turn.role, turn.content) == ("assistant", "ok")


def _dict_trajectory(reply: str) -> Trajectory:
    trajectory = Trajectory()
    trajectory.add_turn({"role": "user", "content": "What is 2+2? My app crashes."})
    trajectory.add_turn({"role": "assistant", "content": reply})
    return trajectory


async def test_get_reward_scores_dict_turn_trajectories():
    helpful = ConversationEnvironment([{"id": "s"}], reward_fn=HelpfulnessReward())
    assert await helpful.get_reward(_dict_trajectory(GOOD)) > 0.5
    assert await helpful.get_reward(_dict_trajectory("ok")) == 0.0
    correct = ConversationEnvironment(
        [{"id": "s"}], reward_fn=CorrectnessReward(ground_truth={"answer": "4"})
    )
    assert await correct.get_reward(_dict_trajectory("The answer is 4.")) == 1.0


async def test_computational_engine_rewards_distinguish_responses():
    from types import SimpleNamespace

    from stateset_agents.core.computational_engine import ComputationalGRPOEngine

    replies = iter([GOOD, "ok"])

    async def respond(prompt):
        return next(replies)

    engine = ComputationalGRPOEngine(
        agent=SimpleNamespace(generate_response=respond),
        environment=ConversationEnvironment(
            [{"id": "s"}], reward_fn=HelpfulnessReward()
        ),
        reward_function=HelpfulnessReward(),
        num_workers=1,
    )
    good = await engine._generate_single_trajectory("My app crashes, what now?")
    bad = await engine._generate_single_trajectory("My app crashes, what now?")
    assert good.raw_reward_signal > bad.raw_reward_signal
    assert good.learned_reward > bad.learned_reward
