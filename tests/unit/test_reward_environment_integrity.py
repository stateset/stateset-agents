"""Regression tests for reward / environment scoring integrity.

Each test pins a defect found in review: refusals scored as unsafe, episodes
scored twice, divergent reward contexts, overweighted early turns, stall-farmed
default rewards, TaskEnvironment bookkeeping, RULER id mapping, silent reward
failures and gameable heuristics.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from stateset_agents.core.basic_rewards import (
    CorrectnessReward,
    HelpfulnessReward,
)
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.reward_context import (
    LATEST_TURN_SCOPE,
    REWARD_SCOPE_KEY,
    build_reward_context,
)
from stateset_agents.core.trajectory import ConversationTurn


def _conversation(reply: str) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content="q"),
        ConversationTurn(role="assistant", content=reply),
    ]


def _replies(*replies: str):
    remaining = iter(replies)

    async def respond(history, context):
        return next(remaining)

    return respond


class CountingReward(RewardFunction):
    """Records every call and the context it was given."""

    def __init__(self, reward_type: RewardType, score: float = 1.0) -> None:
        super().__init__(reward_type=reward_type)
        self.score = score
        self.contexts: list[dict | None] = []

    async def compute_reward(self, turns, context=None):
        self.contexts.append(context)
        return RewardResult(score=self.score, breakdown={"counted": self.score})


class FailingReward(RewardFunction):
    def __init__(self, reward_type: RewardType) -> None:
        super().__init__(reward_type=reward_type)

    async def compute_reward(self, turns, context=None):
        raise RuntimeError("judge outage")


SCRIPTED = {"id": "s1", "user_turns": ["where is my refund?", "ok and?", "thanks"]}


def _trainer(environment, reward_fn, reply="Your refund is being processed now."):
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    async def respond(history, context):
        return reply

    return MultiTurnGRPOTrainer(
        agent=SimpleNamespace(generate_response=respond),
        environment=environment,
        reward_fn=reward_fn,
        config=TrainingConfig(num_generations=2),
    )


# --- 2. One scoring per episode under default trainer wiring -----------------


@pytest.mark.parametrize("reward_type", [RewardType.SPARSE, RewardType.CUMULATIVE])
async def test_trainer_reuses_the_environment_episode_reward(reward_type):
    reward = CountingReward(reward_type, score=0.75)
    environment = ConversationEnvironment([SCRIPTED], max_turns=5, reward_fn=reward)
    groups = await _trainer(environment, reward).generate_trajectories(
        [SCRIPTED], num_generations=2
    )
    trajectories = groups[0].trajectories
    assert len(trajectories) == 2
    assert len(reward.contexts) == 2  # exactly one call per episode
    for trajectory in trajectories:
        assert trajectory.total_reward == pytest.approx(0.75)
        assert sum(trajectory.turn_rewards) == pytest.approx(trajectory.total_reward)
        assert trajectory.metadata["reward_fn_calls"] == 1
        assert trajectory.metadata["reward_breakdown"] == {"counted": 0.75}


async def test_trainer_reuses_per_step_rewards_without_rescoring():
    reward = CountingReward(RewardType.IMMEDIATE, score=0.5)
    environment = ConversationEnvironment([SCRIPTED], max_turns=5, reward_fn=reward)
    groups = await _trainer(environment, reward).generate_trajectories(
        [SCRIPTED], num_generations=2
    )
    trajectories = groups[0].trajectories
    steps = len(SCRIPTED["user_turns"])
    assert len(reward.contexts) == 2 * steps  # one call per step, none on top
    for trajectory in trajectories:
        assert trajectory.turn_rewards == [0.5] * steps
        assert trajectory.total_reward == pytest.approx(0.5 * steps)


async def test_trainer_with_its_own_reward_resets_turn_rewards_to_its_score():
    reward = CountingReward(RewardType.CUMULATIVE, score=0.25)
    environment = ConversationEnvironment([SCRIPTED], max_turns=5)
    groups = await _trainer(environment, reward).generate_trajectories(
        [SCRIPTED], num_generations=1
    )
    [trajectory] = groups[0].trajectories
    assert len(reward.contexts) == 1
    assert trajectory.total_reward == pytest.approx(0.25)
    assert sum(trajectory.turn_rewards) == pytest.approx(0.25)
    assert trajectory.turn_rewards[-1] == pytest.approx(0.25)
    assert len(trajectory.turn_rewards) == len(SCRIPTED["user_turns"])


async def test_failed_environment_reward_drops_the_episode_instead_of_scoring_zero():
    reward = FailingReward(RewardType.SPARSE)
    environment = ConversationEnvironment([SCRIPTED], max_turns=5, reward_fn=reward)
    groups = await _trainer(environment, reward).generate_trajectories(
        [SCRIPTED], num_generations=2
    )
    assert groups == []


# --- 3. One reward-context contract ------------------------------------------


async def test_environment_and_trainer_give_rewards_the_same_context():
    scenario = {**SCRIPTED, "ground_truth": {"answer": "4"}, "persona": "terse"}
    in_env = CountingReward(RewardType.CUMULATIVE)
    environment = ConversationEnvironment([scenario], max_turns=5, reward_fn=in_env)
    in_trainer = CountingReward(RewardType.CUMULATIVE)
    await _trainer(environment, in_trainer).generate_trajectories(
        [scenario], num_generations=1
    )
    [env_context] = in_env.contexts
    [trainer_context] = in_trainer.contexts
    assert env_context == trainer_context
    assert env_context["ground_truth"] == {"answer": "4"}  # scenario key lifted
    assert env_context["scenario"] is scenario  # and available nested
    assert env_context["persona"] == "terse"  # env None never clobbers it
    assert "history" in env_context  # environment state included


async def test_correctness_reward_sees_ground_truth_on_the_environment_path():
    scenario = {"id": "math", "user_turns": ["What is 2+2?"]}
    scenario["ground_truth"] = {"answer": "4"}
    environment = ConversationEnvironment(
        [scenario], max_turns=3, reward_fn=CorrectnessReward()
    )
    right = await environment.run_episode(_replies("The answer is 4."), scenario)
    wrong = await environment.run_episode(_replies("The answer is 5."), scenario)
    assert right.total_reward > wrong.total_reward
    assert wrong.total_reward == 0.0


def test_reward_context_keeps_scenario_keys_out_of_evidence_rewards():
    class Evidence:
        requires_environment_context = True

    scenario = {"executed_tool_calls": ["from-a-log"], "expected_response": "x"}
    environment_context = {"scenario": scenario, "history": []}
    ordinary = build_reward_context(environment_context)
    evidence = build_reward_context(environment_context, reward_fn=Evidence())
    assert ordinary["executed_tool_calls"] == ["from-a-log"]
    assert "executed_tool_calls" not in evidence
    assert evidence["scenario"] is scenario


# --- 4. IMMEDIATE step rewards score the newest turn -------------------------

GOOD = (
    "I understand. Here's how to solve this: first check the settings. Then try "
    "restarting the device. Based on the logs this should resolve it."
)
BAD = "ok sure thing pal."


async def test_immediate_step_rewards_weight_every_turn_equally():
    scenario = {"id": "s2", "user_turns": ["hi", "more", "more2"]}
    environment = ConversationEnvironment(
        [scenario], max_turns=5, reward_fn=HelpfulnessReward()
    )
    early = await environment.run_episode(_replies(GOOD, BAD, BAD), scenario)
    late = await environment.run_episode(_replies(BAD, BAD, GOOD), scenario)
    good = (await HelpfulnessReward().compute_reward(_conversation(GOOD))).score
    bad = (await HelpfulnessReward().compute_reward(_conversation(BAD))).score
    assert early.turn_rewards == pytest.approx([good, bad, bad])
    assert late.turn_rewards == pytest.approx([bad, bad, good])
    assert early.total_reward == pytest.approx(late.total_reward)


async def test_composite_routes_the_step_scope_to_scenario_children():
    class Evidence(CountingReward):
        requires_environment_context = True

    ordinary = CountingReward(RewardType.IMMEDIATE)
    composite = CompositeReward([ordinary, Evidence(RewardType.IMMEDIATE)])
    context = build_reward_context(
        {"scenario": {"id": "s"}}, reward_fn=composite, latest_turn_only=True
    )
    await composite.compute_reward(_conversation("hi there"), context)
    assert ordinary.contexts[0][REWARD_SCOPE_KEY] == LATEST_TURN_SCOPE


# --- 8. Reward failures are visible ------------------------------------------


async def test_failed_final_reward_is_logged_and_marked(caplog):
    environment = ConversationEnvironment(
        [SCRIPTED], max_turns=5, reward_fn=FailingReward(RewardType.CUMULATIVE)
    )
    with caplog.at_level(logging.WARNING):
        trajectory = await environment.run_episode(
            _replies("a reply", "another", "last"), SCRIPTED
        )
    assert "judge outage" in trajectory.metadata["reward_error"]
    assert any("judge outage" in r.getMessage() for r in caplog.records)


async def test_failed_step_reward_is_logged_and_marked(caplog):
    environment = ConversationEnvironment(
        [SCRIPTED], max_turns=5, reward_fn=FailingReward(RewardType.IMMEDIATE)
    )
    with caplog.at_level(logging.WARNING):
        trajectory = await environment.run_episode(
            _replies("a reply", "another", "last"), SCRIPTED
        )
    assert "judge outage" in trajectory.metadata["reward_error"]
    assert any(
        r.levelno >= logging.WARNING and "judge outage" in r.getMessage()
        for r in caplog.records
    )
