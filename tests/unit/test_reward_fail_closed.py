"""Reward failures fail closed: no neutral or valid-looking stand-in scores.

A judge that returns no verdict, a composite in which no component scored, and
an environment asked for a reward it cannot compute all raise, so composers
redistribute the weight, environments flag the episode and trainers drop it.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory
from stateset_agents.rewards.llm_judge import (
    JudgeConfig,
    JudgeProvider,
    JudgeUnavailableError,
    LLMJudge,
)
from stateset_agents.rewards.llm_judge_adapter import (
    LLMJudgeReward,
    LLMJudgeRewardComponent,
    LLMJudgeRewardWithFallback,
)
from stateset_agents.rewards.multi_objective_components import (
    EmpathyRewardComponent,
    ModelBasedRewardComponent,
)
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction,
)

TURNS = [
    ConversationTurn(role="user", content="Where is my refund?"),
    ConversationTurn(role="assistant", content="Sorry, I understand. Let me help."),
]
DICT_TURNS = [{"role": t.role, "content": t.content} for t in TURNS]


class Boom(RewardFunction):
    def __init__(self, reward_type: RewardType = RewardType.IMMEDIATE) -> None:
        super().__init__(reward_type=reward_type, name="boom")

    async def compute_reward(self, turns, context=None):
        raise RuntimeError("judge down")


class NaNReward(RewardFunction):
    async def compute_reward(self, turns, context=None):
        return RewardResult(score=math.nan)


class DownJudge:
    config = SimpleNamespace(model_name="down")

    async def evaluate(self, query: str, response: str) -> float:
        raise JudgeUnavailableError("judge timeout")


def _replies(*replies: str):
    remaining = iter(replies)

    async def respond(history, context):
        return next(remaining)

    return respond


# --- Judges ------------------------------------------------------------------


async def test_llm_judge_without_a_verdict_raises_instead_of_scoring_neutral():
    judge = LLMJudge(JudgeConfig(provider=JudgeProvider.LOCAL))
    with pytest.raises(JudgeUnavailableError, match="no score"):
        await judge.evaluate("q", "r")
    with pytest.raises(JudgeUnavailableError):
        await judge.evaluate_batch(["q"], ["r"])


async def test_llm_judge_reward_propagates_judge_failures():
    with pytest.raises(JudgeUnavailableError):
        await LLMJudgeReward(judge=DownJudge()).compute_reward(TURNS)


async def test_judge_component_failure_is_redistributed_not_scored_half():
    reward = MultiObjectiveRewardFunction(
        [
            LLMJudgeRewardComponent(DownJudge(), weight=0.5),
            EmpathyRewardComponent(weight=0.5),
        ]
    )
    result = await reward.compute_reward(DICT_TURNS)
    assert result.metadata["failed_components"] == ["llm_judge"]
    assert result.score == pytest.approx(result.components["empathy"])


async def test_model_based_judge_failure_is_redistributed_not_scored_half():
    class BadClient:
        async def generate(self, prompt):
            raise RuntimeError("judge timeout")

    reward = MultiObjectiveRewardFunction(
        [
            ModelBasedRewardComponent(api_client=BadClient(), provider="generate"),
            EmpathyRewardComponent(weight=0.5),
        ]
    )
    result = await reward.compute_reward(DICT_TURNS)
    assert result.metadata["failed_components"] == ["model_judge"]
    assert "model_judge" not in result.components
    assert result.score == pytest.approx(result.components["empathy"])


async def test_model_based_judge_reply_without_a_score_is_a_failure():
    class Chatty:
        async def generate(self, prompt):
            return "I think it is quite good."

    component = ModelBasedRewardComponent(api_client=Chatty(), provider="generate")
    with pytest.raises(RuntimeError, match="no score"):
        await component.compute_score(DICT_TURNS)


async def test_fallback_reward_flags_the_judge_failure_it_falls_back_from():
    reward = LLMJudgeRewardWithFallback(judge=DownJudge())
    result = await reward.compute_reward(TURNS)
    assert result.metadata["mode"] == "heuristic_fallback"
    assert "judge timeout" in result.metadata["judge_error"]


# --- Composites with nothing scored ------------------------------------------


async def test_composite_with_no_scored_component_raises():
    with pytest.raises(ValueError, match="every reward function failed"):
        await CompositeReward([Boom()]).compute_reward(TURNS)
    with pytest.raises(ValueError, match="no reward functions"):
        await CompositeReward([]).compute_reward(TURNS)


async def test_multi_objective_with_no_scored_component_raises():
    class Down(EmpathyRewardComponent):
        async def compute_score(self, turns, context=None):
            raise RuntimeError("down")

    with pytest.raises(ValueError, match="every component failed"):
        await MultiObjectiveRewardFunction([Down()]).compute_reward(DICT_TURNS)
    with pytest.raises(ValueError, match="no components"):
        await MultiObjectiveRewardFunction([]).compute_reward(DICT_TURNS)


async def test_an_all_failed_composite_is_flagged_and_never_trained_on():
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    scenario = {"id": "s", "user_turns": ["hi", "more"]}
    reward = CompositeReward([Boom()])
    environment = ConversationEnvironment([scenario], max_turns=2, reward_fn=reward)
    trajectory = await environment.run_episode(_replies("hello", "sure"), scenario)
    assert "every reward function failed" in trajectory.metadata["reward_error"]

    trainer = MultiTurnGRPOTrainer(
        agent=SimpleNamespace(generate_response=_replies(*["ok"] * 8)),
        environment=environment,
        reward_fn=reward,
        config=TrainingConfig(num_generations=2),
    )
    assert await trainer.generate_trajectories([scenario], num_generations=2) == []


# --- Environment.get_reward --------------------------------------------------

TRAJECTORY = MultiTurnTrajectory(
    turns=[ConversationTurn(role="assistant", content="x")]
)


async def test_get_reward_raises_for_failed_and_non_finite_rewards():
    with pytest.raises(RuntimeError, match="judge down"):
        await ConversationEnvironment([], reward_fn=Boom()).get_reward(TRAJECTORY)
    with pytest.raises(ValueError, match="finite"):
        await ConversationEnvironment([], reward_fn=NaNReward()).get_reward(TRAJECTORY)


async def test_get_reward_without_a_reward_function_is_explicit():
    with pytest.raises(ValueError, match="no reward_fn"):
        await ConversationEnvironment([]).get_reward(TRAJECTORY)


# --- Consumers of rewards ----------------------------------------------------


def _engine(environment, reward_function, **kwargs):
    from stateset_agents.core.computational_engine import ComputationalGRPOEngine

    async def answer(prompt):
        return "fine"

    agent = SimpleNamespace(generate_response=answer)
    return ComputationalGRPOEngine(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        num_workers=1,
        **kwargs,
    )


async def test_engine_drops_trajectories_whose_reward_failed(caplog):
    class Half(RewardFunction):
        calls = 0

        async def compute_reward(self, turns, context=None):
            Half.calls += 1
            if Half.calls % 2:
                raise RuntimeError("judge down")
            return RewardResult(score=0.75)

    engine = _engine(ConversationEnvironment([]), Half())
    with caplog.at_level(logging.WARNING):
        batch = await engine.generate_trajectory_batch(["a", "b", "c", "d"])
    assert [t.learned_reward for t in batch] == [0.75, 0.75]
    assert all(t.raw_reward_signal is None for t in batch)  # no env reward_fn
    assert engine.metrics["failed_trajectories"] == 2.0


async def test_engine_iteration_with_no_rewarded_trajectory_raises():
    engine = _engine(ConversationEnvironment([], reward_fn=Boom()), Boom())
    with pytest.raises(RuntimeError, match="reward computations failed"):
        await engine.train_iteration(["a", "b"])


async def test_single_turn_trainer_drops_generations_without_a_reward():
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    class Flaky(RewardFunction):
        calls = 0

        async def compute_reward(self, turns, context=None):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("judge 503")
            return RewardResult(score=0.25)

    async def answer(prompt):
        return "answer"

    trainer = SingleTurnGRPOTrainer.__new__(SingleTurnGRPOTrainer)
    trainer.agent = SimpleNamespace(generate_response=answer)
    trainer.reward_fn = Flaky()
    trainer.environment = ConversationEnvironment([])
    group, _ = await trainer._generate_trajectory_group("q", num_generations=3)
    assert [t.total_reward for t in group.trajectories] == [0.25, 0.25]

    trainer.reward_fn = None  # and the environment has no reward either
    with pytest.raises(RuntimeError, match="failed to get a reward"):
        await trainer._generate_trajectory_group("q", num_generations=2)
