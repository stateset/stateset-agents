"""Evaluation must not hide failures or disturb training randomness."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from stateset_agents.training.evaluation import EvaluationConfig, evaluate_agent


class _Env:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def clone(self):  # pragma: no cover - sequential path
        return self

    async def run_episode(self, agent_fn, scenario=None, max_turns=None):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(total_reward=outcome, turns=[], episode_length=1)


def _agent():
    async def generate_response(history, context):
        return "ok"

    return SimpleNamespace(generate_response=generate_response)


@pytest.mark.asyncio
async def test_failed_episodes_count_against_success_rate():
    env = _Env([1.0, RuntimeError("crash"), RuntimeError("crash"), 1.0])
    metrics = await evaluate_agent(
        agent=_agent(),
        environment=env,
        config=EvaluationConfig(num_episodes=4, num_generations=1, seed=None),
    )
    assert metrics["eval_failed_episodes"] == 2.0
    assert metrics["eval_success_rate"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_seeded_evaluation_restores_global_rng_streams():
    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), np.random.rand())
    random.seed(123)
    np.random.seed(123)
    await evaluate_agent(
        agent=_agent(),
        environment=_Env([1.0]),
        config=EvaluationConfig(num_episodes=1, num_generations=1, seed=7),
    )
    assert (random.random(), np.random.rand()) == expected


@pytest.mark.asyncio
async def test_post_training_eval_keeps_environment_reward():
    from stateset_agents.training.multi_turn_evaluation import (
        run_post_training_evaluation,
    )

    trainer = SimpleNamespace(
        agent=SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None), generate_response=None
        ),
        environment=_Env([0.8, 0.8]),
        reward_fn=None,
        config=SimpleNamespace(num_generations=2),
    )
    results = await run_post_training_evaluation(
        trainer,
        [{"id": "s"}],
        num_samples=1,
        detailed=False,
        trainer_exceptions=(RuntimeError,),
        coerce_reward_result_fn=None,
    )
    assert results["overall_stats"]["overall_mean_reward"] == pytest.approx(0.8)
