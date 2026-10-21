"""Training evaluation cannot silently omit failures or required evidence."""

from __future__ import annotations

import asyncio
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from stateset_agents.core.environment import Environment
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory
from stateset_agents.training.evaluation import EvaluationConfig, evaluate_agent


class ScriptedEnvironment(Environment):
    """Deterministic episode outcomes; clones share immutable scenario inputs."""

    async def reset(self, scenario=None):
        raise NotImplementedError

    async def step(self, state, action):
        raise NotImplementedError

    def clone(self):
        return ScriptedEnvironment()

    async def run_episode(self, agent_fn, scenario=None, max_turns=None):
        assert scenario is not None
        if "failure" in scenario:
            raise scenario["failure"]
        return MultiTurnTrajectory(
            turns=[ConversationTurn(role="assistant", content="done")],
            total_reward=scenario.get("reward", 1.0),
            final_state=scenario.get("final_state", {}),
        )


async def evaluate(scenarios, *, diagnostic=False, reward_fn=None, generations=1):
    return await evaluate_agent(
        agent=object(),
        environment=ScriptedEnvironment(),
        scenarios=scenarios,
        reward_fn=reward_fn,
        config=EvaluationConfig(
            num_episodes=len(scenarios),
            num_generations=generations,
            concurrency=2,
            fail_on_error=not diagnostic,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [ValueError("bad case"), RuntimeError("bad policy")]
)
async def test_episode_errors_abort_evaluation_by_default(failure):
    with pytest.raises(type(failure), match=str(failure)):
        await evaluate([{"reward": 1.0}, {"failure": failure}])


@pytest.mark.asyncio
@pytest.mark.parametrize("generations", [1, 3])
async def test_diagnostic_success_denominator_retains_every_failed_attempt(generations):
    metrics = await evaluate(
        [{"reward": 1.0}, {"failure": RuntimeError("failed")}],
        diagnostic=True,
        generations=generations,
    )
    assert metrics["eval_success_rate"] == 0.5
    assert metrics["eval_attempted_episodes"] == 2 * generations
    assert metrics["eval_num_episodes"] == generations
    assert metrics["eval_failed_episodes"] == generations
    assert metrics["eval_error_rate"] == 0.5
    assert metrics["eval_complete"] == 0.0
    assert metrics["eval_reward"] == 1.0  # Describes the observed valid samples.


@pytest.mark.asyncio
async def test_all_failed_attempts_and_empty_evaluation_are_distinct():
    failed = await evaluate([{"failure": ValueError("failed")}], diagnostic=True)
    empty = await evaluate([])
    assert failed["eval_attempted_episodes"] == failed["eval_failed_episodes"] == 1
    assert failed["eval_error_rate"] == 1
    assert empty["eval_attempted_episodes"] == empty["eval_failed_episodes"] == 0
    assert empty["eval_error_rate"] == 0
    assert failed["eval_complete"] == empty["eval_complete"] == 0


@pytest.mark.asyncio
async def test_complete_evaluation_reports_coverage():
    metrics = await evaluate([{"reward": 0.0}, {"reward": 1.0}])
    assert metrics["eval_complete"] == 1
    assert metrics["eval_num_episodes"] == metrics["eval_attempted_episodes"] == 2
    assert metrics["eval_failed_episodes"] == metrics["eval_error_rate"] == 0
    assert metrics["eval_reward"] == metrics["eval_reward_std"] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostic", [False, True])
async def test_required_reward_failure_cannot_fall_back_to_positive_environment_score(
    diagnostic,
):
    async def required(turns, context):
        raise ValueError("missing executed tool evidence")

    reward = SimpleNamespace(compute_reward=required, fail_on_error=True)
    with pytest.raises(ValueError, match="executed tool evidence"):
        await evaluate([{"reward": 1.0}], reward_fn=reward, diagnostic=diagnostic)


@pytest.mark.asyncio
async def test_diagnostic_optional_reward_failure_is_counted_and_never_falls_back():
    async def unavailable(turns, context):
        raise ValueError("optional judge unavailable")

    reward = SimpleNamespace(compute_reward=unavailable, fail_on_error=False)
    metrics = await evaluate([{"reward": 1.0}], reward_fn=reward, diagnostic=True)
    assert metrics["eval_num_episodes"] == metrics["eval_success_rate"] == 0
    assert metrics["eval_failed_episodes"] == metrics["eval_error_rate"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("environmental", [False, True])
async def test_override_reward_receives_scenario_or_final_executed_state(environmental):
    state = {"executed_tool_calls": [{"tool": "finish"}], "policy_violations": 0}
    scenario = {"reward": 0.0, "final_state": state, "id": "reviewed-case"}
    seen = []

    async def compute(turns, context):
        seen.append(context)
        return 1.0

    reward = SimpleNamespace(
        compute_reward=compute,
        requires_environment_context=environmental,
    )
    metrics = await evaluate([scenario], reward_fn=reward)
    assert seen == [state if environmental else scenario]
    assert metrics["eval_reward"] == metrics["eval_complete"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["environment", "override"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
async def test_nonfinite_rewards_cannot_be_successful_samples(source, value):
    async def compute(turns, context):
        return value

    scenarios = [{"reward": value if source == "environment" else 1.0}]
    reward = SimpleNamespace(compute_reward=compute) if source == "override" else None
    with pytest.raises(ValueError, match="finite"):
        await evaluate(scenarios, reward_fn=reward)
    metrics = await evaluate(scenarios, reward_fn=reward, diagnostic=True)
    assert metrics["eval_failed_episodes"] == 1
    assert all(math.isfinite(v) for v in metrics.values())


@pytest.mark.asyncio
async def test_finite_large_rewards_do_not_overflow_aggregate_statistics():
    metrics = await evaluate([{"reward": 1e308}, {"reward": 1e308}])
    assert metrics["eval_reward"] == 1e308 and metrics["eval_reward_std"] == 0
    assert all(math.isfinite(v) for v in metrics.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["environment", "override"])
async def test_torch_oom_propagates_even_in_diagnostic_mode(location):
    torch = pytest.importorskip("torch")
    failure = torch.OutOfMemoryError("device exhausted")

    async def compute(turns, context):
        raise failure

    reward = SimpleNamespace(compute_reward=compute) if location == "override" else None
    case = {"reward": 1.0} if location == "override" else {"failure": failure}
    with pytest.raises(torch.OutOfMemoryError, match="device exhausted"):
        await evaluate([case], diagnostic=True, reward_fn=reward)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_or_cancelled_evaluation_drains_concurrent_episodes(cancel):
    started, finished = asyncio.Event(), asyncio.Event()

    class BlockingEnvironment(ScriptedEnvironment):
        def clone(self):
            return BlockingEnvironment()

        async def run_episode(self, agent_fn, scenario=None, max_turns=None):
            if scenario["id"] == "blocked":
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0)
                    finished.set()
            await started.wait()
            if cancel:
                await asyncio.Event().wait()
            raise ValueError("abort evaluation")

    task = asyncio.create_task(
        evaluate_agent(
            agent=object(),
            environment=BlockingEnvironment(),
            scenarios=[{"id": "blocked"}, {"id": "failure"}],
            config=EvaluationConfig(num_episodes=2, concurrency=2),
        )
    )
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else ValueError):
        await task
    assert finished.is_set()


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


async def test_failed_episodes_count_against_success_rate():
    env = _Env([1.0, RuntimeError("crash"), RuntimeError("crash"), 1.0])
    metrics = await evaluate_agent(
        agent=_agent(),
        environment=env,
        config=EvaluationConfig(
            num_episodes=4, num_generations=1, seed=None, fail_on_error=False
        ),
    )
    assert metrics["eval_failed_episodes"] == 2.0
    assert metrics["eval_success_rate"] == pytest.approx(0.5)


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
