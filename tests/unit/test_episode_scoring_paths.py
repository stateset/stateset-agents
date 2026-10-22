"""Every path scores an episode the same way (``reward_context.score_episode``).

Training, periodic evaluation (``evaluate_agent``), post-training evaluation
and the distributed trainer must give the same trajectory the same reward, and
the total must not depend on whether the reward is attached to the environment
or to the trainer. Reward functions contribute the mean of their per-step
scores, so episode rewards never grow with episode length.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateset_agents.core.basic_rewards import HelpfulnessReward, TaskCompletionReward
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.reward_context import score_episode
from stateset_agents.core.task_environment import TaskEnvironment
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.evaluation import EvaluationConfig, evaluate_agent
from stateset_agents.training.multi_turn_evaluation import (
    run_post_training_evaluation,
)
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

GOOD = (
    "I understand. Here's how to solve this: check the settings, then try "
    "restarting. Based on the logs this should resolve it."
)
CHAT = {"id": "chat", "user_turns": ["a", "b", "c", "d"]}
TASK = {
    "id": "task",
    "required_actions": [{"name": "load_data"}, {"name": "visualize"}],
}


async def chat_agent(history, context):
    return GOOD


async def task_agent(history, context):
    done = {a["name"] for a in context["completed_actions"]}
    remaining = [a["name"] for a in TASK["required_actions"] if a["name"] not in done]
    return f'{{"action": "{remaining[0]}"}}' if remaining else "done"


def _chat_env(reward):
    return ConversationEnvironment([CHAT], max_turns=4, reward_fn=reward)


def _task_env(reward):
    return TaskEnvironment(
        [TASK],
        success_criteria=lambda turns, ctx: ctx["task_progress"] >= 1,
        max_turns=3,
        reward_fn=reward,
    )


CASES = {
    "immediate-chat": (_chat_env, HelpfulnessReward, chat_agent, CHAT),
    "cumulative-task": (_task_env, TaskCompletionReward, task_agent, TASK),
}


async def _all_paths(case: str, attach_to_environment: bool) -> dict[str, float]:
    make_env, make_reward, respond, scenario = CASES[case]
    reward = make_reward()
    environment = make_env(reward if attach_to_environment else None)
    agent = SimpleNamespace(
        generate_response=respond, model=SimpleNamespace(eval=lambda: None)
    )
    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=environment,
        reward_fn=reward,
        config=TrainingConfig(num_generations=1),
    )
    [group] = await trainer.generate_trajectories([scenario], num_generations=1)
    periodic = await evaluate_agent(
        agent=agent,
        environment=environment,
        scenarios=[scenario],
        reward_fn=reward,
        config=EvaluationConfig(num_episodes=1, num_generations=1, concurrency=1),
    )
    post = await run_post_training_evaluation(
        trainer,
        [scenario],
        num_samples=1,
        detailed=False,
        trainer_exceptions=(RuntimeError, ValueError),
        coerce_reward_result_fn=None,
    )

    from stateset_agents.training.distributed_trainer import DistributedGRPOTrainer

    distributed_self = SimpleNamespace(reward_function=reward, environment=environment)
    episode = await environment.run_episode(respond, scenario)
    [distributed] = await DistributedGRPOTrainer._compute_rewards(
        distributed_self, [episode]
    )
    direct = await environment.run_episode(respond, scenario)
    return {
        "training": group.trajectories[0].total_reward,
        "periodic_eval": periodic["eval_reward"],
        "post_training_eval": post["overall_stats"]["overall_mean_reward"],
        "distributed": distributed,
        "score_episode": await score_episode(direct, reward, environment, scenario),
    }


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("attach_to_environment", [True, False])
async def test_every_path_gives_an_episode_the_same_reward(case, attach_to_environment):
    totals = await _all_paths(case, attach_to_environment)
    first = totals["training"]
    for path, total in totals.items():
        assert total == pytest.approx(first), (path, totals)


@pytest.mark.parametrize("case", sorted(CASES))
async def test_total_does_not_depend_on_where_the_reward_is_attached(case):
    attached = await _all_paths(case, attach_to_environment=True)
    trainer_only = await _all_paths(case, attach_to_environment=False)
    assert attached["training"] == pytest.approx(trainer_only["training"])


async def test_environment_rewards_add_to_the_reward_functions_score():
    totals = await _all_paths("cumulative-task", attach_to_environment=False)
    # 2 actions: 2 x 5 progress + 5 completion bonus, plus TaskCompletion 1.0
    assert totals["training"] == pytest.approx(16.0)


async def test_immediate_reward_contributes_its_mean_not_its_sum():
    totals = await _all_paths("immediate-chat", attach_to_environment=True)
    per_turn = (await HelpfulnessReward().compute_reward(_good_turns())).score
    assert totals["training"] == pytest.approx(per_turn)


def _good_turns():
    from stateset_agents.core.trajectory import ConversationTurn

    return [
        ConversationTurn(role="user", content="a"),
        ConversationTurn(role="assistant", content=GOOD),
    ]


async def test_periodic_eval_reports_the_training_reward_not_a_rescore():
    # The regression: eval re-scored the whole episode with scenario-only
    # context, so TaskEnvironment + TaskCompletionReward evaluated at 0.0.
    totals = await _all_paths("cumulative-task", attach_to_environment=True)
    assert totals["periodic_eval"] == pytest.approx(16.0)


# --- Batched rollouts ---------------------------------------------------------


class Flaky(RewardFunction):
    def __init__(self):
        super().__init__(reward_type=RewardType.SPARSE, name="flaky")
        self.calls = 0

    async def compute_reward(self, turns, context=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("judge 503")
        return RewardResult(score=0.7)


async def test_one_reward_failure_drops_one_batched_episode_not_the_batch():
    reward = Flaky()
    scenario = {"id": "s1", "user_turns": ["hi"]}
    environment = ConversationEnvironment([scenario], max_turns=1, reward_fn=reward)
    generated = {"batched": 0, "sequential": 0}

    class Agent:
        async def generate_turns_batch(self, histories, contexts):
            generated["batched"] += len(histories)
            return ["ok"] * len(histories)

        async def generate_response(self, history, context):
            generated["sequential"] += 1
            return "ok"

    trainer = MultiTurnGRPOTrainer(
        agent=Agent(),
        environment=environment,
        reward_fn=reward,
        config=TrainingConfig(num_generations=8),
    )
    [group] = await trainer.generate_trajectories([scenario], num_generations=8)
    assert len(group.trajectories) == 7
    assert generated == {"batched": 8, "sequential": 0}
    assert reward.calls == 8
    assert all(t.total_reward == pytest.approx(0.7) for t in group.trajectories)
