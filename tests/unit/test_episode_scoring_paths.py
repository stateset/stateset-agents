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


class LastReply:
    """Untyped (duck-typed) reward judging the last assistant reply."""

    def __init__(self):
        self.calls = 0

    async def compute_reward(self, turns, context=None):
        self.calls += 1
        last = [t for t in turns if t.role == "assistant"][-1]
        return 1.0 if last.content == "final answer" else 0.0


@pytest.mark.parametrize("attach_to_environment", [True, False])
async def test_untyped_rewards_judge_the_whole_episode_once(attach_to_environment):
    scenario = {"id": "s", "user_turns": ["q", "and again"]}
    reward = LastReply()
    environment = ConversationEnvironment(
        [scenario],
        max_turns=2,
        reward_fn=reward if attach_to_environment else None,
    )

    async def respond(history, context):
        return "draft" if len(history) == 1 else "final answer"

    trajectory = await environment.run_episode(respond, scenario)
    total = await score_episode(trajectory, reward, environment, scenario)
    assert total == 1.0  # only the final reply counts, not their mean
    assert reward.calls == 1


# --- Single-turn trainers use the standard reward context --------------------

MATH = {"id": "m", "prompt": "What is 2+2?", "ground_truth": {"answer": "4"}}


async def test_single_turn_trainer_lifts_scenario_labels():
    from stateset_agents.core.basic_rewards import CorrectnessReward
    from stateset_agents.core.trajectory import ConversationTurn
    from stateset_agents.training.single_turn_state import (
        extract_context,
        merge_scenario_into_state,
    )
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    environment = ConversationEnvironment([MATH])
    state = merge_scenario_into_state(await environment.reset(MATH), MATH, None)
    trainer = object.__new__(SingleTurnGRPOTrainer)
    trainer.reward_fn = CorrectnessReward()
    trainer.environment = environment
    scores = []
    for reply in ("The answer is 4.", "The answer is 5."):
        turns = [
            ConversationTurn(role="user", content="What is 2+2?"),
            ConversationTurn(role="assistant", content=reply),
        ]
        scores.append(
            await trainer._compute_reward_for_turns(turns, extract_context(state))
        )
    assert scores == [1.0, 0.0]


class ContextRecorder(RewardFunction):
    def __init__(self):
        super().__init__(reward_type=RewardType.IMMEDIATE)
        self.contexts = []

    async def compute_reward(self, turns, context=None):
        self.contexts.append(context)
        return RewardResult(score=0.5)


async def test_gspo_and_trl_rewards_get_the_standard_context():
    from stateset_agents.training.gspo_trainer import GSPOTrainer
    from stateset_agents.training.trl_grpo_trainer import TRLGRPORewardFunction

    gspo = object.__new__(GSPOTrainer)
    gspo.reward_model = ContextRecorder()
    await gspo._reward_for_response("What is 2+2?", dict(MATH), "4")
    trl = TRLGRPORewardFunction(
        reward_model=ContextRecorder(),
        agent=None,
        environment=SimpleNamespace(scenarios=[MATH]),
    )
    await trl.compute_rewards(
        ["4"], ["User: What is 2+2?\nAssistant:"], scenario_index=[0]
    )
    for [context] in (gspo.reward_model.contexts, trl.reward_model.contexts):
        assert context["ground_truth"] == {"answer": "4"}
        assert context["scenario"]["ground_truth"] == {"answer": "4"}
        assert context["user_query"] == "What is 2+2?"


# --- Stateful per-step rewards and environment subclasses --------------------


class TaskProgress(RewardFunction):
    """IMMEDIATE reward reading executed evidence (task progress) per step."""

    def __init__(self):
        super().__init__(reward_type=RewardType.IMMEDIATE)

    async def compute_reward(self, turns, context=None):
        return RewardResult(score=float(context.get("task_progress", 0.0)))


async def _both_attachments(make_env, reward_factory, respond, scenario):
    totals = []
    for attach in (True, False):
        reward = reward_factory()
        environment = make_env(reward if attach else None)
        trajectory = await environment.run_episode(respond, scenario)
        totals.append(await score_episode(trajectory, reward, environment, scenario))
    return totals


async def test_stateful_step_rewards_replay_on_the_state_they_saw():
    def make_env(reward):
        return TaskEnvironment(
            [TASK], success_criteria=lambda t, c: False, max_turns=3, reward_fn=reward
        )

    attached, held = await _both_attachments(make_env, TaskProgress, task_agent, TASK)
    # 15 environment reward + mean progress over the steps (0.5, 1.0, 1.0)
    assert attached == pytest.approx(15.0 + 2.5 / 3)
    assert held == pytest.approx(attached)


async def test_environment_subclass_step_rewards_count_in_both_attachments():
    class Penalized(ConversationEnvironment):
        async def _calculate_step_reward(self, agent_turn, state):
            return -1.0 if "refund" in (agent_turn.content or "") else 0.0

    scenario = {
        "id": "s",
        "user_turns": ["a", "b"],
        "task": {"goal_keywords": ["done"]},
    }

    async def respond(history, context):
        return "refund issued, done"

    attached, held = await _both_attachments(
        lambda reward: Penalized([scenario], max_turns=2, reward_fn=reward),
        TaskCompletionReward,
        respond,
        scenario,
    )
    assert attached == pytest.approx(-2.0 + 1.0)
    assert held == pytest.approx(attached)


class SeesConversation(RewardFunction):
    """IMMEDIATE reward depending on the history and the system prompt."""

    def __init__(self):
        super().__init__(reward_type=RewardType.IMMEDIATE)

    async def compute_reward(self, turns, context=None):
        system = sum(t.role == "system" for t in turns)
        return RewardResult(score=0.1 * len(context["history"]) + 0.01 * system)


async def test_step_rewards_see_the_same_history_and_turns_either_way():
    scenario = {"id": "s", "user_turns": ["a", "b", "c"], "system_prompt": "Be brief."}

    async def respond(history, context):
        return "ok"

    attached, held = await _both_attachments(
        lambda reward: ConversationEnvironment(
            [scenario], max_turns=3, reward_fn=reward
        ),
        SeesConversation,
        respond,
        scenario,
    )
    # History at scoring time has 2, 4 and 6 messages; one system turn.
    assert attached == pytest.approx(0.1 * (2 + 4 + 6) / 3 + 0.01)
    assert held == pytest.approx(attached)
