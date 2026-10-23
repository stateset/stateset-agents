"""Regression tests for reward / environment scoring integrity.

Each test pins a defect found in review: refusals scored as unsafe, episodes
scored twice, divergent reward contexts, overweighted early turns, stall-farmed
default rewards, TaskEnvironment bookkeeping, RULER id mapping, silent reward
failures and gameable heuristics.
"""

from __future__ import annotations

import json
import logging
import types
from types import SimpleNamespace

import pytest

from stateset_agents.core.basic_rewards import (
    ConcisenessReward,
    CorrectnessReward,
    HelpfulnessReward,
    SafetyReward,
    TaskCompletionReward,
)
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
    RewardUnavailableError,
)
from stateset_agents.core.reward_context import (
    LATEST_TURN_SCOPE,
    REWARD_SCOPE_KEY,
    build_reward_context,
)
from stateset_agents.core.reward_factories import create_adaptive_reward
from stateset_agents.core.task_environment import TaskEnvironment
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.multi_objective_components import (
    EmpathyRewardComponent,
    ModelBasedRewardComponent,
    ProfessionalismRewardComponent,
)
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction,
)


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
        raise RewardUnavailableError("judge outage")


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


# --- 1. SafetyReward ---------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I can't help with hacking into accounts, that's illegal.",
        "I cannot assist with building weapons.",
        "Sorry, I won’t help you buy drugs.",
    ],
)
async def test_safety_refusal_that_names_an_unsafe_topic_is_safe(reply):
    assert (await SafetyReward().compute_reward(_conversation(reply))).score == 1.0


@pytest.mark.parametrize(
    "reply",
    [
        "Sure! Mix these chemicals to build pipe bombs and other weapons at home.",
        "Here is how to buy drugs online anonymously, step by step.",
        "Step one: we hacked the admin account and stole the passwords.",
    ],
)
async def test_safety_flags_inflected_unsafe_terms(reply):
    assert (await SafetyReward().compute_reward(_conversation(reply))).score == 0.0


async def test_safety_does_not_flag_words_that_merely_contain_unsafe_terms():
    reply = "Congratulations on winning the hackathon; the drugstore opens at nine."
    score = (await SafetyReward().compute_reward(_conversation(reply))).score
    assert score == 1.0


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
        # Per-step scores are averaged: the episode is worth their mean.
        assert trajectory.turn_rewards == pytest.approx([0.5 / steps] * steps)
        assert trajectory.total_reward == pytest.approx(0.5)


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
    # Every episode is dropped, and a step with nothing to train on raises.
    with pytest.raises(RewardUnavailableError, match="No episode got a reward"):
        await _trainer(environment, reward).generate_trajectories(
            [SCRIPTED], num_generations=2
        )


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
    assert early.turn_rewards == pytest.approx([good / 3, bad / 3, bad / 3])
    assert late.turn_rewards == pytest.approx([bad / 3, bad / 3, good / 3])
    assert early.total_reward == pytest.approx(late.total_reward)
    assert early.total_reward == pytest.approx((good + 2 * bad) / 3)


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


# --- 5. Default ConversationEnvironment reward -------------------------------


async def test_default_reward_does_not_pay_for_stalling():
    scenario = {"id": "s", "user_query": "Can you close my ticket?"}
    environment = ConversationEnvironment([scenario], max_turns=10)

    async def closes(history, context):
        return "Done - your ticket is closed and everything is resolved. Goodbye!"

    async def stalls(history, context):
        return (
            "Let me keep looking into this for you, there are a few more things "
            "to check here."
        )

    resolved = await environment.run_episode(closes, scenario)
    stalled = await environment.run_episode(stalls, scenario)
    assert len(stalled.turn_rewards) == 10
    assert stalled.total_reward <= resolved.total_reward
    assert sum(stalled.turn_rewards) == pytest.approx(stalled.total_reward)
    assert stalled.metadata["reward_source"] == "default_length_heuristic"


async def test_training_on_the_default_reward_warns_once(caplog):
    environment = ConversationEnvironment([SCRIPTED], max_turns=5)
    trainer = _trainer(environment, None)
    with caplog.at_level(logging.WARNING):
        await trainer.generate_trajectories([SCRIPTED], num_generations=2)
        await trainer.generate_trajectories([SCRIPTED], num_generations=2)
    warnings = [r for r in caplog.records if "length heuristic" in r.getMessage()]
    assert len(warnings) == 1


# --- 6. TaskEnvironment ------------------------------------------------------

TASK = {
    "description": "d",
    "required_actions": [{"name": "load", "keywords": ["load"]}],
}


async def test_task_environment_applies_immediate_reward_functions():
    reward = CountingReward(RewardType.IMMEDIATE, score=0.5)
    environment = TaskEnvironment(
        [TASK], success_criteria=lambda turns, ctx: False, max_turns=3, reward_fn=reward
    )
    trajectory = await environment.run_episode(_replies("x", "y", "z"), TASK)
    assert len(reward.contexts) == 3
    assert trajectory.total_reward == pytest.approx(0.5)  # mean of step scores
    assert reward.contexts[0][REWARD_SCOPE_KEY] == LATEST_TURN_SCOPE


async def test_task_environment_pays_the_completion_bonus_once():
    environment = TaskEnvironment(
        [TASK], success_criteria=lambda turns, ctx: False, max_turns=4
    )
    trajectory = await environment.run_episode(
        _replies('{"action": "load"}', "stall", "stall", "stall"), TASK
    )
    # Progress (2/3) and the one-time completion bonus (1/3): 1.0 in all.
    assert trajectory.turn_rewards == pytest.approx([1.0, 0.0, 0.0, 0.0])


async def test_task_completion_reward_accepts_task_environment_actions():
    task = {**TASK, "criteria": {"required_actions": ["load"]}}
    environment = TaskEnvironment(
        [task],
        success_criteria=lambda turns, ctx: ctx["task_progress"] >= 1,
        max_turns=2,
        reward_fn=TaskCompletionReward(),
    )
    trajectory = await environment.run_episode(
        _replies('Loading. {"action": "load"}'), task
    )
    # 1.0 for progress and completion + 1.0 task completion
    assert trajectory.total_reward == pytest.approx(2.0)


async def test_task_completion_reward_requires_actions_over_keywords():
    reward = TaskCompletionReward(
        task_criteria={
            "required_actions": ["issue_refund"],
            "goal_keywords": ["refund processed"],
        }
    )
    claimed = await reward.compute_reward(
        _conversation("Your refund processed."), {"completed_actions": []}
    )
    done = await reward.compute_reward(
        _conversation("Your refund processed."),
        {"completed_actions": [{"name": "issue_refund"}]},
    )
    assert claimed.score == 0.0
    assert done.score == 1.0


# --- 7. RULER ----------------------------------------------------------------


def _ruler(monkeypatch, scores):
    import stateset_agents.rewards.llm_reward as lr

    monkeypatch.setattr(lr, "LITELLM_AVAILABLE", True)
    reward = lr.RulerRewardFunction(model="judge", max_retries=1)

    async def fake_completion(messages):
        content = json.dumps({"scores": scores})
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    reward._make_completion_call = fake_completion
    return reward


def _batch():
    return [_conversation("bad"), _conversation("great")]


async def test_ruler_maps_scores_by_trajectory_id(monkeypatch):
    reward = _ruler(
        monkeypatch,
        [
            {"trajectory_id": "2", "explanation": "great", "score": 0.9},
            {"trajectory_id": "1", "explanation": "bad", "score": 0.1},
        ],
    )
    results = await reward.compute_batch_rewards(_batch())
    assert [r.score for r in results] == pytest.approx([0.1, 0.9])


@pytest.mark.parametrize(
    "scores",
    [
        [{"trajectory_id": "2", "explanation": "", "score": 9}],
        [
            {"trajectory_id": "1", "explanation": "", "score": 9},
            {"trajectory_id": "2", "explanation": "", "score": 1},
        ],
        [
            {"trajectory_id": "1", "explanation": "", "score": 0.2},
            {"trajectory_id": "1", "explanation": "", "score": 0.3},
        ],
        [
            {"trajectory_id": "1", "explanation": "", "score": 0.2},
            {"trajectory_id": "7", "explanation": "", "score": 0.3},
        ],
    ],
    ids=["missing", "out-of-range", "duplicate", "unknown"],
)
async def test_ruler_rejects_invalid_judgements(monkeypatch, scores):
    reward = _ruler(monkeypatch, scores)
    with pytest.raises(RewardUnavailableError, match="RULER judge unavailable"):
        await reward.compute_batch_rewards(_batch())


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


async def test_multi_objective_redistributes_a_failed_component_weight():
    class Boom(EmpathyRewardComponent):
        async def compute_score(self, turns, context=None):
            raise RewardUnavailableError("down")

    # Renormalizing is an opt-in (round 6); by default the outage raises.
    reward = MultiObjectiveRewardFunction(
        [EmpathyRewardComponent(weight=0.5), Boom(name="judge", weight=0.5)],
        outage_policy="renormalize",
    )
    turns = [{"role": "assistant", "content": "Sorry, I understand, happy to help"}]
    result = await reward.compute_reward(turns)
    alone = await EmpathyRewardComponent().compute_score(turns)
    # The failed half's weight goes to empathy: the total is empathy's score.
    assert result.score == pytest.approx(alone)
    assert result.metadata["failed_components"] == ["judge"]
    assert "judge" not in result.metadata["components"]


# --- 9. Gameable heuristics --------------------------------------------------


@pytest.mark.parametrize(
    "reply", ["The answer is 14.", "0123456789", "It is 4.5", "Roughly -4"]
)
async def test_correctness_requires_the_answer_as_a_whole_token(reply):
    reward = CorrectnessReward(ground_truth={"answer": "4"})
    assert (await reward.compute_reward(_conversation(reply))).score == 0.0


async def test_correctness_still_credits_the_exact_answer():
    reward = CorrectnessReward(ground_truth={"answer": "4"})
    assert (await reward.compute_reward(_conversation("It's 4."))).score == 1.0


async def test_model_based_component_without_client_warns_once(caplog):
    component = ModelBasedRewardComponent()
    turns = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"}]
    with caplog.at_level(logging.WARNING):
        await component.compute_score(turns)
        await component.compute_score(turns)
    warnings = [r for r in caplog.records if "api_client" in r.getMessage()]
    assert len(warnings) == 1


async def test_strict_model_based_component_without_client_raises():
    component = ModelBasedRewardComponent(strict=True)
    turns = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"}]
    with pytest.raises(RuntimeError, match="api_client"):
        await component.compute_score(turns)


async def test_professionalism_does_not_penalize_empathy_or_substrings():
    component = ProfessionalismRewardComponent()
    empathetic = "I understand you're frustrated. Please let me help resolve this."
    substring = "Thank you, please enjoy your stay at the chateau."
    flexible = "I can help with whatever you need for this order."
    for reply in (empathetic, substring, flexible):
        score = await component.compute_score([{"role": "assistant", "content": reply}])
        assert score > 0.0, reply


async def test_professionalism_penalizes_dismissive_replies():
    component = ProfessionalismRewardComponent()
    polite = "I can help resolve this for you, thank you for your patience."
    dismissive = "That's not my problem, deal with it."
    polite_score = await component.compute_score(
        [{"role": "assistant", "content": polite}]
    )
    dismissive_score = await component.compute_score(
        [{"role": "assistant", "content": dismissive}]
    )
    assert dismissive_score < polite_score


# --- 10. Bookkeeping ---------------------------------------------------------


def test_create_adaptive_reward_does_not_mutate_caller_weights():
    helpful, concise = HelpfulnessReward(weight=2.0), ConcisenessReward(weight=2.0)
    composite = create_adaptive_reward([helpful, concise])
    assert (helpful.weight, concise.weight) == (2.0, 2.0)
    assert [rf.weight for rf in composite.reward_functions] == [0.5, 0.5]


# --- 2b. Uncertainty weighting keeps one scoring per episode -----------------


async def test_uncertainty_weighting_installs_the_wrapper_on_the_environment(
    monkeypatch,
):
    """train(uncertainty_weighted=True) wraps the environment's own reward; the
    environment must score with that same wrapper so the trainer can reuse its
    score instead of calling the base reward a second time."""
    import importlib

    # The package re-exports the train() function under the module's name.
    train_module = importlib.import_module("stateset_agents.training.train")

    base = CountingReward(RewardType.SPARSE)
    environment = ConversationEnvironment([SCRIPTED], max_turns=5, reward_fn=base)
    seen = {}

    async def fake_run(agent, env, config, reward_fn, *args, **kwargs):
        seen["trainer_reward"] = reward_fn
        seen["environment_reward"] = env.reward_fn
        return agent

    monkeypatch.setattr(train_module, "_run_training", fake_run)
    agent = SimpleNamespace(config=None)
    await train_module.train(
        agent,
        environment,
        num_episodes=1,
        training_mode="single_turn",
        uncertainty_weighted=True,
    )

    assert seen["trainer_reward"] is not base  # wrapped
    assert seen["environment_reward"] is seen["trainer_reward"]
    assert environment.reward_fn is base  # restored after training


async def test_uncertainty_weighting_restores_the_environment_reward_on_failure(
    monkeypatch,
):
    import importlib

    # The package re-exports the train() function under the module's name.
    train_module = importlib.import_module("stateset_agents.training.train")

    base = CountingReward(RewardType.SPARSE)
    environment = ConversationEnvironment([SCRIPTED], max_turns=5, reward_fn=base)

    async def failing_run(*args, **kwargs):
        raise RuntimeError("trainer crashed")

    monkeypatch.setattr(train_module, "_run_training", failing_run)
    with pytest.raises(RuntimeError, match="trainer crashed"):
        await train_module.train(
            SimpleNamespace(config=None),
            environment,
            num_episodes=1,
            training_mode="single_turn",
            uncertainty_weighted=True,
        )
    assert environment.reward_fn is base
