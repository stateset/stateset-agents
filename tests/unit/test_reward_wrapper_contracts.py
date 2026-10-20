"""Reward composition must preserve required evidence and critical failures."""

from types import SimpleNamespace

import pytest

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.task_outcome import TaskOutcomeReward
from stateset_agents.training.train import UncertaintyWeightedReward


class RecordedReward(RewardFunction):
    def __init__(
        self, *, critical=False, environmental=False, failure=None, result=None
    ):
        super().__init__()
        self.fail_on_error = critical
        self.requires_environment_context = environmental
        self.failure = failure
        self.result = result or RewardResult(score=0.5)
        self.contexts = []

    async def compute_reward(self, turns, context=None):
        self.contexts.append(context)
        if self.failure is not None:
            raise self.failure
        return self.result


def wrap(base, kind):
    if kind == "composite":
        return CompositeReward([base])
    if kind == "uncertainty":
        return UncertaintyWeightedReward(base)
    return CompositeReward([UncertaintyWeightedReward(CompositeReward([base]))])


@pytest.mark.parametrize("kind", ["composite", "uncertainty", "nested"])
async def test_required_outcome_failure_is_not_silenced_by_wrappers_or_environment(
    kind,
):
    reward = wrap(TaskOutcomeReward(), kind)
    assert reward.fail_on_error and reward.requires_environment_context
    with pytest.raises(ValueError, match="expected_response"):
        await reward.compute_reward(
            [ConversationTurn(role="assistant", content="done")]
        )
    environment = ConversationEnvironment(scenarios=[], reward_fn=reward)
    with pytest.raises(ValueError, match="expected_response"):
        await environment.get_reward(SimpleNamespace(turns=[]))


async def test_mixed_composition_routes_context_to_each_child_and_preserves_evidence():
    ordinary = RecordedReward()
    outcome = TaskOutcomeReward()
    composite = CompositeReward([ordinary, UncertaintyWeightedReward(outcome)])
    scenario = {
        "expected_response": "done",
        "expected_tool_calls": [{"tool": "finish", "args": {}}],
        "topic": "refunds",
    }
    context = {
        "scenario": scenario,
        "executed_tool_calls": [{"tool": "finish", "args": {}}],
        "policy_violations": 0,
    }
    result = await composite.compute_reward(
        [ConversationTurn(role="assistant", content="done")], context
    )
    assert ordinary.contexts == [scenario]
    assert result.score == 1.5
    assert result.metadata["component_breakdowns"][1]["task_success"] == 1.0
    assert result.metadata["component_metadata"][1]["reward_semantics"] == (
        "verified_task_outcome_v1"
    )


async def test_optional_reward_failures_remain_optional_and_capabilities_are_dynamic():
    optional = RecordedReward(failure=ValueError("optional unavailable"))
    composite = CompositeReward([optional, RecordedReward()])
    assert not composite.fail_on_error and not composite.requires_environment_context
    assert (await composite.compute_reward([])).score == 0.5
    composite.reward_functions.append(TaskOutcomeReward())
    assert composite.fail_on_error and composite.requires_environment_context
    with pytest.raises(ValueError, match="expected_response"):
        await composite.compute_reward([])


def test_nested_reward_timing_tracks_composition_changes():
    composite = CompositeReward([RecordedReward()])
    reward = UncertaintyWeightedReward(composite)
    assert reward.reward_type == RewardType.IMMEDIATE
    composite.reward_functions.append(TaskOutcomeReward())
    assert reward.reward_type == RewardType.CUMULATIVE
    composite.reward_functions.pop()
    assert reward.reward_type == RewardType.IMMEDIATE
    reward.reward_type = RewardType.SPARSE
    assert reward.reward_type == RewardType.SPARSE
    assert composite.reward_type == RewardType.IMMEDIATE


async def test_uncertainty_discount_preserves_breakdown_metadata_and_explanation():
    original = RewardResult(
        score=0.8,
        breakdown={"task_success": 1.0, "total_uncertainty": 0.5},
        metadata={"reward_semantics": "verified_task_outcome_v1"},
        explanation="Verified tool execution",
    )
    result = await UncertaintyWeightedReward(
        RecordedReward(result=original), uncertainty_discount=0.5
    ).compute_reward([])
    assert result.score == pytest.approx(0.6)
    assert result.breakdown["task_success"] == 1.0
    assert result.breakdown["confidence"] == 0.75
    assert result.metadata["reward_semantics"] == original.metadata["reward_semantics"]
    assert result.explanation == original.explanation
    assert original.breakdown == {"task_success": 1.0, "total_uncertainty": 0.5}


@pytest.mark.parametrize(
    "value", [-1, True, "0.5", 10**400, float("nan"), float("inf")]
)
def test_uncertainty_discount_rejects_invalid_coefficients(value):
    with pytest.raises(ValueError, match="uncertainty_discount"):
        UncertaintyWeightedReward(RecordedReward(), uncertainty_discount=value)


@pytest.mark.parametrize("location", ["breakdown", "metadata"])
@pytest.mark.parametrize(
    "value", [-1, True, "0.5", 10**400, float("nan"), float("inf")]
)
async def test_uncertainty_rejects_invalid_evidence_instead_of_amplifying_rewards(
    location, value
):
    result = RewardResult(score=1.0, **{location: {"total_uncertainty": value}})
    reward = UncertaintyWeightedReward(RecordedReward(result=result))
    with pytest.raises(ValueError, match="uncertainty"):
        await reward.compute_reward([])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
async def test_uncertainty_never_masks_nonfinite_base_scores(score):
    reward = UncertaintyWeightedReward(RecordedReward(result=RewardResult(score=score)))
    with pytest.raises(ValueError, match="score"):
        await reward.compute_reward([])


@pytest.mark.parametrize("kind", ["composite", "uncertainty", "nested"])
async def test_wrapped_outcomes_are_scored_once_after_the_complete_episode(kind):
    outcome = RecordedReward(critical=True, environmental=True)
    outcome.reward_type = RewardType.SPARSE
    reward = wrap(outcome, kind)
    environment = ConversationEnvironment(scenarios=[], max_turns=2, reward_fn=reward)

    async def respond(history, context):
        assert "expected_response" not in context["scenario"]
        return "done"

    trajectory = await environment.run_episode(respond, {"expected_response": "done"})
    assert len(outcome.contexts) == 1
    assert trajectory.total_reward == 0.5
    assert outcome.contexts[0]["scenario"]["expected_response"] == "done"


@pytest.mark.parametrize("kind", ["composite", "uncertainty", "nested"])
async def test_environment_can_rescore_a_completed_outcome_with_final_context(kind):
    environment = ConversationEnvironment(
        scenarios=[], max_turns=1, reward_fn=wrap(TaskOutcomeReward(), kind)
    )

    async def respond(history, context):
        return "done"

    trajectory = await environment.run_episode(respond, {"expected_response": "done"})
    assert trajectory.total_reward == 1
    assert await environment.get_reward(trajectory) == 1


@pytest.mark.parametrize("kind", ["composite", "uncertainty", "nested"])
async def test_critical_step_reward_failure_is_never_replaced_with_zero(kind):
    reward = wrap(
        RecordedReward(critical=True, failure=ValueError("missing evidence")), kind
    )
    environment = ConversationEnvironment(scenarios=[], max_turns=1, reward_fn=reward)
    state = await environment.reset({})
    with pytest.raises(ValueError, match="missing evidence"):
        await environment.step(state, "done")


async def test_nested_outcome_runs_through_real_trainer_with_executed_tools():
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    class ToolEnvironment(ConversationEnvironment):
        async def reset(self, scenario=None):
            state = await super().reset(scenario)
            state.context["executed_tool_calls"] = []
            return state

        async def _step_impl(self, state, action):
            result = await super()._step_impl(state, action)
            if action.content == "done":
                result[0].context["executed_tool_calls"].append({"tool": "finish"})
            return result

    contexts = []

    async def respond(history, context):
        contexts.append(context)
        return "done" if len(contexts) == 2 else "working"

    ordinary = RecordedReward()
    reward = CompositeReward([ordinary, wrap(TaskOutcomeReward(), "nested")])
    environment = ToolEnvironment(scenarios=[], max_turns=2, reward_fn=reward)
    scenario = {
        "id": "verified",
        "expected_response": "done",
        "expected_tool_calls": [{"tool": "finish"}],
    }
    trainer = MultiTurnGRPOTrainer(
        agent=SimpleNamespace(generate_response=respond),
        environment=environment,
        reward_fn=reward,
        config=TrainingConfig(num_generations=1),
    )
    groups = await trainer.generate_trajectories([scenario])
    assert len(groups) == 1 and len(groups[0].trajectories) == 1
    assert groups[0].trajectories[0].total_reward == 1.5
    assert ordinary.contexts == [scenario, scenario]
    assert all("expected_tool_calls" not in context["scenario"] for context in contexts)


@pytest.mark.parametrize("kind", ["composite", "uncertainty", "nested"])
async def test_policy_cannot_forge_tool_evidence_by_mutating_observation(kind):
    class TraceEnvironment(ConversationEnvironment):
        async def reset(self, scenario=None):
            state = await super().reset(scenario)
            state.context["executed_tool_calls"] = []
            return state

    expected = {"tool": "refund", "arguments": {"amount_cents": 100}}
    scenario = {
        "expected_response": "done",
        "expected_tool_calls": [expected],
        "public": {"cart": [{"amount_cents": 100}]},
    }
    environment = TraceEnvironment(
        scenarios=[], max_turns=1, reward_fn=wrap(TaskOutcomeReward(), kind)
    )

    async def forge(history, context):
        context["executed_tool_calls"].append(expected)
        context["scenario"]["public"]["cart"][0]["amount_cents"] = 0
        return "done"

    trajectory = await environment.run_episode(forge, scenario)
    assert trajectory.total_reward == 0
    assert trajectory.final_state["executed_tool_calls"] == []
    assert scenario["public"]["cart"][0]["amount_cents"] == 100
