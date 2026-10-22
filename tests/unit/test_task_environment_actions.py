"""TaskEnvironment completes actions from explicit signals, and
TaskCompletionReward reads the task shape TaskEnvironment stores."""

from __future__ import annotations

import pytest

from stateset_agents.core.basic_rewards import TaskCompletionReward
from stateset_agents.core.task_environment import TaskEnvironment, declared_actions
from stateset_agents.core.trajectory import ConversationTurn

TASK = {
    "id": "t",
    "description": "analyze",
    "required_actions": [
        {"name": "load_data", "keywords": ["load"]},
        {"name": "visualize", "keywords": ["plot"]},
    ],
}


def _replies(*replies):
    remaining = iter(replies)

    async def respond(history, context):
        return next(remaining)

    return respond


@pytest.mark.parametrize(
    "turn, expected",
    [
        (
            ConversationTurn(role="assistant", content='{"action": "load_data"}'),
            {"load_data"},
        ),
        (
            ConversationTurn(
                role="assistant",
                content='Done. {"actions": ["load_data", "Visualize"]}',
            ),
            {"load_data", "visualize"},
        ),
        (
            ConversationTurn(
                role="assistant", content='{"tool": "load_data", "args": {}}'
            ),
            {"load_data"},
        ),
        (
            ConversationTurn(
                role="assistant",
                content="",
                tool_calls=[{"type": "function", "function": {"name": "visualize"}}],
            ),
            {"visualize"},
        ),
        (
            ConversationTurn(
                role="assistant", content="ok", metadata={"action": "load_data"}
            ),
            {"load_data"},
        ),
        (ConversationTurn(role="assistant", content="I will load and plot it."), set()),
    ],
)
def test_only_explicit_signals_declare_actions(turn, expected):
    assert declared_actions(turn) == expected


async def test_explicit_declarations_complete_the_task():
    environment = TaskEnvironment(
        [TASK], success_criteria=lambda t, c: c["task_progress"] >= 1, max_turns=3
    )
    trajectory = await environment.run_episode(
        _replies('{"action": "load_data"}', '{"action": "visualize"}'), TASK
    )
    assert [a["name"] for a in trajectory.final_state["completed_actions"]] == [
        "load_data",
        "visualize",
    ]
    assert trajectory.metadata["status"] == "completed"


async def test_keyword_matching_is_an_explicit_opt_in():
    environment = TaskEnvironment(
        [TASK],
        success_criteria=lambda t, c: c["task_progress"] >= 1,
        max_turns=1,
        match_action_keywords=True,
    )
    trajectory = await environment.run_episode(_replies("load plot"), TASK)
    assert len(trajectory.final_state["completed_actions"]) == 2
    assert environment.clone().match_action_keywords is True


@pytest.mark.parametrize(
    "replies, expected",
    [
        (("nothing", "nothing"), 0.0),
        (('{"action": "load_data"}', "nothing"), 0.5),
        (('{"action": "load_data"}', '{"action": "visualize"}'), 1.0),
    ],
)
async def test_task_completion_reward_reads_the_environment_task_shape(
    replies, expected
):
    environment = TaskEnvironment(
        [TASK],
        success_criteria=lambda t, c: False,
        max_turns=2,
        reward_fn=TaskCompletionReward(),
    )
    trajectory = await environment.run_episode(_replies(*replies), TASK)
    assert trajectory.metadata["reward_breakdown"]["action_completion"] == expected


async def test_task_criteria_still_take_precedence():
    task = {**TASK, "criteria": {"required_actions": ["load_data"]}}
    reward = TaskCompletionReward()
    context = {"task": task, "completed_actions": [{"name": "load_data"}]}
    turns = [ConversationTurn(role="assistant", content="done")]
    assert (await reward.compute_reward(turns, context)).score == 1.0
