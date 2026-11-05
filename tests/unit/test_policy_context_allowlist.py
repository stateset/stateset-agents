"""The policy's context is an allowlist: reference and grading data never reach it.

Each environment hands ``agent_fn(history, context)`` a context built by
``Environment.observation_context``. A denylist leaked every reference field
it did not name (``expected_answer``, ``correct_facts``, ``label``, code
``tests``, ``ground_truth``, ``required_actions``), so the context is now built
from the keys an environment declares visible, plus any the caller opts in.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.environments.tool_use import ToolUseEnvironment
from stateset_agents.core.task_environment import TaskEnvironment
from stateset_agents.core.tool_registry import calculator_tool

SECRET = "SECRET-REFERENCE-4721"
#: Every reference or grading field a scenario may carry.
REFERENCE_FIELDS: dict[str, Any] = {
    "ground_truth": SECRET,
    "expected_answer": SECRET,
    "correct_facts": [SECRET],
    "label": SECRET,
    "answer": SECRET,
    "solution": SECRET,
    "reference": SECRET,
    "expected_response": SECRET,
    "expected_tool_calls": [{"tool": SECRET, "args": {}}],
    "tests": [{"args": [1], "expected": SECRET}],
    "hidden_tests": [{"stdin": "1", "expected_stdout": SECRET}],
    "rubric": SECRET,
    "expected_outcome": SECRET,
    "eligible": SECRET,
    "target_expression": SECRET,
}


def _recording_agent(reply: str):
    contexts: list[Any] = []

    async def respond(history: Any, context: Any) -> str:
        contexts.append(context)
        return reply

    respond.contexts = contexts  # type: ignore[attr-defined]
    return respond


def _leaks(contexts: list[Any]) -> bool:
    return SECRET in json.dumps(contexts, default=str)


def test_tool_use_policy_never_sees_reference_fields():
    scenario = {"id": "q", "question": "What is 12*7?", **REFERENCE_FIELDS}
    env = ToolUseEnvironment([scenario], [calculator_tool()], max_tool_calls=1)
    agent = _recording_agent("\\boxed{84}")
    asyncio.run(env.run_episode(agent, scenario))
    assert agent.contexts and not _leaks(agent.contexts)
    assert agent.contexts[0]["scenario"]["question"] == "What is 12*7?"

    async def batch(histories: Any, contexts: Any) -> list[str]:
        agent.contexts.extend(contexts)
        return ["\\boxed{84}" for _ in histories]

    asyncio.run(env.run_episodes_batched(batch, [scenario, scenario]))
    assert not _leaks(agent.contexts)


def test_conversation_policy_never_sees_reference_fields():
    scenario = {
        "id": "c",
        "user_query": "Where is my order?",
        "topic": "shipping",
        **REFERENCE_FIELDS,
    }
    env = ConversationEnvironment([scenario], max_turns=2)
    agent = _recording_agent("Let me check that order for you.")
    asyncio.run(env.run_episode(agent, scenario))
    assert len(agent.contexts) == 2 and not _leaks(agent.contexts)
    context = agent.contexts[0]
    assert context["scenario"]["user_query"] == "Where is my order?"
    assert context["conversation_topic"] == "shipping"
    assert context["history"] == [{"role": "user", "content": "Where is my order?"}]


def test_task_policy_never_sees_reference_fields_or_required_actions():
    scenario = {
        "id": "t",
        "description": "Prepare the quarterly report.",
        "goal": "a report",
        "required_actions": ["secret_load_step", "secret_write_step"],
        **REFERENCE_FIELDS,
    }
    env = TaskEnvironment(
        [scenario], success_criteria=lambda turns, ctx: False, max_turns=2
    )
    agent = _recording_agent('{"action": "secret_load_step"}')
    asyncio.run(env.run_episode(agent, scenario))
    flat = json.dumps(agent.contexts, default=str)
    assert len(agent.contexts) == 2 and not _leaks(agent.contexts)
    assert "secret_write_step" not in flat and "required_actions" not in flat
    assert agent.contexts[0]["task"]["description"] == "Prepare the quarterly report."


def test_reward_side_still_sees_every_reference_field():
    """Hiding fields from the policy never hides them from the reward."""
    from stateset_agents.core.reward_base import (
        RewardFunction,
        RewardResult,
        RewardType,
    )

    seen: list[dict[str, Any]] = []

    class Recorder(RewardFunction):
        def __init__(self) -> None:
            super().__init__(reward_type=RewardType.SPARSE)

        async def compute_reward(self, turns, context=None):
            seen.append(dict(context or {}))
            return RewardResult(score=1.0)

    scenario = {"id": "c", "user_query": "hi", **REFERENCE_FIELDS}
    env = ConversationEnvironment([scenario], max_turns=1, reward_fn=Recorder())
    asyncio.run(env.run_episode(_recording_agent("hello there"), scenario))
    assert seen and seen[0]["ground_truth"] == SECRET
    assert seen[0]["tests"] == REFERENCE_FIELDS["tests"]


@pytest.mark.parametrize(
    "build",
    [
        lambda keys: ConversationEnvironment(
            [{"id": "c", "user_query": "hi", "hint": "use code SAVE10"}],
            max_turns=1,
            visible_context_keys=keys,
        ),
        lambda keys: TaskEnvironment(
            [{"id": "t", "description": "d", "hint": "use code SAVE10"}],
            success_criteria=lambda turns, ctx: False,
            max_turns=1,
            visible_context_keys=keys,
        ),
        lambda keys: ToolUseEnvironment(
            [{"id": "q", "question": "q?", "hint": "use code SAVE10"}],
            [calculator_tool()],
            visible_context_keys=keys,
        ),
    ],
)
def test_extra_visible_keys_are_an_explicit_opt_in(build):
    hidden = _recording_agent("ok")
    asyncio.run(build(()).run_episode(hidden, None))
    assert "SAVE10" not in json.dumps(hidden.contexts, default=str)
    shown = _recording_agent("ok")
    env = build(("hint",))
    asyncio.run(env.run_episode(shown, None))
    assert "SAVE10" in json.dumps(shown.contexts, default=str)
    assert "hint" in env.clone().visible_context_keys
