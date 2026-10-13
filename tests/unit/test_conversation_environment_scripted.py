"""ConversationEnvironment must show the agent the customer's actual messages."""

from __future__ import annotations

import asyncio
from typing import Any

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.trajectory import ConversationTurn


def _run(
    env: ConversationEnvironment, scenario: dict[str, Any]
) -> tuple[Any, list[list[dict[str, str]]]]:
    seen: list[list[dict[str, str]]] = []

    async def agent_fn(history: list[dict[str, str]], context: Any) -> str:
        seen.append([dict(m) for m in history])
        return f"reply {len(seen)}"

    trajectory = asyncio.run(env.run_episode(agent_fn, scenario))
    return trajectory, seen


def test_opening_user_query_is_shown_before_first_agent_turn() -> None:
    scenario = {"id": "s1", "user_query": "Where is my order #123?"}
    env = ConversationEnvironment(scenarios=[scenario], max_turns=1)

    _, seen = _run(env, scenario)

    assert seen[0][-1] == {"role": "user", "content": "Where is my order #123?"}


def test_scripted_user_turns_drive_the_conversation_and_end_it() -> None:
    scenario = {
        "id": "s2",
        "user_query": "I want a refund.",
        "user_turns": ["I want a refund.", "It arrived broken.", "Thanks."],
    }
    env = ConversationEnvironment(scenarios=[scenario], max_turns=10)

    trajectory, seen = _run(env, scenario)

    users = [m["content"] for m in seen[-1] if m["role"] == "user"]
    assert users == ["I want a refund.", "It arrived broken.", "Thanks."]
    # One agent reply per scripted user message; no invented filler turns.
    assert len(seen) == 3
    roles = [t.role for t in trajectory.turns if t.role != "system"]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]


def test_max_turns_still_caps_scripted_conversations() -> None:
    scenario = {"id": "s3", "user_turns": ["a", "b", "c", "d"]}
    env = ConversationEnvironment(scenarios=[scenario], max_turns=2)

    _, seen = _run(env, scenario)

    assert len(seen) == 2


def test_step_reward_sees_the_opening_message() -> None:
    contexts: list[list[str]] = []

    class Recorder:
        async def compute_reward(self, turns: list[ConversationTurn], context: Any):
            contexts.append([t.content for t in turns])
            return 0.0

    scenario = {"id": "s4", "user_turns": ["hello there", "bye"]}
    env = ConversationEnvironment(
        scenarios=[scenario], max_turns=4, reward_fn=Recorder()  # type: ignore[arg-type]
    )

    _run(env, scenario)

    assert contexts[0] == ["hello there", "reply 1"]
    assert contexts[1] == ["hello there", "reply 1", "bye", "reply 2"]


def test_scenarios_without_user_messages_keep_legacy_behaviour() -> None:
    scenario = {"id": "s5", "context": "chit-chat"}
    env = ConversationEnvironment(scenarios=[scenario], max_turns=1)

    _, seen = _run(env, scenario)

    assert [m["role"] for m in seen[0]] == ["system"]
