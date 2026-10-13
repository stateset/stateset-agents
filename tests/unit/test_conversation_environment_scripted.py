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


# --- system prompt ownership and farewell handling ------------------------------


def _replying(text: str):
    async def agent_fn(history: list[dict[str, str]], context: Any) -> str:
        return text

    return agent_fn


def test_no_generic_system_turn_when_nothing_provides_one() -> None:
    """The agent's own ``AgentConfig.system_prompt`` must not be clobbered."""
    scenario = {"id": "s4", "user_query": "hi"}
    env = ConversationEnvironment(scenarios=[scenario], max_turns=1)

    assert asyncio.run(env.get_initial_prompt(scenario)) == ""
    trajectory, seen = _run(env, scenario)
    assert all(m["role"] != "system" for m in seen[0])
    assert all(t.role != "system" for t in trajectory.turns)


def test_explicit_system_prompts_are_emitted() -> None:
    env = ConversationEnvironment(
        scenarios=[], max_turns=1, system_prompt="You are Luna."
    )
    assert asyncio.run(env.get_initial_prompt({})) == "You are Luna."
    assert (
        asyncio.run(env.get_initial_prompt({"context": "Order #1 is late."}))
        == "You are Luna. Context: Order #1 is late."
    )

    plain = ConversationEnvironment(scenarios=[], max_turns=1)
    assert asyncio.run(plain.get_initial_prompt({"system_prompt": "Be brief."})) == (
        "Be brief."
    )
    assert asyncio.run(plain.get_initial_prompt({"context": "VIP customer"})) == (
        "Context: VIP customer"
    )


def test_persona_alone_does_not_create_a_system_turn() -> None:
    env = ConversationEnvironment(scenarios=[], max_turns=1, persona="Calm.")
    assert asyncio.run(env.get_initial_prompt({})) == ""
    assert (
        asyncio.run(env.get_initial_prompt({"system_prompt": "Be brief."}))
        == "Be brief. Calm."
    )


def test_agent_system_prompt_applies_when_environment_has_none() -> None:
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    agent = MultiTurnAgent(
        AgentConfig(
            model_name="stub://test",
            use_stub_model=True,
            system_prompt="You are Byte.",
            use_chat_template=False,
        )
    )
    scenario = {"id": "s5", "user_query": "hi"}
    env = ConversationEnvironment(scenarios=[scenario], max_turns=1)
    _, seen = _run(env, scenario)
    prompt, _ = agent._render_prompt(seen[0])
    assert prompt.startswith("System: You are Byte.\n")

    # An explicit system turn in the history takes precedence (documented).
    prompt, _ = agent._render_prompt(
        [{"role": "system", "content": "Env rules."}, *seen[0]]
    )
    assert "You are Byte." not in prompt and "System: Env rules." in prompt


def test_scripted_conversations_do_not_end_on_a_farewell() -> None:
    scenario = {"id": "s6", "user_turns": ["hi", "one more thing", "thanks"]}
    env = ConversationEnvironment(scenarios=[scenario], max_turns=10)

    trajectory = asyncio.run(env.run_episode(_replying("Goodbye!"), scenario))

    assert sum(t.role == "assistant" for t in trajectory.turns) == 3


def test_unscripted_farewell_ending_is_configurable() -> None:
    scenario = {"id": "s7", "user_query": "hi"}
    default = ConversationEnvironment(scenarios=[scenario], max_turns=4)
    trajectory = asyncio.run(default.run_episode(_replying("Goodbye!"), scenario))
    assert sum(t.role == "assistant" for t in trajectory.turns) == 1

    keep_going = ConversationEnvironment(
        scenarios=[scenario], max_turns=4, end_on_farewell=False
    )
    trajectory = asyncio.run(keep_going.run_episode(_replying("Goodbye!"), scenario))
    assert sum(t.role == "assistant" for t in trajectory.turns) == 4


def test_farewell_matching_uses_word_boundaries() -> None:
    scenario = {"id": "s8", "user_query": "hi"}
    env = ConversationEnvironment(scenarios=[scenario], max_turns=3)
    trajectory = asyncio.run(
        env.run_episode(_replying("Please see your order page."), scenario)
    )
    assert sum(t.role == "assistant" for t in trajectory.turns) == 3


def test_clone_keeps_system_prompt_and_farewell_settings() -> None:
    env = ConversationEnvironment(
        scenarios=[], max_turns=2, system_prompt="S", end_on_farewell=False
    )
    clone = env.clone()
    assert clone.system_prompt == "S" and clone.end_on_farewell is False
