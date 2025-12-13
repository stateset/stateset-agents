from __future__ import annotations

from typing import Any, Dict, List

from stateset_agents.core.environment import ConversationEnvironment, TaskEnvironment


async def test_conversation_environment_clone_isolated_state() -> None:
    scenario: Dict[str, Any] = {
        "id": "s1",
        "topic": "demo",
        "context": "Demo",
        "user_responses": ["ok"],
    }
    env = ConversationEnvironment(scenarios=[scenario], max_turns=2)
    clone = env.clone()

    assert clone is not env
    assert clone.max_turns == env.max_turns
    assert clone.scenarios == env.scenarios
    assert getattr(env, "_last_state", None) is None

    await clone.reset(scenario=scenario)
    assert getattr(clone, "_last_state", None) is not None
    assert getattr(env, "_last_state", None) is None


async def test_task_environment_clone_preserves_configuration() -> None:
    tasks: List[Dict[str, Any]] = [{"id": "t1", "goal": "done", "description": "Demo task"}]

    def _success_criteria(turns: Any, context: Dict[str, Any]) -> bool:
        return True

    env = TaskEnvironment(tasks=tasks, success_criteria=_success_criteria, max_turns=3)
    clone = env.clone()

    assert clone is not env
    assert clone.max_turns == env.max_turns
    assert clone.tasks == env.tasks
    assert clone.success_criteria is env.success_criteria

