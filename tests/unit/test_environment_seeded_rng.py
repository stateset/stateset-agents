"""Environments draw scenarios and simulated replies from their own seeded RNG.

``ConversationEnvironment`` and ``TaskEnvironment`` used the global ``random``
module, so a run's episodes depended on whatever else consumed it, and
seeding one environment reseeded everything.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.task_environment import TaskEnvironment

SCENARIOS = [
    {"id": f"s{i}", "user_query": f"question {i}", "user_responses": ["a", "b", "c"]}
    for i in range(6)
]


def _conversation_run(seed: int) -> list[Any]:
    env = ConversationEnvironment(SCENARIOS, max_turns=4, seed=seed)

    async def agent(history: Any, context: Any) -> str:
        return "Tell me more about it, please."

    async def run() -> list[Any]:
        out = []
        for _ in range(4):
            trajectory = await env.run_episode(agent)
            out.append(
                (
                    trajectory.final_state["scenario"]["id"],
                    [t.content for t in trajectory.turns if t.role == "user"],
                )
            )
        return out

    return asyncio.run(run())


def test_conversation_environment_is_reproducible_per_seed():
    random.seed(1)
    first = _conversation_run(7)
    random.seed(2)  # the global RNG plays no part
    assert _conversation_run(7) == first
    assert _conversation_run(8) != first


def test_conversation_environment_leaves_the_global_rng_alone():
    random.seed(3)
    before = random.getstate()
    _conversation_run(7)
    assert random.getstate() == before


def test_task_environment_is_reproducible_and_leaves_the_global_rng_alone():
    tasks = [{"id": f"t{i}", "description": f"task {i}"} for i in range(6)]

    def picks(seed: int) -> list[Any]:
        env = TaskEnvironment(tasks, success_criteria=lambda t, c: False, seed=seed)
        return [asyncio.run(env.reset()).context["task_id"] for _ in range(8)]

    random.seed(4)
    before = random.getstate()
    first = picks(11)
    assert random.getstate() == before
    assert picks(11) == first and len(set(first)) > 1
    assert ConversationEnvironment(SCENARIOS, seed=5).clone().seed == 5
