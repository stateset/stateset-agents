"""τ-bench retail wrapper: scoring, privacy and fail-closed behaviour."""

from __future__ import annotations

import json
from typing import Any

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.tau_retail import (
    STOP_ACTION,
    EvaluationUnavailable,
    TauRetailEnvironment,
    TauRetailSettings,
    build_system_prompt,
    evaluate_policy,
    success_counts,
)

SETTINGS = TauRetailSettings(user_llm="fake-user", max_turns=4)


class FakeSession:
    """Scores 1.0 when the agent calls ``fix`` before the customer stops."""

    def __init__(self, task_id: str, *, broken_checker: bool = False) -> None:
        self.task_id = task_id
        self.broken_checker = broken_checker
        self.actions: list[str] = []
        self.seed: int | None = None

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        self.seed = seed
        return "user: please fix my order", {"policy": "Be careful.", "tools": []}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        if action not in (STOP_ACTION, "bye"):
            return "tool: ok", 0.0, False, False, {}
        reward = 1.0 if '{"name": "fix"}' in self.actions else 0.0
        detail = "{}" if self.broken_checker else json.dumps({"reward": reward})
        return (
            "",
            reward if not self.broken_checker else 0.0,
            True,
            False,
            {"reward_info": detail},
        )


def scripted(*actions: str):
    """Return a policy that plays ``actions`` in order, one per turn."""

    async def policy(history: Any, context: dict[str, Any]) -> ConversationTurn:
        turn = sum(message["role"] == "assistant" for message in history)
        return ConversationTurn(role="assistant", content=actions[turn])

    return policy


def environment(**kwargs: Any) -> tuple[TauRetailEnvironment, list[FakeSession]]:
    sessions: list[FakeSession] = []

    def factory(task_id: str) -> FakeSession:
        sessions.append(FakeSession(task_id, **kwargs))
        return sessions[-1]

    return TauRetailEnvironment(SETTINGS, env_factory=factory), sessions


@pytest.mark.asyncio
async def test_official_reward_decides_success() -> None:
    env, _ = environment()
    good = await evaluate_policy(
        env, scripted('{"name": "fix"}', "bye"), ["a", "b"], trials=2, seed=7
    )
    bad = await evaluate_policy(env, scripted("bye"), ["a", "b"], trials=2, seed=7)
    assert success_counts(good) == {"a": 2, "b": 2}
    assert success_counts(bad) == {"a": 0, "b": 0}
    assert [row["simulator_seed"] for row in good] == [7, 8, 7, 8]


@pytest.mark.asyncio
async def test_turn_budget_forces_an_official_score() -> None:
    env, sessions = environment()
    rows = await evaluate_policy(
        env, scripted('{"name": "fix"}', "x", "x", "x"), ["a"], trials=1, seed=0
    )
    assert rows[0]["forced_stop"] and rows[0]["success"]
    assert rows[0]["agent_turns"] == SETTINGS.max_turns
    assert sessions[0].actions[-1] == STOP_ACTION


@pytest.mark.asyncio
async def test_missing_checker_detail_is_not_a_failed_task() -> None:
    env, _ = environment(broken_checker=True)
    with pytest.raises(EvaluationUnavailable):
        await evaluate_policy(env, scripted("bye"), ["a"], trials=1, seed=0)


@pytest.mark.asyncio
async def test_policy_sees_only_task_id_and_observation() -> None:
    env, _ = environment()
    seen: list[dict[str, Any]] = []

    async def policy(history: Any, context: dict[str, Any]) -> ConversationTurn:
        seen.append(context)
        return ConversationTurn(role="assistant", content="bye")

    await evaluate_policy(env, policy, ["a"], trials=1, seed=0)
    assert seen == [{"task_id": "a", "observation": "user: please fix my order"}]


def test_rejects_duplicate_tasks_and_bad_scenarios() -> None:
    env, _ = environment()
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(evaluate_policy(env, scripted("bye"), ["a", "a"], trials=1, seed=0))
    with pytest.raises(ValueError):
        asyncio.run(env.reset({"id": 3}))


def test_system_prompt_contains_policy_and_tools() -> None:
    prompt = build_system_prompt("Never refund twice.", [{"name": "lookup"}])
    assert "Never refund twice." in prompt and '"lookup"' in prompt


@pytest.mark.asyncio
async def test_official_retail_checker_with_mocked_customer() -> None:
    """Replay a task's reference actions through the real ``tau2`` checker."""
    pytest.importorskip("tau2")
    from tau2.domains.retail.environment import get_tasks

    from stateset_agents.training.tau_retail import retail_task_ids

    train, test = retail_task_ids("train"), retail_task_ids("test")
    assert len(train) == 74 and len(test) == 40 and not set(train) & set(test)
    task = next(
        t
        for t in get_tasks("train")
        if not t.evaluation_criteria.nl_assertions
        and not t.evaluation_criteria.communicate_info
    )
    reference = [
        json.dumps({"name": a.name, "arguments": a.arguments})
        for a in task.evaluation_criteria.actions
    ]
    settings = TauRetailSettings(
        user_llm="gpt-4o-mini",
        user_llm_args={"mock_response": "Yes, please go ahead."},
        max_turns=len(reference) + 1,
    )
    env = TauRetailEnvironment(settings)
    solved = await evaluate_policy(
        env, scripted(*reference, "All done."), [task.id], trials=1, seed=1
    )
    idle = await evaluate_policy(
        env,
        scripted(*["One moment."] * settings.max_turns),
        [task.id],
        trials=1,
        seed=1,
    )
    assert solved[0]["success"] and not idle[0]["success"]
    assert "evaluation_criteria" not in await env.get_initial_prompt()
