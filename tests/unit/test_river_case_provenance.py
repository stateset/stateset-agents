"""Rollout evidence must identify the input admitted to its private sandbox."""

import asyncio
import copy
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.core.environments.refund_environment import RefundEnvironment
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_environment import (
    river_environment_factory,
    trajectory_case_identity,
)


@pytest.fixture(autouse=True)
def sdk(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )


@pytest.mark.asyncio
async def test_input_is_snapshotted_before_waiting_for_admission():
    entered, resume = asyncio.Event(), asyncio.Event()

    async def admit():
        entered.set()
        await resume.wait()

    row = {
        "order_id": "A",
        "amount_cents": 100,
        "eligible": True,
        "metadata": {"version": 1},
    }
    expected = copy.deepcopy(row)
    env = river_environment_factory(RefundEnvironment, before_reset=admit)()
    pending = asyncio.create_task(env.reset(row))
    await entered.wait()
    row["amount_cents"] = 999
    row["metadata"]["version"] = 2
    resume.set()
    await pending
    assert env.state.context["scenario"] == expected
    trajectory = SimpleNamespace(metrics={})
    await env.on_truncated(trajectory, row, "context_limit")
    assert trajectory_case_identity(trajectory, cases={"A": expected}) == {
        "case_id": "A",
        "case_hash": content_hash(expected),
    }
    with pytest.raises(ValueError, match="reset input"):
        trajectory_case_identity(trajectory, cases={"A": row})


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["admission", "reset", "messages", "nonfinite"])
async def test_failed_reset_cannot_reuse_previous_episode_or_hash(failure):
    fail = False

    async def admit():
        if fail and failure == "admission":
            raise ValueError("admission failed")

    class Sandbox(RefundEnvironment):
        async def reset(self, row):
            if fail and failure == "reset":
                raise ValueError("reset failed")
            state = await super().reset(row)
            if fail and failure == "messages":
                state.context["messages"] = []
            return state

    env = river_environment_factory(Sandbox, before_reset=admit)()
    row = {"order_id": "A", "amount_cents": 100, "eligible": True}
    await env.reset(row)
    first = SimpleNamespace(metrics={})
    await env.on_truncated(first, row, "turns")
    fail = True
    invalid = {**row, "extra": float("nan")} if failure == "nonfinite" else row
    with pytest.raises(ValueError):
        await env.reset(invalid)
    assert env.state is None and env.case_hash is None
    with pytest.raises(RuntimeError, match="reset"):
        await env.on_truncated(SimpleNamespace(metrics={}), row, "turns")
    with pytest.raises(RuntimeError, match="reset"):
        await env.on_turn(first)
    with pytest.raises(RuntimeError, match="termination"):
        await env.reward(first, row)
    fail = False
    second = {**row, "order_id": "B"}
    await env.reset(second)
    trajectory = SimpleNamespace(metrics={})
    await env.on_truncated(trajectory, second, "turns")
    assert trajectory_case_identity(trajectory, cases={"B": second})["case_id"] == "B"


@pytest.mark.parametrize("mutation", ["missing", "different", "case_id", "bad_type"])
def test_trajectory_identity_rejects_missing_or_wrong_reset_input(mutation):
    row = {"order_id": "A", "eligible": True}
    trajectory = SimpleNamespace(
        stateset_episode_id="A", stateset_case_hash=content_hash(row)
    )
    if mutation == "missing":
        del trajectory.stateset_case_hash
    elif mutation == "different":
        trajectory.stateset_case_hash = content_hash({**row, "eligible": False})
    elif mutation == "case_id":
        trajectory.stateset_episode_id = "B"
    else:
        trajectory.stateset_episode_id = ["A"]
    with pytest.raises(ValueError, match="reset input"):
        trajectory_case_identity(trajectory, cases={"A": row})
