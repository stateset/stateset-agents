"""Executed action traces support diagnosis without rewriting model history."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.core.environments.refund_environment import RefundEnvironment
from stateset_agents.remote.river_environment import (
    river_environment_factory,
    trajectory_environment_trace,
)

ROW = {"order_id": "A", "amount_cents": 100, "eligible": True}


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_tracing_requires_an_explicit_boolean_before_sdk_import(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "river_client", None)
    with pytest.raises(ValueError, match="record_trace must be boolean"):
        river_environment_factory(RefundEnvironment, record_trace=value)


@pytest.fixture
def factory(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )
    return river_environment_factory(
        RefundEnvironment, record_trace=True, truncation_reward=-1
    )


async def execute(env, actions):
    messages = await env.reset(ROW)
    trajectory = SimpleNamespace(messages=messages, metrics={}, truncated=None)
    for tool, args in actions:
        trajectory.messages.append(
            {"role": "assistant", "content": json.dumps({"tool": tool, "args": args})}
        )
        observations = await env.on_turn(trajectory)
        if observations is not None:
            trajectory.messages.extend(observations)
    trajectory.reward = await env.reward(trajectory, ROW)
    return trajectory


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", [100, 101])
async def test_trace_records_actions_observations_and_terminal_failures(
    factory, amount
):
    env = factory()
    trajectory = await execute(
        env,
        [
            ("lookup_order", {"order_id": "A"}),
            ("refund", {"order_id": "A", "amount_cents": amount}),
            ("finish", {}),
        ],
    )
    history = copy.deepcopy(trajectory.messages)
    trace = trajectory_environment_trace(trajectory)
    assert len(trace["steps"]) == 3
    assert [step["reward"] for step in trace["steps"]] == [
        0,
        0,
        1 if amount == 100 else -1,
    ]
    assert trace["steps"][-1]["done"] is True
    assert trace["terminal"]["reward"] == trajectory.reward
    assert (
        json.loads(trace["steps"][1]["action"]["content"])["args"]["amount_cents"]
        == amount
    )
    if amount == 101:
        assert trace["steps"][1]["metrics"]["policy_violations"] == 1
        assert (
            "Incorrect refund amount" in trace["steps"][1]["observations"][0]["content"]
        )
    assert trajectory.messages == history
    trace["steps"][0]["metrics"]["tool_calls"] = 900
    trajectory.messages[0]["content"] = "mutated caller history"
    saved = trajectory_environment_trace(trajectory)
    assert saved["steps"][0]["metrics"]["tool_calls"] == 1
    assert saved["initial_messages"] == history[:1]
    await env.reset({**ROW, "order_id": "B"})
    assert trajectory_environment_trace(trajectory) == saved


@pytest.mark.asyncio
async def test_trace_exists_when_budget_exhausts_before_first_action(factory):
    env = factory()
    await env.reset(ROW)
    trajectory = SimpleNamespace(metrics={}, truncated="context")
    trajectory.reward = await env.on_truncated(trajectory, ROW, "context")
    trace = trajectory_environment_trace(trajectory)
    assert trace["steps"] == []
    assert trace["terminal"] == {
        "reward": -1,
        "status": "timeout",
        "truncated": "context",
    }
    assert trace["initial_messages"]


@pytest.mark.asyncio
async def test_environment_timeout_trace_keeps_all_executed_steps(factory):
    trajectory = await execute(factory(), [("lookup_order", {"order_id": "A"})] * 4)
    trace = trajectory_environment_trace(trajectory)
    assert len(trace["steps"]) == 4
    assert trace["terminal"]["truncated"] == "environment_timeout"
    assert trace["terminal"]["reward"] == -1
    assert trace["steps"][-1]["status"] == "timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["absent", "hash", "reward", "cause", "step", "nan"]
)
async def test_trace_reader_rejects_missing_or_inconsistent_evidence(factory, mutation):
    trajectory = await execute(factory(), [("finish", {})])
    trace = trajectory.stateset_environment_trace
    if mutation == "absent":
        del trajectory.stateset_environment_trace
    elif mutation == "hash":
        trace["case_hash"] = "0" * 64
    elif mutation == "reward":
        trace["terminal"]["reward"] = 1
    elif mutation == "cause":
        trace["terminal"]["truncated"] = "turns"
    elif mutation == "step":
        trace["steps"][0] = {}
    else:
        trace["steps"][0]["metrics"]["bad"] = float("nan")
    with pytest.raises(ValueError):
        trajectory_environment_trace(trajectory)


@pytest.mark.asyncio
async def test_generic_adapter_does_not_record_traces_unless_enabled(factory):
    env = river_environment_factory(RefundEnvironment)()
    trajectory = await execute(env, [("finish", {})])
    assert not hasattr(trajectory, "stateset_environment_trace")
