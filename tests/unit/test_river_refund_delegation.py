"""Intent/execution separation and independent replay contracts for River."""

import copy
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.core.environments.refund_delegation_environment import (
    RefundDelegationEnvironment,
)
from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.refund_delegation_trace import replay_delegation_trace
from stateset_agents.remote.river_environment import trajectory_environment_trace
from stateset_agents.remote.river_refund_delegation import (
    river_refund_delegation_factory,
    trajectory_delegation_trace,
)

RESOLVE = '{"tool":"resolve_refund","args":{"intent":"resolve_refund"}}'
FINISH = '{"tool":"finish","args":{}}'
ROW = refund_policy_benchmark("train", count=8, seed=501)[0]


@pytest.fixture
def factory(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )
    return river_refund_delegation_factory()


async def execute(factory, row=ROW, actions=(RESOLVE, FINISH), stop=None):
    env = factory()
    traj = SimpleNamespace(messages=await env.reset(row), metrics={}, truncated=stop)
    for content in actions:
        traj.messages.append({"role": "assistant", "content": content})
        observations = await env.on_turn(traj)
        if observations is not None:
            traj.messages.extend(observations)
    if stop:
        traj.reward = await env.on_truncated(traj, row, stop)
    else:
        traj.reward = await env.reward(traj, row)
    return env, traj


@pytest.mark.parametrize(
    "row",
    refund_policy_benchmark("test", count=32, seed=502),
    ids=lambda r: r["order_id"],
)
async def test_all_families_replay_with_raw_intent_separate_from_execution(
    factory, row
):
    _, traj = await execute(factory, row)
    trace = trajectory_delegation_trace(traj)
    assert trace["schema_version"] == 2
    assert trace["steps"][0]["action"]["content"] == RESOLVE
    assert len(trace["steps"][0]["execution"]) == 2
    assert trace["steps"][0]["execution"][0]["action"]["tool"] == "lookup_order"
    replay = await replay_delegation_trace(row, trace)
    assert replay["success"] and replay["reward"] == 1
    assert replay["tool_calls"] == 2 and replay["executed_tool_calls"] == 3
    with pytest.raises(ValueError):
        trajectory_environment_trace(traj)


@pytest.mark.parametrize(
    "bad",
    [
        "not-json",
        '{"tool":"refund","args":{"amount_cents":999}}',
        '{"tool":"resolve_refund","args":{"intent":"resolve_refund","order_id":"other"}}',
        '{"tool":"resolve_refund","args":{"intent":"bad","intent":"resolve_refund"}}',
    ],
)
async def test_invalid_intent_has_no_effect_and_correction_does_not_erase_failure(
    factory, bad
):
    _, traj = await execute(factory, actions=(bad, RESOLVE, FINISH))
    trace = trajectory_delegation_trace(traj)
    assert trace["steps"][0]["execution"] == []
    replay = await replay_delegation_trace(ROW, trace)
    assert not replay["success"] and replay["policy_violations"] == 1


async def test_duplicate_intents_cannot_create_duplicate_refunds(factory):
    _, traj = await execute(factory, actions=(RESOLVE, RESOLVE, FINISH))
    trace = trajectory_delegation_trace(traj)
    assert trace["steps"][1]["execution"] == []
    replay = await replay_delegation_trace(ROW, trace)
    assert replay["duplicate_intents"] == 1 and not replay["success"]
    assert replay["executed_tool_calls"] == 3


@pytest.mark.parametrize(
    "actions,stop",
    [
        ((), "context"),
        ((RESOLVE,), "output_tokens"),
        ((RESOLVE,) * 4, None),
        ((FINISH,), None),
    ],
)
async def test_premature_finish_and_all_truncation_paths_replay(factory, actions, stop):
    _, traj = await execute(factory, actions=actions, stop=stop)
    replay = await replay_delegation_trace(ROW, trajectory_delegation_trace(traj))
    assert not replay["success"] and replay["reward"] == -1


@pytest.mark.parametrize(
    "mutation",
    [
        "intent",
        "action",
        "receipt",
        "metrics",
        "terminal",
        "schema",
        "protocol",
        "missing",
        "extra",
    ],
)
async def test_auditor_rejects_tampering_in_either_evidence_channel(factory, mutation):
    _, traj = await execute(factory)
    trace = trajectory_delegation_trace(traj)
    if mutation == "intent":
        trace["steps"][0]["action"]["content"] = FINISH
    elif mutation == "action":
        trace["steps"][0]["execution"][1]["action"]["args"]["order_id"] = "other"
    elif mutation == "receipt":
        trace["steps"][0]["observations"][0]["content"] = "{}"
    elif mutation == "metrics":
        trace["steps"][0]["execution"][0]["metrics"]["policy_violations"] = 100
    elif mutation == "terminal":
        trace["terminal"]["reward"] = -1
    elif mutation == "schema":
        trace["schema_version"] = 1
    elif mutation == "protocol":
        trace["protocol"] = "raw-model"
    elif mutation == "missing":
        del trace["steps"][0]["execution"]
    else:
        trace["extra"] = True
    with pytest.raises(ValueError):
        await replay_delegation_trace(ROW, trace)


async def test_readers_and_resets_preserve_prior_completed_evidence(factory):
    env, traj = await execute(factory)
    original = copy.deepcopy(traj.stateset_environment_trace)
    trace = trajectory_delegation_trace(traj)
    trace["steps"][0]["execution"].clear()
    await env.reset({**ROW, "order_id": "another"})
    assert trajectory_delegation_trace(traj) == original


async def test_lookup_or_backend_failure_propagates_instead_of_becoming_policy_reward():
    env = RefundDelegationEnvironment()
    state = await env.reset(ROW)

    async def broken(_):
        raise ValueError("backend outage")

    env._resolver._lookup = broken
    with pytest.raises(ValueError, match="backend outage"):
        await env.step(state, ConversationTurn(role="assistant", content=RESOLVE))
    assert state.context["intent_violations"] == 0


async def test_failed_reset_cannot_reuse_previous_authorized_order():
    env = RefundDelegationEnvironment()
    state = await env.reset(ROW)
    with pytest.raises(ValueError):
        await env.reset({})
    with pytest.raises(ValueError):
        await env.step(state, ConversationTurn(role="assistant", content=RESOLVE))


async def test_admission_occurs_before_delegated_reset(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )

    def deny():
        raise RuntimeError("budget exhausted")

    env = river_refund_delegation_factory(before_reset=deny)()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        await env.reset(ROW)
    assert env.state is None and env.trace is None
