"""Recovery trace attribution, knowledge/effect separation, and replay."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.data.refund_recovery import (
    reference_recovery_action,
    refund_recovery_cases,
)
from stateset_agents.evaluation.refund_recovery_trace import replay_recovery_trace
from stateset_agents.remote.river_environment import trajectory_environment_trace
from stateset_agents.remote.river_refund_delegation import trajectory_delegation_trace
from stateset_agents.remote.river_refund_recovery import (
    river_refund_recovery_factory,
    trajectory_recovery_trace,
)


@pytest.fixture
def factory(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )
    return river_refund_recovery_factory()


async def execute(factory, row, *, stop=None):
    env = factory()
    traj = SimpleNamespace(messages=await env.reset(row), metrics={}, truncated=stop)
    observation = None
    if stop is None:
        for _ in range(6):
            intent = reference_recovery_action(observation)
            traj.messages.append({"role": "assistant", "content": json.dumps(intent)})
            messages = await env.on_turn(traj)
            if messages is None:
                break
            traj.messages.extend(messages)
            observation = json.loads(messages[0]["content"])
        traj.reward = await env.reward(traj, row)
    else:
        traj.reward = await env.on_truncated(traj, row, stop)
    return env, traj


@pytest.mark.parametrize(
    "row",
    refund_recovery_cases("test", count=20, seed=921),
    ids=lambda r: r["order_id"],
)
async def test_every_family_replays_actual_commits_and_reconciliation(factory, row):
    _, traj = await execute(factory, row)
    trace = trajectory_recovery_trace(traj)
    replay = await replay_recovery_trace(row, trace)
    assert replay["success"] and replay["resolution_count"] <= 1
    assert replay["backend_calls"] == traj.metrics["backend_calls"]
    assert trace["schema_version"] == 3 and trace["protocol"] == "refund-recovery-v1"
    for legacy_reader in (trajectory_environment_trace, trajectory_delegation_trace):
        with pytest.raises(ValueError):
            legacy_reader(traj)


@pytest.mark.parametrize(
    "mutation",
    [
        "commit",
        "receipt",
        "intent",
        "observation",
        "metric",
        "terminal",
        "protocol",
        "case",
        "extra",
    ],
)
async def test_auditor_rejects_forged_knowledge_or_effects(factory, mutation):
    row = next(
        r
        for r in refund_recovery_cases("test", count=10, seed=922)
        if r["recovery"] == "lost_ack_applied"
    )
    _, traj = await execute(factory, row)
    trace = trajectory_recovery_trace(traj)
    if mutation == "commit":
        trace["steps"][0]["execution"][1]["result"]["committed"] = False
    elif mutation == "receipt":
        trace["steps"][1]["execution"][0]["result"]["receipt"]["applied"] = False
    elif mutation == "intent":
        trace["steps"][1]["action"][
            "content"
        ] = '{"tool":"resolve_refund","args":{"intent":"resolve_refund"}}'
    elif mutation == "observation":
        trace["steps"][0]["observations"][0]["content"] = '{"status":"applied"}'
    elif mutation == "metric":
        trace["steps"][0]["metrics"]["resolution_count"] = 2
    elif mutation == "terminal":
        trace["terminal"]["reward"] = -1
    elif mutation == "protocol":
        trace["protocol"] = "refund-delegation-v1"
    elif mutation == "case":
        trace["case_hash"] = "0" * 64
    else:
        trace["extra"] = True
    with pytest.raises(ValueError):
        await replay_recovery_trace(row, trace)


async def test_pre_action_truncation_and_detached_evidence(factory):
    row = refund_recovery_cases("test", count=10, seed=923)[0]
    env, traj = await execute(factory, row, stop="context")
    original = copy.deepcopy(trajectory_recovery_trace(traj))
    replay = await replay_recovery_trace(row, original)
    assert (
        replay["reward"] == -1
        and replay["resolution_count"] == replay["backend_calls"] == 0
    )
    await env.reset({**row, "order_id": "other-order"})
    assert trajectory_recovery_trace(traj) == original
