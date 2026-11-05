"""Safe recovery choices, no hidden labels, and exact effect accounting."""

import copy
import json
from collections import Counter

import pytest

from stateset_agents.core.environments.refund_recovery_environment import (
    RECOVERY_FAMILIES,
    RefundRecoveryEnvironment,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.refund_recovery import (
    reference_recovery_action,
    reference_recovery_demonstration,
    refund_recovery_cases,
    replay_recovery_candidate,
)


def action(tool="resolve_refund", **args):
    if tool == "resolve_refund" and not args:
        args = {"intent": "resolve_refund"}
    return ConversationTurn(
        role="assistant", content=json.dumps({"tool": tool, "args": args})
    )


@pytest.mark.parametrize(
    "row",
    refund_recovery_cases("test", count=80, seed=811),
    ids=lambda row: row["order_id"],
)
async def test_public_observation_reference_completes_each_fault_without_duplicates(
    row,
):
    demo = await reference_recovery_demonstration(row)
    checked = await replay_recovery_candidate(row, demo)
    assert checked["passed"] and 2 <= checked["turns"] <= 4


def test_case_factory_balances_ten_families_and_holds_out_windows():
    train = refund_recovery_cases("train", count=160, seed=812)
    test = refund_recovery_cases("test", count=80, seed=812, excluded=train)
    assert Counter(r["recovery"] for r in train) == dict.fromkeys(RECOVERY_FAMILIES, 16)
    assert {r["return_window_days"] for r in train} == {14, 30}
    assert {r["return_window_days"] for r in test} == {7, 45}
    with pytest.raises(ValueError, match="overlap"):
        refund_recovery_cases(
            "train", count=160, seed=812, excluded=[{**train[0], "order_id": "alias"}]
        )


@pytest.mark.parametrize("count,seed", [(True, 1), (0, 1), (10, -1), (10, False)])
def test_factory_rejects_ambiguous_integer_configuration(count, seed):
    with pytest.raises(ValueError):
        refund_recovery_cases("train", count=count, seed=seed)


@pytest.mark.parametrize("fault", RECOVERY_FAMILIES)
async def test_private_fault_labels_and_annotations_never_enter_initial_prompt(fault):
    source = next(
        r
        for r in refund_recovery_cases("train", count=10, seed=813)
        if r["recovery"] == "clean"
    )
    env = RefundRecoveryEnvironment()
    baseline = await env.reset(source)
    state = await env.reset(
        {
            **source,
            "recovery": fault,
            "family": "arbitrary",
            "expected_action": "secret-answer",
        }
    )
    assert state.context["messages"] == baseline.context["messages"]
    assert env.observation_context(state) == {}
    assert "secret-answer" not in state.context["messages"][0]["content"]


@pytest.mark.parametrize(
    "fault",
    [
        "lost_ack_applied",
        "lost_ack_not_applied",
        "reconcile_pending",
        "reconcile_unavailable",
    ],
)
async def test_blind_retry_after_unknown_cannot_create_second_effect_or_erase_violation(
    fault,
):
    row = next(
        r
        for r in refund_recovery_cases("train", count=10, seed=814)
        if r["recovery"] == fault
    )
    env = RefundRecoveryEnvironment()
    state = await env.reset(row)
    state, _, _, info = await env.step(state, action())
    assert json.loads(info["messages"][0]["content"]) == {
        "status": "outcome_unknown",
        "retry_safe": False,
    }
    committed = state.context["resolution_count"]
    state, _, _, info = await env.step(state, action())
    assert state.context["resolution_count"] == committed
    assert env.execution_events() == [] and info["metrics"]["duplicate_attempts"] == 1
    state, _, _, info = await env.step(state, action("reconcile"))
    observation = json.loads(info["messages"][0]["content"])
    for _ in range(3):
        next_action = reference_recovery_action(observation)
        state, reward, done, info = await env.step(
            state, ConversationTurn(role="assistant", content=json.dumps(next_action))
        )
        if done:
            break
        observation = json.loads(info["messages"][0]["content"])
    assert done and reward == -1 and info["metrics"]["task_success"] == 0
    assert info["metrics"]["resolution_count"] <= 1


@pytest.mark.parametrize("fault", ["unauthorized", "authorization_revoked"])
async def test_authorization_failure_applies_nothing_and_cannot_be_overridden(fault):
    row = next(
        r
        for r in refund_recovery_cases("train", count=10, seed=815)
        if r["recovery"] == fault
    )
    env = RefundRecoveryEnvironment()
    state = await env.reset(row)
    state, _, _, info = await env.step(state, action())
    assert json.loads(info["messages"][0]["content"])["status"] == "unauthorized"
    assert info["metrics"]["resolution_count"] == info["metrics"]["refunded_cents"] == 0
    if fault == "unauthorized":
        assert all("facts" not in event["result"] for event in env.execution_events())
        assert info["metrics"]["backend_calls"] == 1
    state, _, _, info = await env.step(state, action())
    assert env.execution_events() == []
    state, reward, done, info = await env.step(
        state, action("finish", outcome="unauthorized")
    )
    assert done and reward == -1 and info["metrics"]["resolution_count"] == 0


@pytest.mark.parametrize(
    "fault,expected", [("stale_chargeback", "escalate"), ("stale_refund", "deny")]
)
async def test_stale_snapshot_rejects_old_action_and_recomputes_new_priority(
    fault, expected
):
    row = next(
        r
        for r in refund_recovery_cases("train", count=10, seed=816)
        if r["recovery"] == fault
    )
    env = RefundRecoveryEnvironment()
    state = await env.reset(row)
    state, _, _, info = await env.step(state, action())
    assert info["metrics"]["resolution_count"] == 0
    assert json.loads(info["messages"][0]["content"])["status"] == "retryable"
    state, _, _, info = await env.step(state, action())
    observation = json.loads(info["messages"][0]["content"])
    assert observation["receipt"]["action"]["tool"] == expected
    assert info["metrics"]["resolution_count"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        '{"tool":"resolve_refund","args":{"intent":"resolve_refund","order_id":"other"}}',
        '{"tool":"resolve_refund","args":{"intent":"bad","intent":"resolve_refund"}}',
        '{"tool":"refund","args":{"order_id":"other","amount_cents":1}}',
        '{"tool":"finish","args":{"outcome":[]}}',
    ],
)
async def test_invalid_model_intents_have_no_backend_effect(bad):
    env = RefundRecoveryEnvironment()
    state = await env.reset(refund_recovery_cases("train", count=10, seed=817)[0])
    state, _, _, info = await env.step(
        state, ConversationTurn(role="assistant", content=bad)
    )
    assert env.execution_events() == []
    assert info["metrics"]["backend_calls"] == info["metrics"]["resolution_count"] == 0
    assert info["metrics"]["policy_violations"] == 1


async def test_unplanned_infrastructure_exception_is_not_a_reward_or_safe_retry():
    env = RefundRecoveryEnvironment()
    state = await env.reset(refund_recovery_cases("train", count=10, seed=818)[0])

    async def broken(_):
        raise OSError("unexpected backend outage")

    env._resolver._submit = broken
    with pytest.raises(OSError, match="unexpected"):
        await env.step(state, action())
    assert state.context["violations"] == 0
    with pytest.raises(RuntimeError, match="uncertain"):
        await env.step(state, action())
    assert state.context["resolution_count"] == 0


async def test_failed_reset_invalidates_old_episode_and_successful_reset_isolated():
    env = RefundRecoveryEnvironment()
    row = refund_recovery_cases("train", count=10, seed=819)[0]
    old = await env.reset(row)
    with pytest.raises(ValueError):
        await env.reset({})
    with pytest.raises(ValueError):
        await env.step(old, action())
    current = await env.reset({**row, "order_id": "new-order"})
    assert current.context["resolution_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["identity", "prompt", "observation", "action", "truncation", "trailing"],
)
async def test_teacher_replay_rejects_fabrication_or_relabelling(mutation):
    row = refund_recovery_cases("train", count=10, seed=820)[0]
    candidate = copy.deepcopy(await reference_recovery_demonstration(row))
    if mutation == "identity":
        candidate["case_hash"] = "0" * 64
    elif mutation == "prompt":
        candidate["messages"][0]["content"] += " altered"
    elif mutation == "observation":
        candidate["messages"][2]["content"] = "{}"
    elif mutation == "action":
        candidate["messages"][1][
            "content"
        ] = '{"tool":"finish","args":{"outcome":"resolved"}}'
    elif mutation == "truncation":
        candidate["truncated"] = "context"
    else:
        candidate["messages"].append({"role": "assistant", "content": "extra"})
    with pytest.raises(ValueError):
        await replay_recovery_candidate(row, candidate)


def test_reference_refuses_unfenced_or_unrecognized_retry_advice():
    for observation in (
        {"status": "not_applied", "retry_safe": True},
        {"status": "retryable", "retry_safe": 1},
        {"status": "unknown"},
    ):
        with pytest.raises(ValueError):
            reference_recovery_action(observation)


@pytest.mark.parametrize(
    "overrides,policy_tool",
    [
        ({"chargeback_open": True}, "escalate"),
        ({"status": "cancelled"}, "deny"),
    ],
)
async def test_applied_denial_or_escalation_is_resolved_not_unauthorized(
    overrides, policy_tool
):
    row = {**refund_recovery_cases("train", count=10, seed=821)[0], **overrides}
    env = RefundRecoveryEnvironment()
    state = await env.reset(row)
    state, _, _, info = await env.step(state, action())
    observation = json.loads(info["messages"][0]["content"])
    assert observation["status"] == "applied"
    assert observation["receipt"]["action"]["tool"] == policy_tool
    state, reward, done, info = await env.step(
        state, action("finish", outcome="unauthorized")
    )
    assert done and reward == -1 and info["metrics"]["policy_violations"] == 1
    assert (
        info["metrics"]["resolution_count"] == 1
        and info["metrics"]["refunded_cents"] == 0
    )
