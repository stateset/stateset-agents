"""Synthetic SDK result objects for orchestration tests, not learning evidence."""

from types import SimpleNamespace

from stateset_agents.remote.river_environment import trajectory_truncation


def fake_trajectory(**values):
    """Attach explicitly synthetic diagnostics to a mocked SDK result."""
    trajectory = SimpleNamespace(**values)
    cause = trajectory_truncation(trajectory)
    trajectory.stateset_environment_trace = {
        "schema_version": 1,
        "case_id": trajectory.stateset_episode_id,
        "case_hash": trajectory.stateset_case_hash,
        "initial_messages": [{"role": "user", "content": "Synthetic SDK fixture"}],
        "steps": [],
        "terminal": {
            "reward": trajectory.reward,
            "truncated": cause,
            "status": "timeout" if cause else "completed",
        },
    }
    return trajectory


async def traced_refund_trajectory(
    row, *, successful=True, environment="refund-policy-v2", termination=None
):
    """Execute a rule-based policy for audit fixtures; token/time counts are fake."""
    import json
    import sys
    from unittest.mock import patch

    from stateset_agents.core.environments.refund_environment import RefundEnvironment
    from stateset_agents.core.environments.refund_policy_environment import (
        RefundPolicyEnvironment,
    )
    from stateset_agents.data.refund_demonstrations import (
        reference_refund_demonstration,
    )
    from stateset_agents.remote.river_environment import (
        river_environment_factory,
    )

    with patch.dict(
        sys.modules, {"river_client": SimpleNamespace(rl=SimpleNamespace(Env=object))}
    ):
        factory = river_environment_factory(
            (
                RefundPolicyEnvironment
                if environment == "refund-policy-v2"
                else RefundEnvironment
            ),
            record_trace=True,
            truncation_reward=-1.0,
        )
    env = factory()
    trajectory = SimpleNamespace(
        messages=await env.reset(row), metrics={}, truncated=None
    )

    def action(tool, **args):
        return {
            "role": "assistant",
            "content": json.dumps({"tool": tool, "args": args}),
        }

    if termination == "environment_timeout":
        actions = [action("lookup_order", order_id=row["order_id"])] * 4
    elif termination is not None:
        actions = []
    elif not successful:
        actions = [action("finish")]
    elif environment == "refund-policy-v2":
        actions = [
            m
            for m in (await reference_refund_demonstration(row))["messages"]
            if m["role"] == "assistant"
        ]
    else:
        resolution = (
            action("refund", order_id=row["order_id"], amount_cents=row["amount_cents"])
            if row["eligible"]
            else action("deny", order_id=row["order_id"])
        )
        actions = [
            action("lookup_order", order_id=row["order_id"]),
            resolution,
            action("finish"),
        ]
    for action in actions:
        trajectory.messages.append(action)
        observations = await env.on_turn(trajectory)
        if observations is not None:
            trajectory.messages.extend(observations)
    if termination is not None and termination != "environment_timeout":
        trajectory.truncated = termination
        trajectory.reward = await env.on_truncated(trajectory, row, termination)
    else:
        trajectory.reward = await env.reward(trajectory, row)
    return trajectory


async def traced_refund_outcome(row, *, successful=True):
    """Build a sandbox-backed outcome with explicitly synthetic usage counts."""
    from stateset_agents.remote.river_environment import (
        trajectory_case_identity,
        trajectory_environment_trace,
    )

    trajectory = await traced_refund_trajectory(row, successful=successful)
    return {
        **trajectory_case_identity(trajectory, cases={row["order_id"]: row}),
        "environment_trace": trajectory_environment_trace(trajectory),
        "success": trajectory.metrics["task_success"] == 1,
        "reward": trajectory.reward,
        "truncated": None,
        "policy_violations": trajectory.metrics["policy_violations"],
        "tool_calls": trajectory.metrics["tool_calls"],
        "generated_tokens": 10,
        "elapsed_seconds": 0.1,
    }
