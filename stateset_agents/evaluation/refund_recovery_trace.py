"""Replay recovery intents, injected faults, backend commits, and receipts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from stateset_agents.core.environment_base import EpisodeStatus
from stateset_agents.core.environments.refund_recovery_environment import (
    RECOVERY_PROTOCOL,
    RefundRecoveryEnvironment,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash


async def replay_recovery_trace(case: dict[str, Any], trace: Any) -> dict[str, Any]:
    """Re-execute all recorded knowledge transitions and financial effect evidence.

    Deterministic injected faults are task fixtures. An external truncation is
    only a declared stop. Real service behavior and payment persistence are not
    attested by local replay. No SDK or provider calls are used here.
    """
    if (
        not isinstance(trace, dict)
        or set(trace)
        != {
            "schema_version",
            "protocol",
            "case_id",
            "case_hash",
            "initial_messages",
            "steps",
            "terminal",
        }
        or type(trace["schema_version"]) is not int
        or trace["schema_version"] != 3
        or trace["protocol"] != RECOVERY_PROTOCOL
        or trace["case_id"] != case.get("order_id")
        or trace["case_hash"] != content_hash(case)
    ):
        raise ValueError("Missing trace or mismatched recovery identity")
    content_hash(trace)

    def same(actual: Any, expected: Any) -> None:
        if content_hash(actual) != content_hash(expected):
            raise ValueError("Recovery trace replay mismatch")

    env = RefundRecoveryEnvironment()
    state = await env.reset(deepcopy(case))
    same(state.context["messages"], trace["initial_messages"])
    if not isinstance(trace["steps"], list) or len(trace["steps"]) > env.max_turns:
        raise ValueError("Invalid recovery step count")
    metrics: dict[str, Any] = {
        "task_success": 0,
        "policy_violations": 0,
        "tool_calls": 0,
        "backend_calls": 0,
        "resolution_count": 0,
        "refunded_cents": 0,
        "duplicate_attempts": 0,
        "reconciliations": 0,
    }
    reward, done = 0.0, False
    for step in trace["steps"]:
        action = step.get("action") if isinstance(step, dict) else None
        if done or not isinstance(action, dict) or action.get("role") != "assistant":
            raise ValueError("Invalid recovery action or action after termination")
        state, increment, done, info = await env.step(
            state,
            ConversationTurn(
                role="assistant",
                content=action.get("content", ""),
                tool_calls=deepcopy(action.get("tool_calls")),
            ),
        )
        same(
            {
                "action": action,
                "reward": increment,
                "done": done,
                "status": state.status.value,
                "observations": info["messages"],
                "metrics": info["metrics"],
                "execution": env.execution_events(),
            },
            step,
        )
        reward += increment
        metrics.update(info["metrics"])
    terminal = trace["terminal"]
    if not isinstance(terminal, dict):
        raise ValueError("Missing recovery terminal outcome")
    cause, external_stop = terminal.get("truncated", "missing"), not done
    if done:
        if cause != (
            "environment_timeout" if state.status == EpisodeStatus.TIMEOUT else None
        ):
            raise ValueError("Recovery completion and truncation disagree")
    else:
        if (
            not isinstance(cause, str)
            or not cause
            or cause in {"missing", "environment_timeout"}
        ):
            raise ValueError("Incomplete recovery needs a declared external stop")
        reward, state.status, metrics["task_success"] = -1.0, EpisodeStatus.TIMEOUT, 0
    same({"reward": reward, "status": state.status.value, "truncated": cause}, terminal)
    return {
        "case_id": trace["case_id"],
        "case_hash": trace["case_hash"],
        "reward": reward,
        "success": metrics["task_success"] == 1,
        "truncated": cause,
        **{key: value for key, value in metrics.items() if key != "task_success"},
        "trace_hash": content_hash(trace),
        "external_stop_declared": external_stop,
    }
