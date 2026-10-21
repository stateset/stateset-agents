"""Offline audit of model intents and their separately executed tool actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from stateset_agents.core.environment_base import EpisodeStatus
from stateset_agents.core.environments.refund_delegation_environment import (
    DELEGATION_PROTOCOL,
    RefundDelegationEnvironment,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash


async def replay_delegation_trace(case: dict[str, Any], trace: Any) -> dict[str, Any]:
    """Replay all intent, execution, observation, and terminal evidence locally.

    External truncation is a declared stop, not an attestation of provider limits.
    No SDK, model, credentials, or live payment backend is used by replay.
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
        or trace["schema_version"] != 2
        or trace["protocol"] != DELEGATION_PROTOCOL
        or trace["case_id"] != case.get("order_id")
        or trace["case_hash"] != content_hash(case)
    ):
        raise ValueError("Missing trace or mismatched delegation identity")
    content_hash(trace)

    def same(actual: Any, expected: Any) -> None:
        if content_hash(actual) != content_hash(expected):
            raise ValueError("Delegation trace replay mismatch")

    env = RefundDelegationEnvironment()
    state = await env.reset(deepcopy(case))
    same(state.context["messages"], trace["initial_messages"])
    if not isinstance(trace["steps"], list) or len(trace["steps"]) > env.max_turns:
        raise ValueError("Invalid delegation step count")
    metrics: dict[str, Any] = {
        "task_success": 0,
        "policy_violations": 0,
        "tool_calls": 0,
        "executed_tool_calls": 0,
        "duplicate_intents": 0,
    }
    reward, done = 0.0, False
    for step in trace["steps"]:
        action = step.get("action") if isinstance(step, dict) else None
        if done or not isinstance(action, dict) or action.get("role") != "assistant":
            raise ValueError("Invalid delegation action or action after termination")
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
        raise ValueError("Missing delegation terminal outcome")
    cause = terminal.get("truncated", "missing")
    external_stop = not done
    if done:
        if cause != (
            "environment_timeout" if state.status == EpisodeStatus.TIMEOUT else None
        ):
            raise ValueError("Delegation completion and truncation disagree")
    else:
        if (
            not isinstance(cause, str)
            or not cause
            or cause in {"missing", "environment_timeout"}
        ):
            raise ValueError("Incomplete delegation requires declared external stop")
        reward, state.status, metrics["task_success"] = -1.0, EpisodeStatus.TIMEOUT, 0
    same({"reward": reward, "status": state.status.value, "truncated": cause}, terminal)
    return {
        "case_id": trace["case_id"],
        "case_hash": trace["case_hash"],
        "reward": reward,
        "success": metrics["task_success"] == 1,
        "policy_violations": metrics["policy_violations"],
        "tool_calls": metrics["tool_calls"],
        "executed_tool_calls": metrics["executed_tool_calls"],
        "duplicate_intents": metrics["duplicate_intents"],
        "truncated": cause,
        "trace_hash": content_hash(trace),
        "external_stop_declared": external_stop,
    }
