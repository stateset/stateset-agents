"""Offline replay of recorded refund actions, including failures and truncations."""

from __future__ import annotations

import copy
import math
from typing import Any

from stateset_agents.core.environment_base import EpisodeStatus
from stateset_agents.core.environments.refund_environment import RefundEnvironment
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash, validate_report
from stateset_agents.evaluation.checkpoint_selection import select_validation_checkpoint
from stateset_agents.evaluation.implementation import package_implementation

_ENVIRONMENTS = {
    "refund-v1": RefundEnvironment,
    "refund-policy-v2": RefundPolicyEnvironment,
}


async def select_replayed_validation_checkpoint(
    evaluations: Any,
    *,
    steps: int,
    cases: dict[str, Any],
    environment: str,
    truncation_reward: float = -1.0,
) -> dict[str, Any]:
    """Select only after every planned validation reward passes sandbox replay.

    All steps are checked, including checkpoints that would not win selection.
    No test cases or provider calls are used by this selection gate.
    """
    selected = select_validation_checkpoint(evaluations, steps=steps, cases=cases)
    for evaluation in evaluations:
        for outcome in evaluation["outcomes"]:
            case_id = outcome["case_id"]
            try:
                replay = await replay_refund_trace(
                    environment,
                    cases[case_id],
                    outcome.get("environment_trace"),
                    truncation_reward=truncation_reward,
                )
                if replay["reward"] != outcome["reward"]:
                    raise ValueError("Validation reward differs from replay")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"Validation replay failed at step {evaluation['step']}, case {case_id}: {exc}"
                ) from exc
    return selected


def _same(actual: Any, recorded: Any, description: str) -> None:
    if content_hash(actual) != content_hash(recorded):
        raise ValueError(f"Trace replay mismatch: {description}")


async def replay_refund_trace(
    environment: str,
    case: dict[str, Any],
    trace: Any,
    *,
    truncation_reward: float = -1.0,
) -> dict[str, Any]:
    """Re-execute sandbox actions and verify every recorded state transition.

    External truncation is treated as a declared stop, never proof that a token
    or time limit was reached. The configured failure reward must still match.
    No model calls, SDK clients, or real refund services are used.
    """
    if environment not in _ENVIRONMENTS:
        raise ValueError("Unsupported refund trace environment")
    try:
        valid_reward = (
            not isinstance(truncation_reward, bool)
            and isinstance(truncation_reward, (int, float))
            and math.isfinite(truncation_reward)
        )
    except OverflowError:
        valid_reward = False
    if not valid_reward:
        raise ValueError("Truncation reward must be finite")
    if (
        not isinstance(trace, dict)
        or set(trace)
        != {
            "schema_version",
            "case_id",
            "case_hash",
            "initial_messages",
            "steps",
            "terminal",
        }
        or type(trace["schema_version"]) is not int
        or trace["schema_version"] != 1
        or trace["case_id"] != case.get("order_id")
        or trace["case_hash"] != content_hash(case)
    ):
        raise ValueError("Missing trace or mismatched trace case identity")
    content_hash(trace)
    env = _ENVIRONMENTS[environment]()
    state = await env.reset(copy.deepcopy(case))
    _same(state.context["messages"], trace["initial_messages"], "initial messages")
    steps, terminal = trace["steps"], trace["terminal"]
    if not isinstance(steps, list) or len(steps) > env.max_turns:
        raise ValueError("Invalid trace step count")
    if not isinstance(terminal, dict):
        raise ValueError("Missing trace terminal outcome")
    metrics: dict[str, Any] = {
        "task_success": 0,
        "policy_violations": 0,
        "tool_calls": 0,
    }
    total_reward = 0.0
    done = False
    for index, recorded in enumerate(steps):
        if done:
            raise ValueError("Trace contains actions after environment completion")
        action = recorded.get("action") if isinstance(recorded, dict) else None
        if not isinstance(action, dict) or action.get("role") != "assistant":
            raise ValueError("Trace action must be an assistant message")
        state, reward, done, info = await env.step(
            state,
            ConversationTurn(
                role="assistant",
                content=action.get("content", ""),
                tool_calls=copy.deepcopy(action.get("tool_calls")),
            ),
        )
        _same(
            {
                "action": action,
                "reward": reward,
                "done": done,
                "status": state.status.value,
                "observations": info.get("messages", []),
                "metrics": info.get("metrics", {}),
            },
            recorded,
            f"step {index + 1}",
        )
        total_reward += reward
        metrics.update(info.get("metrics", {}))
    cause = terminal.get("truncated", "missing")
    external_stop = not done
    if done:
        expected_cause = (
            "environment_timeout" if state.status == EpisodeStatus.TIMEOUT else None
        )
        if cause != expected_cause:
            raise ValueError("Trace completion and truncation disagree")
    else:
        if (
            not isinstance(cause, str)
            or not cause
            or cause in {"missing", "environment_timeout"}
        ):
            raise ValueError("Incomplete trace requires a declared external stop")
        total_reward = float(truncation_reward)
        state.status = EpisodeStatus.TIMEOUT
        metrics["task_success"] = 0
    _same(
        {"reward": total_reward, "status": state.status.value, "truncated": cause},
        terminal,
        "terminal outcome",
    )
    return {
        "case_id": trace["case_id"],
        "case_hash": trace["case_hash"],
        "reward": total_reward,
        "success": metrics["task_success"] == 1,
        "policy_violations": metrics["policy_violations"],
        "tool_calls": metrics["tool_calls"],
        "truncated": cause,
        "trace_hash": content_hash(trace),
        "external_stop_declared": external_stop,
    }


async def audit_refund_traces(
    report: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check all held-out outcomes against source-matched sandbox trace replay."""
    if not isinstance(report, dict):
        raise ValueError("Trace audit requires a test report object")
    validate_report(report)
    if report["environment"] not in _ENVIRONMENTS:
        raise ValueError("Unsupported refund trace environment")
    implementation = package_implementation()
    if report.get("implementation") != implementation:
        raise ValueError("Trace audit requires the report's exact implementation")
    if not isinstance(cases, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("order_id"), str)
        or not row["order_id"]
        for row in cases
    ):
        raise ValueError("Cases must be a list of identified refund scenarios")
    by_id = {row["order_id"]: row for row in cases}
    if len(by_id) != len(cases) or report["case_hashes"] != {
        key: content_hash(row) for key, row in by_id.items()
    }:
        raise ValueError("Trace audit cases differ from the evaluated cases")
    truncation_reward = report["settings"].get("truncation_reward")
    checked, issues = [], []
    for outcome in report["outcomes"]:
        case_id = outcome["case_id"]
        try:
            replay = await replay_refund_trace(
                report["environment"],
                by_id[case_id],
                outcome.get("environment_trace"),
                truncation_reward=truncation_reward,
            )
            for key in (
                "case_hash",
                "reward",
                "success",
                "policy_violations",
                "tool_calls",
                "truncated",
            ):
                if outcome.get(key) != replay[key]:
                    raise ValueError(f"Reported outcome differs from replay: {key}")
            checked.append(replay)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({"case_id": case_id, "reason": str(exc)})
    return {
        "schema_version": 1,
        "scope": "sandbox_action_consistency",
        "passed": not issues and len(checked) == len(cases),
        "report_hash": content_hash(report),
        "cases_hash": content_hash(cases),
        "implementation": implementation,
        "provider_requests": 0,
        "checked": checked,
        "issues": issues,
        "unverified": [
            "provider_execution",
            "model_identity",
            "generated_tokens",
            "elapsed_seconds",
            "external_truncation_trigger",
        ],
    }
