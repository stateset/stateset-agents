"""Replay local sandbox traces without calling a model or provider."""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Callable
from typing import Any

from stateset_agents.core.environment_base import Environment, EpisodeStatus
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash


def _same(actual: Any, recorded: Any, description: str) -> None:
    if content_hash(actual) != content_hash(recorded):
        raise ValueError(f"Trace replay mismatch: {description}")


async def replay_environment_trace(
    factory: Callable[[], Environment],
    case: dict[str, Any],
    trace: Any,
    *,
    case_id: str,
    truncation_reward: float = -1.0,
) -> dict[str, Any]:
    """Verify each transition against a fresh, isolated sandbox instance.

    The caller supplies the trusted environment factory and planned case ID.
    This verifies consistency with that sandbox, not provider execution, model
    identity, business correctness, or the claimed trigger for an external stop.
    The sandbox's optional close hook runs even when reset or replay fails.
    """
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
        not isinstance(case, dict)
        or not isinstance(case_id, str)
        or not case_id
        or not isinstance(trace, dict)
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
        or trace["case_id"] != case_id
        or trace["case_hash"] != content_hash(case)
    ):
        raise ValueError("Missing trace or mismatched trace case identity")
    content_hash(trace)
    env = factory()
    try:
        state = await env.reset(copy.deepcopy(case))
        if state.episode_id != case_id:
            raise ValueError("Sandbox reset differs from planned case identity")
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
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                raise ValueError("Environment reward must be finite")
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
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            closed = close()
            if inspect.isawaitable(closed):
                await closed


async def replay_industry_trace(
    domain: str,
    case: dict[str, Any],
    trace: Any,
    *,
    truncation_reward: float = -1.0,
) -> dict[str, Any]:
    """Replay a logistics, SaaS, travel, services, or telecom policy trace.

    These are fictional policies. Successful replay does not establish
    production expertise or independently reviewed industry quality.
    """
    from stateset_agents.core.environments.industry_support_environment import (
        POLICIES,
        IndustrySupportEnvironment,
    )

    if not isinstance(domain, str) or domain not in POLICIES:
        raise ValueError("Unsupported industry trace environment")
    if not isinstance(case, dict):
        raise ValueError("Industry trace requires a scenario object")
    identity = case.get("record_id")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Missing trace or mismatched trace case identity")
    return await replay_environment_trace(
        lambda: IndustrySupportEnvironment(domain),
        case,
        trace,
        case_id=identity,
        truncation_reward=truncation_reward,
    )
