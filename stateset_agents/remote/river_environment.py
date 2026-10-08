"""Bridge StateSet environments to River's optional native rollout engine."""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash


def river_environment_factory(
    factory: Callable[[], Environment],
    *,
    before_reset: Callable[[], Awaitable[None] | None] | None = None,
    truncation_reward: float = 0.0,
    record_trace: bool = False,
) -> Callable[[], Any]:
    """Create a River Env factory with isolated StateSet state per trajectory.

    StateSet ``reset`` supplies initial messages in ``state.context['messages']``.
    Each nonterminal ``step`` supplies only new observations in ``info['messages']``.
    Step rewards are summed; sparse environments should return their outcome only
    on termination. Infrastructure exceptions propagate and never become rewards.
    Unfinished trajectories are dropped on recovery, avoiding unsafe tool replay.
    An optional admission hook completes before reset or any sampling.
    Hook failures propagate as infrastructure failures, never as rewards.
    Engine truncation receives the configured absolute terminal reward, replacing
    partial rewards. Configure the trainer to preserve it (River defaults to
    zeroing truncated rewards). Environment timeouts retain their own terminal
    reward and are exposed through ``trajectory_truncation`` and a numeric metric.
    Reset inputs must be canonical-JSON serializable. Each reset takes a private
    snapshot before admission and records its hash on the resulting trajectory,
    including trajectories truncated before their first assistant turn.
    With ``record_trace``, retain the sandbox's initial messages and every
    executed action, observation, metric and reward on the finished trajectory.
    Observations are what the sandbox produced; the engine may stop before
    delivering a final observation to the model. Trace data must be JSON-safe.
    """
    if type(record_trace) is not bool:
        raise ValueError("record_trace must be boolean")
    if (
        isinstance(truncation_reward, bool)
        or not isinstance(truncation_reward, (int, float))
        or not math.isfinite(truncation_reward)
    ):
        raise ValueError("truncation_reward must be a finite number")
    try:
        from river_client import rl
    except ImportError as exc:
        raise ImportError("Install stateset-agents[river] on Python >=3.12") from exc

    class StateSetRiverEnv(rl.Env):
        recovery = "drop"

        def __init__(self) -> None:
            self.environment = factory()
            self.state: EnvironmentState | None = None
            self.total_reward = 0.0
            self.done = False
            self.case_hash: str | None = None
            self.trace: dict[str, Any] | None = None

        async def reset(self, row: dict[str, Any]) -> list[dict[str, Any]]:
            # A failed reset must not leave the previous episode usable.
            self.state, self.case_hash = None, None
            self.trace = None
            self.total_reward, self.done = 0.0, False
            scenario = deepcopy(row)
            case_hash = content_hash(scenario)
            if before_reset is not None:
                admission = before_reset()
                if inspect.isawaitable(admission):
                    await admission
            state = await self.environment.reset(scenario)
            messages = state.context.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("StateSet reset must provide context['messages']")
            self.state, self.case_hash = state, case_hash
            if record_trace:
                self.trace = {
                    "schema_version": 1,
                    "case_id": state.episode_id,
                    "case_hash": case_hash,
                    "initial_messages": deepcopy(messages),
                    "steps": [],
                }
            return list(messages)

        async def on_turn(self, traj: Any) -> list[dict[str, Any]] | None:
            if self.state is None or self.done:
                raise RuntimeError("Environment must be reset before stepping")
            traj.stateset_episode_id = self.state.episode_id
            traj.stateset_case_hash = self.case_hash
            message = traj.messages[-1]
            if message.get("role") != "assistant":
                raise ValueError("Expected an assistant action")
            trace_action = deepcopy(message) if self.trace is not None else None
            action = ConversationTurn(
                role="assistant",
                content=message.get("content", ""),
                tool_calls=message.get("tool_calls"),
            )
            self.state, reward, self.done, info = await self.environment.step(
                self.state, action
            )
            if not math.isfinite(reward):
                raise ValueError("Environment reward must be finite")
            self.total_reward += reward
            if self.trace is not None:
                self.trace["steps"].append(
                    {
                        "action": trace_action,
                        "reward": reward,
                        "done": self.done,
                        "status": self.state.status.value,
                        "observations": deepcopy(info.get("messages", [])),
                        "metrics": deepcopy(info.get("metrics", {})),
                    }
                )
            for key, value in info.get("metrics", {}).items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    traj.metrics[key] = float(value)
            if self.state.status == EpisodeStatus.TIMEOUT:
                traj.stateset_truncated = "environment_timeout"
                traj.metrics["environment_timeout"] = 1.0
                traj.metrics["task_success"] = 0.0
            if self.done:
                return None
            messages = info.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    "Nonterminal StateSet step must provide info['messages']"
                )
            return list(messages)

        async def reward(self, traj: Any, row: dict[str, Any]) -> float:
            if not self.done:
                raise RuntimeError(
                    "Final reward requested before environment termination"
                )
            self._finish_trace(traj)
            return self.total_reward

        async def on_truncated(
            self, traj: Any, row: dict[str, Any], cause: str
        ) -> float:
            if self.state is None:
                raise RuntimeError("Environment must be reset before truncation")
            traj.stateset_episode_id = self.state.episode_id
            traj.stateset_case_hash = self.case_hash
            traj.stateset_truncated = cause
            traj.metrics["task_success"] = 0.0
            self.state.status = EpisodeStatus.TIMEOUT
            self.done = True
            self.total_reward = float(truncation_reward)
            self._finish_trace(traj)
            return self.total_reward

        def _finish_trace(self, traj: Any) -> None:
            if self.trace is None:
                return
            assert self.state is not None
            trace = deepcopy(self.trace)
            trace["terminal"] = {
                "reward": self.total_reward,
                "status": self.state.status.value,
                "truncated": trajectory_truncation(traj),
            }
            content_hash(trace)  # Reject non-JSON or nonfinite diagnostic evidence.
            traj.stateset_environment_trace = trace

        async def close(self) -> None:
            close = getattr(self.environment, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

    return StateSetRiverEnv


def trajectory_case_identity(
    trajectory: Any, *, cases: dict[str, Any]
) -> dict[str, str]:
    """Verify a rollout's reset input against the planned case before accepting it.

    This records local input provenance, not independent proof that a provider
    executed the declared model or that an environment implemented its rules.
    """
    case_id = getattr(trajectory, "stateset_episode_id", None)
    case_hash = getattr(trajectory, "stateset_case_hash", None)
    if (
        not isinstance(case_id, str)
        or case_id not in cases
        or not isinstance(case_hash, str)
        or case_hash != content_hash(cases[case_id])
    ):
        raise ValueError("Trajectory reset input does not match planned case")
    return {"case_id": case_id, "case_hash": case_hash}


def trajectory_environment_trace(trajectory: Any) -> dict[str, Any]:
    """Return a detached completed sandbox trace bound to this rollout's outcome.

    This checks attribution and the terminal reward, not replay correctness.
    It deliberately fails when tracing was disabled or evidence was lost.
    """
    trace = getattr(trajectory, "stateset_environment_trace", None)
    terminal = trace.get("terminal") if isinstance(trace, dict) else None
    if (
        not isinstance(trace, dict)
        or type(trace.get("schema_version")) is not int
        or trace["schema_version"] != 1
        or not isinstance(trace.get("case_id"), str)
        or not trace["case_id"]
        or not isinstance(trace.get("case_hash"), str)
        or len(trace["case_hash"]) != 64
        or any(c not in "0123456789abcdef" for c in trace["case_hash"])
        or trace.get("case_id") != getattr(trajectory, "stateset_episode_id", None)
        or trace.get("case_hash") != getattr(trajectory, "stateset_case_hash", None)
        or not isinstance(trace.get("initial_messages"), list)
        or not trace["initial_messages"]
        or not isinstance(trace.get("steps"), list)
        or not isinstance(terminal, dict)
        or isinstance(terminal.get("reward"), bool)
        or not isinstance(terminal.get("reward"), (int, float))
        or not isinstance(terminal.get("status"), str)
        or not terminal["status"]
        or terminal.get("reward") != getattr(trajectory, "reward", None)
        or terminal.get("truncated", "missing") != trajectory_truncation(trajectory)
    ):
        raise ValueError("Missing or inconsistent completed environment trace")
    for step in trace["steps"]:
        if (
            not isinstance(step, dict)
            or not isinstance(step.get("action"), dict)
            or step["action"].get("role") != "assistant"
            or isinstance(step.get("reward"), bool)
            or not isinstance(step.get("reward"), (int, float))
            or type(step.get("done")) is not bool
            or not isinstance(step.get("status"), str)
            or not step["status"]
            or not isinstance(step.get("observations"), list)
            or not isinstance(step.get("metrics"), dict)
        ):
            raise ValueError("Invalid environment trace step")
    content_hash(trace)
    return deepcopy(trace)


def trajectory_truncation(trajectory: Any) -> str | None:
    """Return an engine or environment truncation cause without rewriting SDK state.

    An environment can terminate on its own turn limit before the engine's
    limit. Its timeout must still count in reports and exclude SFT candidates.
    The custom field is retained by River's trajectory checkpoint serialization.
    """
    for cause in (
        getattr(trajectory, "truncated", None),
        getattr(trajectory, "stateset_truncated", None),
    ):
        if cause is not None:
            if not isinstance(cause, str) or not cause:
                raise ValueError("Trajectory truncation cause must be nonempty text")
            return cause
    return None
