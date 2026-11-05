"""River rollout adapter with a distinct intent/execution evidence protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

from stateset_agents.core.environments.refund_delegation_environment import (
    DELEGATION_PROTOCOL,
    RefundDelegationEnvironment,
)
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_environment import (
    river_environment_factory,
    trajectory_environment_trace,
    trajectory_truncation,
)


def river_refund_delegation_factory(
    *,
    before_reset: Callable[[], Awaitable[None] | None] | None = None,
) -> Callable[[], Any]:
    """Create isolated River environments with schema-2 delegation traces.

    Inherit admission, attribution, truncation, and recovery-drop behavior from
    the standard bridge. Legacy schema-1 raw-action traces remain unchanged.
    """
    base: Any = river_environment_factory(
        RefundDelegationEnvironment,
        before_reset=before_reset,
        truncation_reward=-1.0,
        record_trace=True,
    )

    class RiverRefundDelegation(base):
        async def reset(self, row: dict[str, Any]) -> list[dict[str, Any]]:
            messages = await super().reset(row)
            self.trace["schema_version"] = 2
            self.trace["protocol"] = DELEGATION_PROTOCOL
            return cast(list[dict[str, Any]], messages)

        async def on_turn(self, traj: Any) -> list[dict[str, Any]] | None:
            messages = await super().on_turn(traj)
            self.trace["steps"][-1]["execution"] = self.environment.execution_events()
            return cast(list[dict[str, Any]] | None, messages)

    return RiverRefundDelegation


def trajectory_delegation_trace(trajectory: Any) -> dict[str, Any]:
    """Validate attribution and separate executed evidence from raw intent."""
    trace = deepcopy(getattr(trajectory, "stateset_environment_trace", None))
    if (
        not isinstance(trace, dict)
        or type(trace.get("schema_version")) is not int
        or trace["schema_version"] != 2
        or trace.get("protocol") != DELEGATION_PROTOCOL
    ):
        raise ValueError("Missing delegation protocol trace")
    legacy = deepcopy(trace)
    content_hash(trace)
    legacy["schema_version"] = 1
    del legacy["protocol"]
    if not isinstance(legacy.get("steps"), list):
        raise ValueError("Missing delegation steps")
    for step in legacy["steps"]:
        if not isinstance(step, dict) or not isinstance(
            step.pop("execution", None), list
        ):
            raise ValueError("Missing executed action evidence")
    trajectory_environment_trace(
        SimpleNamespace(
            stateset_environment_trace=legacy,
            stateset_episode_id=getattr(trajectory, "stateset_episode_id", None),
            stateset_case_hash=getattr(trajectory, "stateset_case_hash", None),
            reward=getattr(trajectory, "reward", None),
            truncated=trajectory_truncation(trajectory),
        )
    )
    return trace
