"""Native River adapter for explicit synthetic refund recovery decisions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

from stateset_agents.core.environments.refund_recovery_environment import (
    RECOVERY_PROTOCOL,
    RefundRecoveryEnvironment,
)
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_environment import (
    river_environment_factory,
    trajectory_environment_trace,
    trajectory_truncation,
)


def river_refund_recovery_factory(
    *,
    before_reset: Callable[[], Awaitable[None] | None] | None = None,
) -> Callable[[], Any]:
    """Capture schema-3 recovery traces without rewriting earlier protocols."""
    base: Any = river_environment_factory(
        RefundRecoveryEnvironment,
        before_reset=before_reset,
        truncation_reward=-1.0,
        record_trace=True,
    )

    class RiverRefundRecovery(base):
        async def reset(self, row: dict[str, Any]) -> list[dict[str, Any]]:
            messages = await super().reset(row)
            self.trace["schema_version"] = 3
            self.trace["protocol"] = RECOVERY_PROTOCOL
            return cast(list[dict[str, Any]], messages)

        async def on_turn(self, traj: Any) -> list[dict[str, Any]] | None:
            messages = await super().on_turn(traj)
            self.trace["steps"][-1]["execution"] = self.environment.execution_events()
            return cast(list[dict[str, Any]] | None, messages)

    return RiverRefundRecovery


def trajectory_recovery_trace(trajectory: Any) -> dict[str, Any]:
    """Require recovery attribution and separated backend/effect evidence."""
    trace = deepcopy(getattr(trajectory, "stateset_environment_trace", None))
    if (
        not isinstance(trace, dict)
        or type(trace.get("schema_version")) is not int
        or trace["schema_version"] != 3
        or trace.get("protocol") != RECOVERY_PROTOCOL
    ):
        raise ValueError("Missing recovery protocol trace")
    content_hash(trace)
    legacy = deepcopy(trace)
    legacy["schema_version"] = 1
    del legacy["protocol"]
    if not isinstance(legacy.get("steps"), list):
        raise ValueError("Missing recovery steps")
    for step in legacy["steps"]:
        if not isinstance(step, dict) or not isinstance(
            step.pop("execution", None), list
        ):
            raise ValueError("Missing recovery backend evidence")
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
