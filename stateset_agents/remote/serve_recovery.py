"""Recovery for RunPod endpoints interrupted during startup."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.runpod_recovery import (
    LeaseBusy,
    PodLease,
    RecoveryResult,
    recover_pods,
    session_lease_dir,
    valid_pod_id,
)

__all__ = [
    "ServingLease",
    "ServingRecoveryResult",
    "clear_stopped_lease",
    "recover_serving",
    "serving_lease_dir",
    "valid_pod_id",
]
ServingRecoveryResult = RecoveryResult
_NAME = re.compile(r"stateset-serve-[0-9a-f]{12}")


class ServingLease(PodLease):
    """A serving allocation awaiting endpoint handoff."""

    def pod_id(self) -> str | None:
        """Return the ID RunPod reported, if the allocation response arrived."""
        value = self.read_record()["pod_id"]
        return value if isinstance(value, str) else None


def serving_lease_dir() -> Path:
    """Return the directory for interrupted endpoint startups."""
    return session_lease_dir("serving")


def clear_stopped_lease(
    root: Path, name: str, pod_id: str, *, verified_name: bool
) -> None:
    """Retire matching local evidence after an explicitly acknowledged stop."""
    if not _NAME.fullmatch(name):
        return
    lease = PodLease(root / f"{name}.json")
    if not lease.path.exists():
        return
    try:
        with lease.locked():
            if not lease.path.exists():
                return
            recorded_id = lease.read_record()["pod_id"]
            if recorded_id == pod_id or (recorded_id is None and verified_name):
                lease.clear()
    except LeaseBusy:
        return  # The startup or recovery process still owns its cleanup.
    except (ValueError, OSError) as exc:
        raise RemoteExecutionError(
            f"Pod stopped, but cannot read serving record {lease.path}"
        ) from exc


def recover_serving(
    root: Path, get_api: Callable[[], Any], *, terminate: bool = False
) -> list[RecoveryResult]:
    """Inspect interrupted serving launches and optionally clean up confirmed matches."""
    return recover_pods(root, get_api, kind="serving", terminate=terminate)
