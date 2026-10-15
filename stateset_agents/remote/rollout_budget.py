"""Durable, conservative admission limits for native rollout trajectories."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from stateset_agents.remote.river_rl import atomic_json


class RolloutBudgetExceeded(ValueError):
    """A trajectory would exceed the reserved output-token allowance."""


class RolloutAdmissionBudget:
    """Reserve full trajectory allowances before any sampling can start.

    Reservations are never refunded, including after errors, cancellation, or
    recovery. This bounds admitted output generation under the rollout engine's
    per-trajectory contract, not provider billing, input tokens or optimizer work.
    The caller must hold the run directory's exclusive process lock. One shared
    instance serializes admission across its threads and all rollout engines.
    """

    def __init__(
        self,
        path: Path,
        *,
        limit: int,
        trajectory_tokens: int,
        fingerprint: str,
        create: bool = False,
    ) -> None:
        for name, value in (("limit", limit), ("trajectory_tokens", trajectory_tokens)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Budget requires a run fingerprint")
        if limit < trajectory_tokens:
            raise ValueError("Budget must admit at least one full trajectory")
        self.path = path
        self.limit = limit
        self.trajectory_tokens = trajectory_tokens
        self.fingerprint = fingerprint
        self._lock = threading.Lock()
        if create:
            if path.exists():
                raise ValueError("Refusing to reset an existing admission budget")
            atomic_json(
                path,
                {
                    **self._identity(),
                    "admitted_trajectories": 0,
                    "reserved_generated_tokens": 0,
                },
            )
        self.snapshot()

    def _identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fingerprint": self.fingerprint,
            "limit": self.limit,
            "trajectory_tokens": self.trajectory_tokens,
        }

    def _read(self) -> dict[str, Any]:
        state = json.loads(self.path.read_text())
        if not isinstance(state, dict) or any(
            state.get(key) != value for key, value in self._identity().items()
        ):
            raise ValueError("Admission budget belongs to a different run or limit")
        admitted = state.get("admitted_trajectories")
        reserved = state.get("reserved_generated_tokens")
        if (
            type(admitted) is not int
            or admitted < 0
            or type(reserved) is not int
            or reserved < 0
            or reserved != admitted * self.trajectory_tokens
            or reserved > self.limit
        ):
            raise ValueError("Invalid admission budget counters")
        return state

    def snapshot(self) -> dict[str, Any]:
        """Read and validate persisted reservations; never recreate missing state."""
        with self._lock:
            return self._read()

    def reserve(self) -> None:
        """Durably reserve one trajectory, or raise before environment reset."""
        with self._lock:
            state = self._read()
            reserved = state["reserved_generated_tokens"]
            if reserved + self.trajectory_tokens > self.limit:
                raise RolloutBudgetExceeded(
                    f"Rollout admission budget exhausted: {reserved} + "
                    f"{self.trajectory_tokens} > {self.limit}"
                )
            state["admitted_trajectories"] += 1
            state["reserved_generated_tokens"] += self.trajectory_tokens
            atomic_json(self.path, state)

    def ensure_available(self) -> None:
        """Reject an exhausted run before opening provider sessions."""
        with self._lock:
            state = self._read()
            if state["reserved_generated_tokens"] + self.trajectory_tokens > self.limit:
                raise RolloutBudgetExceeded("Rollout admission budget exhausted")
