"""Transport-neutral worker leases for distributed asynchronous rollouts.

This module is the control-plane boundary between remote rollout workers and
the policy-versioned :mod:`async_rollouts` queue.  It deliberately contains no
HTTP, Ray, or cloud-provider dependency: transports authenticate callers, then
delegate registration, heartbeat, and submission to this shared contract.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from .async_rollouts import AsyncRolloutCoordinator, AsyncRolloutError, RolloutRecord
from .policy_artifacts import (
    PolicyArtifact,
    PolicyArtifactError,
    PolicyArtifactUnavailable,
)


class WorkerLeaseError(AsyncRolloutError):
    """Base error for invalid or inactive distributed-worker leases."""


class WorkerLeaseExpired(WorkerLeaseError):
    """Raised when a worker uses a lease after its deadline."""


class WorkerCapacityError(WorkerLeaseError):
    """Raised when the configured number of live workers is exhausted."""


@dataclass(frozen=True)
class DistributedRolloutConfig:
    """Limits for the remote rollout-worker control plane."""

    lease_ttl_seconds: float = 30.0
    max_workers: int = 1_024
    worker_history_capacity: int = 1_000_000
    max_worker_id_length: int = 256
    policy_artifact_capacity: int = 64
    require_policy_artifact: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.lease_ttl_seconds, bool)
            or not isinstance(self.lease_ttl_seconds, (int, float))
            or not math.isfinite(float(self.lease_ttl_seconds))
            or self.lease_ttl_seconds <= 0
        ):
            raise ValueError("lease_ttl_seconds must be finite and positive")
        for name in (
            "max_workers",
            "worker_history_capacity",
            "max_worker_id_length",
            "policy_artifact_capacity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.worker_history_capacity < self.max_workers:
            raise ValueError("worker_history_capacity must be at least max_workers")
        if not isinstance(self.require_policy_artifact, bool):
            raise ValueError("require_policy_artifact must be bool")


@dataclass(frozen=True)
class WorkerLease:
    """A renewable assignment to sample one exact policy snapshot."""

    worker_id: str
    lease_id: str
    generation: int
    policy_version: int
    issued_at: float
    expires_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if not isinstance(self.lease_id, str) or not self.lease_id.strip():
            raise ValueError("lease_id must be a non-empty string")
        for name in ("generation", "policy_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("issued_at", "expires_at"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a stable checkpoint representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerLease:
        """Restore a validated lease from checkpoint state."""
        expected = {
            "worker_id",
            "lease_id",
            "generation",
            "policy_version",
            "issued_at",
            "expires_at",
        }
        if set(value) != expected:
            raise ValueError("worker lease fields do not match schema")
        return cls(**dict(value))


@dataclass(frozen=True)
class WorkerHealth:
    """Non-secret worker state suitable for metrics and health endpoints."""

    worker_id: str
    generation: int
    policy_version: int
    lease_seconds_remaining: float


@dataclass(frozen=True)
class DistributedRolloutStats:
    """Monotonic control-plane counters and current worker capacity."""

    active_workers: int
    total_registrations: int
    replaced_leases: int
    expired_leases: int
    rejected_unknown_worker: int
    rejected_expired_lease: int
    rejected_stale_lease: int
    rejected_policy_assignment: int
    accepted_submissions: int

    def to_dict(self) -> dict[str, int]:
        """Return a stable JSON-compatible metrics document."""
        return asdict(self)


_COUNTER_NAMES = {
    "total_registrations",
    "replaced_leases",
    "expired_leases",
    "rejected_unknown_worker",
    "rejected_expired_lease",
    "rejected_stale_lease",
    "rejected_policy_assignment",
    "accepted_submissions",
}


class DistributedRolloutControlPlane:
    """Lease remote workers and safely admit their policy-versioned samples.

    A repeated registration for the same ``worker_id`` creates a new
    generation and immediately fences the old lease.  Heartbeats renew the
    lease and assign the coordinator's current policy version.  A rollout must
    match that exact assignment before it reaches the existing staleness and
    deduplication checks in :class:`AsyncRolloutCoordinator`.
    """

    def __init__(
        self,
        *,
        coordinator: AsyncRolloutCoordinator,
        config: DistributedRolloutConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(coordinator, AsyncRolloutCoordinator):
            raise TypeError("coordinator must be AsyncRolloutCoordinator")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.coordinator = coordinator
        self.config = config or DistributedRolloutConfig()
        self._clock = clock
        self._condition = asyncio.Condition()
        self._leases: dict[str, WorkerLease] = {}
        self._generations: dict[str, int] = {}
        self._policy_artifacts: dict[int, PolicyArtifact] = {}
        self._counters = dict.fromkeys(_COUNTER_NAMES, 0)
        self._inflight_submissions = 0
        self._checkpointing = False

    def _now(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise WorkerLeaseError("clock must return a finite non-negative value")
        return float(value)

    def _expire_locked(self, now: float) -> tuple[str, ...]:
        expired = tuple(
            sorted(
                worker_id
                for worker_id, lease in self._leases.items()
                if lease.expires_at <= now
            )
        )
        for worker_id in expired:
            del self._leases[worker_id]
        self._counters["expired_leases"] += len(expired)
        return expired

    def _require_locked(self, worker_id: str, lease_id: str, now: float) -> WorkerLease:
        lease = self._leases.get(worker_id)
        if lease is None:
            self._counters["rejected_unknown_worker"] += 1
            raise WorkerLeaseError(f"worker {worker_id!r} is not registered")
        if lease.expires_at <= now:
            del self._leases[worker_id]
            self._counters["expired_leases"] += 1
            self._counters["rejected_expired_lease"] += 1
            raise WorkerLeaseExpired(f"worker {worker_id!r} lease has expired")
        if lease.lease_id != lease_id:
            self._counters["rejected_stale_lease"] += 1
            raise WorkerLeaseError(f"worker {worker_id!r} lease is no longer current")
        return lease

    def _require_current_artifact_locked(self) -> None:
        if (
            self.config.require_policy_artifact
            and self.coordinator.current_policy_version not in self._policy_artifacts
        ):
            raise PolicyArtifactUnavailable(
                "no policy artifact is published for the current policy version"
            )

    def _prune_policy_artifacts_locked(self) -> None:
        while len(self._policy_artifacts) > self.config.policy_artifact_capacity:
            oldest_version = min(self._policy_artifacts)
            del self._policy_artifacts[oldest_version]

    async def register_initial_policy_artifact(
        self, artifact: PolicyArtifact
    ) -> PolicyArtifact:
        """Publish immutable weights for the coordinator's initial version."""
        if not isinstance(artifact, PolicyArtifact):
            raise TypeError("artifact must be PolicyArtifact")
        async with self._condition:
            current = self.coordinator.current_policy_version
            if artifact.policy_version != current:
                raise PolicyArtifactError(
                    "initial artifact version must equal the current policy version"
                )
            existing = self._policy_artifacts.get(current)
            if existing is not None and existing != artifact:
                raise PolicyArtifactError(
                    "a different artifact is already published for this policy version"
                )
            self._policy_artifacts[current] = artifact
            self._prune_policy_artifacts_locked()
            return artifact

    async def publish_policy_artifact(self, artifact: PolicyArtifact) -> PolicyArtifact:
        """Record verified weights, then atomically expose their next version."""
        if not isinstance(artifact, PolicyArtifact):
            raise TypeError("artifact must be PolicyArtifact")
        async with self._condition:
            expected = self.coordinator.current_policy_version + 1
            if artifact.policy_version != expected:
                raise PolicyArtifactError(
                    "published artifact version must be the next policy version"
                )
            existing = self._policy_artifacts.get(expected)
            if existing is not None and existing != artifact:
                raise PolicyArtifactError(
                    "a different artifact is already staged for this policy version"
                )

            # Heartbeats share this condition, so no worker can observe the new
            # version between artifact registration and coordinator advancement.
            self._policy_artifacts[expected] = artifact
            try:
                await self.coordinator.advance_policy(expected)
            except BaseException:
                if existing is None:
                    self._policy_artifacts.pop(expected, None)
                raise
            self._prune_policy_artifacts_locked()
            return artifact

    async def policy_artifact(
        self, policy_version: int | None = None
    ) -> PolicyArtifact | None:
        """Return the immutable descriptor for one policy version, if retained."""
        async with self._condition:
            version = (
                self.coordinator.current_policy_version
                if policy_version is None
                else policy_version
            )
            if isinstance(version, bool) or not isinstance(version, int) or version < 0:
                raise ValueError("policy_version must be a non-negative integer")
            return self._policy_artifacts.get(version)

    async def register(self, worker_id: str) -> WorkerLease:
        """Register or fence-and-replace a worker, returning its new lease."""
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if len(worker_id) > self.config.max_worker_id_length:
            raise ValueError("worker_id exceeds max_worker_id_length")
        now = self._now()
        async with self._condition:
            self._expire_locked(now)
            self._require_current_artifact_locked()
            previous = self._leases.get(worker_id)
            if previous is None and len(self._leases) >= self.config.max_workers:
                raise WorkerCapacityError("distributed rollout worker capacity reached")
            if (
                worker_id not in self._generations
                and len(self._generations) >= self.config.worker_history_capacity
            ):
                raise WorkerCapacityError("worker generation history capacity reached")
            generation = self._generations.get(worker_id, -1) + 1
            lease = WorkerLease(
                worker_id=worker_id,
                lease_id=uuid.uuid4().hex,
                generation=generation,
                policy_version=self.coordinator.current_policy_version,
                issued_at=now,
                expires_at=now + float(self.config.lease_ttl_seconds),
            )
            self._leases[worker_id] = lease
            self._generations[worker_id] = generation
            self._counters["total_registrations"] += 1
            if previous is not None:
                self._counters["replaced_leases"] += 1
            return lease

    async def heartbeat(self, worker_id: str, lease_id: str) -> WorkerLease:
        """Renew a live lease and return the latest policy assignment."""
        now = self._now()
        async with self._condition:
            lease = self._require_locked(worker_id, lease_id, now)
            self._require_current_artifact_locked()
            renewed = replace(
                lease,
                policy_version=self.coordinator.current_policy_version,
                issued_at=now,
                expires_at=now + float(self.config.lease_ttl_seconds),
            )
            self._leases[worker_id] = renewed
            return renewed

    async def submit(
        self,
        worker_id: str,
        lease_id: str,
        record: RolloutRecord,
        *,
        timeout_seconds: float | None = None,
    ) -> bool:
        """Validate a worker assignment, then submit through the rollout queue."""
        if not isinstance(record, RolloutRecord):
            raise TypeError("record must be RolloutRecord")
        async with self._condition:
            await self._condition.wait_for(lambda: not self._checkpointing)
            now = self._now()
            lease = self._require_locked(worker_id, lease_id, now)
            if record.policy_version != lease.policy_version:
                self._counters["rejected_policy_assignment"] += 1
                raise WorkerLeaseError(
                    "rollout policy version does not match the worker assignment"
                )
            artifact = self._policy_artifacts.get(lease.policy_version)
            if artifact is not None:
                if record.policy_artifact_sha256 is None:
                    if self.config.require_policy_artifact:
                        raise PolicyArtifactError(
                            "rollout must identify its assigned policy artifact"
                        )
                elif record.policy_artifact_sha256 != artifact.sha256:
                    raise PolicyArtifactError(
                        "rollout policy artifact does not match the worker assignment"
                    )
            self._inflight_submissions += 1

        accepted = False
        try:
            accepted = await self.coordinator.submit(
                record, timeout_seconds=timeout_seconds
            )
            return accepted
        finally:
            async with self._condition:
                self._inflight_submissions -= 1
                if accepted:
                    self._counters["accepted_submissions"] += 1
                self._condition.notify_all()

    async def unregister(self, worker_id: str, lease_id: str) -> None:
        """Release a live worker lease; stale owners cannot release replacements."""
        now = self._now()
        async with self._condition:
            self._require_locked(worker_id, lease_id, now)
            del self._leases[worker_id]

    async def reap_expired(self) -> tuple[str, ...]:
        """Remove expired workers and return their stable worker identifiers."""
        now = self._now()
        async with self._condition:
            return self._expire_locked(now)

    async def health(self) -> tuple[WorkerHealth, ...]:
        """Return deterministic, lease-ID-free health data for live workers."""
        now = self._now()
        async with self._condition:
            self._expire_locked(now)
            return tuple(
                WorkerHealth(
                    worker_id=lease.worker_id,
                    generation=lease.generation,
                    policy_version=lease.policy_version,
                    lease_seconds_remaining=max(0.0, lease.expires_at - now),
                )
                for lease in sorted(
                    self._leases.values(), key=lambda item: item.worker_id
                )
            )

    async def stats(self) -> DistributedRolloutStats:
        """Return control-plane counters after reaping expired workers."""
        now = self._now()
        async with self._condition:
            self._expire_locked(now)
            return DistributedRolloutStats(
                active_workers=len(self._leases), **self._counters
            )

    async def state_dict(self) -> dict[str, Any]:
        """Checkpoint worker fencing state together with the rollout queue."""
        async with self._condition:
            await self._condition.wait_for(lambda: not self._checkpointing)
            self._checkpointing = True
            try:
                await self._condition.wait_for(lambda: self._inflight_submissions == 0)
                now = self._now()
                self._expire_locked(now)
                coordinator_state = await self.coordinator.state_dict()
                return {
                    "schema_version": 2,
                    "config": asdict(self.config),
                    "leases": [
                        lease.to_dict()
                        for lease in sorted(
                            self._leases.values(), key=lambda item: item.worker_id
                        )
                    ],
                    "generations": dict(sorted(self._generations.items())),
                    "counters": dict(self._counters),
                    "policy_artifacts": [
                        artifact.to_dict()
                        for artifact in sorted(
                            self._policy_artifacts.values(),
                            key=lambda item: item.policy_version,
                        )
                    ],
                    "coordinator": coordinator_state,
                }
            finally:
                self._checkpointing = False
                self._condition.notify_all()

    @classmethod
    async def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        clock: Callable[[], float] = time.time,
    ) -> DistributedRolloutControlPlane:
        """Restore state, retaining only leases that remain live at restart."""
        if not isinstance(state, Mapping) or state.get("schema_version") != 2:
            raise ValueError("distributed rollout state must use schema_version=2")
        expected = {
            "schema_version",
            "config",
            "leases",
            "generations",
            "counters",
            "policy_artifacts",
            "coordinator",
        }
        if set(state) != expected:
            raise ValueError("distributed rollout state fields do not match schema")
        config_value = state["config"]
        leases_value = state["leases"]
        generations_value = state["generations"]
        counters_value = state["counters"]
        artifacts_value = state["policy_artifacts"]
        coordinator_value = state["coordinator"]
        if not isinstance(config_value, Mapping):
            raise ValueError("distributed rollout config must be a mapping")
        if not isinstance(leases_value, Sequence) or isinstance(
            leases_value, (str, bytes, bytearray)
        ):
            raise ValueError("distributed rollout leases must be a sequence")
        if not isinstance(generations_value, Mapping):
            raise ValueError("distributed rollout generations must be a mapping")
        if (
            not isinstance(counters_value, Mapping)
            or set(counters_value) != _COUNTER_NAMES
        ):
            raise ValueError("distributed rollout counters do not match schema")
        if not isinstance(coordinator_value, Mapping):
            raise ValueError("distributed rollout coordinator must be a mapping")
        if not isinstance(artifacts_value, Sequence) or isinstance(
            artifacts_value, (str, bytes, bytearray)
        ):
            raise ValueError("distributed policy artifacts must be a sequence")

        counters: dict[str, int] = {}
        for name in _COUNTER_NAMES:
            value = counters_value[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"counter {name} must be a non-negative integer")
            counters[name] = value
        generations: dict[str, int] = {}
        for worker_id, generation in generations_value.items():
            if not isinstance(worker_id, str) or not worker_id.strip():
                raise ValueError("generation worker IDs must be non-empty strings")
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 0
            ):
                raise ValueError("worker generations must be non-negative integers")
            generations[worker_id] = generation

        coordinator = AsyncRolloutCoordinator.from_state_dict(coordinator_value)
        control_plane = cls(
            coordinator=coordinator,
            config=DistributedRolloutConfig(**dict(config_value)),
            clock=clock,
        )
        if len(generations) > control_plane.config.worker_history_capacity:
            raise ValueError("checkpoint exceeds worker generation history capacity")
        if any(
            len(worker_id) > control_plane.config.max_worker_id_length
            for worker_id in generations
        ):
            raise ValueError("checkpoint worker ID exceeds max_worker_id_length")
        artifacts: dict[int, PolicyArtifact] = {}
        for value in artifacts_value:
            if not isinstance(value, Mapping):
                raise ValueError("policy artifact state must be a mapping")
            artifact = PolicyArtifact.from_dict(value)
            if artifact.policy_version in artifacts:
                raise ValueError("policy artifact versions must be unique")
            if artifact.policy_version > coordinator.current_policy_version:
                raise ValueError("checkpoint contains a future policy artifact")
            artifacts[artifact.policy_version] = artifact
        if len(artifacts) > control_plane.config.policy_artifact_capacity:
            raise ValueError("checkpoint exceeds policy artifact capacity")
        if (
            control_plane.config.require_policy_artifact
            and coordinator.current_policy_version not in artifacts
        ):
            raise ValueError("checkpoint is missing the current policy artifact")
        now = control_plane._now()
        leases: dict[str, WorkerLease] = {}
        expired_on_restore = 0
        for value in leases_value:
            if not isinstance(value, Mapping):
                raise ValueError("worker lease state must be a mapping")
            lease = WorkerLease.from_dict(value)
            if lease.worker_id in leases:
                raise ValueError("worker lease IDs must be unique")
            if generations.get(lease.worker_id) != lease.generation:
                raise ValueError("worker lease generation does not match registry")
            if lease.policy_version > coordinator.current_policy_version:
                raise ValueError("worker lease references a future policy")
            if lease.expires_at <= now:
                expired_on_restore += 1
            else:
                leases[lease.worker_id] = lease
        if len(leases) > control_plane.config.max_workers:
            raise ValueError("checkpoint exceeds distributed worker capacity")

        control_plane._leases = leases
        control_plane._generations = generations
        control_plane._policy_artifacts = artifacts
        control_plane._counters = counters
        control_plane._counters["expired_leases"] += expired_on_restore
        return control_plane


__all__ = [
    "DistributedRolloutConfig",
    "DistributedRolloutControlPlane",
    "DistributedRolloutStats",
    "WorkerCapacityError",
    "WorkerHealth",
    "WorkerLease",
    "WorkerLeaseError",
    "WorkerLeaseExpired",
]
