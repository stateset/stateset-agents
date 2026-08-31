"""Policy-versioned coordination primitives for asynchronous RL rollouts.

The coordinator separates rollout production from learner updates while making
policy staleness explicit.  It is intentionally transport-agnostic: producers
may run in local tasks, worker processes, or remote services as long as they
submit :class:`RolloutRecord` values.

Unlike a plain ``asyncio.Queue``, this control plane rejects future-policy
samples, evicts samples outside a configured lag bound, retains sampler token
log-probabilities for importance correction, and exposes auditable counters.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


class AsyncRolloutError(RuntimeError):
    """Base error raised by asynchronous rollout coordination."""


class AsyncRolloutClosed(AsyncRolloutError):
    """Raised when work is submitted to, or requested from, a closed coordinator."""


class AsyncRolloutTimeout(AsyncRolloutError):
    """Raised when bounded backpressure or batch collection times out."""


@dataclass(frozen=True)
class AsyncRolloutConfig:
    """Safety and batching limits for asynchronous rollout collection."""

    queue_capacity: int = 128
    min_batch_size: int = 1
    max_batch_size: int = 32
    max_policy_lag: int = 1
    max_importance_weight: float = 2.0
    deduplication_capacity: int = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "queue_capacity",
            "min_batch_size",
            "max_batch_size",
            "deduplication_capacity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_batch_size > self.max_batch_size:
            raise ValueError("min_batch_size must not exceed max_batch_size")
        if self.max_batch_size > self.queue_capacity:
            raise ValueError("max_batch_size must not exceed queue_capacity")
        if self.deduplication_capacity < self.queue_capacity:
            raise ValueError("deduplication_capacity must be at least queue_capacity")
        if (
            isinstance(self.max_policy_lag, bool)
            or not isinstance(self.max_policy_lag, int)
            or self.max_policy_lag < 0
        ):
            raise ValueError("max_policy_lag must be a non-negative integer")
        if (
            isinstance(self.max_importance_weight, bool)
            or not isinstance(self.max_importance_weight, (int, float))
            or not math.isfinite(float(self.max_importance_weight))
            or self.max_importance_weight < 1.0
        ):
            raise ValueError("max_importance_weight must be finite and at least 1")


@dataclass(frozen=True)
class RolloutRecord:
    """One rollout tied to the exact policy version that sampled it.

    ``payload`` is deliberately opaque to the coordinator.  It normally holds
    token ids, masks, rewards, advantages, and environment metadata.  Sampler
    log-probabilities remain separate because they are mandatory evidence for
    correcting stale-policy samples.
    """

    rollout_id: str
    policy_version: int
    sampler_log_probs: tuple[float, ...]
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, str) or not self.rollout_id.strip():
            raise ValueError("rollout_id must be a non-empty string")
        if (
            isinstance(self.policy_version, bool)
            or not isinstance(self.policy_version, int)
            or self.policy_version < 0
        ):
            raise ValueError("policy_version must be a non-negative integer")
        if not self.sampler_log_probs:
            raise ValueError("sampler_log_probs must contain at least one token")
        if any(not math.isfinite(float(value)) for value in self.sampler_log_probs):
            raise ValueError("sampler_log_probs must be finite")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        """Return a Python-native checkpoint representation."""
        return {
            "rollout_id": self.rollout_id,
            "policy_version": self.policy_version,
            "sampler_log_probs": list(self.sampler_log_probs),
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RolloutRecord:
        """Restore and validate a record from coordinator state."""
        expected = {
            "rollout_id",
            "policy_version",
            "sampler_log_probs",
            "payload",
        }
        if set(value) != expected:
            raise ValueError("rollout checkpoint fields do not match schema")
        try:
            log_probs = value["sampler_log_probs"]
            if not isinstance(log_probs, Sequence) or isinstance(
                log_probs, (str, bytes, bytearray)
            ):
                raise ValueError("sampler_log_probs must be a sequence")
            return cls(
                rollout_id=value["rollout_id"],
                policy_version=value["policy_version"],
                sampler_log_probs=tuple(log_probs),
                payload=value["payload"],
            )
        except KeyError as exc:
            raise ValueError(f"rollout checkpoint is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class RolloutBatch:
    """A learner batch with the policy lag of every included rollout."""

    learner_policy_version: int
    records: tuple[RolloutRecord, ...]
    policy_lags: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("rollout batch must not be empty")
        if len(self.records) != len(self.policy_lags):
            raise ValueError("records and policy_lags must have equal length")


@dataclass(frozen=True)
class AsyncRolloutStats:
    """Monotonic counters and current coordinator state."""

    current_policy_version: int
    queue_depth: int
    submitted: int
    consumed: int
    dropped_stale: int
    rejected_future: int
    rejected_duplicate: int
    max_observed_policy_lag: int
    closed: bool

    def to_dict(self) -> dict[str, int | bool]:
        """Return a stable JSON-compatible representation."""
        return asdict(self)


def compute_importance_weights(
    learner_log_probs: Sequence[float],
    sampler_log_probs: Sequence[float],
    *,
    max_weight: float = 2.0,
) -> tuple[float, ...]:
    """Return symmetrically clipped per-token learner/sampler ratios.

    Computing in log space avoids underflow for long or low-probability token
    sequences.  Symmetric clipping bounds both amplification and suppression:
    a ``max_weight`` of 2 produces weights in ``[0.5, 2.0]``.
    """
    if len(learner_log_probs) != len(sampler_log_probs):
        raise ValueError("learner and sampler log-probabilities must have equal length")
    if not learner_log_probs:
        raise ValueError("log-probabilities must contain at least one token")
    if (
        isinstance(max_weight, bool)
        or not isinstance(max_weight, (int, float))
        or not math.isfinite(float(max_weight))
        or max_weight < 1.0
    ):
        raise ValueError("max_weight must be finite and at least 1")

    log_bound = math.log(float(max_weight))
    weights: list[float] = []
    for learner, sampler in zip(learner_log_probs, sampler_log_probs, strict=True):
        if not math.isfinite(float(learner)) or not math.isfinite(float(sampler)):
            raise ValueError("log-probabilities must be finite")
        log_ratio = min(log_bound, max(-log_bound, float(learner) - float(sampler)))
        weights.append(math.exp(log_ratio))
    return tuple(weights)


class AsyncRolloutCoordinator:
    """Bounded producer/learner queue with explicit policy-staleness control."""

    def __init__(
        self,
        config: AsyncRolloutConfig | None = None,
        *,
        initial_policy_version: int = 0,
    ) -> None:
        if (
            isinstance(initial_policy_version, bool)
            or not isinstance(initial_policy_version, int)
            or initial_policy_version < 0
        ):
            raise ValueError("initial_policy_version must be a non-negative integer")
        self.config = config or AsyncRolloutConfig()
        self._current_policy_version = initial_policy_version
        self._queue: deque[RolloutRecord] = deque()
        self._condition = asyncio.Condition()
        self._closed = False
        self._submitted = 0
        self._consumed = 0
        self._dropped_stale = 0
        self._rejected_future = 0
        self._rejected_duplicate = 0
        self._max_observed_policy_lag = 0
        self._seen_rollout_ids: set[str] = set()

    @property
    def current_policy_version(self) -> int:
        """Return the policy version currently owned by the learner."""
        return self._current_policy_version

    def _policy_lag(self, record: RolloutRecord) -> int:
        return self._current_policy_version - record.policy_version

    def _evict_stale(self) -> None:
        retained: deque[RolloutRecord] = deque()
        for record in self._queue:
            if self._policy_lag(record) > self.config.max_policy_lag:
                self._dropped_stale += 1
            else:
                retained.append(record)
        self._queue = retained

    async def submit(
        self, record: RolloutRecord, *, timeout_seconds: float | None = None
    ) -> bool:
        """Submit one rollout, applying bounded backpressure.

        Returns ``False`` when a rollout is already outside the configured
        staleness window.  Future-policy records are rejected because they
        indicate a weight-versioning or routing defect.
        """
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        async def _submit() -> bool:
            async with self._condition:
                if self._closed:
                    raise AsyncRolloutClosed("rollout coordinator is closed")
                if record.rollout_id in self._seen_rollout_ids:
                    self._rejected_duplicate += 1
                    return False
                if record.policy_version > self._current_policy_version:
                    self._rejected_future += 1
                    raise AsyncRolloutError(
                        "rollout policy version is newer than the learner policy"
                    )
                lag = self._policy_lag(record)
                self._max_observed_policy_lag = max(self._max_observed_policy_lag, lag)
                if lag > self.config.max_policy_lag:
                    self._dropped_stale += 1
                    return False

                await self._condition.wait_for(
                    lambda: self._closed
                    or len(self._queue) < self.config.queue_capacity
                )
                if self._closed:
                    raise AsyncRolloutClosed("rollout coordinator is closed")

                # The policy may advance while this producer is backpressured.
                if record.rollout_id in self._seen_rollout_ids:
                    self._rejected_duplicate += 1
                    return False
                lag = self._policy_lag(record)
                self._max_observed_policy_lag = max(self._max_observed_policy_lag, lag)
                if lag > self.config.max_policy_lag:
                    self._dropped_stale += 1
                    return False
                if len(self._seen_rollout_ids) >= self.config.deduplication_capacity:
                    raise AsyncRolloutError(
                        "rollout deduplication capacity exhausted; checkpoint and "
                        "start a new coordinator epoch"
                    )
                self._queue.append(record)
                self._seen_rollout_ids.add(record.rollout_id)
                self._submitted += 1
                self._condition.notify_all()
                return True

        try:
            if timeout_seconds is None:
                return await _submit()
            operation = asyncio.create_task(_submit())
            return await asyncio.wait_for(operation, timeout=timeout_seconds)
        except TimeoutError as exc:
            raise AsyncRolloutTimeout(
                "timed out waiting for rollout queue capacity"
            ) from exc

    async def next_batch(
        self,
        *,
        min_size: int | None = None,
        max_size: int | None = None,
        timeout_seconds: float | None = None,
    ) -> RolloutBatch:
        """Return the next FIFO learner batch after removing stale samples."""
        requested_min = self.config.min_batch_size if min_size is None else min_size
        requested_max = self.config.max_batch_size if max_size is None else max_size
        if (
            isinstance(requested_min, bool)
            or not isinstance(requested_min, int)
            or requested_min < 1
        ):
            raise ValueError("min_size must be a positive integer")
        if (
            isinstance(requested_max, bool)
            or not isinstance(requested_max, int)
            or requested_max < requested_min
        ):
            raise ValueError("max_size must be an integer at least min_size")
        if requested_max > self.config.queue_capacity:
            raise ValueError("max_size must not exceed queue_capacity")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        async def _collect() -> RolloutBatch:
            async with self._condition:
                self._evict_stale()
                await self._condition.wait_for(
                    lambda: len(self._queue) >= requested_min or self._closed
                )
                self._evict_stale()
                if not self._queue:
                    raise AsyncRolloutClosed("rollout coordinator is closed")
                if len(self._queue) < requested_min and not self._closed:
                    raise AsyncRolloutError("rollout batch invariant violated")

                count = min(requested_max, len(self._queue))
                records = tuple(self._queue.popleft() for _ in range(count))
                lags = tuple(self._policy_lag(record) for record in records)
                self._consumed += count
                self._condition.notify_all()
                return RolloutBatch(
                    learner_policy_version=self._current_policy_version,
                    records=records,
                    policy_lags=lags,
                )

        try:
            if timeout_seconds is None:
                return await _collect()
            operation = asyncio.create_task(_collect())
            return await asyncio.wait_for(operation, timeout=timeout_seconds)
        except TimeoutError as exc:
            raise AsyncRolloutTimeout("timed out waiting for a rollout batch") from exc

    async def advance_policy(self, new_version: int | None = None) -> int:
        """Advance learner weights and evict samples beyond the lag bound."""
        async with self._condition:
            target = (
                self._current_policy_version + 1 if new_version is None else new_version
            )
            if (
                isinstance(target, bool)
                or not isinstance(target, int)
                or target <= self._current_policy_version
            ):
                raise ValueError("new policy version must increase monotonically")
            self._current_policy_version = target
            self._evict_stale()
            self._condition.notify_all()
            return target

    async def close(self) -> None:
        """Stop producers and allow consumers to drain the remaining queue."""
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    async def state_dict(self) -> dict[str, Any]:
        """Return restart-safe state, including queued and consumed rollout IDs.

        The payload remains Python-native so tensor-bearing records can be saved
        with the caller's checkpoint mechanism.  Consumers that require JSON
        should supply JSON-compatible payload values.
        """
        async with self._condition:
            return {
                "schema_version": 1,
                "config": asdict(self.config),
                "current_policy_version": self._current_policy_version,
                "queue": [record.to_dict() for record in self._queue],
                "seen_rollout_ids": sorted(self._seen_rollout_ids),
                "counters": {
                    "submitted": self._submitted,
                    "consumed": self._consumed,
                    "dropped_stale": self._dropped_stale,
                    "rejected_future": self._rejected_future,
                    "rejected_duplicate": self._rejected_duplicate,
                    "max_observed_policy_lag": self._max_observed_policy_lag,
                },
                "closed": self._closed,
            }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> AsyncRolloutCoordinator:
        """Restore a coordinator without allowing stale or duplicate queue state."""
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("async rollout state must use schema_version=1")
        expected_fields = {
            "schema_version",
            "config",
            "current_policy_version",
            "queue",
            "seen_rollout_ids",
            "counters",
            "closed",
        }
        if set(state) != expected_fields:
            raise ValueError("async rollout state fields do not match schema")
        config_value = state.get("config")
        counters = state.get("counters")
        queue_value = state.get("queue")
        seen_value = state.get("seen_rollout_ids")
        if not isinstance(config_value, Mapping):
            raise ValueError("async rollout state config must be a mapping")
        if not isinstance(counters, Mapping):
            raise ValueError("async rollout state counters must be a mapping")
        if not isinstance(queue_value, Sequence) or isinstance(
            queue_value, (str, bytes, bytearray)
        ):
            raise ValueError("async rollout state queue must be a sequence")
        if not isinstance(seen_value, Sequence) or isinstance(
            seen_value, (str, bytes, bytearray)
        ):
            raise ValueError("async rollout seen IDs must be a sequence")

        config = AsyncRolloutConfig(**dict(config_value))
        coordinator = cls(
            config,
            initial_policy_version=state.get("current_policy_version", -1),
        )
        records = []
        for value in queue_value:
            if not isinstance(value, Mapping):
                raise ValueError("queued rollout state must be a mapping")
            record = RolloutRecord.from_dict(value)
            if record.policy_version > coordinator.current_policy_version:
                raise ValueError("checkpoint contains a future-policy rollout")
            if coordinator._policy_lag(record) > config.max_policy_lag:
                raise ValueError("checkpoint contains a stale rollout")
            records.append(record)
        queue_ids = [record.rollout_id for record in records]
        seen_ids = list(seen_value)
        if any(not isinstance(value, str) or not value for value in seen_ids):
            raise ValueError("async rollout seen IDs must be non-empty strings")
        if len(seen_ids) != len(set(seen_ids)):
            raise ValueError("async rollout seen IDs must be unique")
        if len(queue_ids) != len(set(queue_ids)):
            raise ValueError("queued rollout IDs must be unique")
        if not set(queue_ids).issubset(seen_ids):
            raise ValueError("every queued rollout ID must be in seen_rollout_ids")
        if len(seen_ids) > config.deduplication_capacity:
            raise ValueError("checkpoint exceeds deduplication_capacity")

        counter_names = {
            "submitted",
            "consumed",
            "dropped_stale",
            "rejected_future",
            "rejected_duplicate",
            "max_observed_policy_lag",
        }
        if set(counters) != counter_names:
            raise ValueError("async rollout checkpoint counters are incomplete")
        normalized_counters: dict[str, int] = {}
        for name in counter_names:
            value = counters[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"counter {name} must be a non-negative integer")
            normalized_counters[name] = value
        if normalized_counters["submitted"] < len(seen_ids):
            raise ValueError("submitted counter must cover every seen rollout ID")

        coordinator._queue = deque(records)
        coordinator._seen_rollout_ids = set(seen_ids)
        coordinator._submitted = normalized_counters["submitted"]
        coordinator._consumed = normalized_counters["consumed"]
        coordinator._dropped_stale = normalized_counters["dropped_stale"]
        coordinator._rejected_future = normalized_counters["rejected_future"]
        coordinator._rejected_duplicate = normalized_counters["rejected_duplicate"]
        coordinator._max_observed_policy_lag = normalized_counters[
            "max_observed_policy_lag"
        ]
        closed = state.get("closed")
        if not isinstance(closed, bool):
            raise ValueError("async rollout state closed must be bool")
        coordinator._closed = closed
        return coordinator

    def stats(self) -> AsyncRolloutStats:
        """Return a point-in-time audit snapshot without exposing payloads."""
        return AsyncRolloutStats(
            current_policy_version=self._current_policy_version,
            queue_depth=len(self._queue),
            submitted=self._submitted,
            consumed=self._consumed,
            dropped_stale=self._dropped_stale,
            rejected_future=self._rejected_future,
            rejected_duplicate=self._rejected_duplicate,
            max_observed_policy_lag=self._max_observed_policy_lag,
            closed=self._closed,
        )


__all__ = [
    "AsyncRolloutClosed",
    "AsyncRolloutConfig",
    "AsyncRolloutCoordinator",
    "AsyncRolloutError",
    "AsyncRolloutStats",
    "AsyncRolloutTimeout",
    "RolloutBatch",
    "RolloutRecord",
    "compute_importance_weights",
]
