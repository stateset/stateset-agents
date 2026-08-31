"""Failure-safe producer/learner runtime for asynchronous agent RL.

The runtime composes :mod:`stateset_agents.training.async_rollouts` primitives
into an executable loop.  Rollout workers are decoupled from learner updates,
but a new policy version is never exposed until its publisher callback has
completed.  Worker failures close the coordinator and propagate to the caller
instead of leaving a paid training job waiting forever.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .async_rollouts import (
    AsyncRolloutClosed,
    AsyncRolloutCoordinator,
    AsyncRolloutError,
    AsyncRolloutStats,
    AsyncRolloutTimeout,
    RolloutBatch,
    RolloutRecord,
)

RolloutProducer = Callable[[int, int], Awaitable[RolloutRecord]]
LearnerStep = Callable[[RolloutBatch], Awaitable[Mapping[str, float]]]
PolicyPublisher = Callable[[int], Awaitable[None]]


class AsyncRolloutWorkerError(AsyncRolloutError):
    """Raised when a rollout producer fails."""

    def __init__(self, worker_id: int, cause: Exception) -> None:
        super().__init__(f"rollout worker {worker_id} failed: {cause}")
        self.worker_id = worker_id
        self.cause = cause


@dataclass(frozen=True)
class AsyncRolloutRuntimeConfig:
    """Execution limits for a native asynchronous training run."""

    producer_count: int = 1
    max_updates: int = 1
    batch_timeout_seconds: float = 300.0
    submit_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name in ("producer_count", "max_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "batch_timeout_seconds",
            "submit_timeout_seconds",
            "shutdown_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AsyncRolloutRunResult:
    """Auditable result of a completed producer/learner run."""

    initial_policy_version: int
    final_policy_version: int
    updates_completed: int
    learner_metrics: tuple[Mapping[str, float], ...]
    rollout_stats: AsyncRolloutStats

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible result document."""
        return {
            "initial_policy_version": self.initial_policy_version,
            "final_policy_version": self.final_policy_version,
            "updates_completed": self.updates_completed,
            "learner_metrics": [dict(metrics) for metrics in self.learner_metrics],
            "rollout_stats": self.rollout_stats.to_dict(),
        }


def _validate_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise AsyncRolloutError("learner metrics must be a mapping")
    normalized: dict[str, float] = {}
    for name, metric in value.items():
        if not isinstance(name, str) or not name.strip():
            raise AsyncRolloutError("learner metric names must be non-empty strings")
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise AsyncRolloutError(
                f"learner metric {name!r} must be finite and numeric"
            )
        normalized[name] = float(metric)
    return normalized


class AsyncRolloutRuntime:
    """Run asynchronous rollout producers against one serialized learner."""

    def __init__(
        self,
        *,
        coordinator: AsyncRolloutCoordinator,
        producer: RolloutProducer,
        learner_step: LearnerStep,
        publish_policy: PolicyPublisher,
        config: AsyncRolloutRuntimeConfig | None = None,
    ) -> None:
        if not callable(producer):
            raise TypeError("producer must be callable")
        if not callable(learner_step):
            raise TypeError("learner_step must be callable")
        if not callable(publish_policy):
            raise TypeError("publish_policy must be callable")
        self.coordinator = coordinator
        self.producer = producer
        self.learner_step = learner_step
        self.publish_policy = publish_policy
        self.config = config or AsyncRolloutRuntimeConfig()
        self._has_run = False

    async def _producer_loop(
        self,
        worker_id: int,
        stop: asyncio.Event,
        failures: asyncio.Queue[tuple[int, Exception]],
    ) -> None:
        try:
            while not stop.is_set():
                requested_version = self.coordinator.current_policy_version
                record = await self.producer(worker_id, requested_version)
                if not isinstance(record, RolloutRecord):
                    raise TypeError("producer must return RolloutRecord")
                if record.policy_version != requested_version:
                    raise AsyncRolloutError(
                        "producer returned a rollout for a policy version other "
                        "than the requested snapshot"
                    )
                try:
                    await self.coordinator.submit(
                        record,
                        timeout_seconds=self.config.submit_timeout_seconds,
                    )
                    # A duplicate or immediately accepted record may not
                    # suspend. Yield so a fast producer cannot starve the
                    # learner or shutdown path.
                    await asyncio.sleep(0)
                except AsyncRolloutTimeout:
                    # Bounded backpressure is expected while the learner owns
                    # the accelerator. Retry with a fresh policy snapshot.
                    await asyncio.sleep(0)
                    continue
                except AsyncRolloutClosed:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - propagate through typed channel
            if failures.empty():
                failures.put_nowait((worker_id, exc))
            await self.coordinator.close()

    async def _next_batch_or_failure(
        self,
        failures: asyncio.Queue[tuple[int, Exception]],
    ) -> RolloutBatch:
        batch_task = asyncio.create_task(
            self.coordinator.next_batch(
                timeout_seconds=self.config.batch_timeout_seconds
            )
        )
        failure_task = asyncio.create_task(failures.get())
        try:
            done, _ = await asyncio.wait(
                {batch_task, failure_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if failure_task in done or not failures.empty():
                worker_id, cause = (
                    failure_task.result()
                    if failure_task in done
                    else failures.get_nowait()
                )
                raise AsyncRolloutWorkerError(worker_id, cause) from cause
            return batch_task.result()
        finally:
            for task in (batch_task, failure_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(batch_task, failure_task, return_exceptions=True)

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        """Retrieve a detached worker result to prevent task warnings."""
        if not task.cancelled():
            task.exception()

    async def _stop_producers(
        self, tasks: list[asyncio.Task[None]], stop: asyncio.Event
    ) -> None:
        stop.set()
        await self.coordinator.close()
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        done, pending = await asyncio.wait(
            tasks, timeout=self.config.shutdown_timeout_seconds
        )
        await asyncio.gather(*done, return_exceptions=True)
        if pending:
            # Python cannot forcibly terminate a coroutine that suppresses
            # cancellation. Detach only after the configured deadline and
            # consume its eventual result so the caller is never held open.
            for task in pending:
                task.add_done_callback(self._consume_task_result)
            raise AsyncRolloutError("rollout workers did not shut down in time")

    async def run(self) -> AsyncRolloutRunResult:
        """Execute the configured number of learner updates exactly once."""
        if self._has_run:
            raise AsyncRolloutError("async rollout runtime instances are single-use")
        if self.coordinator.stats().closed:
            raise AsyncRolloutClosed("rollout coordinator is already closed")
        self._has_run = True
        initial_version = self.coordinator.current_policy_version

        # Publish initial weights before any producer sees the version.
        try:
            await self.publish_policy(initial_version)
        except Exception:
            await self.coordinator.close()
            raise

        stop = asyncio.Event()
        failures: asyncio.Queue[tuple[int, Exception]] = asyncio.Queue(maxsize=1)
        tasks = [
            asyncio.create_task(
                self._producer_loop(worker_id, stop, failures),
                name=f"stateset-rollout-producer-{worker_id}",
            )
            for worker_id in range(self.config.producer_count)
        ]
        metrics: list[Mapping[str, float]] = []
        try:
            for _ in range(self.config.max_updates):
                batch = await self._next_batch_or_failure(failures)
                if not failures.empty():
                    worker_id, cause = failures.get_nowait()
                    raise AsyncRolloutWorkerError(worker_id, cause) from cause
                step_metrics = _validate_metrics(await self.learner_step(batch))

                # The learner has updated its weights, but producers must not
                # observe the new version until publication completes.
                target_version = self.coordinator.current_policy_version + 1
                await self.publish_policy(target_version)
                await self.coordinator.advance_policy(target_version)
                metrics.append(step_metrics)
        finally:
            await self._stop_producers(tasks, stop)

        return AsyncRolloutRunResult(
            initial_policy_version=initial_version,
            final_policy_version=self.coordinator.current_policy_version,
            updates_completed=len(metrics),
            learner_metrics=tuple(metrics),
            rollout_stats=self.coordinator.stats(),
        )


__all__ = [
    "AsyncRolloutRunResult",
    "AsyncRolloutRuntime",
    "AsyncRolloutRuntimeConfig",
    "AsyncRolloutWorkerError",
    "LearnerStep",
    "PolicyPublisher",
    "RolloutProducer",
]
