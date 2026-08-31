"""Behavioral tests for the native asynchronous producer/learner runtime."""

from __future__ import annotations

import asyncio
import math

import pytest

import stateset_agents.training as training
from stateset_agents.training.async_rollouts import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    AsyncRolloutError,
    AsyncRolloutTimeout,
    RolloutBatch,
    RolloutRecord,
)
from stateset_agents.training.async_runtime import (
    AsyncRolloutRuntime,
    AsyncRolloutRuntimeConfig,
    AsyncRolloutWorkerError,
)


def _record(label: str, version: int) -> RolloutRecord:
    return RolloutRecord(
        rollout_id=label,
        policy_version=version,
        sampler_log_probs=(-0.5, -0.25),
        payload={"tokens": [1, 2]},
    )


def test_async_runtime_surface_is_public_and_lazy() -> None:
    assert training.AsyncRolloutRuntime is AsyncRolloutRuntime
    assert "AsyncRolloutRuntime" in training.__all__


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"producer_count": 0}, "producer_count"),
        ({"max_updates": 0}, "max_updates"),
        ({"batch_timeout_seconds": 0.0}, "batch_timeout_seconds"),
        ({"submit_timeout_seconds": math.inf}, "submit_timeout_seconds"),
        ({"shutdown_timeout_seconds": -1}, "shutdown_timeout_seconds"),
    ],
)
def test_runtime_config_rejects_unbounded_values(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        AsyncRolloutRuntimeConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_runtime_executes_decoupled_workers_and_serialized_updates() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(
            queue_capacity=8,
            min_batch_size=2,
            max_batch_size=2,
            max_policy_lag=1,
        )
    )
    published: list[int] = []
    next_id = 0
    batches: list[RolloutBatch] = []

    async def publish(version: int) -> None:
        published.append(version)
        await asyncio.sleep(0)

    async def produce(worker_id: int, version: int) -> RolloutRecord:
        nonlocal next_id
        assert version in published
        next_id += 1
        await asyncio.sleep(0)
        return _record(f"worker-{worker_id}-{next_id}", version)

    async def learn(batch: RolloutBatch) -> dict[str, float]:
        batches.append(batch)
        await asyncio.sleep(0)
        return {"loss": 1.0 / (len(batches) + 1), "batch_size": len(batch.records)}

    runtime = AsyncRolloutRuntime(
        coordinator=coordinator,
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
        config=AsyncRolloutRuntimeConfig(
            producer_count=2,
            max_updates=3,
            batch_timeout_seconds=1.0,
            submit_timeout_seconds=1.0,
            shutdown_timeout_seconds=1.0,
        ),
    )

    result = await runtime.run()

    assert published == [0, 1, 2, 3]
    assert result.initial_policy_version == 0
    assert result.final_policy_version == 3
    assert result.updates_completed == 3
    assert len(result.learner_metrics) == 3
    assert all(len(batch.records) == 2 for batch in batches)
    assert all(max(batch.policy_lags) <= 1 for batch in batches)
    assert result.rollout_stats.consumed == 6
    assert result.rollout_stats.closed is True
    assert result.to_dict()["updates_completed"] == 3


@pytest.mark.asyncio
async def test_policy_version_is_not_exposed_until_publish_completes() -> None:
    coordinator = AsyncRolloutCoordinator()
    publication_complete: set[int] = set()
    observed: list[int] = []

    async def publish(version: int) -> None:
        await asyncio.sleep(0.01)
        publication_complete.add(version)

    async def produce(_worker_id: int, version: int) -> RolloutRecord:
        assert version in publication_complete
        observed.append(version)
        return _record(f"rollout-{len(observed)}", version)

    async def learn(_batch: RolloutBatch) -> dict[str, float]:
        return {"loss": 0.1}

    result = await AsyncRolloutRuntime(
        coordinator=coordinator,
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
    ).run()

    assert observed
    assert observed[0] == 0
    assert result.final_policy_version == 1
    assert publication_complete == {0, 1}


@pytest.mark.asyncio
async def test_worker_failure_propagates_and_closes_runtime() -> None:
    coordinator = AsyncRolloutCoordinator()

    async def produce(worker_id: int, _version: int) -> RolloutRecord:
        raise RuntimeError(f"worker-{worker_id}-boom")

    async def learn(_batch: RolloutBatch) -> dict[str, float]:
        pytest.fail("learner must not run after producer failure")

    async def publish(_version: int) -> None:
        return None

    runtime = AsyncRolloutRuntime(
        coordinator=coordinator,
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
        config=AsyncRolloutRuntimeConfig(batch_timeout_seconds=1.0),
    )

    with pytest.raises(AsyncRolloutWorkerError, match="worker-0-boom") as error:
        await runtime.run()

    assert isinstance(error.value.cause, RuntimeError)
    assert coordinator.stats().closed is True


@pytest.mark.asyncio
async def test_wrong_producer_policy_version_fails_closed() -> None:
    async def produce(_worker_id: int, version: int) -> RolloutRecord:
        return _record("wrong-version", version + 1)

    async def learn(_batch: RolloutBatch) -> dict[str, float]:
        return {"loss": 0.1}

    async def publish(_version: int) -> None:
        return None

    runtime = AsyncRolloutRuntime(
        coordinator=AsyncRolloutCoordinator(),
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
        config=AsyncRolloutRuntimeConfig(batch_timeout_seconds=1.0),
    )
    with pytest.raises(AsyncRolloutWorkerError, match="other than the requested"):
        await runtime.run()


@pytest.mark.asyncio
async def test_invalid_learner_metrics_fail_closed() -> None:
    counter = 0

    async def produce(_worker_id: int, version: int) -> RolloutRecord:
        nonlocal counter
        counter += 1
        return _record(f"rollout-{counter}", version)

    async def learn(_batch: RolloutBatch) -> dict[str, float]:
        return {"loss": math.nan}

    async def publish(_version: int) -> None:
        return None

    coordinator = AsyncRolloutCoordinator()
    runtime = AsyncRolloutRuntime(
        coordinator=coordinator,
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
    )
    with pytest.raises(AsyncRolloutError, match="finite and numeric"):
        await runtime.run()
    assert coordinator.stats().closed is True


@pytest.mark.asyncio
async def test_batch_timeout_cancels_sleeping_producer() -> None:
    async def produce(_worker_id: int, version: int) -> RolloutRecord:
        await asyncio.sleep(10)
        return _record("never", version)

    async def learn(_batch: RolloutBatch) -> dict[str, float]:
        return {"loss": 0.1}

    async def publish(_version: int) -> None:
        return None

    coordinator = AsyncRolloutCoordinator()
    runtime = AsyncRolloutRuntime(
        coordinator=coordinator,
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
        config=AsyncRolloutRuntimeConfig(
            batch_timeout_seconds=0.01,
            shutdown_timeout_seconds=1.0,
        ),
    )
    with pytest.raises(AsyncRolloutTimeout, match="batch"):
        await runtime.run()
    assert coordinator.stats().closed is True


@pytest.mark.asyncio
async def test_external_cancellation_cleans_up_internal_waiters() -> None:
    producer_started = asyncio.Event()

    async def produce(_worker_id: int, version: int) -> RolloutRecord:
        producer_started.set()
        await asyncio.sleep(10)
        return _record("never", version)

    async def learn(_batch: RolloutBatch) -> dict[str, float]:
        return {"loss": 0.1}

    async def publish(_version: int) -> None:
        return None

    runtime = AsyncRolloutRuntime(
        coordinator=AsyncRolloutCoordinator(),
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
        config=AsyncRolloutRuntimeConfig(batch_timeout_seconds=10.0),
    )
    run_task = asyncio.create_task(runtime.run())
    await producer_started.wait()
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task
    await asyncio.sleep(0)
    assert not any(
        task.get_name().startswith("stateset-rollout-producer-")
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_shutdown_deadline_bounds_cancellation_resistant_producer() -> None:
    cancellation_delayed = asyncio.Event()
    release_producer = asyncio.Event()

    async def produce(_worker_id: int, version: int) -> RolloutRecord:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancellation_delayed.set()
            await release_producer.wait()
        return _record("delayed-cancellation", version)

    async def learn(_batch: RolloutBatch) -> dict[str, float]:
        return {"loss": 0.1}

    async def publish(_version: int) -> None:
        return None

    runtime = AsyncRolloutRuntime(
        coordinator=AsyncRolloutCoordinator(),
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
        config=AsyncRolloutRuntimeConfig(
            batch_timeout_seconds=0.01,
            shutdown_timeout_seconds=0.01,
        ),
    )

    with pytest.raises(AsyncRolloutError, match="did not shut down in time"):
        await asyncio.wait_for(runtime.run(), timeout=0.5)
    assert cancellation_delayed.is_set()

    release_producer.set()
    await asyncio.sleep(0.01)
    assert not any(
        task.get_name().startswith("stateset-rollout-producer-")
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_runtime_is_single_use_and_initial_publish_failure_closes() -> None:
    async def produce(_worker_id: int, version: int) -> RolloutRecord:
        return _record("one", version)

    async def learn(_batch: RolloutBatch) -> dict[str, float]:
        return {"loss": 0.1}

    async def publish(_version: int) -> None:
        return None

    runtime = AsyncRolloutRuntime(
        coordinator=AsyncRolloutCoordinator(),
        producer=produce,
        learner_step=learn,
        publish_policy=publish,
    )
    await runtime.run()
    with pytest.raises(AsyncRolloutError, match="single-use"):
        await runtime.run()

    async def fail_publish(_version: int) -> None:
        raise RuntimeError("publish failed")

    coordinator = AsyncRolloutCoordinator()
    failing = AsyncRolloutRuntime(
        coordinator=coordinator,
        producer=produce,
        learner_step=learn,
        publish_policy=fail_publish,
    )
    with pytest.raises(RuntimeError, match="publish failed"):
        await failing.run()
    assert coordinator.stats().closed is True
