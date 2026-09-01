"""Behavioral tests for policy-versioned asynchronous rollout coordination."""

from __future__ import annotations

import asyncio
import math

import pytest

import stateset_agents.training as training
from stateset_agents.training.async_rollouts import (
    AsyncRolloutClosed,
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    AsyncRolloutError,
    AsyncRolloutTimeout,
    RolloutRecord,
    compute_importance_weights,
)


def _record(label: str, version: int, *log_probs: float) -> RolloutRecord:
    return RolloutRecord(
        rollout_id=label,
        policy_version=version,
        sampler_log_probs=tuple(log_probs or (-1.0, -2.0)),
        payload={"tokens": [1, 2]},
    )


def test_async_rollout_surface_is_public_and_lazy() -> None:
    assert training.AsyncRolloutCoordinator is AsyncRolloutCoordinator
    assert "AsyncRolloutCoordinator" in training.__all__


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"queue_capacity": 0}, "queue_capacity"),
        ({"min_batch_size": 3, "max_batch_size": 2}, "min_batch_size"),
        ({"queue_capacity": 2, "max_batch_size": 3}, "queue_capacity"),
        ({"max_policy_lag": -1}, "max_policy_lag"),
        ({"max_importance_weight": 0.5}, "max_importance_weight"),
        (
            {
                "queue_capacity": 4,
                "max_batch_size": 4,
                "deduplication_capacity": 3,
            },
            "deduplication_capacity",
        ),
    ],
)
def test_config_rejects_unsafe_limits(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AsyncRolloutConfig(**kwargs)  # type: ignore[arg-type]


def test_record_requires_versioned_finite_sampler_evidence() -> None:
    with pytest.raises(ValueError, match="rollout_id"):
        _record("", 0)
    with pytest.raises(ValueError, match="policy_version"):
        _record("future", -1)
    with pytest.raises(ValueError, match="finite"):
        _record("nan", 0, math.nan)
    with pytest.raises(ValueError, match="at least one token"):
        RolloutRecord("empty", 0, (), {})


def test_importance_weights_are_log_space_and_symmetrically_clipped() -> None:
    weights = compute_importance_weights(
        learner_log_probs=(-9.0, -1.0, -3.0),
        sampler_log_probs=(-1.0, -9.0, -3.0),
        max_weight=2.0,
    )
    assert weights == pytest.approx((0.5, 2.0, 1.0))


def test_importance_weights_reject_mismatched_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        compute_importance_weights((-1.0,), (-1.0, -2.0))
    with pytest.raises(ValueError, match="finite"):
        compute_importance_weights((math.inf,), (-1.0,))
    with pytest.raises(ValueError, match="max_weight"):
        compute_importance_weights((-1.0,), (-1.0,), max_weight=0.0)


@pytest.mark.asyncio
async def test_fifo_batches_report_exact_policy_lag() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=4, min_batch_size=2, max_batch_size=3),
        initial_policy_version=3,
    )
    assert await coordinator.submit(_record("old", 2))
    assert await coordinator.submit(_record("current", 3))

    batch = await coordinator.next_batch()

    assert [record.rollout_id for record in batch.records] == ["old", "current"]
    assert batch.learner_policy_version == 3
    assert batch.policy_lags == (1, 0)
    assert coordinator.stats().consumed == 2


@pytest.mark.asyncio
async def test_policy_advance_evicts_samples_outside_lag_bound() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=4, max_batch_size=4, max_policy_lag=1),
        initial_policy_version=1,
    )
    assert await coordinator.submit(_record("v1", 1))
    await coordinator.advance_policy(3)

    stats = coordinator.stats()
    assert stats.current_policy_version == 3
    assert stats.queue_depth == 0
    assert stats.dropped_stale == 1


@pytest.mark.asyncio
async def test_submit_drops_already_stale_and_rejects_future_policy() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(max_policy_lag=1), initial_policy_version=4
    )
    assert not await coordinator.submit(_record("stale", 2))
    with pytest.raises(AsyncRolloutError, match="newer"):
        await coordinator.submit(_record("future", 5))

    stats = coordinator.stats()
    assert stats.dropped_stale == 1
    assert stats.rejected_future == 1
    assert stats.max_observed_policy_lag == 2


@pytest.mark.asyncio
async def test_duplicate_rollout_ids_are_never_consumed_twice() -> None:
    coordinator = AsyncRolloutCoordinator()
    record = _record("retry-safe", 0)
    assert await coordinator.submit(record)
    assert not await coordinator.submit(record)

    batch = await coordinator.next_batch()

    assert [item.rollout_id for item in batch.records] == ["retry-safe"]
    assert coordinator.stats().rejected_duplicate == 1


@pytest.mark.asyncio
async def test_checkpoint_restores_queue_counters_and_deduplication() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=4, max_batch_size=4),
        initial_policy_version=2,
    )
    consumed = _record("consumed", 1)
    queued = _record("queued", 2)
    assert await coordinator.submit(consumed)
    await coordinator.next_batch()
    assert await coordinator.submit(queued)
    snapshot = await coordinator.state_dict()

    restored = AsyncRolloutCoordinator.from_state_dict(snapshot)
    batch = await restored.next_batch()

    assert [record.rollout_id for record in batch.records] == ["queued"]
    assert batch.policy_lags == (0,)
    assert not await restored.submit(consumed)
    stats = restored.stats()
    assert stats.submitted == 2
    assert stats.consumed == 2
    assert stats.rejected_duplicate == 1


def test_checkpoint_rejects_stale_or_incomplete_state() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        AsyncRolloutCoordinator.from_state_dict({})

    state = {
        "schema_version": 1,
        "config": {
            "queue_capacity": 2,
            "min_batch_size": 1,
            "max_batch_size": 2,
            "max_policy_lag": 0,
            "max_importance_weight": 2.0,
            "deduplication_capacity": 10,
        },
        "current_policy_version": 2,
        "queue": [_record("stale", 1).to_dict()],
        "seen_rollout_ids": ["stale"],
        "counters": {
            "submitted": 1,
            "consumed": 0,
            "dropped_stale": 0,
            "rejected_future": 0,
            "rejected_duplicate": 0,
            "max_observed_policy_lag": 1,
        },
        "closed": False,
    }
    with pytest.raises(ValueError, match="stale rollout"):
        AsyncRolloutCoordinator.from_state_dict(state)

    state["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match schema"):
        AsyncRolloutCoordinator.from_state_dict(state)


def test_rollout_checkpoint_rejects_unknown_fields() -> None:
    value = _record("strict", 0).to_dict()
    value["untrusted"] = True
    with pytest.raises(ValueError, match="fields do not match schema"):
        RolloutRecord.from_dict(value)


def test_rollout_checkpoint_restores_pre_artifact_records() -> None:
    value = _record("legacy", 0).to_dict()
    value.pop("policy_artifact_sha256")
    restored = RolloutRecord.from_dict(value)
    assert restored.rollout_id == "legacy"
    assert restored.policy_artifact_sha256 is None


@pytest.mark.asyncio
async def test_backpressure_rechecks_staleness_after_policy_update() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(
            queue_capacity=1,
            min_batch_size=1,
            max_batch_size=1,
            max_policy_lag=0,
        )
    )
    assert await coordinator.submit(_record("first", 0))
    waiting = asyncio.create_task(
        coordinator.submit(_record("blocked", 0), timeout_seconds=1.0)
    )
    await asyncio.sleep(0)

    await coordinator.advance_policy()

    assert not await waiting
    stats = coordinator.stats()
    assert stats.dropped_stale == 2
    assert stats.queue_depth == 0


@pytest.mark.asyncio
async def test_backpressure_and_batch_waits_have_bounded_timeouts() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=1, max_batch_size=1)
    )
    assert await coordinator.submit(_record("first", 0))
    with pytest.raises(AsyncRolloutTimeout, match="capacity"):
        await coordinator.submit(_record("second", 0), timeout_seconds=0.01)

    await coordinator.next_batch()
    with pytest.raises(AsyncRolloutTimeout, match="batch"):
        await coordinator.next_batch(timeout_seconds=0.01)


@pytest.mark.asyncio
async def test_close_releases_waiters_and_allows_final_partial_batch() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=4, min_batch_size=3, max_batch_size=4)
    )
    assert await coordinator.submit(_record("only", 0))
    waiter = asyncio.create_task(coordinator.next_batch())
    await asyncio.sleep(0)

    await coordinator.close()
    batch = await waiter

    assert [record.rollout_id for record in batch.records] == ["only"]
    assert coordinator.stats().closed is True
    with pytest.raises(AsyncRolloutClosed):
        await coordinator.submit(_record("late", 0))
    with pytest.raises(AsyncRolloutClosed):
        await coordinator.next_batch()


@pytest.mark.asyncio
async def test_policy_versions_must_advance_monotonically() -> None:
    coordinator = AsyncRolloutCoordinator(initial_policy_version=2)
    with pytest.raises(ValueError, match="monotonically"):
        await coordinator.advance_policy(2)
    assert await coordinator.advance_policy() == 3
