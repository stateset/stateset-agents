"""Behavioral tests for the transport-neutral rollout control plane."""

from __future__ import annotations

import asyncio
import math

import pytest

import stateset_agents.training as training
from stateset_agents.training.async_rollouts import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    RolloutRecord,
)
from stateset_agents.training.distributed_rollouts import (
    DistributedRolloutConfig,
    DistributedRolloutControlPlane,
    WorkerCapacityError,
    WorkerLeaseError,
    WorkerLeaseExpired,
)


class FakeClock:
    """Deterministic wall clock for lease and restart tests."""

    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _record(label: str, version: int) -> RolloutRecord:
    return RolloutRecord(
        rollout_id=label,
        policy_version=version,
        sampler_log_probs=(-0.5,),
        payload={"token": 1},
    )


def test_distributed_surface_is_public_and_lazy() -> None:
    assert training.DistributedRolloutControlPlane is DistributedRolloutControlPlane
    assert "DistributedRolloutControlPlane" in training.__all__


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"lease_ttl_seconds": 0}, "lease_ttl_seconds"),
        ({"lease_ttl_seconds": math.inf}, "lease_ttl_seconds"),
        ({"max_workers": 0}, "max_workers"),
        ({"max_workers": True}, "max_workers"),
        ({"worker_history_capacity": 0}, "worker_history_capacity"),
        (
            {"max_workers": 2, "worker_history_capacity": 1},
            "worker_history_capacity",
        ),
        ({"max_worker_id_length": 0}, "max_worker_id_length"),
        ({"policy_artifact_capacity": 0}, "policy_artifact_capacity"),
        ({"require_policy_artifact": 1}, "require_policy_artifact"),
    ],
)
def test_config_rejects_unbounded_values(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DistributedRolloutConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_registration_heartbeat_and_submission_are_policy_exact() -> None:
    clock = FakeClock()
    coordinator = AsyncRolloutCoordinator()
    control = DistributedRolloutControlPlane(coordinator=coordinator, clock=clock)

    lease = await control.register("worker-a")
    assert lease.generation == 0
    assert lease.policy_version == 0
    assert lease.expires_at == 1_030.0

    clock.advance(5)
    renewed = await control.heartbeat("worker-a", lease.lease_id)
    assert renewed.lease_id == lease.lease_id
    assert renewed.expires_at == 1_035.0
    assert await control.submit("worker-a", renewed.lease_id, _record("rollout-0", 0))

    batch = await coordinator.next_batch()
    assert [record.rollout_id for record in batch.records] == ["rollout-0"]
    stats = await control.stats()
    assert stats.active_workers == 1
    assert stats.accepted_submissions == 1


@pytest.mark.asyncio
async def test_heartbeat_assigns_new_policy_and_rejects_wrong_snapshot() -> None:
    coordinator = AsyncRolloutCoordinator()
    control = DistributedRolloutControlPlane(coordinator=coordinator)
    lease = await control.register("worker-a")
    await coordinator.advance_policy()

    # Until its next heartbeat, the worker retains its explicit old assignment.
    assert await control.submit("worker-a", lease.lease_id, _record("old-but-valid", 0))
    renewed = await control.heartbeat("worker-a", lease.lease_id)
    assert renewed.policy_version == 1

    with pytest.raises(WorkerLeaseError, match="assignment"):
        await control.submit("worker-a", renewed.lease_id, _record("wrong-version", 0))
    assert await control.submit("worker-a", renewed.lease_id, _record("current", 1))
    assert (await control.stats()).rejected_policy_assignment == 1


@pytest.mark.asyncio
async def test_reregistration_fences_the_previous_worker_generation() -> None:
    control = DistributedRolloutControlPlane(coordinator=AsyncRolloutCoordinator())
    first = await control.register("worker-a")
    second = await control.register("worker-a")

    assert second.generation == first.generation + 1
    assert second.lease_id != first.lease_id
    with pytest.raises(WorkerLeaseError, match="no longer current"):
        await control.heartbeat("worker-a", first.lease_id)
    assert (await control.stats()).rejected_stale_lease == 1
    assert (await control.stats()).replaced_leases == 1


@pytest.mark.asyncio
async def test_capacity_reclaims_expired_workers() -> None:
    clock = FakeClock()
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(),
        config=DistributedRolloutConfig(lease_ttl_seconds=10, max_workers=1),
        clock=clock,
    )
    await control.register("worker-a")
    with pytest.raises(WorkerCapacityError, match="capacity"):
        await control.register("worker-b")

    clock.advance(10)
    replacement = await control.register("worker-b")
    assert replacement.worker_id == "worker-b"
    stats = await control.stats()
    assert stats.active_workers == 1
    assert stats.expired_leases == 1


@pytest.mark.asyncio
async def test_worker_identity_and_generation_history_are_bounded() -> None:
    clock = FakeClock()
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(),
        config=DistributedRolloutConfig(
            lease_ttl_seconds=1,
            max_workers=1,
            worker_history_capacity=1,
            max_worker_id_length=8,
        ),
        clock=clock,
    )
    with pytest.raises(ValueError, match="max_worker_id_length"):
        await control.register("worker-too-long")

    await control.register("worker-a")
    clock.advance(1)
    with pytest.raises(WorkerCapacityError, match="history capacity"):
        await control.register("worker-b")


@pytest.mark.asyncio
async def test_expired_and_unknown_workers_are_audited_separately() -> None:
    clock = FakeClock()
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(),
        config=DistributedRolloutConfig(lease_ttl_seconds=5),
        clock=clock,
    )
    lease = await control.register("worker-a")
    clock.advance(5)

    with pytest.raises(WorkerLeaseExpired, match="expired"):
        await control.heartbeat("worker-a", lease.lease_id)
    with pytest.raises(WorkerLeaseError, match="not registered"):
        await control.heartbeat("missing", "lease")

    stats = await control.stats()
    assert stats.expired_leases == 1
    assert stats.rejected_expired_lease == 1
    assert stats.rejected_unknown_worker == 1


@pytest.mark.asyncio
async def test_health_is_sorted_and_never_exposes_lease_ids() -> None:
    clock = FakeClock()
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(), clock=clock
    )
    await control.register("worker-b")
    await control.register("worker-a")

    health = await control.health()
    assert [item.worker_id for item in health] == ["worker-a", "worker-b"]
    assert all(item.lease_seconds_remaining == 30.0 for item in health)
    assert "lease_id" not in repr(health)
    assert "lease_id" not in (await control.stats()).to_dict()


@pytest.mark.asyncio
async def test_unregister_requires_the_current_lease() -> None:
    control = DistributedRolloutControlPlane(coordinator=AsyncRolloutCoordinator())
    first = await control.register("worker-a")
    current = await control.register("worker-a")

    with pytest.raises(WorkerLeaseError, match="no longer current"):
        await control.unregister("worker-a", first.lease_id)
    await control.unregister("worker-a", current.lease_id)
    assert (await control.stats()).active_workers == 0


@pytest.mark.asyncio
async def test_checkpoint_restores_queue_fencing_and_live_leases() -> None:
    clock = FakeClock()
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=4, max_batch_size=4)
    )
    control = DistributedRolloutControlPlane(coordinator=coordinator, clock=clock)
    lease = await control.register("worker-a")
    assert await control.submit(
        "worker-a", lease.lease_id, _record("before-restart", 0)
    )

    state = await control.state_dict()
    restored = await DistributedRolloutControlPlane.from_state_dict(state, clock=clock)
    restored_health = await restored.health()
    assert restored_health[0].generation == lease.generation
    assert restored_health[0].policy_version == lease.policy_version

    # The same lease continues safely, while rollout-ID dedup survives restart.
    assert not await restored.submit(
        "worker-a", lease.lease_id, _record("before-restart", 0)
    )
    assert await restored.submit(
        "worker-a", lease.lease_id, _record("after-restart", 0)
    )
    batch = await restored.coordinator.next_batch(min_size=2, max_size=2)
    assert {record.rollout_id for record in batch.records} == {
        "before-restart",
        "after-restart",
    }


@pytest.mark.asyncio
async def test_checkpoint_waits_for_admitted_backpressured_submission() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=1, max_batch_size=1)
    )
    control = DistributedRolloutControlPlane(coordinator=coordinator)
    lease = await control.register("worker-a")
    assert await control.submit("worker-a", lease.lease_id, _record("fills-queue", 0))

    blocked_submit = asyncio.create_task(
        control.submit("worker-a", lease.lease_id, _record("in-flight", 0))
    )
    await asyncio.sleep(0)
    checkpoint = asyncio.create_task(control.state_dict())
    await asyncio.sleep(0)
    assert not blocked_submit.done()
    assert not checkpoint.done()

    first = await coordinator.next_batch()
    assert first.records[0].rollout_id == "fills-queue"
    assert await blocked_submit
    state = await checkpoint
    assert state["counters"]["accepted_submissions"] == 2
    assert state["coordinator"]["queue"][0]["rollout_id"] == "in-flight"


@pytest.mark.asyncio
async def test_cancelled_checkpoint_releases_new_submissions() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=1, max_batch_size=1)
    )
    control = DistributedRolloutControlPlane(coordinator=coordinator)
    lease = await control.register("worker-a")
    assert await control.submit("worker-a", lease.lease_id, _record("first", 0))

    blocked_submit = asyncio.create_task(
        control.submit("worker-a", lease.lease_id, _record("second", 0))
    )
    await asyncio.sleep(0)
    checkpoint = asyncio.create_task(control.state_dict())
    await asyncio.sleep(0)
    checkpoint.cancel()
    with pytest.raises(asyncio.CancelledError):
        await checkpoint

    await coordinator.next_batch()
    assert await blocked_submit
    await coordinator.next_batch()
    assert await asyncio.wait_for(
        control.submit("worker-a", lease.lease_id, _record("third", 0)),
        timeout=0.5,
    )


@pytest.mark.asyncio
async def test_restore_discards_expired_lease_but_preserves_generation_fence() -> None:
    clock = FakeClock()
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(),
        config=DistributedRolloutConfig(lease_ttl_seconds=5),
        clock=clock,
    )
    old = await control.register("worker-a")
    state = await control.state_dict()
    clock.advance(5)

    restored = await DistributedRolloutControlPlane.from_state_dict(state, clock=clock)
    assert (await restored.stats()).expired_leases == 1
    assert (await restored.stats()).active_workers == 0
    replacement = await restored.register("worker-a")
    assert replacement.generation == old.generation + 1


@pytest.mark.asyncio
async def test_checkpoint_schema_rejects_corruption() -> None:
    clock = FakeClock()
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(), clock=clock
    )
    await control.register("worker-a")
    state = await control.state_dict()

    bad_counters = dict(state)
    bad_counters["counters"] = {"accepted_submissions": -1}
    with pytest.raises(ValueError, match="counters"):
        await DistributedRolloutControlPlane.from_state_dict(bad_counters, clock=clock)

    bad_generation = dict(state)
    bad_generation["generations"] = {"worker-a": 99}
    with pytest.raises(ValueError, match="generation"):
        await DistributedRolloutControlPlane.from_state_dict(
            bad_generation, clock=clock
        )


@pytest.mark.asyncio
async def test_invalid_clock_fails_closed_before_mutating_registry() -> None:
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(), clock=lambda: math.nan
    )
    with pytest.raises(WorkerLeaseError, match="clock"):
        await control.register("worker-a")
