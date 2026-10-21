"""Regression tests for asynchronous rollout control-plane edge cases.

Covers checkpointing while producers are backpressured, cancellation of timed
waits after records changed hands, heartbeats during long generations, and
PEFT weight sync to a loader that materializes the iterable first.
"""

from __future__ import annotations

import asyncio

import pytest

from stateset_agents.training.async_rollouts import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    RolloutRecord,
)
from stateset_agents.training.distributed_rollouts import (
    DistributedRolloutControlPlane,
    WorkerLeaseError,
)


def _record(label: str, version: int) -> RolloutRecord:
    return RolloutRecord(
        rollout_id=label,
        policy_version=version,
        sampler_log_probs=(-1.0,),
        payload={},
    )


# --- checkpoint while the queue is full ------------------------------------


@pytest.mark.asyncio
async def test_learner_can_checkpoint_while_a_producer_is_backpressured() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=1, max_batch_size=1)
    )
    control = DistributedRolloutControlPlane(coordinator=coordinator)
    lease = await control.register("worker-a")
    assert await control.submit("worker-a", lease.lease_id, _record("full", 0))
    blocked = asyncio.create_task(
        control.submit("worker-a", lease.lease_id, _record("waiting", 0))
    )
    await asyncio.sleep(0)
    assert not blocked.done()

    # The learner checkpoints between steps, before draining the next batch.
    state = await asyncio.wait_for(control.state_dict(), timeout=2.0)

    assert [r["rollout_id"] for r in state["coordinator"]["queue"]] == ["full"]
    assert state["counters"]["accepted_submissions"] == 1
    await coordinator.next_batch()
    assert await asyncio.wait_for(blocked, timeout=2.0)
    assert (await control.stats()).accepted_submissions == 2


# --- cancellation of timed waits -------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_timed_next_batch_never_loses_records() -> None:
    coordinator = AsyncRolloutCoordinator()
    waiter = asyncio.create_task(coordinator.next_batch(timeout_seconds=5.0))
    await asyncio.sleep(0.01)
    await coordinator.submit(_record("a", 0))
    waiter.cancel()
    try:
        batch = await waiter
    except asyncio.CancelledError:
        # Cancelled before taking the record: it must still be queued.
        stats = coordinator.stats()
        assert stats.queue_depth == 1
        assert stats.consumed == 0
        retry = await coordinator.next_batch(timeout_seconds=1.0)
        assert [r.rollout_id for r in retry.records] == ["a"]
    else:
        assert [r.rollout_id for r in batch.records] == ["a"]


@pytest.mark.asyncio
async def test_cancelled_timed_submit_is_either_admitted_or_absent() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=1, max_batch_size=1)
    )
    assert await coordinator.submit(_record("full", 0))
    submitter = asyncio.create_task(
        coordinator.submit(_record("late", 0), timeout_seconds=5.0)
    )
    await asyncio.sleep(0.01)
    await coordinator.next_batch()
    submitter.cancel()
    try:
        admitted = await submitter
    except asyncio.CancelledError:
        admitted = False
    stats = coordinator.stats()
    # Counted as submitted exactly when it is actually in the queue, and a
    # producer retry of an unadmitted record is not rejected as a duplicate.
    assert stats.queue_depth == stats.submitted - 1
    if not admitted and stats.queue_depth == 0:
        assert await coordinator.submit(_record("late", 0))


@pytest.mark.asyncio
async def test_timeouts_still_raise_when_nothing_arrives() -> None:
    from stateset_agents.training.async_rollouts import AsyncRolloutTimeout

    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=1, max_batch_size=1)
    )
    with pytest.raises(AsyncRolloutTimeout, match="batch"):
        await coordinator.next_batch(timeout_seconds=0.01)
    assert await coordinator.submit(_record("full", 0))
    with pytest.raises(AsyncRolloutTimeout, match="capacity"):
        await coordinator.submit(_record("blocked", 0), timeout_seconds=0.01)
    # The timed-out record was never admitted, so a retry is not a duplicate.
    await coordinator.next_batch()
    assert await coordinator.submit(_record("blocked", 0))


# --- heartbeats during long generations ------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_mid_generation_keeps_rollout_within_policy_lag() -> None:
    coordinator = AsyncRolloutCoordinator(AsyncRolloutConfig(max_policy_lag=1))
    control = DistributedRolloutControlPlane(coordinator=coordinator)
    lease = await control.register("worker-a")
    assert lease.policy_version == 0

    # Generation of a v0 sample outlives a learner step; the worker renews its
    # lease (as it must before the TTL) and is reassigned v1.
    await coordinator.advance_policy()
    renewed = await control.heartbeat("worker-a", lease.lease_id)
    assert renewed.policy_version == 1

    assert await control.submit("worker-a", lease.lease_id, _record("v0", 0))
    assert (await control.stats()).rejected_policy_assignment == 0


@pytest.mark.asyncio
async def test_heartbeat_does_not_admit_samples_beyond_policy_lag() -> None:
    coordinator = AsyncRolloutCoordinator(AsyncRolloutConfig(max_policy_lag=0))
    control = DistributedRolloutControlPlane(coordinator=coordinator)
    lease = await control.register("worker-a")
    await coordinator.advance_policy()
    await control.heartbeat("worker-a", lease.lease_id)

    assert not await control.submit("worker-a", lease.lease_id, _record("v0", 0))
    assert coordinator.stats().dropped_stale == 1


@pytest.mark.asyncio
async def test_record_newer_than_the_assignment_is_still_rejected() -> None:
    coordinator = AsyncRolloutCoordinator()
    control = DistributedRolloutControlPlane(coordinator=coordinator)
    lease = await control.register("worker-a")
    await coordinator.advance_policy()  # learner is at v1, lease still at v0
    with pytest.raises(WorkerLeaseError, match="assignment"):
        await control.submit("worker-a", lease.lease_id, _record("v1", 1))


# --- PEFT weight sync -------------------------------------------------------


def _peft_model():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = torch.nn.Linear(4, 4, bias=False)
            self.head = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):  # pragma: no cover - not called
            return self.head(self.q(x))

    torch.manual_seed(0)
    model = peft.get_peft_model(
        Tiny(),
        peft.LoraConfig(
            r=2,
            target_modules=["q"],
            modules_to_save=["head"],
            init_lora_weights=False,
        ),
    )
    layer = model.get_base_model().q
    merged = (
        layer.base_layer.weight.detach().clone()
        + layer.lora_B["default"].weight
        @ layer.lora_A["default"].weight
        * layer.scaling["default"]
    ).detach()
    return torch, model, merged


def test_materializing_weight_loader_receives_merged_lora_weights() -> None:
    from stateset_agents.training.vllm_backend import _iter_policy_weights

    torch, model, merged = _peft_model()
    weights = dict(list(_iter_policy_weights(model)))  # copy-after-collect loader
    assert torch.allclose(weights["q.weight"], merged)


def test_modules_to_save_sync_under_engine_names() -> None:
    from stateset_agents.training.vllm_backend import _iter_policy_weights

    torch, model, _ = _peft_model()
    trained = model.get_base_model().head.modules_to_save["default"].weight
    with torch.no_grad():
        trained.add_(1.0)
    weights = dict(_iter_policy_weights(model))
    assert set(weights) == {"q.weight", "head.weight"}
    assert torch.allclose(weights["head.weight"], trained)
