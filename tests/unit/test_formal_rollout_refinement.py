"""Bounded operation traces against the rollout safety model in formal/tla."""

from __future__ import annotations

import itertools

import pytest

from stateset_agents.training.async_rollouts import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    AsyncRolloutTimeout,
    RolloutRecord,
)
from stateset_agents.training.distributed_rollouts import (
    DistributedRolloutConfig,
    DistributedRolloutControlPlane,
    WorkerLease,
    WorkerLeaseError,
)
from stateset_agents.training.policy_artifacts import (
    PolicyArtifact,
    PolicyArtifactError,
    PolicyArtifactUnavailable,
)


def _artifact(version: int) -> PolicyArtifact:
    return PolicyArtifact(
        policy_version=version,
        uri=f"s3://stateset-test/policy-{version}",
        sha256=f"{version + 1:x}" * 64,
        size_bytes=1,
        published_at=1_000.0 + version,
    )


async def _assert_snapshot(control: DistributedRolloutControlPlane) -> None:
    state = await control.state_dict()
    queue = state["coordinator"]["queue"]
    seen = state["coordinator"]["seen_rollout_ids"]
    version = state["coordinator"]["current_policy_version"]
    assert len(queue) <= control.coordinator.config.queue_capacity
    assert len({item["rollout_id"] for item in queue}) == len(queue)
    assert {item["rollout_id"] for item in queue}.issubset(seen)
    assert all(
        0
        <= version - item["policy_version"]
        <= control.coordinator.config.max_policy_lag
        for item in queue
    )
    assert await control.policy_artifact(version) is not None
    assert (await control.stats()).accepted_submissions == state["coordinator"][
        "counters"
    ]["submitted"]


@pytest.mark.asyncio
async def test_bounded_rollout_operation_traces_preserve_safety() -> None:
    """Check all four-step traces over the model's core public operations."""
    actions = ("register", "publish", "heartbeat", "submit", "old", "consume")
    for trace in itertools.product(actions, repeat=4):
        coordinator = AsyncRolloutCoordinator(
            AsyncRolloutConfig(queue_capacity=2, max_batch_size=2, max_policy_lag=2)
        )
        control = DistributedRolloutControlPlane(
            coordinator=coordinator,
            config=DistributedRolloutConfig(
                policy_artifact_capacity=1,
                require_policy_artifact=True,
            ),
            clock=lambda: 1_000.0,
        )
        await control.register_initial_policy_artifact(_artifact(0))
        leases: list[WorkerLease] = []
        for step, action in enumerate(trace):
            if action == "register":
                leases.append(await control.register("worker"))
            elif action == "publish":
                if coordinator.current_policy_version < 2:
                    await control.publish_policy_artifact(
                        _artifact(coordinator.current_policy_version + 1)
                    )
            elif action == "heartbeat" and leases:
                leases[-1] = await control.heartbeat("worker", leases[-1].lease_id)
            elif action in {"submit", "old"} and leases:
                lease = leases[0] if action == "old" else leases[-1]
                artifact = await control.policy_artifact(lease.policy_version)
                record = RolloutRecord(
                    rollout_id=f"{step}-{action}",
                    policy_version=lease.policy_version,
                    sampler_log_probs=(-0.5,),
                    payload={},
                    policy_artifact_sha256=(
                        artifact.sha256 if artifact is not None else "f" * 64
                    ),
                )
                try:
                    accepted = await control.submit(
                        "worker", lease.lease_id, record, timeout_seconds=0.001
                    )
                    if lease.lease_id != leases[-1].lease_id:
                        assert not accepted, trace
                except (
                    WorkerLeaseError,
                    PolicyArtifactError,
                    PolicyArtifactUnavailable,
                    AsyncRolloutTimeout,
                ):
                    pass
            elif action == "consume" and coordinator.stats().queue_depth:
                await coordinator.next_batch(min_size=1, max_size=1)
            await _assert_snapshot(control)
