"""Tests for content-addressed distributed policy publication."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

import stateset_agents.training as training
from stateset_agents.training.async_rollouts import (
    AsyncRolloutCoordinator,
    RolloutRecord,
)
from stateset_agents.training.distributed_rollouts import (
    DistributedRolloutConfig,
    DistributedRolloutControlPlane,
)
from stateset_agents.training.policy_artifacts import (
    PolicyArtifact,
    PolicyArtifactError,
    PolicyArtifactUnavailable,
    compute_policy_artifact_sha256,
    verify_policy_artifact,
)


def _artifact(version: int, *, digest_character: str = "a") -> PolicyArtifact:
    return PolicyArtifact(
        policy_version=version,
        uri=f"s3://stateset-policy-artifacts/run-1/policy-{version}.safetensors",
        sha256=digest_character * 64,
        size_bytes=1024 + version,
        published_at=1_000.0 + version,
    )


def test_policy_artifact_surface_is_public_and_lazy() -> None:
    assert training.PolicyArtifact is PolicyArtifact
    assert training.verify_policy_artifact is verify_policy_artifact
    assert "PolicyArtifact" in training.__all__


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"policy_version": -1}, "policy_version"),
        ({"uri": "https://user:secret@example.com/model"}, "credentials"),
        ({"uri": "https://example.com/model?token=secret"}, "query"),
        ({"uri": "relative/model.safetensors"}, "scheme"),
        ({"sha256": "A" * 64}, "sha256"),
        ({"sha256": "a" * 63}, "sha256"),
        ({"size_bytes": 0}, "size_bytes"),
        ({"published_at": float("inf")}, "published_at"),
    ],
)
def test_policy_artifact_rejects_unsafe_or_unbounded_descriptors(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "policy_version": 0,
        "uri": "https://artifacts.example.com/policy.safetensors",
        "sha256": "a" * 64,
        "size_bytes": 10,
        "published_at": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        PolicyArtifact(**values)  # type: ignore[arg-type]


def test_streaming_digest_and_verification_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "policy.safetensors"
    content = b"immutable-policy-weights"
    path.write_bytes(content)
    artifact = PolicyArtifact(
        policy_version=3,
        uri="file:///models/policy.safetensors",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        published_at=1.0,
    )

    assert compute_policy_artifact_sha256(path, chunk_size=3) == artifact.sha256
    verify_policy_artifact(path, artifact)

    path.write_bytes(content + b"tampered")
    with pytest.raises(PolicyArtifactError, match="size mismatch"):
        verify_policy_artifact(path, artifact)

    same_size_tamper = bytearray(content)
    same_size_tamper[0] ^= 1
    path.write_bytes(same_size_tamper)
    with pytest.raises(PolicyArtifactError, match="SHA-256 mismatch"):
        verify_policy_artifact(path, artifact)


@pytest.mark.asyncio
async def test_required_artifact_fails_closed_until_initial_weights_publish() -> None:
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(),
        config=DistributedRolloutConfig(require_policy_artifact=True),
    )
    with pytest.raises(PolicyArtifactUnavailable, match="current policy"):
        await control.register("worker-a")

    artifact = await control.register_initial_policy_artifact(_artifact(0))
    lease = await control.register("worker-a")
    assert artifact.policy_version == lease.policy_version == 0
    assert await control.policy_artifact(lease.policy_version) == artifact


@pytest.mark.asyncio
async def test_policy_artifact_is_visible_before_new_version_assignment() -> None:
    coordinator = AsyncRolloutCoordinator()
    control = DistributedRolloutControlPlane(
        coordinator=coordinator,
        config=DistributedRolloutConfig(require_policy_artifact=True),
    )
    await control.register_initial_policy_artifact(_artifact(0))
    lease = await control.register("worker-a")

    entered_advance = asyncio.Event()
    release_advance = asyncio.Event()
    original_advance = coordinator.advance_policy

    async def delayed_advance(version: int | None = None) -> int:
        entered_advance.set()
        await release_advance.wait()
        return await original_advance(version)

    coordinator.advance_policy = delayed_advance  # type: ignore[method-assign]
    publication = asyncio.create_task(control.publish_policy_artifact(_artifact(1)))
    await entered_advance.wait()
    heartbeat = asyncio.create_task(control.heartbeat("worker-a", lease.lease_id))
    await asyncio.sleep(0)
    assert not heartbeat.done()

    release_advance.set()
    await publication
    renewed = await heartbeat
    assert renewed.policy_version == 1
    assert (await control.policy_artifact(1)) == _artifact(1)


@pytest.mark.asyncio
async def test_artifact_required_rollouts_retain_and_match_exact_weight_bytes() -> None:
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(),
        config=DistributedRolloutConfig(require_policy_artifact=True),
    )
    artifact = await control.register_initial_policy_artifact(_artifact(0))
    lease = await control.register("worker-a")

    def record(label: str, digest: str | None) -> RolloutRecord:
        return RolloutRecord(
            rollout_id=label,
            policy_version=0,
            sampler_log_probs=(-0.5,),
            payload={"token": 1},
            policy_artifact_sha256=digest,
        )

    with pytest.raises(PolicyArtifactError, match="must identify"):
        await control.submit("worker-a", lease.lease_id, record("missing", None))
    with pytest.raises(PolicyArtifactError, match="does not match"):
        await control.submit("worker-a", lease.lease_id, record("wrong", "b" * 64))
    assert await control.submit(
        "worker-a", lease.lease_id, record("exact", artifact.sha256)
    )
    batch = await control.coordinator.next_batch()
    assert batch.records[0].policy_artifact_sha256 == artifact.sha256


@pytest.mark.asyncio
async def test_failed_policy_advance_rolls_back_staged_artifact() -> None:
    coordinator = AsyncRolloutCoordinator()
    control = DistributedRolloutControlPlane(coordinator=coordinator)

    async def fail_advance(_version: int | None = None) -> int:
        raise RuntimeError("optimizer checkpoint rejected")

    coordinator.advance_policy = fail_advance  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="checkpoint rejected"):
        await control.publish_policy_artifact(_artifact(1))
    assert await control.policy_artifact(1) is None
    assert coordinator.current_policy_version == 0


@pytest.mark.asyncio
async def test_artifact_history_is_bounded_and_checkpointed() -> None:
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(),
        config=DistributedRolloutConfig(policy_artifact_capacity=2),
    )
    await control.register_initial_policy_artifact(_artifact(0))
    await control.publish_policy_artifact(_artifact(1))
    await control.publish_policy_artifact(_artifact(2))

    assert await control.policy_artifact(0) is None
    assert await control.policy_artifact(1) == _artifact(1)
    state = await control.state_dict()
    assert state["schema_version"] == 2
    assert [item["policy_version"] for item in state["policy_artifacts"]] == [1, 2]

    restored = await DistributedRolloutControlPlane.from_state_dict(state)
    assert await restored.policy_artifact(1) == _artifact(1)
    assert await restored.policy_artifact(2) == _artifact(2)


@pytest.mark.asyncio
async def test_artifact_versions_are_immutable_and_monotonic() -> None:
    control = DistributedRolloutControlPlane(coordinator=AsyncRolloutCoordinator())
    await control.register_initial_policy_artifact(_artifact(0))
    assert await control.register_initial_policy_artifact(_artifact(0)) == _artifact(0)

    with pytest.raises(PolicyArtifactError, match="different artifact"):
        await control.register_initial_policy_artifact(
            _artifact(0, digest_character="b")
        )
    with pytest.raises(PolicyArtifactError, match="next policy version"):
        await control.publish_policy_artifact(_artifact(2))
