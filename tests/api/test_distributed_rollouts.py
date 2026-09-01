"""End-to-end tests for the authenticated distributed-rollout transport."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from fastapi import FastAPI

from stateset_agents.api import attach_distributed_rollout_control_plane
from stateset_agents.api import config as api_config
from stateset_agents.api.routers.distributed_rollouts import router
from stateset_agents.training.async_rollouts import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
)
from stateset_agents.training.distributed_rollouts import (
    DistributedRolloutConfig,
    DistributedRolloutControlPlane,
)
from stateset_agents.training.policy_artifacts import PolicyArtifact


class FakeClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def preserve_api_config() -> Iterator[None]:
    previous = api_config._config
    yield
    api_config._config = previous


def _configure_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv(
        "API_KEYS",
        "worker-key-12345:rollout_worker,second-key-12345:rollout_worker,"
        "admin-key-12345:admin,reader-key-12345:user",
    )
    monkeypatch.setenv("API_MAX_REQUEST_SIZE_MB", "1")
    api_config.reload_config()


def _app(control: DistributedRolloutControlPlane | None = None) -> FastAPI:
    app = FastAPI()
    if control is not None:
        attach_distributed_rollout_control_plane(app, control)
    app.include_router(router)
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


def test_control_plane_attachment_is_validated() -> None:
    app = FastAPI()
    with pytest.raises(TypeError, match="DistributedRolloutControlPlane"):
        attach_distributed_rollout_control_plane(app, object())  # type: ignore[arg-type]


def _rollout(
    lease_id: str,
    *,
    rollout_id: str = "sample-1",
    version: int = 0,
    artifact_sha256: str | None = None,
):
    return {
        "lease_id": lease_id,
        "rollout": {
            "rollout_id": rollout_id,
            "policy_version": version,
            "sampler_log_probs": [-0.5, -0.2],
            "payload": {"tokens": [1, 2], "reward": 1.0},
            "policy_artifact_sha256": artifact_sha256,
        },
        "timeout_seconds": 1.0,
    }


@pytest.mark.asyncio
async def test_transport_requires_explicit_auth_even_in_anonymous_api_mode(
    monkeypatch: pytest.MonkeyPatch, preserve_api_config: None
) -> None:
    _configure_auth(monkeypatch)
    app = _app(DistributedRolloutControlPlane(coordinator=AsyncRolloutCoordinator()))
    async with _client(app) as client:
        missing = await client.post("/api/v1/rollouts/workers/worker-1/register")
        invalid = await client.post(
            "/api/v1/rollouts/workers/worker-1/register",
            headers={"X-API-Key": "invalid"},
        )
        forbidden = await client.post(
            "/api/v1/rollouts/workers/worker-1/register",
            headers={"X-API-Key": "reader-key-12345"},
        )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert invalid.status_code == 401
    assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_unconfigured_control_plane_reports_not_ready(
    monkeypatch: pytest.MonkeyPatch, preserve_api_config: None
) -> None:
    _configure_auth(monkeypatch)
    async with _client(_app()) as client:
        anonymous = await client.post("/api/v1/rollouts/workers/worker-1/register")
        response = await client.post(
            "/api/v1/rollouts/workers/worker-1/register",
            headers={"X-API-Key": "worker-key-12345"},
        )
    assert anonymous.status_code == 401
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


@pytest.mark.asyncio
async def test_remote_worker_lifecycle_and_admin_observability(
    monkeypatch: pytest.MonkeyPatch, preserve_api_config: None
) -> None:
    _configure_auth(monkeypatch)
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(queue_capacity=4, max_batch_size=4)
    )
    control = DistributedRolloutControlPlane(coordinator=coordinator)
    headers = {"X-API-Key": "worker-key-12345"}
    async with _client(_app(control)) as client:
        registered = await client.post(
            "/api/v1/rollouts/workers/worker-1/register", headers=headers
        )
        assert registered.status_code == 200
        lease = registered.json()
        assert lease["worker_id"] == "worker-1"
        assert lease["policy_version"] == 0

        heartbeat = await client.post(
            "/api/v1/rollouts/workers/worker-1/heartbeat",
            headers=headers,
            json={"lease_id": lease["lease_id"]},
        )
        assert heartbeat.status_code == 200

        submitted = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers=headers,
            json=_rollout(lease["lease_id"]),
        )
        duplicate = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers=headers,
            json=_rollout(lease["lease_id"]),
        )
        assert submitted.json() == {"accepted": True}
        assert duplicate.json() == {"accepted": False}

        worker_health = await client.get("/api/v1/rollouts/workers", headers=headers)
        assert worker_health.status_code == 403
        health = await client.get(
            "/api/v1/rollouts/workers", headers={"X-API-Key": "admin-key-12345"}
        )
        stats = await client.get(
            "/api/v1/rollouts/stats", headers={"X-API-Key": "admin-key-12345"}
        )
        assert health.status_code == 200
        assert "lease_id" not in health.text
        assert health.json()["workers"][0]["worker_id"].endswith(":worker-1")
        assert stats.json()["stats"]["accepted_submissions"] == 1

        removed = await client.request(
            "DELETE",
            "/api/v1/rollouts/workers/worker-1",
            headers=headers,
            json={"lease_id": lease["lease_id"]},
        )
        assert removed.status_code == 204

    batch = await coordinator.next_batch()
    assert batch.records[0].rollout_id == "sample-1"


@pytest.mark.asyncio
async def test_authenticated_principals_cannot_cross_worker_namespaces(
    monkeypatch: pytest.MonkeyPatch, preserve_api_config: None
) -> None:
    _configure_auth(monkeypatch)
    control = DistributedRolloutControlPlane(coordinator=AsyncRolloutCoordinator())
    async with _client(_app(control)) as client:
        first = await client.post(
            "/api/v1/rollouts/workers/shared/register",
            headers={"X-API-Key": "worker-key-12345"},
        )
        first_lease = first.json()["lease_id"]
        cross_tenant = await client.post(
            "/api/v1/rollouts/workers/shared/heartbeat",
            headers={"X-API-Key": "second-key-12345"},
            json={"lease_id": first_lease},
        )
        second = await client.post(
            "/api/v1/rollouts/workers/shared/register",
            headers={"X-API-Key": "second-key-12345"},
        )

    assert cross_tenant.status_code == 409
    assert "not registered" in cross_tenant.json()["detail"]
    assert second.status_code == 200
    assert second.json()["generation"] == 0
    assert (await control.stats()).active_workers == 2


@pytest.mark.asyncio
async def test_transport_maps_fencing_policy_expiry_and_capacity_errors(
    monkeypatch: pytest.MonkeyPatch, preserve_api_config: None
) -> None:
    _configure_auth(monkeypatch)
    clock = FakeClock()
    coordinator = AsyncRolloutCoordinator()
    control = DistributedRolloutControlPlane(
        coordinator=coordinator,
        config=DistributedRolloutConfig(
            lease_ttl_seconds=5, max_workers=1, worker_history_capacity=2
        ),
        clock=clock,
    )
    headers = {"X-API-Key": "worker-key-12345"}
    async with _client(_app(control)) as client:
        first = await client.post(
            "/api/v1/rollouts/workers/worker-1/register", headers=headers
        )
        old_lease = first.json()["lease_id"]
        replacement = await client.post(
            "/api/v1/rollouts/workers/worker-1/register", headers=headers
        )
        current_lease = replacement.json()["lease_id"]

        fenced = await client.post(
            "/api/v1/rollouts/workers/worker-1/heartbeat",
            headers=headers,
            json={"lease_id": old_lease},
        )
        full = await client.post(
            "/api/v1/rollouts/workers/worker-2/register", headers=headers
        )
        wrong_policy = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers=headers,
            json=_rollout(current_lease, version=1),
        )
        clock.value += 5
        expired = await client.post(
            "/api/v1/rollouts/workers/worker-1/heartbeat",
            headers=headers,
            json={"lease_id": current_lease},
        )

    assert fenced.status_code == 409
    assert full.status_code == 429
    assert full.headers["retry-after"] == "1"
    assert wrong_policy.status_code == 409
    assert expired.status_code == 410


@pytest.mark.asyncio
async def test_http_assignment_fails_closed_and_returns_content_addressed_weights(
    monkeypatch: pytest.MonkeyPatch, preserve_api_config: None
) -> None:
    _configure_auth(monkeypatch)
    control = DistributedRolloutControlPlane(
        coordinator=AsyncRolloutCoordinator(),
        config=DistributedRolloutConfig(require_policy_artifact=True),
    )
    headers = {"X-API-Key": "worker-key-12345"}
    initial = PolicyArtifact(
        policy_version=0,
        uri="s3://stateset-policy-artifacts/run-1/policy-0.safetensors",
        sha256="a" * 64,
        size_bytes=1024,
        published_at=1_000.0,
    )
    updated = PolicyArtifact(
        policy_version=1,
        uri="s3://stateset-policy-artifacts/run-1/policy-1.safetensors",
        sha256="b" * 64,
        size_bytes=1025,
        published_at=1_001.0,
    )

    async with _client(_app(control)) as client:
        unavailable = await client.post(
            "/api/v1/rollouts/workers/worker-1/register", headers=headers
        )
        assert unavailable.status_code == 503
        assert unavailable.headers["retry-after"] == "1"

        await control.register_initial_policy_artifact(initial)
        registered = await client.post(
            "/api/v1/rollouts/workers/worker-1/register", headers=headers
        )
        lease_id = registered.json()["lease_id"]
        assert registered.json()["artifact"] == initial.to_dict()

        await control.publish_policy_artifact(updated)
        heartbeat = await client.post(
            "/api/v1/rollouts/workers/worker-1/heartbeat",
            headers=headers,
            json={"lease_id": lease_id},
        )
        missing_digest = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers=headers,
            json=_rollout(lease_id, version=1),
        )
        wrong_digest = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers=headers,
            json=_rollout(lease_id, version=1, artifact_sha256="c" * 64),
        )
        accepted = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers=headers,
            json=_rollout(lease_id, version=1, artifact_sha256=updated.sha256),
        )

    assert heartbeat.status_code == 200
    assert heartbeat.json()["policy_version"] == 1
    assert heartbeat.json()["artifact"] == updated.to_dict()
    assert missing_digest.status_code == 409
    assert wrong_digest.status_code == 409
    assert accepted.json() == {"accepted": True}


@pytest.mark.asyncio
async def test_transport_validation_is_strict_and_bounded(
    monkeypatch: pytest.MonkeyPatch, preserve_api_config: None
) -> None:
    _configure_auth(monkeypatch)
    control = DistributedRolloutControlPlane(coordinator=AsyncRolloutCoordinator())
    headers = {"X-API-Key": "worker-key-12345"}
    async with _client(_app(control)) as client:
        invalid_worker = await client.post(
            "/api/v1/rollouts/workers/not%20allowed/register", headers=headers
        )
        registered = await client.post(
            "/api/v1/rollouts/workers/worker-1/register", headers=headers
        )
        lease_id = registered.json()["lease_id"]
        extra_field = _rollout(lease_id)
        extra_field["unexpected"] = True
        invalid_body = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers=headers,
            json=extra_field,
        )

    assert invalid_worker.status_code == 422
    assert invalid_body.status_code == 422


@pytest.mark.asyncio
async def test_gateway_rejects_fixed_and_chunked_oversized_rollouts(
    monkeypatch: pytest.MonkeyPatch, preserve_api_config: None
) -> None:
    _configure_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    attach_distributed_rollout_control_plane(
        app,
        DistributedRolloutControlPlane(coordinator=AsyncRolloutCoordinator()),
    )
    oversized = b"x" * (1024 * 1024 + 1)

    async def chunks() -> AsyncIterator[bytes]:
        yield oversized[:700_000]
        yield oversized[700_000:]

    async with _client(app) as client:
        fixed = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers={"X-API-Key": "worker-key-12345"},
            content=oversized,
        )
        chunked = await client.post(
            "/api/v1/rollouts/workers/worker-1/submit",
            headers={"X-API-Key": "worker-key-12345"},
            content=chunks(),
        )

    assert fixed.status_code == 413
    assert chunked.status_code == 413, chunked.text
    assert fixed.headers["x-content-type-options"] == "nosniff"
