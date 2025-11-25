"""
High-signal tests for the Ultimate GRPO API service.

Focus on security, validation, and observability guarantees that make the API
production-ready.
"""

import dataclasses

import pytest
from httpx import ASGITransport, AsyncClient

from api import ultimate_grpo_service as service

pytestmark = [
    pytest.mark.api,
    pytest.mark.skipif(
        not service.FASTAPI_AVAILABLE, reason="FastAPI dependencies not available"
    ),
]


@pytest.fixture(autouse=True)
def reset_api_state():
    """Reset global API state between tests."""
    original_config = dataclasses.replace(service.API_CONFIG)

    service.RATE_LIMITER.windows.clear()
    service.API_METRICS.request_counts.clear()
    service.API_METRICS.status_counts.clear()
    service.API_METRICS.latencies.clear()
    service.API_METRICS.rate_limit_hits = 0
    service.training_jobs.clear()
    service.active_conversations.clear()

    yield

    service.API_CONFIG.api_keys = original_config.api_keys
    service.API_CONFIG.rate_limit_per_minute = original_config.rate_limit_per_minute
    service.API_CONFIG.max_prompts = original_config.max_prompts
    service.API_CONFIG.max_prompt_length = original_config.max_prompt_length
    service.API_CONFIG.max_message_length = original_config.max_message_length
    service.API_CONFIG.max_iterations = original_config.max_iterations

    service.RATE_LIMITER.windows.clear()
    service.API_METRICS.request_counts.clear()
    service.API_METRICS.status_counts.clear()
    service.API_METRICS.latencies.clear()
    service.API_METRICS.rate_limit_hits = 0
    service.training_jobs.clear()
    service.active_conversations.clear()


@pytest.fixture
async def async_client():
    """Provide an async client that runs inside the app lifespan."""
    async with service.lifespan(service.app):
        transport = ASGITransport(app=service.app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client


@pytest.mark.asyncio
async def test_auth_required_when_configured(async_client):
    """Ensure endpoints enforce API keys when configured."""
    service.API_CONFIG.api_keys = {"test-key": ["admin"]}
    service.RATE_LIMITER.windows.clear()

    unauthorized = await async_client.get("/api/metrics")
    assert unauthorized.status_code == 401

    authorized = await async_client.get(
        "/api/metrics", headers={"x-api-key": "test-key"}
    )
    assert authorized.status_code == 200
    data = authorized.json()
    assert data["rate_limit"]["auth_enabled"] is True


@pytest.mark.asyncio
async def test_rate_limit_blocks_repeated_requests(async_client):
    """Requests beyond the configured limit receive 429 responses."""
    service.API_CONFIG.api_keys = {}
    service.API_CONFIG.rate_limit_per_minute = 1
    service.RATE_LIMITER.windows.clear()

    first = await async_client.get("/api/metrics")
    assert first.status_code == 200

    second = await async_client.get("/api/metrics")
    assert second.status_code == 429
    assert service.API_METRICS.rate_limit_hits >= 1


@pytest.mark.asyncio
async def test_validation_errors_use_consistent_envelope(async_client):
    """Request validation errors return the standardized error payload."""
    response = await async_client.post(
        "/api/train",
        json={"prompts": [], "strategy": "computational", "num_iterations": 0},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["status_code"] == 422
    assert body["error"]["message"]
    assert body["request_id"]


@pytest.mark.asyncio
async def test_metrics_expose_api_snapshot(async_client):
    """API metrics include request, status, and latency snapshots."""
    await async_client.get("/")  # generate at least one request entry
    metrics_response = await async_client.get("/api/metrics")

    assert metrics_response.status_code == 200
    payload = metrics_response.json()

    assert "api" in payload
    assert "total_requests" in payload["api"]
    assert "status_codes" in payload["api"]
