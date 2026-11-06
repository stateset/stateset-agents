"""Exercise upstream failure semantics through both public inference routes."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from stateset_agents.api import config as api_config
from stateset_agents.api.dependencies import get_inference_service
from stateset_agents.api.services import inference_service
from stateset_agents.api.services.inference_service import (
    InferenceConfig,
    InferenceService,
)


@pytest_asyncio.fixture
async def gateway(monkeypatch):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("API_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setattr(api_config, "_config", None)
    monkeypatch.setattr(inference_service.asyncio, "sleep", AsyncMock())
    service = InferenceService(
        InferenceConfig(backend="openai", base_url="https://upstream.invalid")
    )
    from stateset_agents.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client:
            yield client, service
    finally:
        await service.aclose()


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize(
    "upstream_status,retry_after,expected_status,expected_calls",
    [
        (429, "17", 429, 1),
        (429, "Fri, 06 Nov 2026 16:00:00 GMT", 429, 1),
        (429, None, 429, 1),
        (401, None, 502, 1),
        (403, None, 502, 1),
        (400, None, 422, 1),
        (503, None, 502, 3),
        (503, "17", 502, 1),
    ],
)
async def test_upstream_status_and_retry_policy(
    gateway, route, upstream_status, retry_after, expected_status, expected_calls
):
    client, service = gateway
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(
            upstream_status,
            json={"error": {"message": "upstream diagnostic"}},
            headers={"Retry-After": retry_after} if retry_after else {},
        )

    service._client = httpx.AsyncClient(
        base_url=service.config.base_url, transport=httpx.MockTransport(upstream)
    )
    response = await client.post(
        route,
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == expected_status, response.text
    assert response.headers.get("retry-after") == retry_after
    assert len(calls) == expected_calls
    if upstream_status in {401, 403}:
        assert "upstream diagnostic" not in response.text
        assert "authentication" in response.text.lower()


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/messages"])
async def test_transient_connection_failure_can_recover(gateway, route):
    client, service = gateway
    calls = []

    def upstream(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("temporarily unavailable", request=request)
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    service._client = httpx.AsyncClient(
        base_url=service.config.base_url, transport=httpx.MockTransport(upstream)
    )
    response = await client.post(
        route,
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200, response.text
    assert len(calls) == 2
