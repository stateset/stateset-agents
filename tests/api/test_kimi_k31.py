"""Kimi K3.1 Pro's OpenAI-compatible gateway contract, without network calls."""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio

from stateset_agents.api import config as api_config
from stateset_agents.api.dependencies import get_inference_service
from stateset_agents.api.messages_models import MessagesRequest
from stateset_agents.api.services.inference_service import (
    InferenceConfig,
    InferenceService,
)

MODEL = "moonshotai/Kimi-K3.1-Pro"
ASSISTANT = {
    "role": "assistant",
    "content": None,
    "reasoning_content": "I should look up the order before answering.",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup_order", "arguments": '{"order_id":"123"}'},
        }
    ],
}


@pytest_asyncio.fixture
async def gateway(monkeypatch):
    previous = api_config._config
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "openai")
    monkeypatch.setenv("INFERENCE_BACKEND_URL", "https://api.moonshot.ai")
    monkeypatch.setenv("INFERENCE_BACKEND_API_KEY", "test-moonshot-key")
    monkeypatch.setenv("INFERENCE_MODEL_MAP", json.dumps({MODEL: "kimi-k3.1-pro"}))
    api_config.reload_config()
    captured = []

    def backend(request):
        captured.append(request)
        body = json.loads(request.content)
        assert request.url == "https://api.moonshot.ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-moonshot-key"
        if body.get("stream"):
            chunk = {
                "id": "completion-1",
                "object": "chat.completion.chunk",
                "model": "kimi-k3.1-pro",
                "choices": [{"index": 0, "delta": ASSISTANT, "finish_reason": None}],
            }
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 1,
                "model": "kimi-k3.1-pro",
                "choices": [
                    {"index": 0, "message": ASSISTANT, "finish_reason": "tool_calls"}
                ],
            },
        )

    original_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        if kwargs.get("base_url") == "https://api.moonshot.ai":
            kwargs["transport"] = httpx.MockTransport(backend)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    service = InferenceService(InferenceConfig.from_env())
    from stateset_agents.api.main import create_app

    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: service
    try:
        async with original_client(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client:
            yield client, service, captured
    finally:
        await service.aclose()
        api_config._config = previous


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "high", "max"])
async def test_hosted_kimi_preserves_assistant_tool_history(gateway, effort):
    client, service, captured = gateway
    messages = [
        {"role": "user", "content": "Where is order 123?"},
        ASSISTANT,
        {"role": "tool", "tool_call_id": "call_1", "content": "Shipped"},
    ]
    response = await client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": messages, "reasoning_effort": effort},
    )
    assert response.status_code == 200, response.text
    sent = json.loads(captured[0].content)
    assert sent["model"] == "kimi-k3.1-pro"
    assert sent["messages"] == messages
    assert sent["reasoning_effort"] == effort
    assert "temperature" not in sent
    result = response.json()
    assert result["model"] == MODEL
    assert result["choices"][0]["message"] == ASSISTANT
    assert "test-moonshot-key" not in repr(service.config)


@pytest.mark.asyncio
async def test_kimi_stream_preserves_reasoning_and_tool_deltas(gateway):
    client, _, captured = gateway
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Look up my order."}],
            "reasoning_effort": "high",
            "stream": True,
        },
    )
    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: {")]
    chunk = json.loads(lines[0][6:])
    assert chunk["model"] == MODEL
    assert chunk["choices"][0]["delta"] == ASSISTANT
    assert "data: [DONE]" in response.text
    assert json.loads(captured[0].content)["reasoning_effort"] == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["ultra", "medium", "xhigh", 1, {}, []])
async def test_invalid_effort_is_rejected_before_backend(gateway, effort):
    client, _, captured = gateway
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": effort,
        },
    )
    assert response.status_code == 422
    assert captured == []


@pytest.mark.asyncio
async def test_reasoning_history_obeys_message_limits(gateway):
    client, _, captured = gateway
    limit = api_config.get_config().validation.max_message_length
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{**ASSISTANT, "reasoning_content": "x" * (limit + 1)}],
        },
    )
    assert response.status_code == 422
    assert captured == []


@pytest.mark.asyncio
async def test_invalid_reasoning_history_is_rejected_before_backend(gateway):
    client, _, captured = gateway
    response = await client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{**ASSISTANT, "reasoning_content": {}}]},
    )
    assert response.status_code == 422
    assert captured == []


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("target", ["kimi-k3.1-pro", "kimi-k3.1-pro-highspeed"])
async def test_alias_rejects_unsupported_effort_before_backend(
    gateway, route, stream, target
):
    client, service, captured = gateway
    service.config.model_map["support-model"] = target
    response = await client.post(
        route,
        json={
            "model": "support-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "medium",
            "stream": stream,
        },
    )
    assert response.status_code == 422, response.text
    assert "low, high, or max" in response.text
    assert captured == []


async def test_direct_service_validates_resolved_model(gateway):
    _, service, captured = gateway
    service.config.model_map["support-model"] = "kimi-k3.1-pro"
    request = MessagesRequest(
        model="support-model",
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="medium",
    )
    with pytest.raises(ValueError, match="low, high, or max"):
        await service.create_openai_response(request)
    assert captured == []


@pytest.mark.parametrize("effort", ["low", "high", "max"])
async def test_alias_preserves_supported_effort_and_public_name(gateway, effort):
    client, service, captured = gateway
    service.config.model_map["support-model"] = "kimi-k3.1-pro"
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "support-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": effort,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "support-model"
    sent = json.loads(captured[0].content)
    assert sent["model"] == "kimi-k3.1-pro"
    assert sent["reasoning_effort"] == effort
