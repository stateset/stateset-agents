"""QwenCloud contracts exercised through real HTTP clients with fake transport."""

import json

import httpx
import pytest
from fastapi import FastAPI

from stateset_agents.api import config as api_config
from stateset_agents.api.dependencies import get_inference_service
from stateset_agents.api.openai_models import OpenAIChatCompletionRequest
from stateset_agents.api.routers.openai import router
from stateset_agents.api.services.inference_service import (
    InferenceConfig,
    InferenceService,
)

BASE_URL = "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1"


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    for key in (
        "INFERENCE_BACKEND_URL",
        "INFERENCE_DEFAULT_MODEL",
        "INFERENCE_API_KEY",
        "DASHSCOPE_API_KEY",
        "INFERENCE_HEALTH_PATH",
        "INFERENCE_MAX_RETRIES",
        "INFERENCE_DEFAULT_MAX_TOKENS",
        "INFERENCE_MODEL_MAP",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("INFERENCE_BACKEND", "qwencloud")
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    previous = api_config._config
    api_config.reload_config()
    yield
    api_config._config = previous


def test_qwencloud_environment_defaults(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", " test-key ")
    config = InferenceConfig.from_env()
    assert config.backend == "qwencloud"
    assert config.base_url == BASE_URL
    assert config.default_model == "qwen4-max"
    assert config.api_key == "test-key"
    assert config.health_path == "/models"
    assert config.default_max_tokens == 8192
    assert config.max_retries == 0
    assert "test-key" not in repr(config)


def test_qwencloud_environment_overrides(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "regional-key")
    monkeypatch.setenv("INFERENCE_API_KEY", " explicit-key ")
    monkeypatch.setenv("INFERENCE_BACKEND_URL", " https://example.test/prefix/v1/ ")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "custom-model")
    monkeypatch.setenv("INFERENCE_MAX_RETRIES", "1")
    monkeypatch.setenv("INFERENCE_DEFAULT_MAX_TOKENS", "2048")
    config = InferenceConfig.from_env()
    assert config.api_key == "explicit-key"
    assert config.base_url == "https://example.test/prefix/v1/"
    assert config.default_model == "custom-model"
    assert config.max_retries == 1
    assert config.default_max_tokens == 2048


@pytest.mark.asyncio
async def test_missing_key_fails_before_creating_client():
    service = InferenceService(InferenceConfig.from_env())
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        await service._get_client()
    assert service._client is None


def _request(**kwargs):
    return OpenAIChatCompletionRequest(
        model="qwen4-max", messages=[{"role": "user", "content": "Hello"}], **kwargs
    )


def _completion():
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen4-max",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "reasoning_content": "Recorded provider reasoning",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


async def _mock_service(monkeypatch, handler, config=None):
    # Exercise production URL joining and authentication, not a mock of post().
    factory = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "stateset_agents.api.services.inference_service.httpx.AsyncClient",
        lambda **kwargs: factory(transport=transport, **kwargs),
    )
    service = InferenceService(config or InferenceConfig.qwencloud(api_key="test-key"))
    await service._get_client()
    # Restore the factory so the ASGI client below uses its own transport.
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return service


@pytest.mark.asyncio
async def test_authenticated_requests_health_and_discovery(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": []})
        return httpx.Response(200, json=_completion())

    service = await _mock_service(monkeypatch, handler)
    try:
        response = await service.create_openai_response(_request())
        assert response == _completion()
        assert await service.check_health()
        assert await service.list_models() == {"object": "list", "data": []}
        assert [str(r.url) for r in seen] == [
            f"{BASE_URL}/chat/completions",
            f"{BASE_URL}/models",
            f"{BASE_URL}/models",
        ]
        assert json.loads(seen[0].content)["model"] == "qwen4-max"
    finally:
        client = service._client
        await service.aclose()
    assert client.is_closed
    assert service._client is None


@pytest.mark.asyncio
async def test_openai_route_preserves_reasoning_tools_vision_and_json_schema(
    monkeypatch,
):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    service = await _mock_service(monkeypatch, handler)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_inference_service] = lambda: service
    messages = [
        {"role": "system", "content": "Be helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/a.png"},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.test/a.mp4"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Recorded provider reasoning",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "Found"},
    ]
    payload = {
        "model": "qwen4-max",
        "messages": messages,
        "max_completion_tokens": 8192,
        "enable_thinking": True,
        "reasoning_effort": "low",
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        "tool_choice": "auto",
    }
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200, response.text
        assert captured == [payload]
        assert response.json() == _completion()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_stream_preserves_reasoning_tool_deltas_and_usage(monkeypatch):
    chunks = [
        {
            "model": "qwen4-max",
            "choices": [{"delta": {"reasoning_content": "Thinking"}}],
        },
        {
            "model": "qwen4-max",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]
                    }
                }
            ],
        },
        {
            "model": "qwen4-max",
            "choices": [{"delta": {"content": "Done"}, "finish_reason": "stop"}],
        },
        {
            "model": "qwen4-max",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        },
    ]

    def handler(request):
        assert str(request.url) == f"{BASE_URL}/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content)["stream"] is True
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    service = await _mock_service(monkeypatch, handler)
    try:
        events = [event async for event in service.stream_openai(_request(stream=True))]
        assert events[-1] == "data: [DONE]\n\n"
        assert [
            json.loads(event.removeprefix("data: ")) for event in events[:-1]
        ] == chunks
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_default_does_not_replay_failed_paid_requests(monkeypatch, status):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={"error": {"message": "rejected"}})

    service = await _mock_service(monkeypatch, handler)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await service.create_openai_response(_request())
        assert len(seen) == 1
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_discovery_auth_failure_is_visible(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "Invalid API-key"}})

    service = await _mock_service(monkeypatch, handler)
    try:
        assert await service.check_health() is False
        with pytest.raises(httpx.HTTPStatusError):
            await service.list_models()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_stopping_stream_closes_upstream_response(monkeypatch):
    class ResponseStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            pytest.fail("The caller stopped before requesting another chunk")

        async def aclose(self):
            self.closed = True

    upstream_stream = ResponseStream()

    def handler(request):
        return httpx.Response(200, stream=upstream_stream)

    service = await _mock_service(monkeypatch, handler)
    stream = service.stream_openai(_request(stream=True))
    try:
        assert "Hello" in await anext(stream)
        await stream.aclose()
        assert upstream_stream.closed
    finally:
        await stream.aclose()
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing_slash", [False, True])
async def test_custom_endpoint_and_model_alias(monkeypatch, trailing_slash):
    base_url = "https://example.test/custom/compatible-mode/v1"
    config = InferenceConfig.qwencloud(
        api_key="custom-key", base_url=base_url + ("/" if trailing_slash else "")
    )
    config.model_map = {"public-qwen": "qwen4-max"}

    def handler(request):
        assert request.headers["Authorization"] == "Bearer custom-key"
        assert str(request.url) == f"{base_url}/chat/completions"
        assert json.loads(request.content)["model"] == "qwen4-max"
        return httpx.Response(200, json=_completion())

    service = await _mock_service(monkeypatch, handler, config)
    try:
        request = _request()
        request.model = "public-qwen"
        response = await service.create_openai_response(request)
        assert response["model"] == "public-qwen"
        assert [item["id"] for item in (await service.list_models())["data"]] == [
            "public-qwen"
        ]
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_vllm_paths_and_unauthenticated_defaults_are_preserved(monkeypatch):
    from stateset_agents.api.messages_models import MessagesRequest

    seen = []

    def handler(request):
        seen.append((request.method, str(request.url)))
        assert "Authorization" not in request.headers
        return httpx.Response(200, json=_completion())

    service = await _mock_service(monkeypatch, handler, InferenceConfig())
    try:
        await service.create_openai_response(_request())
        await service.create_openai_response(
            MessagesRequest(
                model="local-model",
                system="Be helpful",
                messages=[{"role": "user", "content": "Hello"}],
            )
        )
        assert await service.check_health()
        await service.list_models()
        assert seen == [
            ("POST", "http://localhost:8000/v1/chat/completions"),
            ("POST", "http://localhost:8000/v1/chat/completions"),
            ("GET", "http://localhost:8000/health"),
            ("GET", "http://localhost:8000/v1/models"),
        ]
    finally:
        await service.aclose()


def test_vllm_does_not_pick_up_qwencloud_key(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "vllm")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "private-qwen-key")
    config = InferenceConfig.from_env()
    assert config.base_url == "http://localhost:8000"
    assert config.default_model is None
    assert config.max_retries == 2
    assert config.api_key is None


@pytest.mark.parametrize("budget", ["max_tokens", "max_completion_tokens"])
def test_native_request_sends_only_explicit_token_budget(budget):
    service = InferenceService(InferenceConfig.qwencloud(api_key="test-key"))
    payload = service._build_openai_payload(_request(**{budget: 4096}))
    assert payload[budget] == 4096
    other = (
        "max_tokens" if budget == "max_completion_tokens" else "max_completion_tokens"
    )
    assert other not in payload
