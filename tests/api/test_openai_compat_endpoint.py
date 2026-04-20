import httpx
import pytest
from unittest.mock import AsyncMock

from stateset_agents.api import config as api_config


@pytest.fixture
def preserve_api_config():
    prev = api_config._config
    yield
    api_config._config = prev


def _client_for_app(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_openai_chat_completions_stub(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_openai_chat_completions_accepts_null_content_with_tool_calls(
    monkeypatch, preserve_api_config
):
    """OpenAI compatibility: assistant messages may omit/NULL content with tool_calls."""
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "max_tokens": 32,
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"x"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                    {"role": "user", "content": "Thanks"},
                ],
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_openai_chat_completions_stream_stub(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ) as response:
            assert response.status_code == 200
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert "data:" in body
    assert "[DONE]" in body


@pytest.mark.asyncio
async def test_openai_chat_completions_stream_uses_public_model_id(
    monkeypatch, preserve_api_config
):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "internal/model")
    monkeypatch.setenv("INFERENCE_MODEL_MAP", '{"public-model": "internal/model"}')
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "public-model",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ) as response:
            assert response.status_code == 200
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert '"model":"public-model"' in body


@pytest.mark.asyncio
async def test_openai_models_list_stub(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"]
    assert payload["data"][0]["id"] == "moonshotai/Kimi-K2.5"


@pytest.mark.asyncio
async def test_openai_models_list_hides_internal_ids(
    monkeypatch, preserve_api_config
):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "internal/model")
    monkeypatch.setenv("INFERENCE_MODEL_MAP", '{"public-model": "internal/model"}')
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    model_ids = [model["id"] for model in payload["data"]]
    assert model_ids == ["public-model"]


@pytest.mark.asyncio
async def test_openai_chat_completions_returns_public_model_id_when_mapped(
    monkeypatch, preserve_api_config
):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "internal/model")
    monkeypatch.setenv("INFERENCE_MODEL_MAP", '{"public-model": "internal/model"}')
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "public-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "public-model"


@pytest.mark.asyncio
async def test_openai_chat_completions_returns_500_for_internal_failures(
    monkeypatch, preserve_api_config
):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    app.state.inference_service.create_openai_response = AsyncMock(
        side_effect=RuntimeError("boom")
    )
    async with _client_for_app(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 500
    assert "Internal server error" in response.text


@pytest.mark.asyncio
async def test_openai_chat_completions_rejects_too_long_content(
    monkeypatch, preserve_api_config
):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    config = api_config.get_config()
    long_message = "x" * (config.validation.max_message_length + 1)

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "messages": [{"role": "user", "content": long_message}],
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_openai_chat_completions_rejects_too_many_messages(
    monkeypatch, preserve_api_config
):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    config = api_config.get_config()
    messages = [
        {"role": "user", "content": "hello"}
        for _ in range(config.validation.max_conversation_messages + 1)
    ]

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "messages": messages,
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint(monkeypatch, preserve_api_config):
    try:
        import prometheus_client  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("prometheus_client not installed")

    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "# HELP" in response.text
