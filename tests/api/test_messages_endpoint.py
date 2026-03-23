import httpx
import pytest

from stateset_agents.api import config as api_config


@pytest.fixture
def preserve_api_config():
    prev = api_config._config
    yield
    api_config._config = prev


def _client_for_app(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def test_messages_endpoint_stub(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "assistant"
    assert payload["usage"]["output_tokens"] > 0


async def test_messages_endpoint_openai_format(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "response_format": "openai",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("object") == "chat.completion"


async def test_messages_endpoint_stream(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "moonshotai/Kimi-K2.5")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ) as response:
            assert response.status_code == 200
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert "event: message_start" in body
    assert "event: message_stop" in body


@pytest.mark.asyncio
async def test_messages_endpoint_stream_uses_public_model_id(
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
            "/v1/messages",
            json={
                "model": "public-model",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ) as response:
            assert response.status_code == 200
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert '"model":"public-model"' in body


async def test_messages_endpoint_rejects_too_long_content(
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
            "/v1/messages",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "messages": [{"role": "user", "content": long_message}],
            },
        )

    assert response.status_code == 422


async def test_messages_endpoint_rejects_too_many_messages(
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
            "/v1/messages",
            json={
                "model": "moonshotai/Kimi-K2.5",
                "messages": messages,
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_messages_endpoint_hides_internal_model_ids(
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
async def test_messages_endpoint_returns_public_model_id_with_mapping(
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
            "/v1/messages",
            json={
                "model": "public-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "public-model"
