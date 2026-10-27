"""Tests for generation parameter ceilings and the passthrough allowlist."""

from __future__ import annotations

import logging

import httpx
import pytest

from stateset_agents.api import config as api_config
from stateset_agents.api.config import ValidationConfig
from stateset_agents.api.generation_params import (
    ALLOWED_PASSTHROUGH_FIELDS,
    sanitize_passthrough_params,
    validate_generation_request,
    validate_max_tokens,
)
from stateset_agents.api.messages_models import MessagesRequest
from stateset_agents.api.services.inference_service import (
    InferenceConfig,
    InferenceService,
)


@pytest.fixture
def preserve_api_config():
    prev = api_config._config
    yield
    api_config._config = prev


def _client_for_app(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _request(**kwargs) -> MessagesRequest:
    return MessagesRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kwargs
    )


def test_validation_config_generation_defaults(monkeypatch):
    for key in (
        "API_MAX_GENERATION_TOKENS",
        "API_MAX_COMPLETIONS_PER_REQUEST",
        "API_MAX_LOGIT_BIAS_ENTRIES",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = ValidationConfig.from_env()
    assert cfg.max_generation_tokens == 32768
    assert cfg.max_completions_per_request == 4
    assert cfg.max_logit_bias_entries == 300


def test_validation_config_generation_env_override(monkeypatch):
    monkeypatch.setenv("API_MAX_GENERATION_TOKENS", "2048")
    monkeypatch.setenv("API_MAX_COMPLETIONS_PER_REQUEST", "2")
    cfg = ValidationConfig.from_env()
    assert cfg.max_generation_tokens == 2048
    assert cfg.max_completions_per_request == 2


def test_validate_max_tokens_rejects_above_ceiling():
    limits = ValidationConfig(max_generation_tokens=100)
    validate_max_tokens(None, limits)
    validate_max_tokens(100, limits)
    with pytest.raises(ValueError, match="max_tokens"):
        validate_max_tokens(101, limits)


def test_sanitize_drops_unknown_fields_and_logs_debug(caplog):
    limits = ValidationConfig()
    with caplog.at_level(logging.DEBUG, logger="stateset_agents.api.generation_params"):
        result = sanitize_passthrough_params(
            {
                "seed": 7,
                "presence_penalty": 0.5,
                "ignore_eos": True,
                "guided_json": {"type": "object"},
                "max_completion_tokens": 10**9,
            },
            limits,
        )
    assert result == {"seed": 7, "presence_penalty": 0.5}
    assert "ignore_eos" in caplog.text


def test_sanitize_accepts_known_sampling_fields():
    limits = ValidationConfig()
    extra = {
        "presence_penalty": -1.0,
        "frequency_penalty": 1.5,
        "repetition_penalty": 1.1,
        "min_p": 0.05,
        "seed": 42,
        "logit_bias": {"50256": -100},
        "n": 2,
        "best_of": 3,
        "logprobs": True,
        "top_logprobs": 5,
        "user": "user-123",
        "min_tokens": 4,
        "parallel_tool_calls": False,
    }
    assert set(extra) <= ALLOWED_PASSTHROUGH_FIELDS
    assert sanitize_passthrough_params(extra, limits, max_tokens=64) == extra


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"n": 5}, "'n'"),
        ({"n": 0}, "'n'"),
        ({"n": True}, "'n'"),
        ({"best_of": 50}, "'best_of'"),
        ({"n": 3, "best_of": 2}, "best_of"),
        ({"presence_penalty": 9}, "presence_penalty"),
        ({"repetition_penalty": 0}, "repetition_penalty"),
        ({"min_p": float("nan")}, "min_p"),
        ({"top_logprobs": 100}, "top_logprobs"),
        ({"logit_bias": {"abc": 1}}, "logit_bias"),
        ({"logit_bias": {"1": 1000}}, "logit_bias"),
        ({"logit_bias": [1, 2]}, "logit_bias"),
        ({"min_tokens": 1000}, "min_tokens"),
        ({"user": "x" * 1000}, "user"),
    ],
)
def test_sanitize_rejects_out_of_bounds(extra, message):
    limits = ValidationConfig(max_completions_per_request=4)
    with pytest.raises(ValueError, match=message):
        sanitize_passthrough_params(extra, limits, max_tokens=64)


def test_sanitize_caps_logit_bias_size():
    limits = ValidationConfig(max_logit_bias_entries=2)
    with pytest.raises(ValueError, match="too many entries"):
        sanitize_passthrough_params({"logit_bias": {"1": 1, "2": 1, "3": 1}}, limits)


def test_validate_generation_request_checks_stop_sequences():
    limits = ValidationConfig(max_stop_sequences=2)
    with pytest.raises(ValueError, match="stop sequences"):
        validate_generation_request(_request(stop_sequences=["a", "b", "c"]), limits)


def test_build_openai_payload_forwards_only_allowlisted_extras(
    monkeypatch, preserve_api_config
):
    api_config.reload_config()
    service = InferenceService(InferenceConfig(backend="vllm", default_model="m"))
    request = _request(
        max_tokens=16,
        seed=3,
        frequency_penalty=0.2,
        ignore_eos=True,
        best_of=1,
        guided_regex=".*",
        skip_special_tokens=False,
    )
    payload = service._build_openai_payload(request)
    assert payload["seed"] == 3
    assert payload["frequency_penalty"] == 0.2
    assert payload["best_of"] == 1
    assert payload["max_tokens"] == 16
    for key in ("ignore_eos", "guided_regex", "skip_special_tokens"):
        assert key not in payload


def test_build_openai_payload_rejects_oversized_max_tokens(
    monkeypatch, preserve_api_config
):
    monkeypatch.setenv("API_MAX_GENERATION_TOKENS", "64")
    api_config.reload_config()
    service = InferenceService(InferenceConfig(backend="vllm", default_model="m"))
    with pytest.raises(ValueError, match="max_tokens"):
        service._build_openai_payload(_request(max_tokens=65))


def _stub_env(monkeypatch):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("INFERENCE_BACKEND", "stub")
    monkeypatch.setenv("INFERENCE_DEFAULT_MODEL", "stub-model")
    monkeypatch.setenv("API_MAX_GENERATION_TOKENS", "1000")
    monkeypatch.setenv("API_MAX_COMPLETIONS_PER_REQUEST", "2")
    api_config.reload_config()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/messages", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
async def test_endpoints_reject_max_tokens_above_ceiling(
    monkeypatch, preserve_api_config, path, stream
):
    _stub_env(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            path,
            json={
                "model": "stub-model",
                "max_tokens": 1001,
                "stream": stream,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert response.status_code == 422
    assert "max_tokens" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/messages", "/v1/chat/completions"])
async def test_endpoints_reject_excessive_fan_out(
    monkeypatch, preserve_api_config, path
):
    _stub_env(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            path,
            json={
                "model": "stub-model",
                "max_tokens": 16,
                "n": 64,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert response.status_code == 422
    assert "'n'" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/messages", "/v1/chat/completions"])
async def test_endpoints_accept_known_fields_and_ignore_unknown(
    monkeypatch, preserve_api_config, path
):
    _stub_env(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            path,
            json={
                "model": "stub-model",
                "max_tokens": 1000,
                "seed": 1,
                "presence_penalty": 0.1,
                "ignore_eos": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_v1_conversations_reject_max_tokens_above_ceiling(
    monkeypatch, preserve_api_config
):
    _stub_env(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.post(
            "/api/v1/conversations",
            json={"message": "Hello", "max_tokens": 5000},
        )
    assert response.status_code == 422
    assert "max_tokens" in response.text
