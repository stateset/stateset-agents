"""Tests for training-lab auth gating and feature flag."""

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from stateset_agents.api import config as api_config


@pytest.fixture
def preserve_api_config():
    prev = api_config._config
    yield
    api_config._config = prev


def _client_for_app(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def test_lab_not_mounted_when_disabled(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "false")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        response = await client.get("/api/lab/experiments")

    assert response.status_code == 404


async def test_lab_requires_auth_when_enabled(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "true")
    monkeypatch.setenv("API_KEYS", "unit-lab-key:admin")
    monkeypatch.setenv("API_JWT_SECRET", "unit-test-secret")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        unauth = await client.get("/api/lab/experiments")
        assert unauth.status_code == 401

        auth = await client.get(
            "/api/lab/experiments", headers={"X-API-Key": "unit-lab-key"}
        )
        assert auth.status_code == 200


def test_lab_ws_rejects_unauthenticated(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "true")
    monkeypatch.setenv("API_KEYS", "unit-lab-key:admin")
    monkeypatch.setenv("API_JWT_SECRET", "unit-test-secret")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/lab/experiments/does-not-exist/ws"):
            pass


def test_lab_ws_accepts_valid_api_key(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "true")
    monkeypatch.setenv("API_KEYS", "unit-lab-key:admin")
    monkeypatch.setenv("API_JWT_SECRET", "unit-test-secret")
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    app = create_app()
    client = TestClient(app)

    with client.websocket_connect(
        "/api/lab/experiments/does-not-exist/ws?api_key=unit-lab-key"
    ) as ws:
        ws.send_text("ping")
        message = ws.receive_text()
        assert "pong" in message
