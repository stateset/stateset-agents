"""Tests for training-lab auth gating and feature flag."""

import httpx
import pytest
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


class _FakeWebSocket:
    """Small protocol double; avoids TestClient's AnyIO thread portal."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self.query_params = {"api_key": api_key} if api_key else {}
        self.headers = {}
        self.accepted = False
        self.close_code: int | None = None
        self.sent: list[str] = []
        self._received_ping = False

    async def accept(self) -> None:
        self.accepted = True

    async def close(self, code: int) -> None:
        self.close_code = code

    async def send_text(self, message: str) -> None:
        self.sent.append(message)

    async def receive_text(self) -> str:
        if not self._received_ping:
            self._received_ping = True
            return "ping"
        raise WebSocketDisconnect()


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


async def test_lab_ws_rejects_unauthenticated(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "true")
    monkeypatch.setenv("API_KEYS", "unit-lab-key:admin")
    monkeypatch.setenv("API_JWT_SECRET", "unit-test-secret")
    api_config.reload_config()

    from stateset_agents.api.routers.training_lab import experiment_ws

    websocket = _FakeWebSocket()
    await experiment_ws(websocket, "does-not-exist")  # type: ignore[arg-type]

    assert websocket.accepted is True
    assert websocket.close_code == 4401


async def test_lab_ws_accepts_valid_api_key(monkeypatch, preserve_api_config):
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "true")
    monkeypatch.setenv("API_KEYS", "unit-lab-key:admin")
    monkeypatch.setenv("API_JWT_SECRET", "unit-test-secret")
    api_config.reload_config()

    from stateset_agents.api.routers.training_lab import experiment_ws

    websocket = _FakeWebSocket(api_key="unit-lab-key")
    await experiment_ws(websocket, "does-not-exist")  # type: ignore[arg-type]

    assert websocket.accepted is True
    assert websocket.close_code is None
    assert any("pong" in message for message in websocket.sent)
