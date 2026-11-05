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
        # Header credentials: query-string tokens are opt-in
        # (API_LAB_WS_QUERY_TOKEN) because they leak into access logs.
        self.query_params = {}
        self.headers = {"x-api-key": api_key} if api_key else {}
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

    from stateset_agents.api.routers import training_lab
    from stateset_agents.api.routers.training_lab import experiment_ws

    # Only existing experiments can be subscribed to.
    monkeypatch.setitem(training_lab._experiments, "ws-exp", {"metrics": {}})
    websocket = _FakeWebSocket(api_key="unit-lab-key")
    await experiment_ws(websocket, "ws-exp")  # type: ignore[arg-type]

    assert websocket.accepted is True
    assert websocket.close_code is None
    assert any("pong" in message for message in websocket.sent)
    # Disconnect leaves no empty subscriber list behind.
    assert "ws-exp" not in training_lab._metrics_subscribers


class _SubprotocolWebSocket(_FakeWebSocket):
    """Fake that records the subprotocol selected on accept."""

    def __init__(self, *, subprotocols=(), query=None, headers=None) -> None:
        super().__init__()
        self.scope = {"subprotocols": list(subprotocols)}
        self.query_params = dict(query or {})
        self.headers = dict(headers or {})
        self.selected: str | None = None

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepted = True
        self.selected = subprotocol


def _lab_auth_env(monkeypatch):
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "true")
    monkeypatch.setenv("API_KEYS", "unit-lab-key:admin")
    monkeypatch.setenv("API_JWT_SECRET", "unit-test-secret")
    monkeypatch.delenv("API_LAB_WS_QUERY_TOKEN", raising=False)
    api_config.reload_config()


async def test_lab_ws_rejects_unknown_experiment_without_leaking_keys(
    monkeypatch, preserve_api_config
):
    _lab_auth_env(monkeypatch)
    from stateset_agents.api.routers import training_lab

    for index in range(5):
        websocket = _FakeWebSocket(api_key="unit-lab-key")
        await training_lab.experiment_ws(  # type: ignore[arg-type]
            websocket, f"never-created-{index}"
        )
        assert websocket.close_code == 4404
        assert websocket.sent == []

    assert not any(
        key.startswith("never-created-") for key in training_lab._metrics_subscribers
    )


async def test_lab_ws_accepts_bearer_subprotocol_without_echoing_it(
    monkeypatch, preserve_api_config
):
    _lab_auth_env(monkeypatch)
    from stateset_agents.api.routers import training_lab

    monkeypatch.setitem(training_lab._experiments, "ws-sub", {"metrics": {}})
    websocket = _SubprotocolWebSocket(
        subprotocols=[training_lab.LAB_WS_SUBPROTOCOL, "bearer.unit-lab-key"]
    )
    await training_lab.experiment_ws(websocket, "ws-sub")  # type: ignore[arg-type]

    assert websocket.close_code is None
    assert websocket.selected == training_lab.LAB_WS_SUBPROTOCOL
    assert any("pong" in message for message in websocket.sent)

    wrong = _SubprotocolWebSocket(subprotocols=["bearer.not-a-key"])
    await training_lab.experiment_ws(wrong, "ws-sub")  # type: ignore[arg-type]
    assert wrong.close_code == 4401
    assert wrong.selected is None


async def test_lab_ws_query_credentials_log_deprecation(
    monkeypatch, preserve_api_config, caplog
):
    _lab_auth_env(monkeypatch)
    monkeypatch.setenv("API_LAB_WS_QUERY_TOKEN", "true")
    api_config.reload_config()
    from stateset_agents.api.routers import training_lab

    monkeypatch.setitem(training_lab._experiments, "ws-query", {"metrics": {}})
    websocket = _SubprotocolWebSocket(query={"token": "unit-lab-key"})
    with caplog.at_level("WARNING", logger=training_lab.logger.name):
        await training_lab.experiment_ws(websocket, "ws-query")  # type: ignore[arg-type]

    assert websocket.close_code is None
    assert "Deprecated" in caplog.text
    assert "unit-lab-key" not in caplog.text

    header = _SubprotocolWebSocket(headers={"x-api-key": "unit-lab-key"})
    caplog.clear()
    with caplog.at_level("WARNING", logger=training_lab.logger.name):
        await training_lab.experiment_ws(header, "ws-query")  # type: ignore[arg-type]
    assert "Deprecated" not in caplog.text
