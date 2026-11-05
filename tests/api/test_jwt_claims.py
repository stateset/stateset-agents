"""JWT verification requires an expiry claim."""

from __future__ import annotations

import time

import jwt
import pytest

from stateset_agents.api.auth import JWTHandler

SECRET = "unit-test-secret-key-with-at-least-32-chars"


@pytest.fixture
def handler() -> JWTHandler:
    return JWTHandler(SECRET)


def _token(payload: dict) -> str:
    return jwt.encode(payload, SECRET, algorithm="HS256")


def test_token_with_future_exp_is_accepted(handler):
    payload = handler.decode(_token({"sub": "u", "exp": int(time.time()) + 60}))
    assert payload is not None
    assert payload["sub"] == "u"


def test_token_without_exp_is_rejected(handler):
    assert handler.decode(_token({"sub": "u", "roles": ["admin"]})) is None


def test_expired_token_is_rejected(handler):
    assert handler.decode(_token({"sub": "u", "exp": int(time.time()) - 60})) is None


def test_generated_tokens_round_trip(monkeypatch):
    from stateset_agents.api import config as api_config
    from stateset_agents.api.auth import generate_token, get_jwt_handler

    prev = api_config._config
    monkeypatch.setenv("API_JWT_SECRET", SECRET)
    api_config.reload_config()
    try:
        token = generate_token("user-1", ["user"])
        payload = get_jwt_handler().decode(token)
        assert payload is not None
        assert payload["sub"] == "user-1"
        assert "exp" in payload
    finally:
        api_config._config = prev
