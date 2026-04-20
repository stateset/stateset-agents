"""Regression tests for API auth cache refresh on config reload."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from stateset_agents.api import auth as api_auth
from stateset_agents.api import config as api_config
from stateset_agents.api.errors import UnauthorizedError


def _request_with_bearer_token(token: str) -> Request:
    """Build a minimal ASGI request carrying a bearer token."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/probe",
            "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )


def test_authenticate_request_refreshes_jwt_handler_after_config_reload(monkeypatch):
    """Changing JWT config should invalidate old tokens and accept new ones."""
    previous_config = api_config._config
    previous_jwt_handler = api_auth._jwt_handler
    previous_jwt_handler_config = api_auth._jwt_handler_config

    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv(
        "API_JWT_SECRET", "first-secret-key-for-testing-purposes-only-1234567890"
    )

    try:
        api_config.reload_config()

        old_token = api_auth.generate_token("rotating-user", ["user"])
        authenticated = api_auth.authenticate_request(
            _request_with_bearer_token(old_token)
        )
        assert authenticated.user_id == "rotating-user"
        assert authenticated.auth_method == "jwt"

        old_handler = api_auth.get_jwt_handler()

        monkeypatch.setenv(
            "API_JWT_SECRET", "second-secret-key-for-testing-purposes-only-0987654321"
        )
        api_config.reload_config()

        refreshed_handler = api_auth.get_jwt_handler()
        assert refreshed_handler is not old_handler

        with pytest.raises(UnauthorizedError, match="Invalid API key or token"):
            api_auth.authenticate_request(_request_with_bearer_token(old_token))

        new_token = api_auth.generate_token("rotating-user", ["admin"])
        refreshed_user = api_auth.authenticate_request(
            _request_with_bearer_token(new_token)
        )
        assert refreshed_user.user_id == "rotating-user"
        assert refreshed_user.roles == ["admin"]
        assert refreshed_user.auth_method == "jwt"
    finally:
        api_config._config = previous_config
        api_auth._jwt_handler = previous_jwt_handler
        api_auth._jwt_handler_config = previous_jwt_handler_config
