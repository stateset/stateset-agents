"""Regression tests for the API auth service provider."""

from __future__ import annotations

import pytest

from stateset_agents.api import config as api_config
from stateset_agents.api.services import auth_service as auth_provider


@pytest.mark.asyncio
async def test_get_auth_service_refreshes_after_config_reload(monkeypatch):
    """AuthService should rebuild when the configured JWT secret changes."""
    previous_config = api_config._config
    previous_auth_service = auth_provider._auth_service
    previous_auth_service_secret = auth_provider._auth_service_secret

    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.setenv(
        "API_JWT_SECRET", "provider-secret-key-for-testing-purposes-only-123456"
    )

    try:
        api_config.reload_config()
        first_service = await auth_provider.get_auth_service()
        assert first_service.secret_key == (
            "provider-secret-key-for-testing-purposes-only-123456"
        )

        monkeypatch.setenv(
            "API_JWT_SECRET", "provider-secret-key-for-testing-purposes-only-abcdef"
        )
        api_config.reload_config()

        refreshed_service = await auth_provider.get_auth_service()
        assert refreshed_service is not first_service
        assert refreshed_service.secret_key == (
            "provider-secret-key-for-testing-purposes-only-abcdef"
        )
    finally:
        api_config._config = previous_config
        auth_provider._auth_service = previous_auth_service
        auth_provider._auth_service_secret = previous_auth_service_secret
