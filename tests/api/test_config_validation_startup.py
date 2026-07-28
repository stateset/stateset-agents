"""Tests for wiring config.validate() into app startup."""

import logging

import pytest

from stateset_agents.api import config as api_config
from stateset_agents.api.config import (
    APIConfig,
    ConfigurationError,
    Environment,
    SecurityConfig,
)


@pytest.fixture
def preserve_api_config():
    prev = api_config._config
    yield
    api_config._config = prev


def test_development_with_no_keys_logs_warning(monkeypatch, preserve_api_config, caplog):
    monkeypatch.setenv("API_ENVIRONMENT", "development")
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.delenv("API_KEYS", raising=False)
    api_config.reload_config()

    from stateset_agents.api.main import create_app

    with caplog.at_level(logging.WARNING):
        create_app()

    assert any(
        "No API keys configured" in record.message for record in caplog.records
    )


def test_production_auth_enabled_no_credentials_raises(preserve_api_config):
    # Bypass from_env() parse-time checks by constructing the config directly,
    # mirroring a misconfiguration that could slip through (e.g. env drift
    # between parse-time and runtime, or a manually constructed config).
    bad_config = APIConfig(
        environment=Environment.PRODUCTION,
        security=SecurityConfig(api_keys=[], jwt_secret=None, require_auth=True),
    )
    api_config._config = bad_config

    from stateset_agents.api.main import create_app

    with pytest.raises(ConfigurationError):
        create_app()


def test_production_with_jwt_secret_only_does_not_raise(preserve_api_config, caplog):
    # A JWT secret alone is a valid credential source, so this should not
    # escalate to a hard failure even though no API keys are configured.
    ok_config = APIConfig(
        environment=Environment.PRODUCTION,
        security=SecurityConfig(
            api_keys=[], jwt_secret="unit-test-secret", require_auth=True
        ),
    )
    api_config._config = ok_config

    from stateset_agents.api.main import create_app

    with caplog.at_level(logging.WARNING):
        create_app()  # should not raise

    assert any(
        "No API keys configured" in record.message for record in caplog.records
    )
