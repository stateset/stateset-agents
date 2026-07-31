"""Tests for provider name resolution."""

from __future__ import annotations

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.local import LocalExecutor
from stateset_agents.remote.registry import available_providers, get_executor


class TestGetExecutor:
    def test_resolves_local(self):
        assert isinstance(get_executor("local"), LocalExecutor)

    def test_resolution_is_case_insensitive(self):
        assert isinstance(get_executor("LOCAL"), LocalExecutor)

    def test_unknown_provider_names_the_valid_options(self):
        with pytest.raises(RemoteExecutionError) as excinfo:
            get_executor("aws-batch")

        message = str(excinfo.value)
        assert "aws-batch" in message
        assert "local" in message


class TestAvailableProviders:
    def test_lists_local_and_modal(self):
        """Modal is listed even when its SDK is absent — install guidance
        belongs at submit time, not at discovery time."""
        providers = available_providers()

        assert "local" in providers
        assert "modal" in providers
