"""Environment lookup by name: built-ins, process registrations and plugins."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.core import environment_registry as registry
from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
)


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_REGISTERED", {})


def test_builtins_are_listed_and_built_without_registration() -> None:
    assert {"refund-policy", "tau-retail"} <= set(registry.available_environments())
    environment = registry.make_environment("refund-policy", scenarios=[])
    assert isinstance(environment, RefundPolicyEnvironment)


def test_tau_retail_factory_takes_settings_fields() -> None:
    environment = registry.make_environment("tau-retail", user_llm="fake", max_turns=5)
    assert environment.max_turns == 5 and environment.settings.user_llm == "fake"


def test_process_registration_and_name_collisions() -> None:
    registry.register_environment("mine", lambda **_: RefundPolicyEnvironment([]))
    assert "mine" in registry.available_environments()
    assert isinstance(registry.make_environment("mine"), RefundPolicyEnvironment)
    for name in ("mine", "refund-policy"):
        with pytest.raises(ValueError, match="already registered"):
            registry.register_environment(name, lambda: None)
    registry.register_environment("mine", lambda: "not an environment", replace=True)
    with pytest.raises(TypeError, match="not an Environment"):
        registry.make_environment("mine")


def test_installed_plugins_are_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = SimpleNamespace(
        name="third-party", load=lambda: lambda **_: RefundPolicyEnvironment([])
    )

    def entry_points(*, group: str) -> list[Any]:
        assert group == "stateset_agents.environments"
        return [plugin]

    monkeypatch.setattr(registry.metadata, "entry_points", entry_points)
    assert "third-party" in registry.available_environments()
    assert isinstance(registry.make_environment("third-party"), RefundPolicyEnvironment)


def test_unknown_name_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="refund-policy"):
        registry.make_environment("missing")
    with pytest.raises(ValueError):
        registry.register_environment(" ", lambda: None)
