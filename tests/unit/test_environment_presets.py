from __future__ import annotations

from stateset_agents.core.environment import CONVERSATION_CONFIGS, TASK_CONFIGS
from stateset_agents.core.environment_presets import LazyConfigRegistry


def test_lazy_config_registry_loads_once() -> None:
    load_calls = 0

    def _loader() -> dict[str, dict[str, int]]:
        nonlocal load_calls
        load_calls += 1
        return {"demo": {"value": 1}}

    registry = LazyConfigRegistry(_loader)

    assert load_calls == 0
    assert repr(registry) == "LazyConfigRegistry(<not loaded>)"
    assert registry["demo"]["value"] == 1
    assert "demo" in registry
    assert len(registry) == 1
    assert registry.copy()["demo"]["value"] == 1
    assert load_calls == 1


def test_environment_presets_expose_mapping_api() -> None:
    customer_service = CONVERSATION_CONFIGS["customer_service"].copy()
    data_analysis = TASK_CONFIGS.get("data_analysis")

    assert "scenarios" in customer_service
    assert customer_service["max_turns"] >= 1
    assert data_analysis is not None
    assert "tasks" in data_analysis
