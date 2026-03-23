import importlib
import sys


def test_core_package_import_is_lazy() -> None:
    for module_name in (
        "stateset_agents.core",
        "stateset_agents.core.agent_config",
        "stateset_agents.core.agent",
        "stateset_agents.core.reward",
        "stateset_agents.core.value_function",
    ):
        sys.modules.pop(module_name, None)

    core = importlib.import_module("stateset_agents.core")

    assert "stateset_agents.core.agent_config" not in sys.modules
    assert "stateset_agents.core.agent" not in sys.modules
    assert "stateset_agents.core.reward" not in sys.modules
    assert "stateset_agents.core.value_function" not in sys.modules

    assert core.AgentConfig is not None
    assert "stateset_agents.core.agent_config" in sys.modules
    assert "stateset_agents.core.agent" not in sys.modules
    assert "stateset_agents.core.reward" not in sys.modules


def test_core_package_exposes_distinct_runtime_and_typed_conversation_turns() -> None:
    sys.modules.pop("stateset_agents.core", None)

    core = importlib.import_module("stateset_agents.core")

    assert core.ConversationTurn is not core.TypedConversationTurn
