import importlib
import sys


def test_root_package_import_is_lazy() -> None:
    for module_name in (
        "stateset_agents",
        "stateset_agents.core.agent",
        "stateset_agents.core.reward",
        "stateset_agents.training.train",
    ):
        sys.modules.pop(module_name, None)

    sa = importlib.import_module("stateset_agents")

    assert "stateset_agents.core.agent" not in sys.modules
    assert "stateset_agents.core.reward" not in sys.modules
    assert "stateset_agents.training.train" not in sys.modules

    assert sa.Agent is not None
    assert "stateset_agents.core.agent" in sys.modules
    assert "stateset_agents.training.train" not in sys.modules


def test_root_package_exposes_distinct_runtime_and_typed_conversation_turns() -> None:
    sys.modules.pop("stateset_agents", None)

    sa = importlib.import_module("stateset_agents")

    assert sa.ConversationTurn is not sa.TypedConversationTurn
