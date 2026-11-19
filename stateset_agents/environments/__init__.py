"""StateSet environments with lazy v0.74.0 compatibility exports."""

from importlib import import_module
from typing import Any

from .symbolic_physics import (
    SymbolicPhysicsEnvironment,
    SymbolicPhysicsTask,
    load_symbolic_tasks,
)

__all__ = [
    "SymbolicPhysicsEnvironment",
    "SymbolicPhysicsTask",
    "load_symbolic_tasks",
    "ConversationSimulator",
    "ConversationSimulatorConfig",
    "UserSimulator",
]


def __getattr__(name: str) -> Any:
    """Load experimental exports when a caller requests them."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(
        import_module(
            "stateset_agents.experimental.sim_to_real.conversation_simulator"
        ),
        name,
    )
    globals()[name] = value
    return value
