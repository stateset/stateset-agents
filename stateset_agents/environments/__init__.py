"""
Environments for stateset-agents.

The conversation simulator moved to
``stateset_agents.experimental.sim_to_real.conversation_simulator`` in v0.64.0.
Its names are still served from this package, with a ``DeprecationWarning``,
until v0.65.0.
"""

import warnings
from importlib import import_module
from typing import Any

from .symbolic_physics import (
    SymbolicPhysicsEnvironment,
    SymbolicPhysicsTask,
    load_symbolic_tasks,
)

#: Deprecated names served for one release from their experimental home.
_DEPRECATED_EXPORTS = frozenset(
    {"ConversationSimulator", "ConversationSimulatorConfig", "UserSimulator"}
)
_EXPERIMENTAL_MODULE = "stateset_agents.experimental.sim_to_real.conversation_simulator"

__all__ = [
    "SymbolicPhysicsEnvironment",
    "SymbolicPhysicsTask",
    "load_symbolic_tasks",
]


def __getattr__(name: str) -> Any:
    """Serve moved conversation-simulator names with a deprecation warning."""
    if name not in _DEPRECATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"stateset_agents.environments.{name} is deprecated and moved to "
        f"{_EXPERIMENTAL_MODULE}.{name}; the old import path will be removed "
        "in v0.65.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module(_EXPERIMENTAL_MODULE), name)
