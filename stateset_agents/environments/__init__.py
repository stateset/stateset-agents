"""
Environments for stateset-agents.

``ToolUseEnvironment`` (provisional) runs verifiable tool-use episodes; see
``docs/TOOL_USE_RL.md``.

The conversation simulator lives in
``stateset_agents.experimental.sim_to_real.conversation_simulator`` (moved in
v0.64.0; the re-exports from this package were removed in v0.65.0).
"""

from stateset_agents.core.environments.tool_use import ToolUseEnvironment
from stateset_agents.core.tool_registry import (
    ToolObservation,
    ToolRegistry,
    ToolSpec,
    calculator_tool,
)

from .symbolic_physics import (
    SymbolicPhysicsEnvironment,
    SymbolicPhysicsTask,
    load_symbolic_tasks,
)

__all__ = [
    "SymbolicPhysicsEnvironment",
    "SymbolicPhysicsTask",
    "load_symbolic_tasks",
    # Tool-use RL (provisional; docs/TOOL_USE_RL.md).
    "ToolObservation",
    "ToolRegistry",
    "ToolSpec",
    "ToolUseEnvironment",
    "calculator_tool",
]
