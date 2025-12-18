"""
Environments for stateset-agents.

Includes conversation simulation environments for sim-to-real transfer.
"""
try:
    from environments import *  # noqa: F401, F403
except ImportError:
    pass

# Conversation Simulator for sim-to-real transfer
from .conversation_simulator import (
    ConversationSimulator,
    ConversationSimulatorConfig,
    UserSimulator,
)

__all__ = [
    "ConversationSimulator",
    "ConversationSimulatorConfig",
    "UserSimulator",
]

