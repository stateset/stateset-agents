"""
Compatibility facade for environment types, factories, and preset registries.
"""

from __future__ import annotations

from typing import Any, cast

from .conversation_environment import ConversationEnvironment
from .environment_base import (
    ENVIRONMENT_EXCEPTIONS,
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from .environment_presets import CONVERSATION_CONFIGS, TASK_CONFIGS
from .task_environment import TaskEnvironment
from .trajectory import ConversationTurn


def create_environment(env_type: str, config: dict[str, Any]) -> Environment:
    """Factory function for creating environments."""
    if env_type == "conversation":
        return ConversationEnvironment(**config)
    if env_type == "task":
        return cast(Environment, TaskEnvironment(**config))
    raise ValueError(f"Unknown environment type: {env_type}")


__all__ = [
    "ENVIRONMENT_EXCEPTIONS",
    "EpisodeStatus",
    "EnvironmentState",
    "Environment",
    "ConversationEnvironment",
    "TaskEnvironment",
    "ConversationTurn",
    "create_environment",
    "CONVERSATION_CONFIGS",
    "TASK_CONFIGS",
]
