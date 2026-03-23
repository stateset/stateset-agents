"""
Preset environment configurations with lazy loading.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .environment_base import ENVIRONMENT_EXCEPTIONS


class LazyConfigRegistry(dict[str, Any]):
    """Dict-like registry that materializes its contents on first access."""

    def __init__(self, loader: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self._loader = loader
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        loaded = self._loader()
        dict.clear(self)
        dict.update(self, loaded)
        self._loaded = True

    def __getitem__(self, key: str) -> Any:
        self._ensure_loaded()
        return dict.__getitem__(self, key)

    def __contains__(self, key: object) -> bool:
        self._ensure_loaded()
        return dict.__contains__(self, key)

    def __iter__(self):
        self._ensure_loaded()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._ensure_loaded()
        return dict.__len__(self)

    def __bool__(self) -> bool:
        self._ensure_loaded()
        return bool(dict.__len__(self))

    def __setitem__(self, key: str, value: Any) -> None:
        self._ensure_loaded()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key: str) -> None:
        self._ensure_loaded()
        dict.__delitem__(self, key)

    def clear(self) -> None:
        self._ensure_loaded()
        dict.clear(self)

    def copy(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self)

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return dict.get(self, key, default)

    def items(self):
        self._ensure_loaded()
        return dict.items(self)

    def keys(self):
        self._ensure_loaded()
        return dict.keys(self)

    def pop(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return dict.pop(self, key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return dict.setdefault(self, key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_loaded()
        dict.update(self, *args, **kwargs)

    def values(self):
        self._ensure_loaded()
        return dict.values(self)

    def __repr__(self) -> str:
        if not self._loaded:
            return f"{type(self).__name__}(<not loaded>)"
        return dict.__repr__(self)


def _load_conversation_configs() -> dict[str, Any]:
    """Load environment presets from config, with hardcoded fallbacks."""
    configs: dict[str, Any] = {}

    try:
        from stateset_agents.config import (
            get_environment_preset,
            list_environment_presets,
        )

        for preset_name in list_environment_presets():
            try:
                configs[preset_name] = get_environment_preset(preset_name)
            except ENVIRONMENT_EXCEPTIONS:
                pass
    except ImportError:
        pass

    fallback_configs = {
        "customer_service": {
            "scenarios": [
                {
                    "topic": "product_inquiry",
                    "user_goal": "Learn about product features",
                    "context": "User is interested in purchasing a product",
                },
                {
                    "topic": "technical_support",
                    "user_goal": "Resolve technical issue",
                    "context": "User is experiencing a problem with their device",
                },
            ],
            "persona": "You are a professional customer service representative.",
            "max_turns": 15,
        },
        "technical_support": {
            "scenarios": [
                {
                    "topic": "app_crash",
                    "user_goal": "Fix an application crashing on launch",
                    "context": "User reports the app crashes immediately after opening.",
                },
                {
                    "topic": "network_issue",
                    "user_goal": "Restore connectivity",
                    "context": "User cannot connect to Wi-Fi after a recent update.",
                },
                {
                    "topic": "performance",
                    "user_goal": "Improve slow system performance",
                    "context": "User notices the computer is slow and wants troubleshooting steps.",
                },
            ],
            "persona": "You are a user seeking technical support. Provide symptoms and answer questions.",
            "max_turns": 15,
        },
        "sales": {
            "scenarios": [
                {
                    "topic": "plan_selection",
                    "user_goal": "Choose the right plan",
                    "context": "User is comparing tiers and wants a recommendation.",
                },
                {
                    "topic": "pricing",
                    "user_goal": "Understand pricing and discounts",
                    "context": "User asks about pricing, billing, and whether there are promotions.",
                },
                {
                    "topic": "objection_handling",
                    "user_goal": "Address concerns and decide",
                    "context": "User is interested but worried about cost and onboarding effort.",
                },
            ],
            "persona": "You are a potential customer exploring products. Ask questions and raise objections.",
            "max_turns": 15,
        },
        "tutoring": {
            "scenarios": [
                {
                    "topic": "math_help",
                    "user_goal": "Understand a math concept",
                    "context": "Student needs help with algebra",
                },
                {
                    "topic": "essay_writing",
                    "user_goal": "Improve writing skills",
                    "context": "Student working on an essay",
                },
            ],
            "persona": "You are a patient and encouraging tutor.",
            "max_turns": 20,
        },
    }

    for name, config in fallback_configs.items():
        if name not in configs:
            configs[name] = config

    return configs


def _load_task_configs() -> dict[str, Any]:
    return {
        "data_analysis": {
            "tasks": [
                {
                    "description": "Analyze the provided dataset and generate insights",
                    "goal": "Complete data analysis",
                    "type": "analysis",
                    "required_actions": [
                        {"name": "load_data", "keywords": ["load", "read", "import"]},
                        {
                            "name": "explore_data",
                            "keywords": ["explore", "summary", "describe"],
                        },
                        {"name": "visualize", "keywords": ["plot", "chart", "graph"]},
                        {
                            "name": "insights",
                            "keywords": ["insight", "conclusion", "finding"],
                        },
                    ],
                }
            ]
        }
    }


CONVERSATION_CONFIGS = LazyConfigRegistry(_load_conversation_configs)
TASK_CONFIGS = LazyConfigRegistry(_load_task_configs)

__all__ = ["LazyConfigRegistry", "CONVERSATION_CONFIGS", "TASK_CONFIGS"]
