"""
Examples for GRPO Agent Framework

This module contains example implementations and tutorials for training
different types of conversational agents using the framework.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CustomerServiceAgent",
    "CustomerServiceEnvironment",
]


def __getattr__(name: str) -> Any:
    """Load local-training examples only when their public exports are used."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".customer_service_agent", __name__), name)
    globals()[name] = value
    return value
