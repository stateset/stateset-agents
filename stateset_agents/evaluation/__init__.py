"""
Evaluation utilities for stateset-agents.

Includes metrics for sim-to-real transfer evaluation and agent performance.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SimToRealMetrics",
    "SimToRealEvaluator",
    "compute_distribution_divergence",
    "compute_response_statistics",
]


def __getattr__(name: str) -> Any:
    """Keep outcome comparisons independent of optional tensor dependencies."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".sim_to_real_metrics", __name__), name)
    globals()[name] = value
    return value
