"""
Evaluation utilities for stateset-agents.

The sim-to-real metrics that used to live here moved to
``stateset_agents.experimental.sim_to_real.metrics`` in v0.64.0. They are
still served from this package, with a ``DeprecationWarning``, until v0.65.0.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

#: Deprecated names served for one release from their experimental home.
_DEPRECATED_EXPORTS = frozenset(
    {
        "SimToRealMetrics",
        "SimToRealEvaluator",
        "compute_distribution_divergence",
        "compute_response_statistics",
    }
)
_EXPERIMENTAL_MODULE = "stateset_agents.experimental.sim_to_real.metrics"

__all__: list[str] = []


def __getattr__(name: str) -> Any:
    """Serve moved sim-to-real names with a deprecation warning."""
    if name not in _DEPRECATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"stateset_agents.evaluation.{name} is deprecated and moved to "
        f"{_EXPERIMENTAL_MODULE}.{name}; the old import path will be removed "
        "in v0.65.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module(_EXPERIMENTAL_MODULE), name)
