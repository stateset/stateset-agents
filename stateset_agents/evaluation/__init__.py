"""StateSet evaluation with lazy v0.74.0 compatibility exports."""

from importlib import import_module
from typing import Any

__all__ = [
    "SimToRealMetrics",
    "SimToRealEvaluator",
    "compute_distribution_divergence",
    "compute_response_statistics",
]


def __getattr__(name: str) -> Any:
    """Load experimental exports when a caller requests them."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(
        import_module("stateset_agents.experimental.sim_to_real.metrics"), name
    )
    globals()[name] = value
    return value
