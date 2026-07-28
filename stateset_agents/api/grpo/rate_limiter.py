"""Deprecated re-export shim.

The unified rate limiter now lives at ``stateset_agents.api.rate_limiter``
because it is shared infrastructure used by ``middleware.py`` (not specific
to the secondary ``grpo`` app). This module re-exports the same symbols so
existing imports keep working, without emitting a warning on import — the
rate limiter itself is not deprecated, only the rest of the ``grpo`` app
surface is.
"""

from ..rate_limiter import (
    MAX_BUCKETS,
    RateLimitResult,
    UnifiedRateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)

__all__ = [
    "MAX_BUCKETS",
    "RateLimitResult",
    "UnifiedRateLimiter",
    "get_rate_limiter",
    "reset_rate_limiter",
]
