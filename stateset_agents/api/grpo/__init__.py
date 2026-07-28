"""
GRPO Service Module

Modular components for the GRPO Service API.

This package provides a refactored, modular implementation of the
GRPO (Group Relative Policy Optimization) service with:

- Configuration management (config.py)
- Request/response models (models.py)
- State management with TTL (state.py)
- Metrics collection (metrics.py)
- Unified rate limiting (rate_limiter.py)
- Request handlers (handlers.py)
- FastAPI application (service.py)
"""

import importlib
import warnings

from .config import GRPOConfig, get_grpo_config, reset_config
from .handlers import ConversationHandler, TrainingHandler, WebSocketHandler
from .metrics import GRPOMetrics, get_grpo_metrics, reset_metrics
from .models import (
    GRPOConversationRequest,
    GRPOConversationResponse,
    GRPOHealthResponse,
    GRPOMetricsResponse,
    GRPOScaleRequest,
    GRPOScaleResponse,
    GRPOTrainingRequest,
    GRPOTrainingResponse,
    TrainingMetrics,
)
from .rate_limiter import (
    RateLimitResult,
    UnifiedRateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)
from .state import (
    ConversationState,
    StateManager,
    TrainingJob,
    TTLDict,
    get_state_manager,
    reset_state_manager,
)

#: Symbols that belong to the deprecated *app surface* of this package (the
#: standalone GRPO service app). Config/handlers/metrics/models/state/
#: rate_limiter are shared infrastructure re-exported here for convenience
#: and are NOT deprecated, so importing this package alone must not warn.
_DEPRECATED_APP_SUBMODULES = {
    "service",
    "service_routes",
    "router_v1",
    "auth",
}

_DEPRECATION_MESSAGE = (
    "stateset_agents.api.grpo.{name} is part of a secondary GRPO API app "
    "and is deprecated; use stateset_agents.api.main instead."
)


def __getattr__(name: str):
    if name in _DEPRECATED_APP_SUBMODULES:
        warnings.warn(
            _DEPRECATION_MESSAGE.format(name=name),
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config
    "GRPOConfig",
    "get_grpo_config",
    "reset_config",
    # Handlers
    "ConversationHandler",
    "TrainingHandler",
    "WebSocketHandler",
    # Metrics
    "GRPOMetrics",
    "get_grpo_metrics",
    "reset_metrics",
    # Models
    "GRPOConversationRequest",
    "GRPOConversationResponse",
    "GRPOHealthResponse",
    "GRPOMetricsResponse",
    "GRPOScaleRequest",
    "GRPOScaleResponse",
    "GRPOTrainingRequest",
    "GRPOTrainingResponse",
    "TrainingMetrics",
    # Rate Limiter
    "RateLimitResult",
    "UnifiedRateLimiter",
    "get_rate_limiter",
    "reset_rate_limiter",
    # State
    "ConversationState",
    "StateManager",
    "TrainingJob",
    "TTLDict",
    "get_state_manager",
    "reset_state_manager",
]
