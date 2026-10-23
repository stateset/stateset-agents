"""Deprecated secondary GRPO service app (removed in v0.71.0).

This package is a second FastAPI app (``service.create_app``) with its own
config, handlers, metrics, models, state and auth. Nothing outside the package
uses it: the supported gateway is :mod:`stateset_agents.api.main`, which has
its own rate limiter (:mod:`stateset_agents.api.rate_limiter`). The app
surface (``service``, ``service_routes``, ``router_v1``, ``auth``) has been
deprecated since v0.16.0; since v0.70.0 importing any part of the package
warns, and the whole package will be removed in v0.71.0.
"""

import importlib
from typing import Any

from ..._deprecation import warn_deprecated
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

warn_deprecated(
    "stateset_agents.api.grpo (the secondary GRPO service app)",
    removal="0.71.0",
    replacement="stateset_agents.api.main",
)

#: App modules loaded on attribute access (the package import already warned).
_APP_SUBMODULES = {"service", "service_routes", "router_v1", "auth"}


def __getattr__(name: str) -> Any:
    if name in _APP_SUBMODULES:
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
