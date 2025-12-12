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
