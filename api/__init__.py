"""
StateSet Agents API

Production-ready REST API for training and deploying AI agents using
reinforcement learning techniques (GRPO/GSPO).

Quick Start:
    from stateset_agents.api import create_app

    app = create_app()

    # Run with uvicorn
    # uvicorn stateset_agents.api:app --host 0.0.0.0 --port 8000

Configuration:
    Set environment variables or create a .env file. See .env.example for options.

    Required in production:
    - API_JWT_SECRET: Secret for JWT token signing
    - API_CORS_ORIGINS: Allowed CORS origins
    - API_KEYS: API keys with roles

Modules:
    - config: Configuration management
    - auth: Authentication and authorization
    - errors: Error handling and responses
    - middleware: Security, rate limiting, metrics
    - models: Request/response models
    - observability: Logging and tracing
    - v1: API v1 router and endpoints
"""

from .v1.router import create_app, router
from .config import get_config, APIConfig
from .auth import get_current_user, require_roles, AuthenticatedUser
from .errors import APIError, NotFoundError, ValidationError
from .models import (
    TrainingRequest,
    TrainingResponse,
    ConversationRequest,
    ConversationResponse,
    HealthResponse,
)

__all__ = [
    # Application
    "create_app",
    "router",
    # Configuration
    "get_config",
    "APIConfig",
    # Authentication
    "get_current_user",
    "require_roles",
    "AuthenticatedUser",
    # Errors
    "APIError",
    "NotFoundError",
    "ValidationError",
    # Models
    "TrainingRequest",
    "TrainingResponse",
    "ConversationRequest",
    "ConversationResponse",
    "HealthResponse",
]

__version__ = "2.0.0"
