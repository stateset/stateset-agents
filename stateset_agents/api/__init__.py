"""
StateSet Agents API

Production-ready REST API for training and deploying AI agents using
reinforcement learning techniques (GRPO/GSPO).

Quick Start:
    from stateset_agents.api.main import create_app

    app = create_app()

    # Run with uvicorn
    # uvicorn stateset_agents.api.main:app --host 0.0.0.0 --port 8000

Configuration:
    Set environment variables or create a .env file. See .env.example for options.
"""

from .main import create_app
from .config import get_config, APIConfig
from .schemas import (
    TrainingRequest,
    TrainingResponse,
    ConversationRequest,
    ConversationResponse,
    HealthResponse,
)

__all__ = [
    "create_app",
    "get_config",
    "APIConfig",
    "TrainingRequest",
    "TrainingResponse",
    "ConversationRequest",
    "ConversationResponse",
    "HealthResponse",
]

__version__ = "2.0.0"