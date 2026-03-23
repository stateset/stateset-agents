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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import APIConfig, get_config
from .schemas import (
    ConversationRequest,
    ConversationResponse,
    HealthResponse,
    TrainingRequest,
    TrainingResponse,
)
from .. import __version__ as _PACKAGE_VERSION

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(*args: Any, **kwargs: Any) -> FastAPI:
    """Create the FastAPI app without importing the gateway at package import time."""
    from .main import create_app as _create_app

    return _create_app(*args, **kwargs)

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

__version__ = _PACKAGE_VERSION
