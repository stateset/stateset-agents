#!/usr/bin/env python3
"""
StateSet Agents API Server

Production-ready REST API for training and deploying AI agents.

Usage:
    python -m stateset_agents.api.main

    # Or with uvicorn directly:
    uvicorn stateset_agents.api.main:app --host 0.0.0.0 --port 8000

Environment Variables:
    See .env.example for all configuration options.
"""

import logging
import sys

from .config import get_config, ConfigurationError
from .observability import setup_observability
from .v1.router import create_app

# Set up logging first
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the API server."""
    try:
        # Load and validate configuration
        config = get_config()
        warnings = config.validate()

        # Set up observability (logging, tracing)
        setup_observability()

        # Log configuration warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        # Create application
        app = create_app()

        # Log startup info
        logger.info(
            "Starting StateSet Agents API",
            extra={
                "environment": config.environment.value,
                "host": config.host,
                "port": config.port,
                "api_version": config.api_version,
                "auth_required": config.security.require_auth,
                "rate_limit_enabled": config.rate_limit.enabled,
            }
        )

        # Run with uvicorn
        import uvicorn
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level=config.observability.log_level.lower(),
            access_log=False,  # We handle logging ourselves
        )

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Failed to start API server: {e}")
        sys.exit(1)


# Create app instance for uvicorn
try:
    config = get_config()
    setup_observability()
    app = create_app()
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Create a minimal error app
    from fastapi import FastAPI
    app = FastAPI(title="StateSet Agents API - Configuration Error")

    @app.get("/")
    async def configuration_error():
        return {
            "error": "Configuration error",
            "message": str(e),
            "hint": "Check your environment variables. See .env.example for required settings."
        }


if __name__ == "__main__":
    main()
