import logging
import time
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config import get_config
from .errors import setup_exception_handlers
from .middleware import setup_middleware
from .openapi import add_documentation_routes, setup_openapi
from .resilience import HealthChecker, HealthStatus, get_all_circuit_stats
from .routers import agents, messages, metrics, openai, training, training_lab, v1
from .services.inference_service import InferenceConfig, InferenceService

logger = logging.getLogger(__name__)

API_LIFESPAN_EXCEPTIONS = (ImportError, OSError, RuntimeError, ValueError)


class LazyApp:
    """ASGI-compatible proxy that creates the FastAPI app on first use."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def _get_app(self) -> FastAPI:
        if self._app is None:
            self._app = create_app()
        return self._app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._get_app()(scope, receive, send)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_app(), name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with resource initialization."""
    config = getattr(app.state, "config", get_config())
    strict_startup = (
        os.getenv("STATESET_AGENTS_STRICT_STARTUP", "false").lower() in {"1", "true", "on", "yes"}
        or config.is_production()
    )
    health_checker = getattr(app.state, "health_checker", HealthChecker())
    app.state.health_checker = health_checker
    inference_service = getattr(app.state, "inference_service", None)
    if inference_service is None:
        inference_service = InferenceService(InferenceConfig.from_env())
        app.state.inference_service = inference_service

    logger.info("Starting StateSet Agents API v%s", config.api_version)

    # Initialize distributed cache (optional)
    close_cache_fn = None
    try:
        from .distributed_cache import CacheConfig
        from .distributed_cache import close_cache as _close_cache
        from .distributed_cache import init_cache

        cache_config = CacheConfig.from_env()
        await init_cache(cache_config)
        close_cache_fn = _close_cache
        logger.info("Cache initialized with backend: %s", cache_config.backend.value)
    except API_LIFESPAN_EXCEPTIONS as e:
        if strict_startup:
            logger.error("Cache initialization failed during startup: %s", e)
            raise
        logger.warning("Cache initialization skipped: %s", e)

    # Initialize database (optional)
    close_database_fn = None
    try:
        from .persistence import DatabaseConfig
        from .persistence import close_database as _close_database
        from .persistence import init_database

        db_config = DatabaseConfig.from_env()
        await init_database(db_config)
        close_database_fn = _close_database
        logger.info("Database initialized with backend: %s", db_config.backend.value)
    except API_LIFESPAN_EXCEPTIONS as e:
        if strict_startup:
            logger.error("Database initialization failed during startup: %s", e)
            raise
        logger.warning("Database initialization skipped: %s", e)

    # Initialize agent service
    from .services.agent_service import AgentService
    from stateset_agents.utils.security import SecurityMonitor as _SecurityMonitor

    agent_service = getattr(app.state, "agent_service", None)
    if agent_service is None:
        agent_service = AgentService(_SecurityMonitor())
        app.state.agent_service = agent_service

    # Initialize training service
    from .services.training_service import TrainingService

    training_service = getattr(app.state, "training_service", None)
    if training_service is None:
        training_service = TrainingService()
        app.state.training_service = training_service

    # Register health checks
    health_checker.add_check("api", lambda: True)
    if inference_service is not None:
        health_checker.add_check("inference_backend", inference_service.check_health)

    yield

    # Cleanup resources
    logger.info("Shutting down StateSet Agents API")

    if close_cache_fn is not None:
        try:
            await close_cache_fn()
        except API_LIFESPAN_EXCEPTIONS:
            pass

    if close_database_fn is not None:
        try:
            await close_database_fn()
        except API_LIFESPAN_EXCEPTIONS:
            pass

    if inference_service is not None:
        try:
            await inference_service.aclose()
        except API_LIFESPAN_EXCEPTIONS:
            pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    health_checker = HealthChecker()
    app = FastAPI(
        title=config.title,
        description="""
        ## StateSet Agents API

        A comprehensive REST API for training and deploying AI agents using
        reinforcement learning techniques.

        ### Features

        - 🤖 **Agent Management**: Create and manage AI agents
        - 💬 **Conversations**: Interactive conversations with agents
        - 🎯 **Training**: Train agents using reinforcement learning
        - 📊 **Monitoring**: Real-time metrics and monitoring
        - 🔒 **Security**: Authentication and authorization
        - 📈 **Analytics**: Performance analytics and insights
        """,
        version=config.api_version,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.health_checker = health_checker
    app.state.started_at = time.time()
    if not getattr(app.state, "inference_service", None):
        app.state.inference_service = InferenceService(InferenceConfig.from_env())

    # Middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allowed_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allowed_methods,
        allow_headers=config.cors.allowed_headers,
        max_age=config.cors.max_age,
    )

    # Advanced Middleware (Rate Limiting, Security Headers, etc.)
    setup_middleware(app)

    # Include routers
    app.include_router(agents.router)
    app.include_router(agents.conversation_router)
    app.include_router(training.router)
    app.include_router(metrics.router)
    app.include_router(v1.router)
    app.include_router(messages.router)
    app.include_router(openai.router)
    app.include_router(training_lab.router)

    # Compatibility aliases for legacy tests/clients
    app.add_api_route(
        "/conversation",
        v1.chat_v1,
        methods=["POST"],
        response_model=v1.ConversationResponseV1,
        include_in_schema=False,
    )

    # Setup OpenAPI documentation
    setup_openapi(app)
    add_documentation_routes(app)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Root endpoint
    @app.get(
        "/",
        summary="API Root",
        description="Welcome endpoint with API information and available endpoints.",
        tags=["health"],
    )
    async def root() -> dict[str, Any]:
        """Get API information."""
        return {
            "message": "Welcome to StateSet Agents API",
            "version": config.api_version,
            "documentation": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "endpoints": {
                "agents": "/agents",
                "conversations": "/conversations",
                "training": "/training",
                "metrics": "/metrics",
                "messages": "/v1/messages",
                "chat_completions": "/v1/chat/completions",
            },
        }

    # Ready endpoint for Kubernetes
    @app.get(
        "/ready",
        summary="Readiness Check",
        description="Check if the API is ready to receive traffic.",
        tags=["health"],
    )
    async def readiness_check() -> Any:
        """Kubernetes readiness probe endpoint."""
        await health_checker.check_all()
        overall_status = health_checker.overall_status

        if overall_status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "Dependencies unhealthy"},
            )

        return {"status": "ready"}

    # Liveness endpoint for Kubernetes
    @app.get(
        "/live",
        summary="Liveness Check",
        description="Check if the API is alive.",
        tags=["health"],
    )
    async def liveness_check() -> dict[str, Any]:
        """Kubernetes liveness probe endpoint."""
        return {"status": "alive", "timestamp": datetime.utcnow().isoformat() + "Z"}

    # Circuit breaker status endpoint
    @app.get(
        "/circuits",
        summary="Circuit Breaker Status",
        description="Get the status of all circuit breakers.",
        tags=["metrics"],
    )
    async def circuit_status() -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return {
            "circuits": get_all_circuit_stats(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    return app


app = LazyApp()

if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "stateset_agents.api.main:app",
        host=config.host,
        port=config.port,
        reload=not config.is_production(),
        log_level=config.observability.log_level.lower(),
    )
