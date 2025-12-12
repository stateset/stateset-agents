from contextlib import asynccontextmanager
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from .config import get_config
from .routers import agents, training, metrics
from .middleware import setup_middleware
from .errors import setup_exception_handlers
from .openapi import setup_openapi, add_documentation_routes
from .resilience import HealthChecker, HealthStatus, get_all_circuit_stats

logger = logging.getLogger(__name__)

# Load config
config = get_config()

# Global health checker
health_checker = HealthChecker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with resource initialization."""
    logger.info("Starting StateSet Agents API v%s", config.api_version)

    # Initialize distributed cache (optional)
    try:
        from .distributed_cache import init_cache, close_cache, CacheConfig
        cache_config = CacheConfig.from_env()
        await init_cache(cache_config)
        logger.info("Cache initialized with backend: %s", cache_config.backend.value)
    except Exception as e:
        logger.warning("Cache initialization skipped: %s", e)

    # Initialize database (optional)
    try:
        from .persistence import init_database, close_database, DatabaseConfig
        db_config = DatabaseConfig.from_env()
        await init_database(db_config)
        logger.info("Database initialized with backend: %s", db_config.backend.value)
    except Exception as e:
        logger.warning("Database initialization skipped: %s", e)

    # Register health checks
    health_checker.add_check("api", lambda: True)

    yield

    # Cleanup resources
    logger.info("Shutting down StateSet Agents API")

    try:
        from .distributed_cache import close_cache
        await close_cache()
    except Exception:
        pass

    try:
        from .persistence import close_database
        await close_database()
    except Exception:
        pass

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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
    async def root() -> Dict[str, Any]:
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
            },
        }

    # Health endpoint
    @app.get(
        "/health",
        summary="Health Check",
        description="Check the health status of the API and its dependencies.",
        tags=["health"],
    )
    async def health_check() -> Dict[str, Any]:
        """
        Comprehensive health check endpoint.

        Returns the health status of the API and all registered dependencies.
        """
        results = await health_checker.check_all()
        overall_status = health_checker.overall_status

        return {
            "status": overall_status.value,
            "version": config.api_version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": {
                name: {
                    "status": result.status.value,
                    "latency_ms": round(result.latency_ms, 2),
                    "message": result.message,
                }
                for name, result in results.items()
            },
        }

    # Ready endpoint for Kubernetes
    @app.get(
        "/ready",
        summary="Readiness Check",
        description="Check if the API is ready to receive traffic.",
        tags=["health"],
    )
    async def readiness_check() -> Dict[str, Any]:
        """Kubernetes readiness probe endpoint."""
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
    async def liveness_check() -> Dict[str, Any]:
        """Kubernetes liveness probe endpoint."""
        return {"status": "alive", "timestamp": datetime.utcnow().isoformat() + "Z"}

    # Circuit breaker status endpoint
    @app.get(
        "/circuits",
        summary="Circuit Breaker Status",
        description="Get the status of all circuit breakers.",
        tags=["metrics"],
    )
    async def circuit_status() -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return {
            "circuits": get_all_circuit_stats(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        reload=not config.is_production(),
        log_level=config.observability.log_level.lower(),
    )