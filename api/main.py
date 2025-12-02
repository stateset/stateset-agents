from contextlib import asynccontextmanager
import logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from .config import get_config
from .routers import agents, training, metrics
from .middleware import setup_middleware

logger = logging.getLogger(__name__)

# Load config
config = get_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting StateSet Agents API")
    yield
    logger.info("Shutting down StateSet Agents API")

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

    # Root endpoint
    @app.get("/", summary="API Root", description="Welcome endpoint with API information")
    async def root():
        """Get API information."""
        return {
            "message": "Welcome to StateSet Agents API",
            "version": config.api_version,
            "documentation": "/docs",
            "health": "/health",
        }

    # Exception Handlers
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An internal error occurred",
                "timestamp": datetime.utcnow().timestamp(),
            },
        )

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