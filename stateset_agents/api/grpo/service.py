"""
GRPO Service - Refactored Modular Implementation

Main FastAPI application for the GRPO service with modular architecture.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

try:
    import uvicorn
    from fastapi import (
        FastAPI,
        HTTPException,
        Request,
    )
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from stateset_agents.exceptions import INFERENCE_EXCEPTIONS, MODEL_IO_EXCEPTIONS

from ..logging_config import (
    clear_request_id,
    configure_logging,
    get_logger,
    set_request_id,
)
from ..shutdown import ShutdownPhase, get_shutdown_manager
from .auth import verify_request
from .config import get_grpo_config
from .handlers import ConversationHandler, TrainingHandler, WebSocketHandler
from .metrics import get_grpo_metrics
from .router_v1 import create_v1_router
from .service_routes import register_routes
from .state import get_state_manager

logger = get_logger(__name__)

GRPO_INIT_EXCEPTIONS = MODEL_IO_EXCEPTIONS
GRPO_REQUEST_EXCEPTIONS = (HTTPException, *INFERENCE_EXCEPTIONS)


# Global services and handlers
_services: dict[str, Any] = {}
_training_handler: TrainingHandler | None = None
_conversation_handler: ConversationHandler | None = None
_websocket_handler: WebSocketHandler | None = None


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


def _build_error_response(
    request: Request,
    status_code: int,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    """Build consistent error response."""
    request_id = getattr(request.state, "request_id", "unknown-request")
    payload: dict[str, Any] = {
        "error": {"message": message, "status_code": status_code},
        "request_id": request_id,
        "path": str(request.url.path),
        "timestamp": datetime.utcnow().isoformat(),
    }
    if details is not None:
        payload["error"]["details"] = details

    return JSONResponse(status_code=status_code, content=payload)


class LightweightDemoEngine:
    """Lightweight demo engine for testing without heavy compute."""

    def __init__(self) -> None:
        self.scale_factor = 1.0
        self.total_trajectories = 0
        self.total_reward = 0.0

    async def train_iteration(self, prompts: list) -> dict[str, Any]:
        """Simulate a training iteration."""
        trajectories = len(prompts)
        average_reward = 0.5
        computation_used = trajectories * self.scale_factor

        self.total_trajectories += trajectories
        self.total_reward += average_reward * trajectories

        return {
            "trajectories_generated": trajectories,
            "average_reward": average_reward,
            "total_computation_used": computation_used,
            "scale_factor": self.scale_factor,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return engine metrics."""
        average_reward = (
            self.total_reward / self.total_trajectories
            if self.total_trajectories
            else 0.0
        )
        return {
            "total_trajectories": self.total_trajectories,
            "average_reward": average_reward,
            "scale_factor": self.scale_factor,
        }

    def scale_computation(self, scale_factor: float) -> dict[str, Any]:
        """Adjust computation scale."""
        self.scale_factor = scale_factor
        return {"scale_factor": scale_factor}

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


def _get_training_handler() -> TrainingHandler:
    """Return the active training handler, creating it lazily when needed."""
    global _training_handler
    if _training_handler is None:
        _training_handler = TrainingHandler(_services)
    elif _training_handler.services is not _services:
        _training_handler.services = _services
    return _training_handler


def _get_conversation_handler() -> ConversationHandler:
    """Return the active conversation handler, creating it lazily when needed."""
    global _conversation_handler
    if _conversation_handler is None:
        _conversation_handler = ConversationHandler(_services)
    elif _conversation_handler.services is not _services:
        _conversation_handler.services = _services
    return _conversation_handler


def _get_websocket_handler() -> WebSocketHandler:
    """Return the active WebSocket handler, creating it lazily when needed."""
    global _websocket_handler
    training_handler = _get_training_handler()
    conversation_handler = _get_conversation_handler()
    if _websocket_handler is None:
        _websocket_handler = WebSocketHandler(
            _services,
            training_handler,
            conversation_handler,
        )
    else:
        _websocket_handler.services = _services
        _websocket_handler.training_handler = training_handler
        _websocket_handler.conversation_handler = conversation_handler
    return _websocket_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app."""
    logger.info("Starting GRPO Service")
    config = get_grpo_config()
    shutdown_manager = get_shutdown_manager()

    # Validate configuration
    issues = config.validate()
    if issues:
        for issue in issues:
            logger.warning("Configuration issue: %s", issue)

    # Initialize services
    try:
        from stateset_agents.utils.cache import CacheService
        from stateset_agents.utils.monitoring import MonitoringService

        _services["monitoring"] = MonitoringService(
            enable_prometheus=config.enable_prometheus
        )
        _services["cache"] = CacheService()
    except ImportError:
        logger.warning("Optional services not available")

    # Initialize engine
    if config.use_lightweight_engine:
        _services["demo_engine"] = LightweightDemoEngine()
        logger.info("Initialized lightweight demo engine")
    else:
        try:
            from stateset_agents.core.agent import Agent
            from stateset_agents.core.agent_config import AgentConfig
            from stateset_agents.core.computational_engine import (
                create_computational_engine,
            )
            from stateset_agents.core.environment import Environment
            from stateset_agents.core.environment_base import (
                EnvironmentState,
                EpisodeStatus,
            )
            from stateset_agents.core.reward import RewardFunction, RewardResult
            from stateset_agents.core.trajectory import ConversationTurn

            class DemoAgent(Agent):
                async def generate_response(
                    self,
                    messages: str | list[dict[str, str]],
                    context: dict[str, Any] | None = None,
                ) -> str:
                    prompt = (
                        messages
                        if isinstance(messages, str)
                        else messages[-1]["content"] if messages else ""
                    )
                    return f"Response to: {prompt[:50]}..."

            class DemoEnvironment(Environment):
                async def reset(
                    self, scenario: dict[str, Any] | None = None
                ) -> EnvironmentState:
                    return EnvironmentState(
                        episode_id=str(uuid.uuid4()),
                        turn_count=0,
                        status=EpisodeStatus.ONGOING,
                        context={"scenario": scenario or {}, "state": "initial"},
                    )

                async def step(
                    self, state: EnvironmentState, action: ConversationTurn
                ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
                    next_state = state.copy()
                    next_state.turn_count += 1
                    done = next_state.turn_count >= self.max_turns
                    next_state.status = (
                        EpisodeStatus.COMPLETED if done else EpisodeStatus.ONGOING
                    )
                    return next_state, 0.5, done, {"response": action.content}

                async def get_reward(self, trajectory) -> float:
                    return 0.5

            class DemoReward(RewardFunction):
                async def compute_reward(self, turns, context=None):
                    return RewardResult(score=0.5, breakdown={})

            model_config = {
                "model_type": "gpt-oss",
                "model_name": "openai/gpt-oss-120b",
            }
            _services["demo_engine"] = create_computational_engine(
                DemoAgent(
                    AgentConfig(
                        model_name=model_config["model_name"],
                        use_stub_model=True,
                    )
                ),
                DemoEnvironment(),
                DemoReward(),
                num_workers=1,
            )
            logger.info("Initialized computational engine")
        except GRPO_INIT_EXCEPTIONS as e:
            logger.warning("Could not create computational engine: %s", e)
            _services["demo_engine"] = LightweightDemoEngine()

    # Initialize multi-turn agent
    try:
        from stateset_agents.core.multiturn_agent import (
            DialogueDatabase,
            MultiTurnAgent,
        )

        model_config = {"model_type": "gpt-oss", "model_name": "openai/gpt-oss-120b"}
        _services["multiturn_agent"] = MultiTurnAgent(
            model_config,
            dialogue_database=DialogueDatabase([]),
        )
        logger.info("Initialized multi-turn agent")
    except GRPO_INIT_EXCEPTIONS as e:
        logger.warning("Could not create multi-turn agent: %s", e)

    # Initialize handlers lazily but ensure they are available during startup.
    _get_training_handler()
    _get_conversation_handler()
    _get_websocket_handler()

    # Register shutdown tasks
    async def cleanup_jobs():
        """Cancel pending training jobs."""
        state = get_state_manager()
        pending_jobs = [
            job_id
            for job_id in list(state.jobs.keys())
            if state.get_job(job_id)
            and state.get_job(job_id).status in ("starting", "running")
        ]
        for job_id in pending_jobs:
            state.update_job(job_id, status="cancelled")
            logger.info("Cancelled training job: %s", job_id)

    async def cleanup_conversations():
        """End active conversations."""
        state = get_state_manager()
        active_convs = list(state.conversations.keys())
        for conv_id in active_convs:
            state.end_conversation(conv_id)
            logger.debug("Ended conversation: %s", conv_id)
        logger.info("Cleaned up %d conversations", len(active_convs))

    async def cleanup_state():
        """Cleanup state manager."""
        state = get_state_manager()
        cleanup_counts = state.cleanup()
        logger.info("State cleanup: %s", cleanup_counts)

    async def cleanup_engines():
        """Cleanup computational engines."""
        state = get_state_manager()
        for engine_id in list(state.engines.keys()):
            engine = state.unregister_engine(engine_id)
            if hasattr(engine, "cleanup"):
                engine.cleanup()
            logger.debug("Cleaned up engine: %s", engine_id)

        if "demo_engine" in _services:
            engine = _services["demo_engine"]
            if hasattr(engine, "cleanup"):
                engine.cleanup()
            logger.info("Cleaned up demo engine")

    shutdown_manager.register_task(
        "cleanup_jobs",
        ShutdownPhase.CLEANUP_JOBS,
        cleanup_jobs,
        timeout_seconds=10.0,
    )
    shutdown_manager.register_task(
        "cleanup_conversations",
        ShutdownPhase.CLEANUP_CONNECTIONS,
        cleanup_conversations,
        timeout_seconds=5.0,
    )
    shutdown_manager.register_task(
        "cleanup_state",
        ShutdownPhase.CLEANUP_STATE,
        cleanup_state,
        timeout_seconds=5.0,
    )
    shutdown_manager.register_task(
        "cleanup_engines",
        ShutdownPhase.CLEANUP_ENGINES,
        cleanup_engines,
        timeout_seconds=10.0,
    )

    logger.info("GRPO Service initialized")

    yield

    # Execute graceful shutdown
    logger.info("Initiating graceful shutdown")
    await shutdown_manager.execute_shutdown()
    logger.info("GRPO Service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is required but not available")

    config = get_grpo_config()

    app = FastAPI(
        title="GRPO Service",
        description="""
Comprehensive GRPO (Group Relative Policy Optimization) training and inference API.

## API Versioning

This API supports versioning via URL prefix. The current version is v1.

- **Current endpoints**: `/api/*` (default, latest)
- **Versioned endpoints**: `/v1/*` (stable, recommended for production)

All endpoints are available under both prefixes with identical behavior.

## Authentication

All endpoints require authentication via API key or allow anonymous access
if configured. Use the `X-API-Key` header or `Authorization: Bearer <key>`.

## Rate Limiting

Rate limiting is enforced per API key or client IP. Default: 60 requests/minute.
        """,
        version="2.0.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "training", "description": "Training job management"},
            {"name": "conversations", "description": "Multi-turn conversations"},
            {"name": "scaling", "description": "Resource scaling"},
            {"name": "batch", "description": "Batch operations for bulk processing"},
            {"name": "monitoring", "description": "Health and metrics"},
            {"name": "v1-training", "description": "Training (API v1)"},
            {"name": "v1-conversations", "description": "Conversations (API v1)"},
            {"name": "v1-batch", "description": "Batch operations (API v1)"},
        ],
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Request-ID",
        ],
    )

    # Request middleware
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        """Attach request ID and capture metrics."""
        shutdown_manager = get_shutdown_manager()

        # Reject requests during shutdown
        if shutdown_manager.is_shutting_down:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Service is shutting down",
                        "status_code": 503,
                    },
                    "path": str(request.url.path),
                },
                headers={"Retry-After": "30"},
            )

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.monotonic()

        # Track in-flight request
        shutdown_manager.increment_in_flight()

        # Set request ID in logging context
        token = set_request_id(request_id)

        try:
            response = await call_next(request)
        except GRPO_REQUEST_EXCEPTIONS:
            logger.exception("Unhandled exception on %s", request.url.path)
            response = _build_error_response(
                request,
                500,
                "Internal server error",
            )
        finally:
            # Clear request ID from context
            clear_request_id(token)
            # Decrement in-flight count
            shutdown_manager.decrement_in_flight()

        # Record metrics
        latency_ms = (time.monotonic() - start_time) * 1000
        metrics = get_grpo_metrics()
        metrics.record_request(
            request.url.path,
            request.method,
            response.status_code,
            latency_ms,
        )

        # Log request completion
        logger.info(
            "Request completed: %s %s -> %d (%.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            extra={
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )

        response.headers["x-request-id"] = request_id
        return response

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return _build_error_response(request, exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return _build_error_response(
            request,
            422,
            "Invalid request payload",
            details=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on path %s", request.url.path)
        return _build_error_response(
            request,
            500,
            "Internal server error",
        )

    # Register routes
    _register_routes(app)

    # Register versioned router (v1)
    # This is registered after handlers are initialized in lifespan
    # but we need to register it here for route discovery
    v1_router = create_v1_router(
        training_handler=_get_training_handler(),
        conversation_handler=_get_conversation_handler(),
        services=_services,
        verify_request=verify_request,
    )
    app.include_router(v1_router)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    register_routes(
        app,
        get_services=lambda: _services,
        get_training_handler=_get_training_handler,
        get_conversation_handler=_get_conversation_handler,
        get_websocket_handler=_get_websocket_handler,
        verify_request=verify_request,
    )


app = LazyApp() if FASTAPI_AVAILABLE else None


def main():
    """Main entry point."""
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI is required but not available")
        return

    # Configure structured logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    json_logs = os.getenv("LOG_FORMAT", "text").lower() == "json"
    configure_logging(level=log_level, json_format=json_logs, include_request_id=True)

    print("\n" + "=" * 60)
    print("GRPO SERVICE - Modular Architecture")
    print("=" * 60)
    print("\nModules:")
    print("  - config.py: Configuration management")
    print("  - models.py: Request/response models")
    print("  - state.py: State management with TTL")
    print("  - metrics.py: Metrics collection")
    print("  - rate_limiter.py: Unified rate limiting")
    print("  - handlers.py: Request handlers")
    print("  - service.py: FastAPI application")
    print("=" * 60 + "\n")

    uvicorn.run(
        "stateset_agents.api.grpo.service:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
    )


if __name__ == "__main__":
    main()
