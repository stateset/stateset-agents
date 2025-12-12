"""
GRPO Service - Refactored Modular Implementation

Main FastAPI application for the GRPO service with modular architecture.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import uvicorn
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        HTTPException,
        Request,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..logging_config import (
    configure_logging,
    get_logger,
    set_request_id,
    clear_request_id,
)
from ..shutdown import (
    GracefulShutdownManager,
    ShutdownPhase,
    get_shutdown_manager,
)
from .config import GRPOConfig, get_grpo_config
from .handlers import ConversationHandler, TrainingHandler, WebSocketHandler
from .router_v1 import create_v1_router
from .metrics import get_grpo_metrics
from .models import (
    BatchCancelRequest,
    BatchCancelResponse,
    BatchItemResult,
    BatchJobStatusRequest,
    BatchJobStatusResponse,
    BatchTrainingRequest,
    BatchTrainingResponse,
    GRPOConversationRequest,
    GRPOConversationResponse,
    GRPOHealthResponse,
    GRPOMetricsResponse,
    GRPOScaleRequest,
    GRPOScaleResponse,
    GRPOTrainingRequest,
    GRPOTrainingResponse,
)
from .rate_limiter import get_rate_limiter
from .state import get_state_manager

logger = get_logger(__name__)


# Global services and handlers
_services: Dict[str, Any] = {}
_training_handler: Optional[TrainingHandler] = None
_conversation_handler: Optional[ConversationHandler] = None
_websocket_handler: Optional[WebSocketHandler] = None


class RequestContext:
    """Authenticated request context."""

    def __init__(
        self,
        request_id: str,
        user_id: str,
        roles: list,
        api_key: Optional[str],
        client: str,
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.roles = roles
        self.api_key = api_key
        self.client = client


def _extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from request headers."""
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    api_key = request.headers.get("x-api-key")
    return api_key.strip() if api_key else None


async def verify_request(request: Request) -> RequestContext:
    """Authenticate request and enforce rate limits."""
    config = get_grpo_config()
    rate_limiter = get_rate_limiter()
    metrics = get_grpo_metrics()

    client_ip = request.client.host if request.client else "unknown"
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    api_key = _extract_api_key(request)

    if config.api_keys:
        if not api_key or api_key not in config.api_keys:
            raise HTTPException(
                status_code=401,
                detail="A valid API key is required for this API",
            )
        roles = config.api_keys[api_key]
        user_id = request.headers.get("x-user-id", client_ip)
    elif config.allow_anonymous:
        roles = ["anonymous"]
        user_id = request.headers.get("x-user-id", client_ip)
    else:
        raise HTTPException(
            status_code=401,
            detail="API key required. Set GRPO_ALLOW_ANONYMOUS=true to enable unauthenticated access.",
        )

    # Rate limiting
    limit_key = api_key or client_ip
    if not rate_limiter.allow(limit_key, config.rate_limit_per_minute):
        metrics.record_rate_limit_hit()
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please retry after a short delay.",
        )

    return RequestContext(
        request_id=request_id,
        user_id=user_id,
        roles=roles,
        api_key=api_key,
        client=client_ip,
    )


def _build_error_response(
    request: Request,
    status_code: int,
    message: str,
    details: Optional[Any] = None,
) -> JSONResponse:
    """Build consistent error response."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    payload = {
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

    async def train_iteration(self, prompts: list) -> Dict[str, Any]:
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

    def get_metrics(self) -> Dict[str, Any]:
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

    def scale_computation(self, scale_factor: float) -> Dict[str, Any]:
        """Adjust computation scale."""
        self.scale_factor = scale_factor
        return {"scale_factor": scale_factor}

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app."""
    global _services, _training_handler, _conversation_handler, _websocket_handler

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
        from utils.monitoring import MonitoringService
        from utils.cache import CacheService

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
            from stateset_agents.core.environment import Environment
            from stateset_agents.core.reward import RewardFunction, RewardResult
            from stateset_agents.core.computational_engine import create_computational_engine

            class DemoAgent(Agent):
                async def generate_response(self, prompt: str) -> str:
                    return f"Response to: {prompt[:50]}..."

            class DemoEnvironment(Environment):
                async def reset(self) -> Dict[str, Any]:
                    return {"state": "initial"}

                async def step(self, action: str) -> Dict[str, Any]:
                    return {"reward": 0.5, "done": False}

                async def get_reward(self, trajectory) -> float:
                    return 0.5

            class DemoReward(RewardFunction):
                async def compute_reward(self, turns, context=None):
                    return RewardResult(score=0.5, breakdown={})

            model_config = {"model_type": "gpt-oss", "model_name": "openai/gpt-oss-120b"}
            _services["demo_engine"] = create_computational_engine(
                DemoAgent(model_config),
                DemoEnvironment(),
                DemoReward(),
                num_workers=1,
            )
            logger.info("Initialized computational engine")
        except Exception as e:
            logger.warning("Could not create computational engine: %s", e)
            _services["demo_engine"] = LightweightDemoEngine()

    # Initialize multi-turn agent
    try:
        from stateset_agents.core.multiturn_agent import DialogueDatabase, MultiTurnAgent

        model_config = {"model_type": "gpt-oss", "model_name": "openai/gpt-oss-120b"}
        _services["multiturn_agent"] = MultiTurnAgent(
            model_config,
            dialogue_database=DialogueDatabase([]),
        )
        logger.info("Initialized multi-turn agent")
    except Exception as e:
        logger.warning("Could not create multi-turn agent: %s", e)

    # Initialize handlers
    _training_handler = TrainingHandler(_services)
    _conversation_handler = ConversationHandler(_services)
    _websocket_handler = WebSocketHandler(
        _services,
        _training_handler,
        _conversation_handler,
    )

    # Register shutdown tasks
    async def cleanup_jobs():
        """Cancel pending training jobs."""
        state = get_state_manager()
        pending_jobs = [
            job_id for job_id in list(state.jobs.keys())
            if state.get_job(job_id) and state.get_job(job_id).status in ("starting", "running")
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
            "X-User-ID",
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
                    "error": {"message": "Service is shutting down", "status_code": 503},
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
        except Exception:
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
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
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
        training_handler=_training_handler if _training_handler else TrainingHandler({}),
        conversation_handler=_conversation_handler if _conversation_handler else ConversationHandler({}),
        services=_services,
        verify_request=verify_request,
    )
    app.include_router(v1_router)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.get("/", tags=["monitoring"])
    async def root():
        """Root endpoint with service information."""
        config = get_grpo_config()
        return {
            "title": "GRPO Service",
            "version": "2.0.0",
            "description": "Comprehensive GRPO training and inference API",
            "security": {
                "auth_enabled": bool(config.api_keys),
                "allow_anonymous": config.allow_anonymous,
                "rate_limit_per_minute": config.rate_limit_per_minute,
            },
            "endpoints": {
                "training": "/api/train",
                "conversations": "/api/chat",
                "scaling": "/api/scale",
                "metrics": "/api/metrics",
                "health": "/health",
                "websocket": "/ws",
            },
            "api_versions": {
                "current": "2.0.0",
                "supported": ["v1"],
                "deprecated": [],
                "v1_prefix": "/v1",
            },
        }

    @app.get("/health", response_model=GRPOHealthResponse, tags=["monitoring"])
    async def health_check():
        """Health check endpoint."""
        return GRPOHealthResponse(
            status="healthy",
            services={
                "monitoring": "monitoring" in _services,
                "cache": "cache" in _services,
                "demo_engine": "demo_engine" in _services,
                "multiturn_agent": "multiturn_agent" in _services,
            },
        )

    @app.get("/api/metrics", response_model=GRPOMetricsResponse, tags=["monitoring"])
    async def get_metrics(ctx: RequestContext = Depends(verify_request)):
        """Get comprehensive system metrics."""
        state = get_state_manager()
        metrics = get_grpo_metrics()

        response = GRPOMetricsResponse(
            system=state.stats(),
            training_jobs={},
            engines={},
            conversations={},
            api=metrics.get_summary(),
            rate_limit={
                "requests_per_minute": get_grpo_config().rate_limit_per_minute,
            },
        )

        # Add job details
        for job_id in list(state.jobs.keys()):
            job = state.get_job(job_id)
            if job:
                response.training_jobs[job_id] = {
                    "status": job.status,
                    "strategy": job.strategy,
                    "iterations_completed": job.iterations_completed,
                }

        # Add engine metrics
        if "demo_engine" in _services:
            response.engines["demo_engine"] = _services["demo_engine"].get_metrics()

        # Add conversation metrics
        if "multiturn_agent" in _services:
            agent = _services["multiturn_agent"]
            response.conversations = {
                "active_count": len(agent.get_active_conversations()),
                "strategies_available": list(agent.strategies.keys()),
            }

        return response

    @app.post(
        "/api/train",
        response_model=GRPOTrainingResponse,
        tags=["training"],
    )
    async def train_agent(
        request: GRPOTrainingRequest,
        background_tasks: BackgroundTasks,
        ctx: RequestContext = Depends(verify_request),
    ):
        """Start GRPO training."""
        response = await _training_handler.start_training(
            request,
            ctx.user_id,
            ctx.request_id,
        )

        # Launch training in background
        if request.strategy == "distributed":
            background_tasks.add_task(
                _training_handler.run_distributed_training,
                response.job_id,
                request.prompts,
                request.num_iterations,
                request.distributed_config or {},
            )
        else:
            background_tasks.add_task(
                _training_handler.run_computational_training,
                response.job_id,
                request.prompts,
                request.num_iterations,
                request.use_neural_rewards,
                request.use_ruler_rewards,
            )

        return response

    @app.get(
        "/api/status/{job_id}",
        response_model=GRPOTrainingResponse,
        tags=["training"],
    )
    async def get_training_status(
        job_id: str,
        ctx: RequestContext = Depends(verify_request),
    ):
        """Get training job status."""
        response = _training_handler.get_job_status(job_id)
        if not response:
            raise HTTPException(status_code=404, detail="Training job not found")
        return response

    @app.delete("/api/jobs/{job_id}", tags=["training"])
    async def cancel_training_job(
        job_id: str,
        ctx: RequestContext = Depends(verify_request),
    ):
        """Cancel a training job."""
        if not _training_handler.cancel_job(job_id):
            raise HTTPException(
                status_code=404,
                detail="Training job not found or already completed",
            )
        return {"message": "Training job cancelled", "job_id": job_id}

    @app.post(
        "/api/chat",
        response_model=GRPOConversationResponse,
        tags=["conversations"],
    )
    async def chat(
        request: GRPOConversationRequest,
        ctx: RequestContext = Depends(verify_request),
    ):
        """Send a conversation message."""
        try:
            return await _conversation_handler.handle_message(request, ctx.user_id)
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/conversations/{conversation_id}", tags=["conversations"])
    async def end_conversation(
        conversation_id: str,
        ctx: RequestContext = Depends(verify_request),
    ):
        """End a conversation."""
        result = _conversation_handler.end_conversation(conversation_id)
        if not result:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation ended", **result}

    @app.post(
        "/api/scale",
        response_model=GRPOScaleResponse,
        tags=["scaling"],
    )
    async def scale_computation(
        request: GRPOScaleRequest,
        ctx: RequestContext = Depends(verify_request),
    ):
        """Scale computational resources."""
        results = {}
        state = get_state_manager()

        if request.apply_to_all:
            # Scale all engines
            for engine_id in list(state.engines.keys()):
                engine = state.get_engine(engine_id)
                if hasattr(engine, "scale_computation"):
                    try:
                        result = engine.scale_computation(request.scale_factor)
                        results[engine_id] = result
                    except Exception as e:
                        results[engine_id] = {"error": str(e)}

        # Scale demo engine
        if "demo_engine" in _services:
            try:
                result = _services["demo_engine"].scale_computation(request.scale_factor)
                results["demo_engine"] = result
            except Exception as e:
                results["demo_engine"] = {"error": str(e)}

        return GRPOScaleResponse(
            message="Computational resources scaled",
            scale_factor=request.scale_factor,
            results=results,
        )

    # ========================================================================
    # Batch Operations
    # ========================================================================

    @app.post(
        "/api/batch/train",
        response_model=BatchTrainingResponse,
        tags=["batch"],
        summary="Submit batch training jobs",
        description="Submit multiple training jobs in a single request",
    )
    async def batch_train(
        request: BatchTrainingRequest,
        background_tasks: BackgroundTasks,
        ctx: RequestContext = Depends(verify_request),
    ):
        """Submit multiple training jobs in batch."""
        import asyncio

        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        results: list = []
        accepted = 0
        rejected = 0

        for i, item in enumerate(request.items):
            try:
                # Create training request
                training_req = GRPOTrainingRequest(
                    prompts=item.prompts,
                    strategy=item.strategy,
                    num_iterations=item.num_iterations,
                    idempotency_key=item.idempotency_key,
                )

                # Start training
                response = await _training_handler.start_training(
                    training_req,
                    ctx.user_id,
                    ctx.request_id,
                )

                # Schedule background task
                background_tasks.add_task(
                    _training_handler.run_computational_training,
                    response.job_id,
                    training_req.prompts,
                    training_req.num_iterations,
                    True,  # use_neural_rewards
                    False,  # use_ruler_rewards
                )

                results.append(BatchItemResult(
                    index=i,
                    job_id=response.job_id,
                    status="accepted",
                ))
                accepted += 1

            except Exception as e:
                logger.warning("Batch item %d failed: %s", i, e)
                results.append(BatchItemResult(
                    index=i,
                    job_id=None,
                    status="rejected",
                    error=str(e),
                ))
                rejected += 1

                if request.fail_fast:
                    break

        return BatchTrainingResponse(
            batch_id=batch_id,
            total_items=len(request.items),
            accepted=accepted,
            rejected=rejected,
            results=results,
        )

    @app.post(
        "/api/batch/status",
        response_model=BatchJobStatusResponse,
        tags=["batch"],
        summary="Get status of multiple jobs",
    )
    async def batch_job_status(
        request: BatchJobStatusRequest,
        ctx: RequestContext = Depends(verify_request),
    ):
        """Get status of multiple training jobs."""
        jobs: dict = {}
        not_found: list = []

        for job_id in request.job_ids:
            response = _training_handler.get_job_status(job_id)
            if response:
                jobs[job_id] = response
            else:
                not_found.append(job_id)

        return BatchJobStatusResponse(
            jobs=jobs,
            not_found=not_found,
        )

    @app.post(
        "/api/batch/cancel",
        response_model=BatchCancelResponse,
        tags=["batch"],
        summary="Cancel multiple jobs",
    )
    async def batch_cancel(
        request: BatchCancelRequest,
        ctx: RequestContext = Depends(verify_request),
    ):
        """Cancel multiple training jobs."""
        cancelled: list = []
        not_found: list = []
        already_completed: list = []

        for job_id in request.job_ids:
            job = _training_handler.state.get_job(job_id)

            if not job:
                not_found.append(job_id)
            elif job.status in ("completed", "failed", "cancelled"):
                already_completed.append(job_id)
            else:
                _training_handler.cancel_job(job_id)
                cancelled.append(job_id)

        return BatchCancelResponse(
            cancelled=cancelled,
            not_found=not_found,
            already_completed=already_completed,
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time interactions."""
        config = get_grpo_config()
        rate_limiter = get_rate_limiter()

        # Authenticate before accepting
        if config.api_keys:
            api_key = websocket.headers.get("x-api-key") or websocket.headers.get(
                "authorization"
            )
            if api_key and api_key.lower().startswith("bearer "):
                api_key = api_key.split(" ", 1)[1].strip()

            if not api_key or api_key not in config.api_keys:
                await websocket.close(code=1008, reason="Unauthorized")
                return

            # Rate limit
            limit_key = f"ws:{api_key}"
            if not rate_limiter.allow(limit_key, config.rate_limit_per_minute):
                await websocket.close(code=1008, reason="Rate limit exceeded")
                return

        await websocket.accept()

        try:
            await _websocket_handler.handle_connection(websocket)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error("WebSocket error: %s", e)
            await websocket.close()


# Create the app instance
app = create_app() if FASTAPI_AVAILABLE else None


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

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
