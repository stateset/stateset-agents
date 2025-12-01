"""
API v1 Router

Main API router with all endpoints, versioned under /api/v1.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import (
    AuthenticatedUser,
    RequestContext,
    get_current_user,
    get_request_context,
    require_admin,
    require_trainer,
    require_user,
)
from ..config import get_config, APIConfig
from ..errors import (
    APIError,
    NotFoundError,
    InternalError,
    TrainingError,
    setup_exception_handlers,
)
from ..middleware import setup_middleware, get_metrics
from ..models import (
    AgentConfig,
    ConversationRequest,
    ConversationResponse,
    HealthResponse,
    ComponentHealth,
    MetricsResponse,
    ScaleRequest,
    ScaleResponse,
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    TrainingMetrics,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Global State (would be replaced with Redis/DB in production)
# ============================================================================

_services: Dict[str, Any] = {}
_training_jobs: Dict[str, Dict[str, Any]] = {}
_active_conversations: Dict[str, Dict[str, Any]] = {}
_start_time: float = time.time()


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/api/v1", tags=["API v1"])


# ============================================================================
# Health & Metrics Endpoints
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and its components.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check() -> HealthResponse:
    """Comprehensive health check endpoint."""
    components = []

    # Check core services
    for name, service in _services.items():
        start = time.monotonic()
        try:
            if hasattr(service, "health_check"):
                await service.health_check()
            status = "healthy"
        except Exception as e:
            logger.warning(f"Health check failed for {name}: {e}")
            status = "unhealthy"

        latency = (time.monotonic() - start) * 1000
        components.append(ComponentHealth(
            name=name,
            status=status,
            latency_ms=round(latency, 2),
        ))

    # Determine overall status
    overall_status = "healthy" if all(c.status == "healthy" for c in components) else "degraded"
    if not components:
        overall_status = "healthy"  # No components to check

    return HealthResponse(
        status=overall_status,
        version="2.0.0",
        uptime_seconds=round(time.time() - _start_time, 2),
        components=components,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get Metrics",
    description="Get comprehensive system and API metrics.",
    dependencies=[Depends(require_admin)],
)
async def get_system_metrics(
    ctx: RequestContext = Depends(get_request_context),
) -> MetricsResponse:
    """Get system metrics (admin only)."""
    api_metrics = get_metrics().get_summary()

    return MetricsResponse(
        request_id=ctx.request_id,
        system={
            "uptime_seconds": round(time.time() - _start_time, 2),
            "active_services": len(_services),
            "active_jobs": len(_training_jobs),
            "active_conversations": len(_active_conversations),
        },
        api=api_metrics,
        training={
            "total_jobs": len(_training_jobs),
            "running_jobs": sum(1 for j in _training_jobs.values() if j.get("status") == "running"),
            "completed_jobs": sum(1 for j in _training_jobs.values() if j.get("status") == "completed"),
            "failed_jobs": sum(1 for j in _training_jobs.values() if j.get("status") == "failed"),
        },
        conversations={
            "active_count": len(_active_conversations),
        },
    )


# ============================================================================
# Training Endpoints
# ============================================================================

@router.post(
    "/training",
    response_model=TrainingResponse,
    status_code=202,
    summary="Start Training Job",
    description="Start a new GRPO/GSPO training job with the specified configuration.",
    responses={
        202: {"description": "Training job accepted and started"},
        400: {"description": "Invalid training configuration"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        422: {"description": "Validation error"},
    },
)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context),
) -> TrainingResponse:
    """Start a training job."""
    # Check for idempotency
    if request.idempotency_key:
        for job_id, job in _training_jobs.items():
            if job.get("idempotency_key") == request.idempotency_key:
                logger.info(f"Returning existing job for idempotency key: {request.idempotency_key}")
                return _build_training_response(job_id, job, ctx.request_id)

    # Create new job
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()

    _training_jobs[job_id] = {
        "status": TrainingStatus.STARTING.value,
        "strategy": request.strategy.value,
        "prompts": request.prompts,
        "num_iterations": request.num_iterations,
        "iterations_completed": 0,
        "total_trajectories": 0,
        "average_reward": 0.0,
        "computation_used": 0.0,
        "results": [],
        "error": None,
        "started_at": now,
        "completed_at": None,
        "user_id": ctx.user.user_id,
        "request_id": ctx.request_id,
        "idempotency_key": request.idempotency_key,
    }

    logger.info(
        "Training job created",
        extra={
            "job_id": job_id,
            "strategy": request.strategy.value,
            "num_prompts": len(request.prompts),
            "num_iterations": request.num_iterations,
            "user_id": ctx.user.user_id,
            "request_id": ctx.request_id,
        }
    )

    # Start training in background
    background_tasks.add_task(
        _run_training_job,
        job_id,
        request,
    )

    return _build_training_response(job_id, _training_jobs[job_id], ctx.request_id)


@router.get(
    "/training/{job_id}",
    response_model=TrainingResponse,
    summary="Get Training Status",
    description="Get the status and metrics of a training job.",
    responses={
        200: {"description": "Training job status"},
        404: {"description": "Training job not found"},
    },
)
async def get_training_status(
    job_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> TrainingResponse:
    """Get training job status."""
    if job_id not in _training_jobs:
        raise NotFoundError("Training job", job_id)

    return _build_training_response(job_id, _training_jobs[job_id], ctx.request_id)


@router.delete(
    "/training/{job_id}",
    summary="Cancel Training Job",
    description="Cancel a running training job.",
    responses={
        200: {"description": "Training job cancelled"},
        404: {"description": "Training job not found"},
        409: {"description": "Job cannot be cancelled"},
    },
)
async def cancel_training(
    job_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> Dict[str, Any]:
    """Cancel a training job."""
    if job_id not in _training_jobs:
        raise NotFoundError("Training job", job_id)

    job = _training_jobs[job_id]
    if job["status"] in (TrainingStatus.COMPLETED.value, TrainingStatus.FAILED.value):
        from ..errors import ConflictError
        raise ConflictError(f"Job is already {job['status']} and cannot be cancelled")

    job["status"] = TrainingStatus.CANCELLED.value
    job["completed_at"] = datetime.utcnow()

    logger.info(f"Training job {job_id} cancelled by {ctx.user.user_id}")

    return {
        "message": "Training job cancelled",
        "job_id": job_id,
        "request_id": ctx.request_id,
    }


# ============================================================================
# Conversation Endpoints
# ============================================================================

@router.post(
    "/conversations",
    response_model=ConversationResponse,
    summary="Chat with Agent",
    description="Send a message to an agent and get a response.",
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid request"},
        401: {"description": "Authentication required"},
        404: {"description": "Conversation not found"},
    },
)
async def create_conversation(
    request: ConversationRequest,
    ctx: RequestContext = Depends(get_request_context),
) -> ConversationResponse:
    """Create or continue a conversation."""
    start_time = time.monotonic()

    # Get or create conversation ID
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

    # Get or create conversation context
    if conversation_id not in _active_conversations:
        _active_conversations[conversation_id] = {
            "user_id": ctx.user.user_id,
            "created_at": datetime.utcnow(),
            "messages": [],
            "strategy": request.strategy,
        }

    conv = _active_conversations[conversation_id]

    # Build message content
    if request.message:
        user_message = request.message
    else:
        # Use last user message from messages list
        user_messages = [m for m in request.messages if m.role.value == "user"]
        user_message = user_messages[-1].content if user_messages else ""

    # Generate response (mock for now - would use actual agent)
    response_text = await _generate_response(conversation_id, user_message, request)

    # Calculate metrics
    processing_time = (time.monotonic() - start_time) * 1000
    tokens_used = len(response_text.split()) + len(user_message.split())  # Rough estimate

    # Update conversation history
    conv["messages"].append({"role": "user", "content": user_message})
    conv["messages"].append({"role": "assistant", "content": response_text})

    return ConversationResponse(
        request_id=ctx.request_id,
        conversation_id=conversation_id,
        response=response_text,
        tokens_used=tokens_used,
        processing_time_ms=round(processing_time, 2),
        context={
            "turn_count": len(conv["messages"]) // 2,
            "strategy": request.strategy,
        },
        metadata={
            "model": "stateset-agent-v2",
        },
    )


@router.get(
    "/conversations/{conversation_id}",
    summary="Get Conversation",
    description="Get conversation history and metadata.",
    responses={
        200: {"description": "Conversation details"},
        404: {"description": "Conversation not found"},
    },
)
async def get_conversation(
    conversation_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> Dict[str, Any]:
    """Get conversation details."""
    if conversation_id not in _active_conversations:
        raise NotFoundError("Conversation", conversation_id)

    conv = _active_conversations[conversation_id]
    return {
        "conversation_id": conversation_id,
        "user_id": conv["user_id"],
        "created_at": conv["created_at"].isoformat(),
        "message_count": len(conv["messages"]),
        "messages": conv["messages"],
        "request_id": ctx.request_id,
    }


@router.delete(
    "/conversations/{conversation_id}",
    summary="End Conversation",
    description="End a conversation and clean up resources.",
    responses={
        200: {"description": "Conversation ended"},
        404: {"description": "Conversation not found"},
    },
)
async def end_conversation(
    conversation_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> Dict[str, Any]:
    """End a conversation."""
    if conversation_id not in _active_conversations:
        raise NotFoundError("Conversation", conversation_id)

    conv = _active_conversations.pop(conversation_id)

    return {
        "message": "Conversation ended",
        "conversation_id": conversation_id,
        "total_messages": len(conv["messages"]),
        "request_id": ctx.request_id,
    }


# ============================================================================
# Scaling Endpoints
# ============================================================================

@router.post(
    "/scale",
    response_model=ScaleResponse,
    summary="Scale Resources",
    description="Scale computational resources for training engines.",
    dependencies=[Depends(require_admin)],
)
async def scale_resources(
    request: ScaleRequest,
    ctx: RequestContext = Depends(get_request_context),
) -> ScaleResponse:
    """Scale computational resources."""
    results = {}

    # Scale engines (mock implementation)
    if request.apply_to_all:
        for name, service in _services.items():
            if hasattr(service, "scale"):
                try:
                    result = await service.scale(request.scale_factor)
                    results[name] = {"status": "scaled", "result": result}
                except Exception as e:
                    results[name] = {"status": "error", "error": str(e)}
    elif request.target_engines:
        for engine_id in request.target_engines:
            if engine_id in _services:
                service = _services[engine_id]
                if hasattr(service, "scale"):
                    try:
                        result = await service.scale(request.scale_factor)
                        results[engine_id] = {"status": "scaled", "result": result}
                    except Exception as e:
                        results[engine_id] = {"status": "error", "error": str(e)}
                else:
                    results[engine_id] = {"status": "skipped", "reason": "no scale method"}
            else:
                results[engine_id] = {"status": "not_found"}

    return ScaleResponse(
        request_id=ctx.request_id,
        scale_factor=request.scale_factor,
        results=results,
        message=f"Scaling completed with factor {request.scale_factor}",
    )


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time interactions."""
    config = get_config()

    # Authenticate WebSocket connection
    if config.security.require_auth:
        api_key = (
            websocket.headers.get("X-API-Key") or
            websocket.headers.get("Authorization", "").replace("Bearer ", "")
        )

        if not api_key or api_key not in config.security.api_keys:
            await websocket.close(code=1008, reason="Unauthorized")
            return

    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "unknown")

            if msg_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })

            elif msg_type == "chat":
                # Handle chat message
                message = data.get("message", "")
                response = f"Echo: {message}"  # Mock response
                await websocket.send_json({
                    "type": "chat_response",
                    "response": response,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            elif msg_type == "subscribe_job":
                # Subscribe to job updates
                job_id = data.get("job_id")
                if job_id in _training_jobs:
                    await websocket.send_json({
                        "type": "job_update",
                        "job_id": job_id,
                        "status": _training_jobs[job_id]["status"],
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Job {job_id} not found",
                    })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")


# ============================================================================
# Helper Functions
# ============================================================================

def _build_training_response(job_id: str, job: Dict[str, Any], request_id: str) -> TrainingResponse:
    """Build TrainingResponse from job data."""
    return TrainingResponse(
        request_id=request_id,
        job_id=job_id,
        status=TrainingStatus(job["status"]),
        strategy=job["strategy"],
        metrics=TrainingMetrics(
            iterations_completed=job["iterations_completed"],
            total_trajectories=job["total_trajectories"],
            average_reward=job["average_reward"],
            computation_used=job["computation_used"],
        ),
        error=job.get("error"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


async def _run_training_job(job_id: str, request: TrainingRequest) -> None:
    """Run training job in background."""
    job = _training_jobs[job_id]
    job["status"] = TrainingStatus.RUNNING.value

    try:
        for i in range(request.num_iterations):
            # Check for cancellation
            if job["status"] == TrainingStatus.CANCELLED.value:
                logger.info(f"Job {job_id} was cancelled")
                return

            # Simulate training iteration
            await asyncio.sleep(0.5)

            # Update metrics
            job["iterations_completed"] = i + 1
            job["total_trajectories"] += len(request.prompts)
            job["average_reward"] = 0.5 + (i * 0.02)  # Mock improvement
            job["computation_used"] += len(request.prompts) * 1.5

            job["results"].append({
                "iteration": i + 1,
                "trajectories": len(request.prompts),
                "reward": job["average_reward"],
            })

            logger.debug(f"Job {job_id}: iteration {i+1}/{request.num_iterations}")

        job["status"] = TrainingStatus.COMPLETED.value
        job["completed_at"] = datetime.utcnow()
        logger.info(f"Training job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
        job["status"] = TrainingStatus.FAILED.value
        job["error"] = str(e)
        job["completed_at"] = datetime.utcnow()


async def _generate_response(
    conversation_id: str,
    message: str,
    request: ConversationRequest,
) -> str:
    """Generate agent response (mock implementation)."""
    # In production, this would use the actual MultiTurnAgent
    await asyncio.sleep(0.1)  # Simulate processing

    # Mock response
    responses = [
        "I'd be happy to help you with that!",
        "That's an interesting question. Let me think about it.",
        "Based on my understanding, here's what I can tell you:",
        "I can assist you with various tasks. What specifically would you like help with?",
    ]

    import random
    base_response = random.choice(responses)

    if "hello" in message.lower() or "hi" in message.lower():
        return "Hello! How can I assist you today?"

    return f"{base_response}\n\nYou asked: {message[:100]}..."


# ============================================================================
# Application Factory
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _start_time
    _start_time = time.time()

    logger.info("Starting StateSet Agents API v1")

    # Initialize services
    # In production, initialize actual services here
    _services["demo_engine"] = type("MockEngine", (), {"health_check": lambda self: None})()

    yield

    # Cleanup
    logger.info("Shutting down StateSet Agents API v1")
    _services.clear()
    _training_jobs.clear()
    _active_conversations.clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()

    app = FastAPI(
        title=config.title,
        description="""
## StateSet Agents API

Production-ready REST API for training and deploying AI agents using
reinforcement learning techniques (GRPO/GSPO).

### Features

- **Training**: Start and monitor GRPO/GSPO training jobs
- **Conversations**: Multi-turn conversational AI interactions
- **Scaling**: Dynamic resource allocation
- **Real-time**: WebSocket support for live updates

### Authentication

All endpoints require authentication via API key or JWT token:

```
Authorization: Bearer <token>
```
or
```
X-API-Key: <api-key>
```

### Rate Limits

- Default: 60 requests per minute
- Rate limit headers included in all responses

### Versioning

This is API version 1. All endpoints are prefixed with `/api/v1`.
        """,
        version="2.0.0",
        contact={
            "name": "StateSet Team",
            "email": "team@stateset.ai",
            "url": "https://stateset.ai",
        },
        license_info={
            "name": "Business Source License 1.1",
            "url": "https://github.com/stateset/stateset-agents/blob/main/LICENSE",
        },
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Setup CORS
    if config.cors.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors.allowed_origins,
            allow_credentials=config.cors.allow_credentials,
            allow_methods=config.cors.allowed_methods,
            allow_headers=config.cors.allowed_headers,
            max_age=config.cors.max_age,
        )

    # Add compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Setup custom middleware
    setup_middleware(app)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Include router
    app.include_router(router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": config.title,
            "version": "2.0.0",
            "api_version": "v1",
            "documentation": "/docs",
            "health": "/api/v1/health",
        }

    return app
