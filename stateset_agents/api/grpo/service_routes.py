"""Route registration helpers for the GRPO API service."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, WebSocket
from fastapi import WebSocketDisconnect

from stateset_agents.exceptions import INFERENCE_EXCEPTIONS

from ..logging_config import get_logger
from .config import get_grpo_config
from .handlers import ConversationHandler, TrainingHandler, WebSocketHandler
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

GRPO_ENGINE_EXCEPTIONS = INFERENCE_EXCEPTIONS
GRPO_BATCH_EXCEPTIONS = INFERENCE_EXCEPTIONS
GRPO_WS_EXCEPTIONS = INFERENCE_EXCEPTIONS


def register_routes(
    app: FastAPI,
    *,
    get_services: Callable[[], dict[str, Any]],
    get_training_handler: Callable[[], TrainingHandler],
    get_conversation_handler: Callable[[], ConversationHandler],
    get_websocket_handler: Callable[[], WebSocketHandler],
    verify_request: Callable[..., Any],
) -> None:
    """Register all GRPO REST and WebSocket routes."""

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
        services = get_services()
        return GRPOHealthResponse(
            status="healthy",
            services={
                "monitoring": "monitoring" in services,
                "cache": "cache" in services,
                "demo_engine": "demo_engine" in services,
                "multiturn_agent": "multiturn_agent" in services,
            },
        )

    @app.get("/api/metrics", response_model=GRPOMetricsResponse, tags=["monitoring"])
    async def get_metrics(ctx=Depends(verify_request)):
        """Get comprehensive system metrics."""
        del ctx
        services = get_services()
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

        for job_id in list(state.jobs.keys()):
            job = state.get_job(job_id)
            if job:
                response.training_jobs[job_id] = {
                    "status": job.status,
                    "strategy": job.strategy,
                    "iterations_completed": job.iterations_completed,
                }

        if "demo_engine" in services:
            response.engines["demo_engine"] = services["demo_engine"].get_metrics()

        if "multiturn_agent" in services:
            agent = services["multiturn_agent"]
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
        ctx=Depends(verify_request),
    ):
        """Start GRPO training."""
        training_handler = get_training_handler()
        response = await training_handler.start_training(
            request,
            ctx.user_id,
            ctx.request_id,
        )

        if request.strategy == "distributed":
            background_tasks.add_task(
                training_handler.run_distributed_training,
                response.job_id,
                request.prompts,
                request.num_iterations,
                request.distributed_config or {},
            )
        else:
            background_tasks.add_task(
                training_handler.run_computational_training,
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
        ctx=Depends(verify_request),
    ):
        """Get training job status."""
        response = get_training_handler().get_job_status(
            job_id,
            ctx.user_id,
            ctx.roles,
        )
        if not response:
            raise HTTPException(status_code=404, detail="Training job not found")
        return response

    @app.delete("/api/jobs/{job_id}", tags=["training"])
    async def cancel_training_job(
        job_id: str,
        ctx=Depends(verify_request),
    ):
        """Cancel a training job."""
        if not get_training_handler().cancel_job(
            job_id,
            ctx.user_id,
            ctx.roles,
        ):
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
        ctx=Depends(verify_request),
    ):
        """Send a conversation message."""
        try:
            return await get_conversation_handler().handle_message(
                request,
                ctx.user_id,
                ctx.roles,
            )
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e)) from e
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/api/conversations/{conversation_id}", tags=["conversations"])
    async def end_conversation(
        conversation_id: str,
        ctx=Depends(verify_request),
    ):
        """End a conversation."""
        try:
            result = get_conversation_handler().end_conversation(
                conversation_id,
                ctx.user_id,
                ctx.roles,
            )
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
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
        ctx=Depends(verify_request),
    ):
        """Scale computational resources."""
        del ctx
        results = {}
        state = get_state_manager()
        services = get_services()

        if request.apply_to_all:
            for engine_id in list(state.engines.keys()):
                engine = state.get_engine(engine_id)
                if engine is not None and hasattr(engine, "scale_computation"):
                    try:
                        result = engine.scale_computation(request.scale_factor)
                        results[engine_id] = result
                    except GRPO_ENGINE_EXCEPTIONS as e:
                        results[engine_id] = {"error": str(e)}

        if "demo_engine" in services:
            try:
                result = services["demo_engine"].scale_computation(request.scale_factor)
                results["demo_engine"] = result
            except GRPO_ENGINE_EXCEPTIONS as e:
                results["demo_engine"] = {"error": str(e)}

        return GRPOScaleResponse(
            message="Computational resources scaled",
            scale_factor=request.scale_factor,
            results=results,
        )

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
        ctx=Depends(verify_request),
    ):
        """Submit multiple training jobs in batch."""
        training_handler = get_training_handler()
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        results: list[BatchItemResult] = []
        accepted = 0
        rejected = 0

        for i, item in enumerate(request.items):
            try:
                training_req = GRPOTrainingRequest(
                    prompts=item.prompts,
                    strategy=item.strategy,
                    num_iterations=item.num_iterations,
                    idempotency_key=item.idempotency_key,
                )

                response = await training_handler.start_training(
                    training_req,
                    ctx.user_id,
                    ctx.request_id,
                )

                background_tasks.add_task(
                    training_handler.run_computational_training,
                    response.job_id,
                    training_req.prompts,
                    training_req.num_iterations,
                    True,
                    False,
                )

                results.append(
                    BatchItemResult(
                        index=i,
                        job_id=response.job_id,
                        status="accepted",
                    )
                )
                accepted += 1
            except GRPO_BATCH_EXCEPTIONS as e:
                logger.warning("Batch item %d failed: %s", i, e)
                results.append(
                    BatchItemResult(
                        index=i,
                        job_id=None,
                        status="rejected",
                        error=str(e),
                    )
                )
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
        ctx=Depends(verify_request),
    ):
        """Get status of multiple training jobs."""
        training_handler = get_training_handler()
        jobs: dict[str, Any] = {}
        not_found: list[str] = []

        for job_id in request.job_ids:
            response = training_handler.get_job_status(
                job_id,
                ctx.user_id,
                ctx.roles,
            )
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
        ctx=Depends(verify_request),
    ):
        """Cancel multiple training jobs."""
        training_handler = get_training_handler()
        cancelled: list[str] = []
        not_found: list[str] = []
        already_completed: list[str] = []

        for job_id in request.job_ids:
            response = training_handler.get_job_status(
                job_id,
                ctx.user_id,
                ctx.roles,
            )

            if not response:
                not_found.append(job_id)
            elif response.status in ("completed", "failed", "cancelled"):
                already_completed.append(job_id)
            else:
                training_handler.cancel_job(
                    job_id,
                    ctx.user_id,
                    ctx.roles,
                )
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

        if config.api_keys:
            api_key = websocket.headers.get("x-api-key") or websocket.headers.get(
                "authorization"
            )
            if api_key and api_key.lower().startswith("bearer "):
                api_key = api_key.split(" ", 1)[1].strip()

            if not api_key or api_key not in config.api_keys:
                await websocket.close(code=1008, reason="Unauthorized")
                return

            limit_key = f"ws:{api_key}"
            if not rate_limiter.allow(limit_key, config.rate_limit_per_minute):
                await websocket.close(code=1008, reason="Rate limit exceeded")
                return

        await websocket.accept()

        try:
            await get_websocket_handler().handle_connection(websocket)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except GRPO_WS_EXCEPTIONS as e:
            logger.error("WebSocket error: %s", e)
            await websocket.close()
