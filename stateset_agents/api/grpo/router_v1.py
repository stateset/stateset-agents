"""
GRPO API v1 Router

Versioned API router for backward compatibility.
"""

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from .handlers import ConversationHandler, TrainingHandler
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
from ..logging_config import get_logger

logger = get_logger(__name__)

# API v1 Router
router_v1 = APIRouter(prefix="/v1", tags=["v1"])


def create_v1_router(
    training_handler: TrainingHandler,
    conversation_handler: ConversationHandler,
    services: Dict[str, Any],
    verify_request,
) -> APIRouter:
    """
    Create the v1 API router with all endpoints.

    Args:
        training_handler: Training request handler.
        conversation_handler: Conversation request handler.
        services: Service dictionary.
        verify_request: Request verification dependency.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(prefix="/v1")

    # Training endpoints
    @router.post(
        "/train",
        response_model=GRPOTrainingResponse,
        tags=["v1-training"],
        summary="Start training (v1)",
    )
    async def train_v1(
        request: GRPOTrainingRequest,
        background_tasks: BackgroundTasks,
        ctx=Depends(verify_request),
    ):
        """Start GRPO training (API v1)."""
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

    @router.get(
        "/jobs/{job_id}",
        response_model=GRPOTrainingResponse,
        tags=["v1-training"],
        summary="Get job status (v1)",
    )
    async def get_job_v1(
        job_id: str,
        ctx=Depends(verify_request),
    ):
        """Get training job status (API v1)."""
        response = training_handler.get_job_status(job_id)
        if not response:
            raise HTTPException(status_code=404, detail="Job not found")
        return response

    @router.delete(
        "/jobs/{job_id}",
        tags=["v1-training"],
        summary="Cancel job (v1)",
    )
    async def cancel_job_v1(
        job_id: str,
        ctx=Depends(verify_request),
    ):
        """Cancel training job (API v1)."""
        if not training_handler.cancel_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found or completed")
        return {"message": "Job cancelled", "job_id": job_id}

    # Conversation endpoints
    @router.post(
        "/chat",
        response_model=GRPOConversationResponse,
        tags=["v1-conversations"],
        summary="Send message (v1)",
    )
    async def chat_v1(
        request: GRPOConversationRequest,
        ctx=Depends(verify_request),
    ):
        """Send conversation message (API v1)."""
        try:
            return await conversation_handler.handle_message(request, ctx.user_id)
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "/conversations/{conversation_id}",
        tags=["v1-conversations"],
        summary="End conversation (v1)",
    )
    async def end_conversation_v1(
        conversation_id: str,
        ctx=Depends(verify_request),
    ):
        """End conversation (API v1)."""
        result = conversation_handler.end_conversation(conversation_id)
        if not result:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation ended", **result}

    # Batch endpoints
    @router.post(
        "/batch/train",
        response_model=BatchTrainingResponse,
        tags=["v1-batch"],
        summary="Batch training (v1)",
    )
    async def batch_train_v1(
        request: BatchTrainingRequest,
        background_tasks: BackgroundTasks,
        ctx=Depends(verify_request),
    ):
        """Submit batch training jobs (API v1)."""
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        results = []
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

                results.append(BatchItemResult(
                    index=i,
                    job_id=response.job_id,
                    status="accepted",
                ))
                accepted += 1

            except Exception as e:
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

    @router.post(
        "/batch/status",
        response_model=BatchJobStatusResponse,
        tags=["v1-batch"],
        summary="Batch status (v1)",
    )
    async def batch_status_v1(
        request: BatchJobStatusRequest,
        ctx=Depends(verify_request),
    ):
        """Get batch job status (API v1)."""
        jobs = {}
        not_found = []

        for job_id in request.job_ids:
            response = training_handler.get_job_status(job_id)
            if response:
                jobs[job_id] = response
            else:
                not_found.append(job_id)

        return BatchJobStatusResponse(
            jobs=jobs,
            not_found=not_found,
        )

    @router.post(
        "/batch/cancel",
        response_model=BatchCancelResponse,
        tags=["v1-batch"],
        summary="Batch cancel (v1)",
    )
    async def batch_cancel_v1(
        request: BatchCancelRequest,
        ctx=Depends(verify_request),
    ):
        """Cancel batch jobs (API v1)."""
        cancelled = []
        not_found = []
        already_completed = []

        for job_id in request.job_ids:
            job = training_handler.state.get_job(job_id)

            if not job:
                not_found.append(job_id)
            elif job.status in ("completed", "failed", "cancelled"):
                already_completed.append(job_id)
            else:
                training_handler.cancel_job(job_id)
                cancelled.append(job_id)

        return BatchCancelResponse(
            cancelled=cancelled,
            not_found=not_found,
            already_completed=already_completed,
        )

    return router
