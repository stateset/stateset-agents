"""
Training Router Module

API endpoints for training job management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..schemas import TrainingRequest, TrainingResponse, ErrorResponse
from ..models import PaginatedResponse, TrainingStatus
from ..services.training_service import TrainingService
from ..dependencies import require_role, get_current_user, AuthenticatedUser
from ..errors import (
    TrainingJobNotFoundError,
    TrainingConfigError,
    InternalError,
)

router = APIRouter(prefix="/training", tags=["training"])
training_service = TrainingService()


# ============================================================================
# Response Models
# ============================================================================

class TrainingJobDetail(BaseModel):
    """Detailed training job information."""
    training_id: str = Field(..., description="Training job identifier")
    status: str = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    current_episode: int = Field(0, description="Current training episode")
    total_episodes: int = Field(0, description="Total episodes")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Training metrics")
    error: Optional[str] = Field(None, description="Error message if failed")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")


class TrainingJobListResponse(PaginatedResponse):
    """Paginated list of training jobs."""
    items: List[TrainingJobDetail] = Field(default_factory=list, description="List of training jobs")


class TrainingCancelResponse(BaseModel):
    """Response for training cancellation."""
    training_id: str = Field(..., description="Training job identifier")
    status: str = Field(..., description="New status (cancelled)")
    message: str = Field(..., description="Status message")


# ============================================================================
# Training Endpoints
# ============================================================================

@router.post(
    "",
    response_model=TrainingResponse,
    status_code=202,
    summary="Start Training",
    description="Start a new training job with the specified configuration.",
    responses={
        202: {"description": "Training started successfully", "model": TrainingResponse},
        400: {"description": "Invalid configuration", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Insufficient permissions", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def start_training(
    request: TrainingRequest,
    user: AuthenticatedUser = Depends(require_role("trainer")),
) -> TrainingResponse:
    """
    Start a new training job.

    Requires the 'trainer' role. The training runs asynchronously in the background.

    Args:
        request: Training configuration including prompts, strategy, and parameters.
        user: Authenticated user with trainer role.

    Returns:
        TrainingResponse with job ID and initial status.

    Raises:
        TrainingConfigError: If the configuration is invalid.
        InternalError: If training fails to start.
    """
    try:
        # Validate training configuration
        if request.num_episodes < 1:
            raise TrainingConfigError("num_episodes must be at least 1")

        if not request.environment_scenarios:
            raise HTTPException(
                status_code=422,
                detail="environment_scenarios must contain at least one scenario",
            )

        training_id = await training_service.start_training(request)

        return TrainingResponse(
            training_id=training_id,
            status="running",
            message="Training started successfully",
            estimated_time=request.num_episodes * 30,  # Rough estimate: 30s per episode
        )

    except TrainingConfigError:
        raise
    except HTTPException:
        raise
    except ValueError as e:
        raise TrainingConfigError(str(e))
    except Exception as e:
        raise InternalError("Failed to start training", internal_error=e)


@router.get(
    "",
    response_model=TrainingJobListResponse,
    summary="List Training Jobs",
    description="Get a paginated list of training jobs.",
    responses={
        200: {"description": "List of training jobs", "model": TrainingJobListResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
    },
)
async def list_training_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    user: AuthenticatedUser = Depends(get_current_user),
) -> TrainingJobListResponse:
    """
    Get a paginated list of training jobs.

    Args:
        page: Page number (1-indexed).
        page_size: Number of items per page (max 100).
        status: Optional filter by job status.
        user: Authenticated user.

    Returns:
        Paginated list of training job details.
    """
    # Get all jobs from service
    all_jobs = list(training_service.jobs.items())

    # Apply status filter if provided
    if status:
        all_jobs = [(jid, j) for jid, j in all_jobs if j.get("status") == status]

    total = len(all_jobs)

    # Sort by creation time (newest first)
    all_jobs.sort(key=lambda x: x[1].get("created_at", datetime.min), reverse=True)

    # Paginate
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_jobs = all_jobs[start_idx:end_idx]

    items = [
        TrainingJobDetail(
            training_id=job_id,
            status=job.get("status", "unknown"),
            created_at=job.get("created_at", datetime.utcnow()),
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            progress=job.get("progress", 0.0),
            current_episode=job.get("current_episode", 0),
            total_episodes=job.get("total_episodes", 0),
            metrics=job.get("metrics", {}),
            error=job.get("error"),
            config=job.get("config", {}),
        )
        for job_id, job in page_jobs
    ]

    return TrainingJobListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=end_idx < total,
        has_prev=page > 1,
    )


@router.get(
    "/{training_id}",
    response_model=TrainingJobDetail,
    summary="Get Training Status",
    description="Get detailed status of a specific training job.",
    responses={
        200: {"description": "Training job details", "model": TrainingJobDetail},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Training job not found", "model": ErrorResponse},
    },
)
async def get_training_status(
    training_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> TrainingJobDetail:
    """
    Get detailed status of a training job.

    Args:
        training_id: The training job's unique identifier.
        user: Authenticated user.

    Returns:
        Detailed training job information including metrics.

    Raises:
        TrainingJobNotFoundError: If the training job doesn't exist.
    """
    job = training_service.get_training_status(training_id)

    if not job:
        raise TrainingJobNotFoundError(training_id)

    return TrainingJobDetail(
        training_id=training_id,
        status=job.get("status", "unknown"),
        created_at=job.get("created_at", datetime.utcnow()),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        progress=job.get("progress", 0.0),
        current_episode=job.get("current_episode", 0),
        total_episodes=job.get("total_episodes", 0),
        metrics=job.get("metrics", {}),
        error=job.get("error"),
        config=job.get("config", {}),
    )


@router.delete(
    "/{training_id}",
    response_model=TrainingCancelResponse,
    summary="Cancel Training",
    description="Cancel a running training job.",
    responses={
        200: {"description": "Training cancelled", "model": TrainingCancelResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Insufficient permissions", "model": ErrorResponse},
        404: {"description": "Training job not found", "model": ErrorResponse},
    },
)
async def cancel_training(
    training_id: str,
    user: AuthenticatedUser = Depends(require_role("trainer")),
) -> TrainingCancelResponse:
    """
    Cancel a running training job.

    Requires the 'trainer' role. Only running jobs can be cancelled.

    Args:
        training_id: The training job's unique identifier.
        user: Authenticated user with trainer role.

    Returns:
        Confirmation of cancellation.

    Raises:
        TrainingJobNotFoundError: If the training job doesn't exist.
    """
    job = training_service.get_training_status(training_id)

    if not job:
        raise TrainingJobNotFoundError(training_id)

    # Mark as cancelled
    if training_id in training_service.jobs:
        training_service.jobs[training_id]["status"] = "cancelled"
        training_service.jobs[training_id]["completed_at"] = datetime.utcnow()

    return TrainingCancelResponse(
        training_id=training_id,
        status="cancelled",
        message="Training job cancelled successfully",
    )
