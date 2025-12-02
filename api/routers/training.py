from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from ..schemas import TrainingRequest, TrainingResponse, ErrorResponse
from ..services.training_service import TrainingService
from ..dependencies import require_role, get_current_user

router = APIRouter(prefix="/training", tags=["training"])
training_service = TrainingService()

@router.post(
    "",
    response_model=TrainingResponse,
    summary="Start Training",
    description="Start training an agent with the specified configuration",
    responses={
        202: {"description": "Training started successfully"},
        400: {"description": "Invalid configuration", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Insufficient permissions", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def start_training(
    request: TrainingRequest, 
    user_data: Dict = Depends(require_role("trainer"))
):
    """Start a training job."""
    try:
        training_id = await training_service.start_training(request)

        return TrainingResponse(
            training_id=training_id,
            status="running",
            message="Training started successfully",
            estimated_time=request.num_episodes * 30,  # Rough estimate
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to start training")


@router.get(
    "/{training_id}",
    summary="Get Training Status",
    description="Get the status of a training job",
    responses={
        200: {"description": "Training status retrieved"},
        401: {"description": "Authentication required", "model": ErrorResponse},
        404: {"description": "Training job not found", "model": ErrorResponse},
    },
)
async def get_training_status(
    training_id: str, 
    user_data: Dict = Depends(get_current_user)
):
    """Get training job status."""
    try:
        return training_service.get_training_status(training_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get training status")
