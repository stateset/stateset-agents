"""
GRPO Request/Response Models

Pydantic models for the GRPO service API with input sanitization.
"""

import html
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .config import get_grpo_config


def sanitize_string(value: str) -> str:
    """
    Sanitize a string input.

    - Strips leading/trailing whitespace
    - Removes null bytes
    - Normalizes line endings
    - Escapes HTML entities for safety
    """
    if not value:
        return value

    # Remove null bytes
    cleaned = value.replace("\x00", "")

    # Normalize line endings
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Strip whitespace
    cleaned = cleaned.strip()

    return cleaned


def sanitize_html(value: str) -> str:
    """
    Sanitize a string that might contain HTML.

    - Performs basic sanitization
    - Escapes HTML entities
    """
    cleaned = sanitize_string(value)
    if cleaned:
        # Escape HTML entities
        cleaned = html.escape(cleaned)
    return cleaned


class GRPOTrainingRequest(BaseModel):
    """Request model for GRPO training."""

    prompts: List[str] = Field(
        ...,
        description="Training prompts (1-8 prompts)",
        min_length=1,
    )
    strategy: str = Field(
        "computational",
        description="Training strategy: 'computational' or 'distributed'",
    )
    num_iterations: int = Field(
        1,
        ge=1,
        description="Number of training iterations",
    )
    parallel_batch_size: Optional[int] = Field(
        None,
        ge=1,
        description="Batch size for parallel training",
    )
    use_neural_rewards: bool = Field(
        True,
        description="Enable neural reward models",
    )
    use_ruler_rewards: bool = Field(
        False,
        description="Enable RULER LLM judges",
    )
    distributed_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration for distributed training",
    )
    idempotency_key: Optional[str] = Field(
        None,
        description="Unique key for request deduplication",
    )

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, prompts: List[str]) -> List[str]:
        """Validate, sanitize, and clean prompts."""
        config = get_grpo_config()

        # Sanitize and clean
        cleaned = [sanitize_string(p) for p in prompts if p]
        cleaned = [p for p in cleaned if p]  # Remove empty after sanitization

        if not cleaned:
            raise ValueError("At least one non-empty prompt is required")

        if len(cleaned) > config.max_prompts:
            raise ValueError(
                f"Maximum {config.max_prompts} prompts allowed per request"
            )

        for i, prompt in enumerate(cleaned):
            if len(prompt) > config.max_prompt_length:
                raise ValueError(
                    f"Prompt {i+1} exceeds {config.max_prompt_length} characters"
                )

        return cleaned

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, key: Optional[str]) -> Optional[str]:
        """Validate and sanitize idempotency key."""
        if key is None:
            return None
        cleaned = sanitize_string(key)
        # Only allow alphanumeric and dashes
        if not re.match(r"^[a-zA-Z0-9\-_]+$", cleaned):
            raise ValueError(
                "idempotency_key must contain only alphanumeric characters, dashes, and underscores"
            )
        if len(cleaned) > 128:
            raise ValueError("idempotency_key must not exceed 128 characters")
        return cleaned

    @field_validator("num_iterations")
    @classmethod
    def validate_iterations(cls, iterations: int) -> int:
        """Validate iteration count."""
        config = get_grpo_config()
        if iterations > config.max_iterations:
            raise ValueError(
                f"num_iterations cannot exceed {config.max_iterations}"
            )
        return iterations

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, strategy: str) -> str:
        """Validate training strategy."""
        allowed = {"computational", "distributed", "grpo", "gspo"}
        if strategy not in allowed:
            raise ValueError(
                f"strategy must be one of: {', '.join(sorted(allowed))}"
            )
        return strategy

    class Config:
        json_schema_extra = {
            "example": {
                "prompts": ["What is machine learning?", "Explain neural networks"],
                "strategy": "computational",
                "num_iterations": 10,
                "use_neural_rewards": True,
            }
        }


class TrainingMetrics(BaseModel):
    """Training metrics."""

    iterations_completed: int = 0
    total_trajectories: int = 0
    average_reward: float = 0.0
    computation_used: float = 0.0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None


class GRPOTrainingResponse(BaseModel):
    """Response model for GRPO training."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    iterations_completed: int = Field(0, description="Completed iterations")
    total_trajectories: int = Field(0, description="Total trajectories generated")
    average_reward: float = Field(0.0, description="Average reward")
    computation_used: float = Field(0.0, description="Total computation used")
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed metrics",
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    request_id: Optional[str] = Field(None, description="Original request ID")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "running",
                "iterations_completed": 5,
                "total_trajectories": 500,
                "average_reward": 0.75,
                "computation_used": 150.5,
                "metrics": {"strategy": "computational"},
                "started_at": "2024-01-15T10:30:00.000Z",
            }
        }


class GRPOConversationRequest(BaseModel):
    """Request model for GRPO conversations."""

    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(
        None,
        description="Existing conversation ID to continue",
    )
    strategy: str = Field("default", description="Response generation strategy")
    user_id: Optional[str] = Field(None, description="User identifier")
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context",
    )
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum response tokens")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")

    @field_validator("message")
    @classmethod
    def validate_message(cls, message: str) -> str:
        """Validate and sanitize message content."""
        config = get_grpo_config()

        cleaned = sanitize_string(message)
        if not cleaned:
            raise ValueError("message cannot be empty")

        if len(cleaned) > config.max_message_length:
            raise ValueError(
                f"message exceeds {config.max_message_length} characters"
            )

        return cleaned

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, conv_id: Optional[str]) -> Optional[str]:
        """Validate and sanitize conversation ID."""
        if conv_id is None:
            return None
        cleaned = sanitize_string(conv_id)
        # Only allow UUID-like format
        if not re.match(r"^[a-zA-Z0-9\-_]+$", cleaned):
            raise ValueError("conversation_id contains invalid characters")
        if len(cleaned) > 128:
            raise ValueError("conversation_id must not exceed 128 characters")
        return cleaned

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, user_id: Optional[str]) -> Optional[str]:
        """Validate and sanitize user ID."""
        if user_id is None:
            return None
        cleaned = sanitize_string(user_id)
        # Only allow alphanumeric, dashes, underscores, and dots
        if not re.match(r"^[a-zA-Z0-9\-_.@]+$", cleaned):
            raise ValueError("user_id contains invalid characters")
        if len(cleaned) > 128:
            raise ValueError("user_id must not exceed 128 characters")
        return cleaned

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, strategy: str) -> str:
        """Validate conversation strategy."""
        cleaned = sanitize_string(strategy) if strategy else "default"
        allowed = {"default", "creative", "precise", "balanced"}
        if cleaned not in allowed:
            raise ValueError(f"strategy must be one of: {', '.join(sorted(allowed))}")
        return cleaned

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how can you help me?",
                "strategy": "default",
                "max_tokens": 256,
                "temperature": 0.7,
            }
        }


class GRPOConversationResponse(BaseModel):
    """Response model for GRPO conversations."""

    conversation_id: str = Field(..., description="Conversation identifier")
    response: str = Field(..., description="Agent's response")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation context",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    tokens_used: int = Field(0, description="Tokens used in response")
    processing_time_ms: float = Field(0.0, description="Processing time")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_550e8400",
                "response": "Hello! I'm here to help you with any questions.",
                "context": {"turn_count": 1},
                "metadata": {"strategy": "default"},
                "tokens_used": 15,
                "processing_time_ms": 234.5,
            }
        }


class GRPOScaleRequest(BaseModel):
    """Request model for scaling computation."""

    scale_factor: float = Field(
        ...,
        gt=0,
        le=10,
        description="Scaling factor (0-10)",
    )
    apply_to_all: bool = Field(
        False,
        description="Apply scaling to all engines",
    )
    target_engines: Optional[List[str]] = Field(
        None,
        description="Specific engine IDs to scale",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "scale_factor": 2.0,
                "apply_to_all": True,
            }
        }


class GRPOScaleResponse(BaseModel):
    """Response model for scaling requests."""

    message: str = Field(..., description="Status message")
    scale_factor: float = Field(..., description="Applied scale factor")
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results by engine",
    )


class GRPOHealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )
    version: str = Field("2.0.0", description="API version")
    services: Dict[str, bool] = Field(
        default_factory=dict,
        description="Service availability",
    )


class GRPOMetricsResponse(BaseModel):
    """Metrics response."""

    system: Dict[str, Any] = Field(default_factory=dict, description="System metrics")
    training_jobs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training job metrics",
    )
    engines: Dict[str, Any] = Field(default_factory=dict, description="Engine metrics")
    conversations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation metrics",
    )
    api: Dict[str, Any] = Field(default_factory=dict, description="API metrics")
    rate_limit: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rate limit info",
    )


# ============================================================================
# Batch Operation Models
# ============================================================================


class BatchTrainingItem(BaseModel):
    """Single item in a batch training request."""

    prompts: List[str] = Field(
        ...,
        description="Training prompts for this item",
        min_length=1,
    )
    strategy: str = Field("computational", description="Training strategy")
    num_iterations: int = Field(1, ge=1, le=50, description="Iterations")
    idempotency_key: Optional[str] = Field(
        None,
        description="Unique key for this item",
    )

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, prompts: List[str]) -> List[str]:
        """Validate and sanitize prompts."""
        cleaned = [sanitize_string(p) for p in prompts if p]
        cleaned = [p for p in cleaned if p]
        if not cleaned:
            raise ValueError("At least one non-empty prompt required")
        if len(cleaned) > 8:
            raise ValueError("Maximum 8 prompts per item")
        return cleaned


class BatchTrainingRequest(BaseModel):
    """Request model for batch training operations."""

    items: List[BatchTrainingItem] = Field(
        ...,
        description="Training items to process",
        min_length=1,
        max_length=100,
    )
    parallel: bool = Field(
        True,
        description="Process items in parallel",
    )
    max_concurrent: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum concurrent items",
    )
    fail_fast: bool = Field(
        False,
        description="Stop on first failure",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "prompts": ["What is AI?"],
                        "strategy": "computational",
                        "num_iterations": 5,
                    },
                    {
                        "prompts": ["Explain ML"],
                        "strategy": "computational",
                        "num_iterations": 5,
                    },
                ],
                "parallel": True,
                "max_concurrent": 10,
            }
        }


class BatchItemResult(BaseModel):
    """Result of a single batch item."""

    index: int = Field(..., description="Item index in request")
    job_id: Optional[str] = Field(None, description="Created job ID")
    status: str = Field(..., description="Item status")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchTrainingResponse(BaseModel):
    """Response model for batch training operations."""

    batch_id: str = Field(..., description="Unique batch identifier")
    total_items: int = Field(..., description="Total items in batch")
    accepted: int = Field(0, description="Items accepted for processing")
    rejected: int = Field(0, description="Items rejected")
    results: List[BatchItemResult] = Field(
        default_factory=list,
        description="Results per item",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "batch_550e8400",
                "total_items": 5,
                "accepted": 4,
                "rejected": 1,
                "results": [
                    {"index": 0, "job_id": "job_123", "status": "accepted"},
                    {"index": 1, "job_id": "job_124", "status": "accepted"},
                ],
            }
        }


class BatchJobStatusRequest(BaseModel):
    """Request for batch job status."""

    job_ids: List[str] = Field(
        ...,
        description="Job IDs to query",
        min_length=1,
        max_length=100,
    )

    @field_validator("job_ids")
    @classmethod
    def validate_job_ids(cls, job_ids: List[str]) -> List[str]:
        """Validate job IDs."""
        cleaned = [sanitize_string(j) for j in job_ids if j]
        cleaned = [j for j in cleaned if j]
        if not cleaned:
            raise ValueError("At least one job ID required")
        return cleaned


class BatchJobStatusResponse(BaseModel):
    """Response for batch job status."""

    jobs: Dict[str, GRPOTrainingResponse] = Field(
        default_factory=dict,
        description="Job statuses by ID",
    )
    not_found: List[str] = Field(
        default_factory=list,
        description="Job IDs not found",
    )


class BatchCancelRequest(BaseModel):
    """Request to cancel multiple jobs."""

    job_ids: List[str] = Field(
        ...,
        description="Job IDs to cancel",
        min_length=1,
        max_length=100,
    )

    @field_validator("job_ids")
    @classmethod
    def validate_job_ids(cls, job_ids: List[str]) -> List[str]:
        """Validate job IDs."""
        cleaned = [sanitize_string(j) for j in job_ids if j]
        return [j for j in cleaned if j]


class BatchCancelResponse(BaseModel):
    """Response for batch cancel operation."""

    cancelled: List[str] = Field(
        default_factory=list,
        description="Successfully cancelled job IDs",
    )
    not_found: List[str] = Field(
        default_factory=list,
        description="Job IDs not found",
    )
    already_completed: List[str] = Field(
        default_factory=list,
        description="Jobs already completed",
    )
