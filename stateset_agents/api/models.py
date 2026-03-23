"""
API Request/Response Models

Unified Pydantic models for all API services with comprehensive validation.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from .config import get_config

# ============================================================================
# Enums
# ============================================================================


class TrainingStrategy(str, Enum):
    """Available training strategies."""

    COMPUTATIONAL = "computational"
    DISTRIBUTED = "distributed"
    GRPO = "grpo"
    GSPO = "gspo"


class TrainingStatus(str, Enum):
    """Training job status."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Valid message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# ============================================================================
# Base Models
# ============================================================================


class APIResponse(BaseModel):
    """Base response model with common fields."""

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(None, description="Request tracking ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    @field_serializer("timestamp")
    def _serialize_timestamp(self, timestamp: datetime) -> str:
        # Consistent 'Z' suffix for naive UTC datetimes.
        return timestamp.isoformat() + "Z"


class PaginatedResponse(APIResponse):
    """Base model for paginated responses."""

    total: int = Field(..., description="Total number of items")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Items per page")
    has_next: bool = Field(False, description="Whether there are more pages")
    has_prev: bool = Field(False, description="Whether there are previous pages")


# ============================================================================
# Message Models
# ============================================================================


class Message(BaseModel):
    """Conversation message."""

    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: str | None = Field(
        None, description="Optional name for the message author"
    )
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate message content."""
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty")
        config = get_config()
        if len(v) > config.validation.max_message_length:
            raise ValueError(
                f"Message content exceeds maximum length of {config.validation.max_message_length} characters"
            )
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Validate optional name."""
        if v is not None:
            v = v.strip()
            if len(v) > 64:
                raise ValueError("Name cannot exceed 64 characters")
            if not re.match(r"^[a-zA-Z0-9_-]+$", v):
                raise ValueError(
                    "Name can only contain alphanumeric characters, underscores, and hyphens"
                )
        return v


# ============================================================================
# Agent Configuration Models
# ============================================================================


class AgentConfig(BaseModel):
    """Agent configuration."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "model_name": "gpt2",
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "system_prompt": "You are a helpful AI assistant.",
                "enable_planning": True,
                "planning_config": {"max_steps": 4},
            }
        },
    )

    model_name: str = Field(..., description="Model name or path")
    max_new_tokens: int = Field(
        512, ge=1, le=4096, description="Maximum tokens to generate"
    )
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(50, ge=1, le=1000, description="Top-k sampling")
    system_prompt: str | None = Field(
        None, description="System prompt for the agent"
    )
    use_chat_template: bool = Field(True, description="Whether to use chat template")
    enable_planning: bool = Field(False, description="Enable long-term planning")
    planning_config: dict[str, Any] | None = Field(
        None, description="Planning configuration overrides"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name."""
        v = v.strip()
        if not v:
            raise ValueError("Model name cannot be empty")
        if len(v) > 256:
            raise ValueError("Model name cannot exceed 256 characters")
        return v

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: str | None) -> str | None:
        """Validate system prompt."""
        if v is not None:
            v = v.strip()
            if len(v) > 10000:
                raise ValueError("System prompt cannot exceed 10,000 characters")
        return v

    @field_validator("planning_config")
    @classmethod
    def validate_planning_config(
        cls, v: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Validate planning configuration overrides."""
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError("planning_config must be a dictionary")
        return v


# ============================================================================
# Training Request/Response Models
# ============================================================================


class TrainingRequest(BaseModel):
    """Training job request."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "prompts": ["What is machine learning?", "Explain neural networks"],
                "strategy": "computational",
                "num_iterations": 10,
                "use_neural_rewards": True,
            }
        },
    )

    prompts: list[str] = Field(..., min_length=1, description="Training prompts")
    strategy: TrainingStrategy = Field(
        TrainingStrategy.COMPUTATIONAL, description="Training strategy to use"
    )
    num_iterations: int = Field(1, ge=1, description="Number of training iterations")
    parallel_batch_size: int | None = Field(
        None, ge=1, description="Batch size for parallel training"
    )
    use_neural_rewards: bool = Field(True, description="Enable neural reward models")
    use_ruler_rewards: bool = Field(False, description="Enable RULER LLM judges")
    distributed_config: dict[str, Any] | None = Field(
        None, description="Distributed training configuration"
    )
    agent_config: AgentConfig | None = Field(
        None, description="Agent configuration override"
    )
    idempotency_key: str | None = Field(
        None, description="Idempotency key for request deduplication"
    )

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, v: list[str]) -> list[str]:
        """Validate and clean prompts."""
        config = get_config()

        # Clean prompts
        cleaned = [p.strip() for p in v if p and p.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty prompt is required")

        if len(cleaned) > config.validation.max_prompts:
            raise ValueError(
                f"Maximum {config.validation.max_prompts} prompts allowed per request"
            )

        for i, prompt in enumerate(cleaned):
            if len(prompt) > config.validation.max_prompt_length:
                raise ValueError(
                    f"Prompt {i+1} exceeds maximum length of {config.validation.max_prompt_length} characters"
                )

        return cleaned

    @field_validator("num_iterations")
    @classmethod
    def validate_iterations(cls, v: int) -> int:
        """Validate iteration count."""
        config = get_config()
        if v > config.validation.max_iterations:
            raise ValueError(
                f"num_iterations cannot exceed {config.validation.max_iterations}"
            )
        return v

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, v: str | None) -> str | None:
        """Validate idempotency key format."""
        if v is not None:
            v = v.strip()
            if len(v) > 64:
                raise ValueError("Idempotency key cannot exceed 64 characters")
            if not re.match(r"^[a-zA-Z0-9_-]+$", v):
                raise ValueError(
                    "Idempotency key can only contain alphanumeric characters, underscores, and hyphens"
                )
        return v


class TrainingMetrics(BaseModel):
    """Training metrics."""

    iterations_completed: int = Field(0, description="Number of completed iterations")
    total_trajectories: int = Field(0, description="Total trajectories generated")
    average_reward: float = Field(0.0, description="Average reward across iterations")
    computation_used: float = Field(0.0, description="Total computation used")
    loss: float | None = Field(None, description="Current loss value")
    learning_rate: float | None = Field(None, description="Current learning rate")


class TrainingResponse(APIResponse):
    """Training job response."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "running",
                "strategy": "computational",
                "metrics": {
                    "iterations_completed": 5,
                    "total_trajectories": 500,
                    "average_reward": 0.75,
                },
                "started_at": "2024-01-15T10:30:00.000Z",
            }
        },
    )

    job_id: str = Field(..., description="Unique job identifier")
    status: TrainingStatus = Field(..., description="Current job status")
    strategy: TrainingStrategy = Field(..., description="Training strategy used")
    metrics: TrainingMetrics = Field(
        default_factory=TrainingMetrics, description="Training metrics"
    )
    error: str | None = Field(None, description="Error message if failed")
    started_at: datetime | None = Field(None, description="Job start time")
    completed_at: datetime | None = Field(None, description="Job completion time")
    estimated_completion: datetime | None = Field(
        None, description="Estimated completion time"
    )


# ============================================================================
# Conversation Request/Response Models
# ============================================================================


class ConversationRequest(BaseModel):
    """Conversation request."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "message": "Hello, how can you help me?",
                "conversation_id": "conv_550e8400",
                "max_tokens": 256,
                "temperature": 0.7,
                "context": {
                    "goal": "Plan a weekend in Austin",
                    "plan_update": {"action": "advance"},
                    "plan_goal": "Plan a weekend in Dallas",
                },
            }
        },
    )

    message: str | None = Field(None, description="User message (single-turn)")
    messages: list[Message] | None = Field(
        None, description="Full message history (multi-turn)"
    )
    conversation_id: str | None = Field(None, description="Existing conversation ID")
    strategy: str = Field("default", description="Response generation strategy")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum response tokens")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(False, description="Enable streaming response")
    context: dict[str, Any] | None = Field(None, description="Additional context")

    @model_validator(mode="after")
    def validate_message_input(self):
        """Validate that either message or messages is provided."""
        if not self.message and not self.messages:
            raise ValueError("Either 'message' or 'messages' must be provided")
        if self.message and self.messages:
            raise ValueError("Provide either 'message' or 'messages', not both")
        return self

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str | None) -> str | None:
        """Validate single message."""
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Message cannot be empty")
            config = get_config()
            if len(v) > config.validation.max_message_length:
                raise ValueError(
                    f"Message exceeds maximum length of {config.validation.max_message_length} characters"
                )
        return v

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message] | None) -> list[Message] | None:
        """Validate message list."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("Messages list cannot be empty")
            if len(v) > 100:
                raise ValueError("Maximum 100 messages allowed in conversation history")
        return v

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: str | None) -> str | None:
        """Validate conversation ID format."""
        if v is not None:
            v = v.strip()
            if not re.match(r"^[a-zA-Z0-9_-]+$", v):
                raise ValueError("Invalid conversation ID format")
            if len(v) > 64:
                raise ValueError("Conversation ID cannot exceed 64 characters")
        return v


class ConversationResponse(APIResponse):
    """Conversation response."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_id": "conv_550e8400",
                "response": "Hello! I'm here to help you with any questions you have.",
                "tokens_used": 15,
                "processing_time_ms": 234.5,
            }
        },
    )

    conversation_id: str = Field(..., description="Conversation identifier")
    response: str = Field(..., description="Agent's response")
    tokens_used: int = Field(0, description="Number of tokens used")
    processing_time_ms: float = Field(
        0.0, description="Processing time in milliseconds"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Conversation context"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# ============================================================================
# Scaling Request/Response Models
# ============================================================================


class ScaleRequest(BaseModel):
    """Resource scaling request."""

    scale_factor: float = Field(..., gt=0, le=10, description="Scaling factor")
    apply_to_all: bool = Field(False, description="Apply to all engines")
    target_engines: list[str] | None = Field(
        None, description="Specific engines to scale"
    )

    @field_validator("target_engines")
    @classmethod
    def validate_target_engines(cls, v: list[str] | None) -> list[str] | None:
        """Validate target engine IDs."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("target_engines cannot be empty if provided")
            if len(v) > 100:
                raise ValueError("Maximum 100 target engines allowed")
        return v


class ScaleResponse(APIResponse):
    """Resource scaling response."""

    scale_factor: float = Field(..., description="Applied scale factor")
    results: dict[str, Any] = Field(
        default_factory=dict, description="Scaling results by engine"
    )
    message: str = Field(..., description="Status message")


# ============================================================================
# Health/Metrics Models
# ============================================================================


class ComponentHealth(BaseModel):
    """Individual component health."""

    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status")
    latency_ms: float | None = Field(None, description="Component response latency")
    details: dict[str, Any] | None = Field(None, description="Additional details")


class HealthResponse(APIResponse):
    """Health check response."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "uptime_seconds": 3600.5,
                "components": [
                    {"name": "database", "status": "healthy", "latency_ms": 5.2},
                    {"name": "cache", "status": "healthy", "latency_ms": 1.1},
                ],
            }
        },
    )

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")
    components: list[ComponentHealth] = Field(
        default_factory=list, description="Component health"
    )


class MetricsResponse(APIResponse):
    """System metrics response."""

    system: dict[str, Any] = Field(default_factory=dict, description="System metrics")
    api: dict[str, Any] = Field(default_factory=dict, description="API metrics")
    training: dict[str, Any] = Field(
        default_factory=dict, description="Training metrics"
    )
    conversations: dict[str, Any] = Field(
        default_factory=dict, description="Conversation metrics"
    )
