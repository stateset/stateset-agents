"""
API Request/Response Models

Unified Pydantic models for all API services with comprehensive validation.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

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
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


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
    name: Optional[str] = Field(None, description="Optional name for the message author")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

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
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional name."""
        if v is not None:
            v = v.strip()
            if len(v) > 64:
                raise ValueError("Name cannot exceed 64 characters")
            if not re.match(r"^[a-zA-Z0-9_-]+$", v):
                raise ValueError("Name can only contain alphanumeric characters, underscores, and hyphens")
        return v


# ============================================================================
# Agent Configuration Models
# ============================================================================

class AgentConfig(BaseModel):
    """Agent configuration."""
    model_name: str = Field(..., description="Model name or path")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(50, ge=1, le=1000, description="Top-k sampling")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    use_chat_template: bool = Field(True, description="Whether to use chat template")

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
    def validate_system_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate system prompt."""
        if v is not None:
            v = v.strip()
            if len(v) > 10000:
                raise ValueError("System prompt cannot exceed 10,000 characters")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "gpt2",
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "system_prompt": "You are a helpful AI assistant."
            }
        }


# ============================================================================
# Training Request/Response Models
# ============================================================================

class TrainingRequest(BaseModel):
    """Training job request."""
    prompts: List[str] = Field(..., min_length=1, description="Training prompts")
    strategy: TrainingStrategy = Field(
        TrainingStrategy.COMPUTATIONAL,
        description="Training strategy to use"
    )
    num_iterations: int = Field(1, ge=1, description="Number of training iterations")
    parallel_batch_size: Optional[int] = Field(None, ge=1, description="Batch size for parallel training")
    use_neural_rewards: bool = Field(True, description="Enable neural reward models")
    use_ruler_rewards: bool = Field(False, description="Enable RULER LLM judges")
    distributed_config: Optional[Dict[str, Any]] = Field(None, description="Distributed training configuration")
    agent_config: Optional[AgentConfig] = Field(None, description="Agent configuration override")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for request deduplication")

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, v: List[str]) -> List[str]:
        """Validate and clean prompts."""
        config = get_config()

        # Clean prompts
        cleaned = [p.strip() for p in v if p and p.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty prompt is required")

        if len(cleaned) > config.validation.max_prompts:
            raise ValueError(f"Maximum {config.validation.max_prompts} prompts allowed per request")

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
            raise ValueError(f"num_iterations cannot exceed {config.validation.max_iterations}")
        return v

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate idempotency key format."""
        if v is not None:
            v = v.strip()
            if len(v) > 64:
                raise ValueError("Idempotency key cannot exceed 64 characters")
            if not re.match(r"^[a-zA-Z0-9_-]+$", v):
                raise ValueError("Idempotency key can only contain alphanumeric characters, underscores, and hyphens")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "prompts": ["What is machine learning?", "Explain neural networks"],
                "strategy": "computational",
                "num_iterations": 10,
                "use_neural_rewards": True
            }
        }


class TrainingMetrics(BaseModel):
    """Training metrics."""
    iterations_completed: int = Field(0, description="Number of completed iterations")
    total_trajectories: int = Field(0, description="Total trajectories generated")
    average_reward: float = Field(0.0, description="Average reward across iterations")
    computation_used: float = Field(0.0, description="Total computation used")
    loss: Optional[float] = Field(None, description="Current loss value")
    learning_rate: Optional[float] = Field(None, description="Current learning rate")


class TrainingResponse(APIResponse):
    """Training job response."""
    job_id: str = Field(..., description="Unique job identifier")
    status: TrainingStatus = Field(..., description="Current job status")
    strategy: TrainingStrategy = Field(..., description="Training strategy used")
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics, description="Training metrics")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "running",
                "strategy": "computational",
                "metrics": {
                    "iterations_completed": 5,
                    "total_trajectories": 500,
                    "average_reward": 0.75
                },
                "started_at": "2024-01-15T10:30:00.000Z"
            }
        }


# ============================================================================
# Conversation Request/Response Models
# ============================================================================

class ConversationRequest(BaseModel):
    """Conversation request."""
    message: Optional[str] = Field(None, description="User message (single-turn)")
    messages: Optional[List[Message]] = Field(None, description="Full message history (multi-turn)")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    strategy: str = Field("default", description="Response generation strategy")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum response tokens")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(False, description="Enable streaming response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

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
    def validate_message(cls, v: Optional[str]) -> Optional[str]:
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
    def validate_messages(cls, v: Optional[List[Message]]) -> Optional[List[Message]]:
        """Validate message list."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("Messages list cannot be empty")
            if len(v) > 100:
                raise ValueError("Maximum 100 messages allowed in conversation history")
        return v

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate conversation ID format."""
        if v is not None:
            v = v.strip()
            if not re.match(r"^[a-zA-Z0-9_-]+$", v):
                raise ValueError("Invalid conversation ID format")
            if len(v) > 64:
                raise ValueError("Conversation ID cannot exceed 64 characters")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how can you help me?",
                "max_tokens": 256,
                "temperature": 0.7
            }
        }


class ConversationResponse(APIResponse):
    """Conversation response."""
    conversation_id: str = Field(..., description="Conversation identifier")
    response: str = Field(..., description="Agent's response")
    tokens_used: int = Field(0, description="Number of tokens used")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")
    context: Dict[str, Any] = Field(default_factory=dict, description="Conversation context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_550e8400",
                "response": "Hello! I'm here to help you with any questions you have.",
                "tokens_used": 15,
                "processing_time_ms": 234.5
            }
        }


# ============================================================================
# Scaling Request/Response Models
# ============================================================================

class ScaleRequest(BaseModel):
    """Resource scaling request."""
    scale_factor: float = Field(..., gt=0, le=10, description="Scaling factor")
    apply_to_all: bool = Field(False, description="Apply to all engines")
    target_engines: Optional[List[str]] = Field(None, description="Specific engines to scale")

    @field_validator("target_engines")
    @classmethod
    def validate_target_engines(cls, v: Optional[List[str]]) -> Optional[List[str]]:
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
    results: Dict[str, Any] = Field(default_factory=dict, description="Scaling results by engine")
    message: str = Field(..., description="Status message")


# ============================================================================
# Health/Metrics Models
# ============================================================================

class ComponentHealth(BaseModel):
    """Individual component health."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status")
    latency_ms: Optional[float] = Field(None, description="Component response latency")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class HealthResponse(APIResponse):
    """Health check response."""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")
    components: List[ComponentHealth] = Field(default_factory=list, description="Component health")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "uptime_seconds": 3600.5,
                "components": [
                    {"name": "database", "status": "healthy", "latency_ms": 5.2},
                    {"name": "cache", "status": "healthy", "latency_ms": 1.1}
                ]
            }
        }


class MetricsResponse(APIResponse):
    """System metrics response."""
    system: Dict[str, Any] = Field(default_factory=dict, description="System metrics")
    api: Dict[str, Any] = Field(default_factory=dict, description="API metrics")
    training: Dict[str, Any] = Field(default_factory=dict, description="Training metrics")
    conversations: Dict[str, Any] = Field(default_factory=dict, description="Conversation metrics")
