from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator

class AgentConfigRequest(BaseModel):
    """Configuration for creating an agent."""

    model_name: str = Field(..., description="Name of the model to use", example="gpt2")
    max_new_tokens: int = Field(
        512, description="Maximum tokens to generate", ge=1, le=4096
    )
    temperature: float = Field(0.8, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Top-p sampling parameter", ge=0.0, le=1.0)
    top_k: int = Field(50, description="Top-k sampling parameter", ge=1, le=1000)
    system_prompt: Optional[str] = Field(
        None, description="System prompt for the agent"
    )
    use_chat_template: bool = Field(True, description="Whether to use chat template")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "gpt2",
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "system_prompt": "You are a helpful AI assistant.",
            }
        }


class ConversationRequest(BaseModel):
    """Request for agent conversation."""

    messages: List[Dict[str, str]] = Field(
        ..., description="List of conversation messages"
    )
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    max_tokens: int = Field(
        512, description="Maximum tokens in response", ge=1, le=4096
    )
    temperature: float = Field(0.8, description="Response temperature", ge=0.0, le=2.0)
    stream: bool = Field(False, description="Whether to stream the response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    @validator("messages")
    def validate_messages(cls, v):
        """Validate conversation messages."""
        if not v:
            raise ValueError("Messages cannot be empty")

        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    "Message role must be 'system', 'user', or 'assistant'"
                )
            if not isinstance(msg["content"], str) or not msg["content"].strip():
                raise ValueError("Message content must be a non-empty string")

        return v

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello! How can you help me?"},
                ],
                "max_tokens": 256,
                "temperature": 0.7,
                "stream": False,
            }
        }


class TrainingRequest(BaseModel):
    """Request for training an agent."""

    agent_config: AgentConfigRequest
    environment_scenarios: List[Dict[str, Any]] = Field(
        ..., description="Training scenarios"
    )
    reward_config: Dict[str, Any] = Field(
        ..., description="Reward function configuration"
    )
    num_episodes: int = Field(
        100, description="Number of training episodes", ge=1, le=10000
    )
    profile: str = Field("balanced", description="Training profile")

    class Config:
        schema_extra = {
            "example": {
                "agent_config": {
                    "model_name": "gpt2",
                    "max_new_tokens": 256,
                    "temperature": 0.7,
                },
                "environment_scenarios": [
                    {
                        "id": "customer_support",
                        "topic": "support",
                        "user_responses": ["I need help", "Thank you"],
                    }
                ],
                "reward_config": {"helpfulness_weight": 0.7, "safety_weight": 0.3},
                "num_episodes": 50,
                "profile": "balanced",
            }
        }


class ConversationResponse(BaseModel):
    """Response from conversation endpoint."""

    response: str = Field(..., description="Agent's response")
    conversation_id: str = Field(..., description="Conversation identifier")
    tokens_used: int = Field(..., description="Number of tokens used")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class TrainingResponse(BaseModel):
    """Response from training endpoint."""

    training_id: str = Field(..., description="Training job identifier")
    status: str = Field(..., description="Training status")
    estimated_time: Optional[float] = Field(
        None, description="Estimated completion time"
    )
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component health status")


class MetricsResponse(BaseModel):
    """Metrics response."""

    timestamp: float = Field(..., description="Metrics timestamp")
    system_metrics: Dict[str, Union[int, float]] = Field(
        ..., description="System metrics"
    )
    performance_metrics: Dict[str, Union[int, float]] = Field(
        ..., description="Performance metrics"
    )
    security_events: int = Field(..., description="Number of security events")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: float = Field(..., description="Error timestamp")
