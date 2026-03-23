"""
Models for the /v1/messages endpoint (Anthropic-style with OpenAI compatibility).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageInput(BaseModel):
    """Input message for the Messages API."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="Message role"
    )
    # OpenAI compatibility: assistant messages that contain tool_calls can have
    # `content: null` (or omit content entirely). We keep this optional and
    # validate downstream when building the backend payload.
    content: str | list[dict[str, Any]] | None = Field(
        None, description="Message content (string or structured blocks)"
    )
    name: str | None = Field(None, description="Optional name for the author")
    tool_call_id: str | None = Field(
        None, description="Tool call identifier for tool responses"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="OpenAI-style tool call metadata"
    )


class MessagesRequest(BaseModel):
    """Request model for /v1/messages."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model identifier")
    messages: list[MessageInput] = Field(..., description="Conversation messages")
    system: str | list[dict[str, Any]] | None = Field(
        None, description="System prompt (string or content blocks)"
    )
    max_tokens: int | None = Field(
        None, ge=1, description="Maximum tokens to generate"
    )
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p")
    top_k: int | None = Field(None, ge=0, description="Top-k")
    stop_sequences: list[str] | None = Field(
        None, description="Stop sequences (Anthropic style)"
    )
    stream: bool = Field(False, description="Stream response tokens")
    tools: list[dict[str, Any]] | None = Field(
        None, description="Tool definitions (Anthropic or OpenAI format)"
    )
    tool_choice: str | dict[str, Any] | None = Field(
        None, description="Tool choice configuration"
    )
    metadata: dict[str, Any] | None = Field(None, description="Arbitrary metadata")
    response_format: Literal["anthropic", "openai"] | None = Field(
        None, description="Response format preference"
    )


class UsageInfo(BaseModel):
    """Token usage statistics."""

    input_tokens: int = Field(..., description="Prompt tokens used")
    output_tokens: int = Field(..., description="Completion tokens used")


class MessagesResponse(BaseModel):
    """Anthropic-style response model for /v1/messages."""

    id: str = Field(..., description="Message identifier")
    type: Literal["message"] = Field("message", description="Response type")
    role: Literal["assistant"] = Field("assistant", description="Assistant role")
    model: str = Field(..., description="Model identifier")
    content: list[dict[str, Any]] = Field(..., description="Response content blocks")
    stop_reason: str | None = Field(None, description="Stop reason")
    stop_sequence: str | None = Field(None, description="Stop sequence")
    usage: UsageInfo = Field(..., description="Token usage")
