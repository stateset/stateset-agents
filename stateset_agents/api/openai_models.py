"""
Minimal OpenAI-compatible models for /v1/chat/completions.

We keep these intentionally permissive (extra fields allowed) to avoid breaking
clients as the OpenAI schema evolves.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIChatMessage(BaseModel):
    """OpenAI chat message input."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="Message role"
    )
    content: str | list[dict[str, Any]] | None = Field(
        None, description="Message content"
    )
    name: str | None = Field(None, description="Optional author name")
    tool_call_id: str | None = Field(None, description="Tool call id (tool role)")
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="Tool call metadata (assistant role)"
    )


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI chat completion request payload."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model identifier")
    messages: list[OpenAIChatMessage] = Field(..., description="Chat messages")

    max_tokens: int | None = Field(None, ge=1, description="Max tokens to generate")
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling")
    stream: bool = Field(False, description="Stream partial responses")

    stop: str | list[str] | None = Field(None, description="Stop sequence(s)")
    tools: list[dict[str, Any]] | None = Field(None, description="Tool definitions")
    tool_choice: str | dict[str, Any] | None = Field(
        None, description="Tool choice"
    )


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI chat completion response payload (minimal)."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
