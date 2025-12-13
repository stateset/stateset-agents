"""API v1 compatibility router.

This router provides a small, stable surface under ``/api/v1`` that is used by
the test-suite and supports basic training job and conversation workflows.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..auth import RequestContext, get_request_context
from ..config import get_config


router = APIRouter(prefix="/api/v1", tags=["v1"])


class TrainingRequestV1(BaseModel):
    """Request schema for starting a v1 training job."""

    prompts: List[str] = Field(..., description="Training prompts", min_length=1)
    strategy: Literal["computational", "distributed"] = Field(
        "computational", description="Training strategy"
    )
    num_iterations: int = Field(1, ge=1, description="Number of training iterations")
    idempotency_key: Optional[str] = Field(
        None, description="Optional idempotency key for safe retries"
    )


class TrainingCreateResponseV1(BaseModel):
    """Response schema for a created v1 training job."""

    job_id: str
    status: str


class TrainingStatusResponseV1(BaseModel):
    """Response schema for a v1 training job status."""

    job_id: str
    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class TrainingCancelResponseV1(BaseModel):
    """Response schema for cancelling a v1 training job."""

    job_id: str
    status: str
    message: str


class MessageV1(BaseModel):
    """Chat message schema for v1 conversations."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ConversationRequestV1(BaseModel):
    """Request schema for creating/continuing a v1 conversation."""

    message: Optional[str] = None
    messages: Optional[List[MessageV1]] = None
    conversation_id: Optional[str] = None

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ConversationResponseV1(BaseModel):
    """Response schema for v1 conversations."""

    conversation_id: str
    response: str
    tokens_used: int
    processing_time_ms: int


def _require_app_state(request: Request, attr: str, default: Any) -> Any:
    if not hasattr(request.app.state, attr):
        setattr(request.app.state, attr, default)
    return getattr(request.app.state, attr)


def _validate_prompts(prompts: List[str]) -> List[str]:
    cleaned: List[str] = []
    for prompt in prompts:
        if prompt is None:
            raise HTTPException(status_code=422, detail="Prompt cannot be null")
        stripped = prompt.strip()
        if not stripped:
            raise HTTPException(status_code=422, detail="Prompt cannot be empty")
        cleaned.append(stripped)
    return cleaned


def _approx_tokens(text: str) -> int:
    tokens = len(text.split())
    return max(1, tokens)


@router.get("/health", summary="Health Check (v1)")
async def health_v1(request: Request) -> Dict[str, Any]:
    """Health check endpoint for API v1."""
    config = get_config()
    started_at = getattr(request.app.state, "started_at", None)
    now = time.time()
    uptime_seconds = int(now - started_at) if isinstance(started_at, (int, float)) else 0

    components = [
        {"name": "api", "status": "healthy"},
        {
            "name": "auth",
            "status": "enabled" if config.security.require_auth else "disabled",
        },
        {
            "name": "rate_limit",
            "status": "enabled" if config.rate_limit.enabled else "disabled",
        },
    ]

    return {
        "status": "healthy",
        "version": config.api_version,
        "uptime_seconds": uptime_seconds,
        "components": components,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

@router.post(
    "/training",
    response_model=TrainingCreateResponseV1,
    status_code=202,
    summary="Start Training (v1)",
)
async def start_training_v1(
    request: Request,
    payload: TrainingRequestV1,
    ctx: RequestContext = Depends(get_request_context),
) -> TrainingCreateResponseV1:
    """Start a new v1 training job."""
    config = get_config()

    prompts = _validate_prompts(payload.prompts)
    if len(prompts) > config.validation.max_prompts:
        raise HTTPException(
            status_code=422,
            detail=f"Too many prompts (max {config.validation.max_prompts})",
        )
    if any(len(p) > config.validation.max_prompt_length for p in prompts):
        raise HTTPException(
            status_code=422,
            detail=f"Prompt exceeds max length ({config.validation.max_prompt_length})",
        )
    if payload.num_iterations > config.validation.max_iterations:
        raise HTTPException(
            status_code=422,
            detail=f"num_iterations exceeds max ({config.validation.max_iterations})",
        )

    jobs: Dict[str, Dict[str, Any]] = _require_app_state(request, "v1_training_jobs", {})
    idempotency: Dict[str, str] = _require_app_state(
        request, "v1_training_idempotency", {}
    )

    if payload.idempotency_key:
        existing = idempotency.get(payload.idempotency_key)
        if existing and existing in jobs:
            return TrainingCreateResponseV1(job_id=existing, status="starting")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "created_at": datetime.utcnow(),
        "user_id": ctx.user.user_id,
        "strategy": payload.strategy,
        "num_iterations": payload.num_iterations,
        "prompts": prompts,
        "metrics": {
            "iterations_completed": 0,
            "total_trajectories": 0,
        },
    }

    if payload.idempotency_key:
        idempotency[payload.idempotency_key] = job_id

    return TrainingCreateResponseV1(job_id=job_id, status="starting")


@router.get(
    "/training/{job_id}",
    response_model=TrainingStatusResponseV1,
    summary="Get Training Status (v1)",
)
async def get_training_status_v1(
    request: Request,
    job_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> TrainingStatusResponseV1:
    """Get v1 training job status."""
    jobs: Dict[str, Dict[str, Any]] = _require_app_state(request, "v1_training_jobs", {})
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return TrainingStatusResponseV1(
        job_id=job_id,
        status=job.get("status", "unknown"),
        metrics=job.get("metrics", {}),
    )


@router.delete(
    "/training/{job_id}",
    response_model=TrainingCancelResponseV1,
    summary="Cancel Training (v1)",
)
async def cancel_training_v1(
    request: Request,
    job_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> TrainingCancelResponseV1:
    """Cancel a v1 training job."""
    jobs: Dict[str, Dict[str, Any]] = _require_app_state(request, "v1_training_jobs", {})
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["status"] = "cancelled"
    return TrainingCancelResponseV1(
        job_id=job_id,
        status="cancelled",
        message="Training job cancelled",
    )


@router.post(
    "/conversations",
    response_model=ConversationResponseV1,
    summary="Chat (v1)",
)
async def chat_v1(
    request: Request,
    payload: ConversationRequestV1,
    ctx: RequestContext = Depends(get_request_context),
) -> ConversationResponseV1:
    """Create or continue a v1 conversation."""
    config = get_config()
    start = time.monotonic()

    if payload.message and payload.messages:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'message' or 'messages', not both",
        )
    if not payload.message and not payload.messages:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'message' or 'messages'",
        )

    conversations: Dict[str, Dict[str, Any]] = _require_app_state(
        request, "v1_conversations", {}
    )

    if payload.message is not None:
        user_text = payload.message.strip()
        if not user_text:
            raise HTTPException(status_code=422, detail="Message cannot be empty")
        if len(user_text) > config.validation.max_message_length:
            raise HTTPException(status_code=422, detail="Message too long")
        messages: List[Dict[str, str]] = [{"role": "user", "content": user_text}]
    else:
        # payload.messages validated by Pydantic
        messages = [
            {"role": msg.role, "content": msg.content} for msg in (payload.messages or [])
        ]
        user_text = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_text = msg["content"].strip()
                break
        if not user_text:
            raise HTTPException(status_code=422, detail="No user message provided")
        if len(user_text) > config.validation.max_message_length:
            raise HTTPException(status_code=422, detail="Message too long")

    conversation_id = payload.conversation_id or str(uuid.uuid4())
    convo = conversations.setdefault(conversation_id, {"messages": []})

    # Append incoming messages (either one user message or the provided list)
    convo["messages"].extend(messages)

    # Generate a minimal deterministic response (no model dependency in tests).
    response_text = f"Echo: {user_text}"
    convo["messages"].append({"role": "assistant", "content": response_text})

    elapsed_ms = int((time.monotonic() - start) * 1000)
    if elapsed_ms <= 0:
        elapsed_ms = 1

    tokens_used = _approx_tokens(user_text) + _approx_tokens(response_text)

    return ConversationResponseV1(
        conversation_id=conversation_id,
        response=response_text,
        tokens_used=tokens_used,
        processing_time_ms=elapsed_ms,
    )


@router.get(
    "/conversations/{conversation_id}",
    summary="Get Conversation (v1)",
)
async def get_conversation_v1(
    request: Request,
    conversation_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> Dict[str, Any]:
    """Get a v1 conversation transcript."""
    conversations: Dict[str, Dict[str, Any]] = _require_app_state(
        request, "v1_conversations", {}
    )
    convo = conversations.get(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": convo.get("messages", [])}


@router.delete(
    "/conversations/{conversation_id}",
    summary="End Conversation (v1)",
)
async def end_conversation_v1(
    request: Request,
    conversation_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> Dict[str, Any]:
    """End a v1 conversation."""
    conversations: Dict[str, Dict[str, Any]] = _require_app_state(
        request, "v1_conversations", {}
    )
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del conversations[conversation_id]
    return {"status": "ended", "conversation_id": conversation_id}
