"""API v1 compatibility router.

This router provides a small, stable surface under ``/api/v1`` that is used by
the test-suite and supports basic training job and conversation workflows.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..auth import RequestContext, get_request_context
from ..dependencies import get_inference_service
from ..config import get_config
from ..messages_models import MessageInput, MessagesRequest
from ..services.inference_service import InferenceService

router = APIRouter(prefix="/api/v1", tags=["v1"])
logger = logging.getLogger(__name__)

V1_TRAINING_JOB_TTL_SECONDS = 60 * 60
V1_TRAINING_IDEMPOTENCY_TTL_SECONDS = 60 * 60
V1_CONVERSATION_TTL_SECONDS = 24 * 60 * 60

V1_MAX_TRAINING_JOBS = 1000
V1_MAX_TRAINING_IDEMPOTENCY_KEYS = 2000
V1_MAX_CONVERSATIONS = 500


class TrainingRequestV1(BaseModel):
    """Request schema for starting a v1 training job."""

    prompts: list[str] = Field(..., description="Training prompts", min_length=1)
    strategy: Literal["computational", "distributed"] = Field(
        "computational", description="Training strategy"
    )
    num_iterations: int = Field(1, ge=1, description="Number of training iterations")
    idempotency_key: str | None = Field(
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
    metrics: dict[str, Any] = Field(default_factory=dict)


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

    message: str | None = None
    messages: list[MessageV1] | None = None
    conversation_id: str | None = None
    model: str | None = None

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)


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


def _cleanup_timed_state(
    values: dict[str, Any],
    metadata: dict[str, float],
    *,
    ttl_seconds: int,
    now: float,
    max_items: int | None = None,
) -> None:
    """Drop expired or over-budget items from in-memory state stores."""
    # Give in-memory entries a one-second grace period to avoid boundary churn
    # from whole-second timestamps and request scheduling jitter.
    expiry_cutoff = now - ttl_seconds - 1.0

    # Remove stale metadata entries.
    for key in list(metadata):
        if key not in values:
            metadata.pop(key, None)
            continue
        if metadata[key] < expiry_cutoff:
            values.pop(key, None)
            metadata.pop(key, None)

    # Seed metadata for legacy values that predate TTL tracking.
    for key in values:
        if key not in metadata:
            metadata[key] = now

    if max_items is None or len(values) <= max_items:
        return

    # Evict oldest entries when above cap to avoid unbounded growth.
    oldest_keys = sorted(metadata.items(), key=lambda item: item[1])
    excess = len(values) - max_items
    for key, _ in oldest_keys[:excess]:
        values.pop(key, None)
        metadata.pop(key, None)


def _get_timed_state(
    request: Request,
    state_attr: str,
    meta_attr: str,
    ttl_seconds: int,
    max_items: int,
) -> tuple[dict[str, Any], dict[str, float], float]:
    values = _require_app_state(request, state_attr, {})
    metadata = _require_app_state(request, meta_attr, {})
    now = time.time()
    _cleanup_timed_state(
        values,
        metadata,
        ttl_seconds=ttl_seconds,
        now=now,
        max_items=max_items,
    )
    return values, metadata, now


def _touch_timed_state(metadata: dict[str, float], key: str, now: float) -> None:
    metadata[key] = now


def _delete_timed_state(
    values: dict[str, Any], metadata: dict[str, float], key: str
) -> None:
    values.pop(key, None)
    metadata.pop(key, None)


def _resolve_idempotency_job_id(
    idempotency: dict[str, Any], key: str
) -> str | None:
    mapped = idempotency.get(key)
    if isinstance(mapped, str):
        return mapped
    if isinstance(mapped, dict):
        value = mapped.get("job_id")
        return value if isinstance(value, str) and value else None
    return None


def _is_v1_conversation_owner(conversation: dict[str, Any], user_id: str) -> bool:
    owner = conversation.get("user_id")
    return owner is None or owner == user_id


def _validate_prompts(prompts: list[str]) -> list[str]:
    cleaned: list[str] = []
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


def _extract_response_text(payload: dict[str, Any]) -> str:
    """Extract assistant content from an OpenAI response payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    message = choices[0]
    if not isinstance(message, dict):
        return ""

    raw_message = message.get("message", {})
    if not isinstance(raw_message, dict):
        return ""

    content = raw_message.get("content")
    if content is None:
        content = raw_message.get("text")
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    return str(content)


def _extract_usage_tokens(usage_payload: Any) -> int:
    """Return best-effort total token count from an OpenAI usage block."""
    if not isinstance(usage_payload, dict):
        return 0

    total_tokens = usage_payload.get("total_tokens")
    if isinstance(total_tokens, int) and total_tokens >= 0:
        return total_tokens

    input_tokens = usage_payload.get("prompt_tokens")
    output_tokens = usage_payload.get("completion_tokens")
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens
    if isinstance(input_tokens, int):
        return input_tokens
    if isinstance(output_tokens, int):
        return output_tokens
    return 0


def _trim_messages(
    messages: list[dict[str, str]], max_messages: int
) -> list[dict[str, str]]:
    """Keep only the newest `max_messages` messages."""
    if max_messages <= 0:
        return []
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def _extract_inference_error_text(exc: Exception) -> str:
    """Build a safe, compact error message from inference backend exceptions."""
    response = getattr(exc, "response", None)
    if isinstance(response, httpx.Response):
        try:
            payload = response.json()
        except ValueError:
            payload = response.text.strip()

        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            if isinstance(error, str) and error:
                return error
        elif isinstance(payload, str) and payload:
            return payload

    return "Inference backend request failed"


@router.get("/health", summary="Health Check (v1)")
async def health_v1(request: Request) -> dict[str, Any]:
    """Health check endpoint for API v1."""
    config = get_config()
    started_at = getattr(request.app.state, "started_at", None)
    now = time.time()
    uptime_seconds = (
        int(now - started_at) if isinstance(started_at, (int, float)) else 0
    )

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
    user_id = ctx.user.user_id

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

    requested_idempotency_key = (
        payload.idempotency_key.strip() if payload.idempotency_key else None
    )

    jobs, job_access, now = _get_timed_state(
        request,
        "v1_training_jobs",
        "v1_training_jobs_access",
        ttl_seconds=V1_TRAINING_JOB_TTL_SECONDS,
        max_items=V1_MAX_TRAINING_JOBS,
    )
    idempotency, idempotency_access, _ = _get_timed_state(
        request,
        "v1_training_idempotency",
        "v1_training_idempotency_access",
        ttl_seconds=V1_TRAINING_IDEMPOTENCY_TTL_SECONDS,
        max_items=V1_MAX_TRAINING_IDEMPOTENCY_KEYS,
    )

    if requested_idempotency_key:
        scoped_key = f"{user_id}:{requested_idempotency_key}"
        existing = _resolve_idempotency_job_id(idempotency, scoped_key)
        if existing is None:
            existing = _resolve_idempotency_job_id(
                idempotency, requested_idempotency_key
            )
            if existing:
                existing_job = jobs.get(existing, {})
                if existing_job.get("user_id") == user_id:
                    _touch_timed_state(idempotency_access, requested_idempotency_key, now)
                else:
                    existing = None
        if existing and existing in jobs and jobs[existing].get("user_id") == user_id:
            _touch_timed_state(job_access, existing, now)
            _touch_timed_state(idempotency_access, scoped_key, now)
            existing_status = jobs[existing].get("status", "starting")
            return TrainingCreateResponseV1(job_id=existing, status=str(existing_status))

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
    job_access[job_id] = now
    _cleanup_timed_state(
        jobs,
        job_access,
        ttl_seconds=V1_TRAINING_JOB_TTL_SECONDS,
        now=now,
        max_items=V1_MAX_TRAINING_JOBS,
    )

    if requested_idempotency_key:
        idempotency[requested_idempotency_key] = job_id
        _touch_timed_state(idempotency_access, requested_idempotency_key, now)
        idempotency_key = f"{user_id}:{requested_idempotency_key}"
        idempotency[idempotency_key] = job_id
        _touch_timed_state(idempotency_access, idempotency_key, now)
        _cleanup_timed_state(
            idempotency,
            idempotency_access,
            ttl_seconds=V1_TRAINING_IDEMPOTENCY_TTL_SECONDS,
            now=now,
            max_items=V1_MAX_TRAINING_IDEMPOTENCY_KEYS,
        )

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
    jobs, job_access, now = _get_timed_state(
        request,
        "v1_training_jobs",
        "v1_training_jobs_access",
        ttl_seconds=V1_TRAINING_JOB_TTL_SECONDS,
        max_items=V1_MAX_TRAINING_JOBS,
    )
    job = jobs.get(job_id)
    if not job or job.get("user_id") != ctx.user.user_id:
        raise HTTPException(status_code=404, detail="Job not found")
    _touch_timed_state(job_access, job_id, now)
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
    jobs, job_access, now = _get_timed_state(
        request,
        "v1_training_jobs",
        "v1_training_jobs_access",
        ttl_seconds=V1_TRAINING_JOB_TTL_SECONDS,
        max_items=V1_MAX_TRAINING_JOBS,
    )
    job = jobs.get(job_id)
    if not job or job.get("user_id") != ctx.user.user_id:
        raise HTTPException(status_code=404, detail="Job not found")
    job["status"] = "cancelled"
    _touch_timed_state(job_access, job_id, now)
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
    service: InferenceService = Depends(get_inference_service),
    ctx: RequestContext = Depends(get_request_context),
) -> ConversationResponseV1:
    """Create or continue a v1 conversation."""
    config = get_config()
    max_messages = int(config.validation.max_conversation_messages)
    if max_messages < 1:
        max_messages = 128
    start = time.monotonic()
    request_id = getattr(request.state, "request_id", "unknown")

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

    conversations, conversation_access, now = _get_timed_state(
        request,
        "v1_conversations",
        "v1_conversations_access",
        ttl_seconds=V1_CONVERSATION_TTL_SECONDS,
        max_items=V1_MAX_CONVERSATIONS,
    )
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    user_id = ctx.user.user_id
    convo = conversations.get(conversation_id)
    if convo is not None and not _is_v1_conversation_owner(convo, user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    if convo is None:
        convo = {"messages": [], "user_id": user_id}
        conversations[conversation_id] = convo
        _touch_timed_state(conversation_access, conversation_id, now)
        _cleanup_timed_state(
            conversations,
            conversation_access,
            ttl_seconds=V1_CONVERSATION_TTL_SECONDS,
            now=now,
            max_items=V1_MAX_CONVERSATIONS,
        )
    else:
        convo_user = convo.get("user_id")
        if convo_user is None:
            convo["user_id"] = user_id
        _touch_timed_state(conversation_access, conversation_id, now)

    history = convo.get("messages")
    if not isinstance(history, list):
        history = []
        convo["messages"] = history

    history = _trim_messages(
        [msg for msg in history if isinstance(msg, dict)], max_messages
    )
    convo["messages"] = history

    if payload.message is not None:
        user_text = payload.message.strip()
        if not user_text:
            raise HTTPException(status_code=422, detail="Message cannot be empty")
        if len(user_text) > config.validation.max_message_length:
            raise HTTPException(status_code=422, detail="Message too long")
        incoming_messages: list[dict[str, str]] = [
            {"role": "user", "content": user_text}
        ]
    else:
        # payload.messages validated by Pydantic
        incoming_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in (payload.messages or [])
        ]
        if len(incoming_messages) > max_messages:
            raise HTTPException(
                status_code=422,
                detail=f"Too many messages in request (max {max_messages})",
            )
        for msg in incoming_messages:
            if len(msg["content"]) > config.validation.max_message_length:
                raise HTTPException(
                    status_code=422,
                    detail="Message too long",
                )
        user_text = ""
        for msg in reversed(incoming_messages):
            if msg["role"] == "user":
                user_text = msg["content"].strip()
                break
        if not user_text:
            raise HTTPException(status_code=422, detail="No user message provided")
        if len(user_text) > config.validation.max_message_length:
            raise HTTPException(status_code=422, detail="Message too long")

    messages = _trim_messages(history + incoming_messages, max_messages)

    model = payload.model.strip() if isinstance(payload.model, str) else None
    if model == "":
        model = None
    if model is None:
        default_model = getattr(service.config, "default_model", None)
        if isinstance(default_model, str) and default_model:
            model = default_model
        elif service.is_stub:
            model = "stub://v1-conversation"
        else:
            raise HTTPException(
                status_code=422,
                detail="No model specified. Set payload.model or INFERENCE_DEFAULT_MODEL.",
            )

    logger.info(
        "v1.conversations request received",
        extra={
            "request_id": request_id,
            "conversation_id": conversation_id,
            "user_id": ctx.user.user_id,
            "model": model,
            "message_count": len(messages),
            "is_stub": service.is_stub,
        },
    )

    try:
        openai_request = MessagesRequest(
            model=model,
            messages=[MessageInput(**message) for message in messages],
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
        openai_payload = await service.create_openai_response(openai_request)
    except ValueError as exc:
        logger.warning(
            "v1.conversations invalid request payload",
            extra={
                "request_id": request_id,
                "conversation_id": conversation_id,
                "user_id": ctx.user.user_id,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status_code = getattr(exc.response, "status_code", None)
        detail = _extract_inference_error_text(exc)
        if isinstance(status_code, int) and status_code >= 500:
            logger.error(
                "v1.conversations upstream service error",
                extra={
                    "request_id": request_id,
                    "conversation_id": conversation_id,
                    "user_id": ctx.user.user_id,
                    "status_code": status_code,
                },
            )
            raise HTTPException(
                status_code=502, detail="Inference backend unavailable"
            ) from exc

        logger.warning(
            "v1.conversations upstream request rejected",
            extra={
                "request_id": request_id,
                "conversation_id": conversation_id,
                "user_id": ctx.user.user_id,
                "status_code": status_code,
                "detail": detail,
            },
        )
        raise HTTPException(status_code=422, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.warning(
            "v1.conversations upstream unavailable",
            extra={
                "request_id": request_id,
                "conversation_id": conversation_id,
                "user_id": ctx.user.user_id,
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=502, detail="Inference backend unavailable"
        ) from exc
    except Exception as exc:
        logger.exception(
            "v1.conversations unexpected inference failure",
            extra={
                "request_id": request_id,
                "conversation_id": conversation_id,
                "user_id": ctx.user.user_id,
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    response_text = _extract_response_text(openai_payload)
    if not response_text:
        response_text = ""
    # Append only after successful inference so failed requests do not pollute
    # conversation state.
    convo["messages"].extend(incoming_messages)
    convo["messages"].append({"role": "assistant", "content": response_text})
    convo["messages"] = _trim_messages(convo["messages"], max_messages)
    _touch_timed_state(conversation_access, conversation_id, time.time())

    elapsed_ms = int((time.monotonic() - start) * 1000)
    if elapsed_ms <= 0:
        elapsed_ms = 1

    tokens_used = _extract_usage_tokens(openai_payload.get("usage"))
    if tokens_used <= 0:
        tokens_used = _approx_tokens(user_text) + _approx_tokens(response_text)

    logger.info(
        "v1.conversations request completed",
        extra={
            "request_id": request_id,
            "conversation_id": conversation_id,
            "user_id": ctx.user.user_id,
            "elapsed_ms": elapsed_ms,
            "tokens_used": tokens_used,
            "response_chars": len(response_text),
        },
    )

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
) -> dict[str, Any]:
    """Get a v1 conversation transcript."""
    conversations, conversation_access, now = _get_timed_state(
        request,
        "v1_conversations",
        "v1_conversations_access",
        ttl_seconds=V1_CONVERSATION_TTL_SECONDS,
        max_items=V1_MAX_CONVERSATIONS,
    )
    convo = conversations.get(conversation_id)
    if not convo or not _is_v1_conversation_owner(convo, ctx.user.user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    _touch_timed_state(conversation_access, conversation_id, now)
    return {"conversation_id": conversation_id, "messages": convo.get("messages", [])}


@router.delete(
    "/conversations/{conversation_id}",
    summary="End Conversation (v1)",
)
async def end_conversation_v1(
    request: Request,
    conversation_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> dict[str, Any]:
    """End a v1 conversation."""
    conversations, conversation_access, _ = _get_timed_state(
        request,
        "v1_conversations",
        "v1_conversations_access",
        ttl_seconds=V1_CONVERSATION_TTL_SECONDS,
        max_items=V1_MAX_CONVERSATIONS,
    )
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not _is_v1_conversation_owner(
        conversations[conversation_id], ctx.user.user_id
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")
    _delete_timed_state(conversations, conversation_access, conversation_id)
    return {"status": "ended", "conversation_id": conversation_id}
