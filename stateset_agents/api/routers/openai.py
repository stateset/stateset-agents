"""
OpenAI-compatible router.

This exposes a minimal `/v1/chat/completions` endpoint that proxies to the
configured inference backend (typically vLLM's OpenAI server).
"""

from __future__ import annotations

import json
import logging
import httpx
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import get_config
from ..dependencies import AuthenticatedUser, get_inference_service, require_auth_if_enabled
from ..messages_models import MessageInput, MessagesRequest
from ..messages_utils import validate_tool_choice, validate_tools
from ..openai_models import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from ..services.inference_service import InferenceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


def _estimate_message_content_length(content: Any) -> int:
    """Estimate length of message content for validation checks."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        try:
            return len(json.dumps(content))
        except (TypeError, ValueError):
            return len(str(content))
    if isinstance(content, dict):
        try:
            return len(json.dumps(content))
        except (TypeError, ValueError):
            return len(str(content))
    return len(str(content))


def _validate_openai_request_messages(
    messages: list[MessageInput], max_messages: int, max_len: int
) -> None:
    if len(messages) > max_messages:
        raise HTTPException(
            status_code=422,
            detail=f"Too many messages (max {max_messages})",
        )

    for index, msg in enumerate(messages):
        if _estimate_message_content_length(msg.content) > max_len:
            raise HTTPException(
                status_code=422,
                detail=f"Message content too long at index {index}",
            )


def _extract_inference_error_text(exc: Exception) -> str:
    """Build a compact inference error message from backend responses."""
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


@router.get(
    "/models",
    summary="Models (OpenAI Compatible)",
)
async def list_models(
    http_request: Request,
    user: AuthenticatedUser | None = Depends(require_auth_if_enabled),
    service: InferenceService = Depends(get_inference_service),
) -> Any:
    try:
        payload = await service.list_models()
        return JSONResponse(content=payload)
    except Exception as exc:
        logger.error("OpenAI models listing failed: %s", exc)
        raise HTTPException(
            status_code=502, detail="Inference backend unavailable"
        ) from exc


@router.post(
    "/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    summary="Chat Completions (OpenAI Compatible)",
)
async def chat_completions(
    payload: OpenAIChatCompletionRequest,
    http_request: Request,
    user: AuthenticatedUser | None = Depends(require_auth_if_enabled),
    service: InferenceService = Depends(get_inference_service),
) -> Any:
    config = get_config()

    try:
        validate_tools(payload.tools)
        validate_tool_choice(payload.tool_choice)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    raw_payload = payload.model_dump(exclude_none=True)

    stop_sequences = None
    stop = raw_payload.pop("stop", None)
    if isinstance(stop, str):
        stop_sequences = [stop]
    elif isinstance(stop, list):
        stop_sequences = [s for s in stop if isinstance(s, str) and s]

    raw_messages = raw_payload.pop("messages", [])
    if len(raw_messages) > config.validation.max_conversation_messages:
        raise HTTPException(
            status_code=422,
            detail=f"Too many messages (max {config.validation.max_conversation_messages})",
        )

    messages_request = MessagesRequest(
        model=raw_payload.pop("model"),
        messages=[MessageInput(**m) for m in raw_messages],
        max_tokens=raw_payload.pop("max_tokens", None),
        temperature=raw_payload.pop("temperature", None),
        top_p=raw_payload.pop("top_p", None),
        stop_sequences=stop_sequences,
        stream=bool(raw_payload.pop("stream", False)),
        tools=raw_payload.pop("tools", None),
        tool_choice=raw_payload.pop("tool_choice", None),
        **raw_payload,
    )
    _validate_openai_request_messages(
        messages_request.messages,
        config.validation.max_conversation_messages,
        config.validation.max_message_length,
    )

    request_id = getattr(http_request.state, "request_id", None)
    start_time = time.monotonic()
    logger.info(
        "OpenAI chat.completions request received",
        extra={
            "request_id": request_id,
            "model": payload.model,
            "stream": payload.stream,
        },
    )

    if messages_request.stream:
        generator = service.stream_openai(messages_request)
        return StreamingResponse(generator, media_type="text/event-stream")

    try:
        openai_payload = await service.create_openai_response(messages_request)
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "OpenAI chat.completions request completed",
            extra={"request_id": request_id, "elapsed_ms": elapsed_ms},
        )
        return JSONResponse(content=openai_payload)
    except ValueError as exc:
        logger.warning(
            "OpenAI chat.completions invalid request payload",
            extra={"request_id": request_id, "error": str(exc)},
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status_code = getattr(exc.response, "status_code", None)
        detail = _extract_inference_error_text(exc)
        if isinstance(status_code, int) and 500 <= status_code < 600:
            logger.error(
                "OpenAI chat.completions upstream service error",
                extra={
                    "request_id": request_id,
                    "status_code": status_code,
                    "detail": detail,
                },
            )
            raise HTTPException(
                status_code=502, detail="Inference backend unavailable"
            ) from exc

        logger.warning(
            "OpenAI chat.completions request rejected",
            extra={
                "request_id": request_id,
                "status_code": status_code,
                "detail": detail,
            },
        )
        raise HTTPException(status_code=422, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.warning(
            "OpenAI chat.completions upstream unavailable",
            extra={"request_id": request_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=502, detail="Inference backend unavailable"
        ) from exc
    except Exception as exc:
        logger.error("OpenAI inference failed: %s", exc)
        raise HTTPException(
            status_code=502, detail="Inference backend unavailable"
        ) from exc
