"""
Messages API router (Anthropic-style with OpenAI compatibility).
"""

from __future__ import annotations

import json
import logging
import httpx
import time
from typing import Any, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import get_config
from ..dependencies import AuthenticatedUser, get_inference_service, require_auth_if_enabled
from ..messages_models import MessageInput, MessagesRequest, MessagesResponse
from ..messages_utils import validate_tool_choice, validate_tools
from ..openai_models import OpenAIChatCompletionResponse
from ..services.inference_service import InferenceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["messages"])


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


def _validate_message_limits(messages: list[MessageInput], max_messages: int, max_len: int) -> None:
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


@router.post(
    "/messages",
    response_model=Union[MessagesResponse, OpenAIChatCompletionResponse],  # noqa: UP007
    summary="Create Message",
)
async def create_message(
    request: MessagesRequest,
    http_request: Request,
    user: AuthenticatedUser | None = Depends(require_auth_if_enabled),
    service: InferenceService = Depends(get_inference_service),
) -> Any:
    """
    Anthropic-compatible Messages endpoint.

    Supports OpenAI-style messages as input and can return OpenAI-style responses
    when `response_format="openai"` is provided.
    """

    config = get_config()

    try:
        validate_tools(request.tools)
        validate_tool_choice(request.tool_choice)
        _validate_message_limits(
            request.messages,
            config.validation.max_conversation_messages,
            config.validation.max_message_length,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    request_id = getattr(http_request.state, "request_id", None)
    start_time = time.monotonic()
    logger.info(
        "Messages request received",
        extra={
            "request_id": request_id,
            "model": request.model,
            "stream": request.stream,
            "response_format": request.response_format or "anthropic",
        },
    )

    if request.stream:
        if request.response_format == "openai":
            generator = service.stream_openai(request)
        else:
            generator = service.stream_anthropic(request)
        return StreamingResponse(generator, media_type="text/event-stream")

    try:
        if request.response_format == "openai":
            openai_payload = await service.create_openai_response(request)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.info(
                "Messages request completed",
                extra={"request_id": request_id, "elapsed_ms": elapsed_ms},
            )
            return JSONResponse(content=openai_payload)
        response_payload = await service.create_anthropic_response(request)
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "Messages request completed",
            extra={"request_id": request_id, "elapsed_ms": elapsed_ms},
        )
        return MessagesResponse(**response_payload)
    except ValueError as exc:
        logger.warning(
            "Messages request invalid request payload",
            extra={"request_id": request_id, "error": str(exc)},
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status_code = getattr(exc.response, "status_code", None)
        detail = _extract_inference_error_text(exc)
        if isinstance(status_code, int) and 500 <= status_code < 600:
            logger.error(
                "Messages upstream service error",
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
            "Messages upstream request rejected",
            extra={
                "request_id": request_id,
                "status_code": status_code,
                "detail": detail,
            },
        )
        raise HTTPException(status_code=422, detail=detail) from exc
    except httpx.HTTPError as exc:
        logger.warning(
            "Messages upstream unavailable",
            extra={"request_id": request_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=502, detail="Inference backend unavailable"
        ) from exc
    except Exception as exc:
        logger.exception("Messages internal request handling failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
