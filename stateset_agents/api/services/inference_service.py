"""
Inference service that proxies /v1/messages to a vLLM OpenAI-compatible backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, cast
from collections.abc import AsyncIterator

import httpx

from ..messages_models import MessagesRequest
from ..messages_utils import (
    anthropic_messages_to_openai,
    extract_last_user_text,
    openai_response_to_anthropic,
    tool_choice_to_openai,
    tools_to_openai,
)

logger = logging.getLogger(__name__)

INFERENCE_EXCEPTIONS = (
    httpx.HTTPError,
    RuntimeError,
    TypeError,
    ValueError,
    asyncio.TimeoutError,
)


def _compact_json(data: dict[str, Any]) -> str:
    """Serialize SSE payloads deterministically and without cosmetic whitespace."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _parse_int_env(
    key: str,
    default: int,
    min_value: int = 0,
    *,
    invalid_fallback: int | None = None,
) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    if invalid_fallback is None:
        invalid_fallback = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid integer value for %s=%r; using %s",
            key,
            raw,
            invalid_fallback,
        )
        return invalid_fallback
    if value < min_value:
        logger.warning(
            "Value for %s=%s is below minimum %s; using %s",
            key,
            value,
            min_value,
            min_value,
        )
        return min_value
    return value


def _parse_float_env(key: str, default: float, min_value: float = 0.0) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid float value for %s=%r; using %s", key, raw, default)
        return default
    if value < min_value:
        logger.warning(
            "Value for %s=%s is below minimum %s; using %s",
            key,
            value,
            min_value,
            min_value,
        )
        return min_value
    return value


def _parse_model_map(raw_map: str) -> dict[str, str]:
    raw_map = (raw_map or "").strip()
    if not raw_map:
        return {}

    model_map: dict[str, str] = {}
    try:
        loaded = json.loads(raw_map)
    except json.JSONDecodeError:
        logger.warning(
            "Invalid JSON in INFERENCE_MODEL_MAP; "
            "falling back to key=value parsing"
        )
        for pair in raw_map.split(","):
            if "=" not in pair:
                logger.debug("Skipping invalid model map pair without '=': %r", pair)
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                model_map[key] = value
            else:
                logger.warning(
                    "Ignoring empty model mapping pair in INFERENCE_MODEL_MAP: %r",
                    pair,
                )
        return model_map

    if not isinstance(loaded, dict):
        logger.warning(
            "INFERENCE_MODEL_MAP expected a JSON object, got %s; using fallback parser.",
            type(loaded).__name__,
        )
        return model_map

    for public_model, internal_model in loaded.items():
        if not isinstance(public_model, str) or not isinstance(internal_model, str):
            continue
        public_model = public_model.strip()
        internal_model = internal_model.strip()
        if public_model and internal_model:
            model_map[public_model] = internal_model

    return model_map


def _normalize_env_string(value: str | None, *, default: str) -> str:
    value = (value or "").strip()
    if not value:
        return default
    return value


def _normalize_path(value: str | None, *, default: str) -> str:
    path = _normalize_env_string(value, default=default)
    if not path:
        return default
    return path if path.startswith("/") else f"/{path}"


@dataclass
class InferenceConfig:
    """Configuration for inference backend."""

    backend: str = "vllm"  # "vllm" | "stub"
    base_url: str = "http://localhost:8000"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    default_model: str | None = None
    default_max_tokens: int = 1024
    health_path: str = "/health"
    include_stream_usage: bool = False
    model_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> InferenceConfig:
        environment = os.getenv("API_ENVIRONMENT", "production").lower()
        default_backend = "stub" if environment != "production" else "vllm"
        backend = os.getenv("INFERENCE_BACKEND", default_backend).strip().lower()
        if backend not in ("stub", "vllm"):
            logger.warning(
                "Unknown INFERENCE_BACKEND=%s; using %s",
                backend,
                default_backend,
            )
            backend = default_backend

        raw_map = os.getenv("INFERENCE_MODEL_MAP", "")
        model_map = _parse_model_map(raw_map)

        return cls(
            backend=backend,
            base_url=_normalize_env_string(
                os.getenv("INFERENCE_BACKEND_URL"),
                default="http://localhost:8000",
            ),
            timeout_seconds=_parse_float_env(
                "INFERENCE_TIMEOUT_SECONDS", 120.0, min_value=0.1
            ),
            max_retries=_parse_int_env("INFERENCE_MAX_RETRIES", 2),
            default_max_tokens=_parse_int_env(
                "INFERENCE_DEFAULT_MAX_TOKENS",
                1024,
                min_value=1,
                invalid_fallback=1,
            ),
            health_path=_normalize_path(
                os.getenv("INFERENCE_HEALTH_PATH"),
                default="/health",
            ),
            include_stream_usage=os.getenv(
                "INFERENCE_STREAM_INCLUDE_USAGE", "false"
            ).lower()
            in ("1", "true", "yes"),
            model_map=model_map,
            default_model=_normalize_env_string(
                os.getenv("INFERENCE_DEFAULT_MODEL"), default=""
            )
            or None,
        )

    def resolve_model(self, model_name: str | None) -> str:
        if model_name:
            return self.model_map.get(model_name, model_name)
        if self.default_model:
            return self.default_model
        raise ValueError("Model name is required and no default model is configured.")


class InferenceService:
    """Inference service routing requests to vLLM or stub backends."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig.from_env()
        self._client: httpx.AsyncClient | None = None

    @property
    def is_stub(self) -> bool:
        """Whether this service is running in stub mode."""
        return self.config.backend == "stub"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    async def aclose(self) -> None:
        """Close any underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> bool:
        """Check backend health."""
        if self.is_stub:
            return True
        client = await self._get_client()
        response = await client.get(self.config.health_path)
        # When mypy skips following imports, httpx can become `Any` and the
        # comparison result becomes `Any`. Force a concrete bool.
        return bool(response.status_code == 200)

    async def list_models(self) -> dict[str, Any]:
        """Return OpenAI-style model list payload.

        This is primarily for OpenAI client compatibility (`GET /v1/models`).
        When using vLLM, we proxy the backend's own model list.
        """
        # If we have an explicit public->internal mapping, expose the public names
        # instead of leaking internal paths into client-visible model IDs.
        if self.config.model_map:
            model_ids = sorted(self.config.model_map.keys())
            if self.config.default_model and self.config.default_model not in model_ids:
                if self.config.default_model not in self.config.model_map.values():
                    model_ids.append(self.config.default_model)
                model_ids = sorted(set(model_ids))
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "stateset",
                    }
                    for model_id in model_ids
                ],
            }

        if self.is_stub:
            model_id = self.config.default_model or "unknown"
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "stateset",
                    }
                ],
            }

        client = await self._get_client()
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.get("/v1/models")
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
            except INFERENCE_EXCEPTIONS as exc:
                if attempt >= self.config.max_retries:
                    logger.error("Inference backend failed listing models: %s", exc)
                    break
                await asyncio.sleep(0.5 * (attempt + 1))

        model_id = self.config.default_model or "unknown"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "stateset",
                }
            ],
        }

    def _build_openai_payload(self, request: MessagesRequest) -> dict[str, Any]:
        model_name = self.config.resolve_model(request.model)

        openai_messages = anthropic_messages_to_openai(
            [m.model_dump(exclude_none=True) for m in request.messages],
            system=request.system,
        )

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": openai_messages,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        else:
            payload["max_tokens"] = self.config.default_max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        openai_tools = tools_to_openai(request.tools)
        if openai_tools:
            payload["tools"] = openai_tools

        openai_tool_choice = tool_choice_to_openai(request.tool_choice)
        if openai_tool_choice is not None:
            payload["tool_choice"] = openai_tool_choice

        # Pass through any extra fields on the request that vLLM/OpenAI backends
        # might understand (e.g. presence_penalty, seed, logit_bias, etc.).
        # We explicitly exclude fields that we transform or that are gateway-only.
        extra = request.model_dump(exclude_none=True)
        for key in (
            "model",
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "response_format",
        ):
            extra.pop(key, None)
        payload.update(extra)

        return payload

    async def _stub_response(self, request: MessagesRequest) -> dict[str, Any]:
        model_name = self.config.resolve_model(request.model)
        openai_messages = anthropic_messages_to_openai(
            [m.model_dump(exclude_none=True) for m in request.messages],
            system=request.system,
        )
        user_text = extract_last_user_text(openai_messages)
        response_text = f"Echo: {user_text}" if user_text else "Echo"

        return {
            "id": f"chatcmpl_{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_text.split()) if user_text else 0,
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_text.split()) + len(response_text.split()),
            },
        }

    def _rewrite_model_name(
        self, payload: dict[str, Any], *, public_model: str, internal_model: str
    ) -> None:
        model = payload.get("model")
        if (
            isinstance(model, str)
            and model == internal_model
            and public_model
            and public_model != internal_model
        ):
            payload["model"] = public_model

    async def create_openai_response(self, request: MessagesRequest) -> dict[str, Any]:
        """Return OpenAI-style response dict."""
        public_model = request.model
        internal_model = self.config.resolve_model(request.model)

        if self.is_stub:
            stub_payload = await self._stub_response(request)
            self._rewrite_model_name(
                stub_payload, public_model=public_model, internal_model=internal_model
            )
            return stub_payload

        payload = self._build_openai_payload(request)
        client = await self._get_client()

        for attempt in range(self.config.max_retries + 1):
            try:
                http_response = await client.post("/v1/chat/completions", json=payload)
                http_response.raise_for_status()
                data = cast(dict[str, Any], http_response.json())
                self._rewrite_model_name(
                    data, public_model=public_model, internal_model=internal_model
                )
                return data
            except INFERENCE_EXCEPTIONS as exc:
                if attempt >= self.config.max_retries:
                    logger.error("Inference backend failed: %s", exc)
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

        raise RuntimeError("Inference backend unavailable")

    async def stream_openai(self, request: MessagesRequest) -> AsyncIterator[str]:
        """Yield OpenAI-style SSE lines from the backend."""
        public_model = request.model
        internal_model = self.config.resolve_model(request.model)

        if self.is_stub:
            stub_response = await self._stub_response(request)
            data = _compact_json(
                {
                    "id": stub_response["id"],
                    "object": "chat.completion.chunk",
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": stub_response["choices"][0]["message"][
                                    "content"
                                ]
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        payload = self._build_openai_payload(request)
        payload["stream"] = True
        if self.config.include_stream_usage:
            stream_options = payload.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
            stream_options.setdefault("include_usage", True)
            payload["stream_options"] = stream_options
        client = await self._get_client()

        async with client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    raw = line.replace("data:", "", 1).strip()
                    if raw and raw != "[DONE]":
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            yield f"{line}\n\n"
                            continue
                        if isinstance(chunk, dict):
                            self._rewrite_model_name(
                                chunk,
                                public_model=public_model,
                                internal_model=internal_model,
                            )
                            yield f"data: {_compact_json(chunk)}\n\n"
                            continue
                    yield f"{line}\n\n"

    async def stream_anthropic(self, request: MessagesRequest) -> AsyncIterator[str]:
        """Yield Anthropic-style SSE lines derived from OpenAI stream."""
        message_id = f"msg_{uuid.uuid4().hex}"
        block_id = f"block_{uuid.uuid4().hex[:8]}"
        stop_sent = False
        output_tokens = 0
        backend_usage: dict[str, Any] | None = None
        tool_block_indices: dict[int, int] = {}
        tool_block_ids: dict[int, str] = {}

        yield _sse_event(
            "message_start",
            {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "model": request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        )
        yield _sse_event(
            "content_block_start",
            {
                "index": 0,
                "content_block": {"type": "text", "text": "", "id": block_id},
            },
        )

        async for line in self.stream_openai(request):
            if not line.startswith("data:"):
                continue
            payload = line.replace("data:", "", 1).strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("usage"), dict):
                backend_usage = cast(dict[str, Any], chunk["usage"])

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text_delta = delta.get("content")
            if text_delta:
                output_tokens += max(1, len(text_delta.split()))
                yield _sse_event(
                    "content_block_delta",
                    {
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text_delta},
                    },
                )

            tool_calls = delta.get("tool_calls") or []
            for tool in tool_calls:
                tool_index = tool.get("index", 0)
                if tool_index not in tool_block_indices:
                    tool_block_indices[tool_index] = 1 + len(tool_block_indices)
                    tool_block_ids[tool_index] = (
                        tool.get("id") or f"tool_{uuid.uuid4().hex[:8]}"
                    )
                    function = tool.get("function") or {}
                    yield _sse_event(
                        "content_block_start",
                        {
                            "index": tool_block_indices[tool_index],
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_block_ids[tool_index],
                                "name": function.get("name", ""),
                                "input": {},
                            },
                        },
                    )
                function = tool.get("function") or {}
                args_delta = function.get("arguments")
                if args_delta:
                    yield _sse_event(
                        "content_block_delta",
                        {
                            "index": tool_block_indices[tool_index],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_delta,
                            },
                        },
                    )

            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                stop_reason = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "content_filter": "content_filter",
                }.get(finish_reason, finish_reason)
                reported_output_tokens = output_tokens
                if backend_usage is not None:
                    completion_tokens = backend_usage.get("completion_tokens")
                    if isinstance(completion_tokens, int) and completion_tokens >= 0:
                        reported_output_tokens = completion_tokens
                yield _sse_event("content_block_stop", {"index": 0})
                for _tool_index, block_index in tool_block_indices.items():
                    yield _sse_event("content_block_stop", {"index": block_index})
                yield _sse_event(
                    "message_delta",
                    {
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": reported_output_tokens},
                    },
                )
                yield _sse_event("message_stop", {"id": message_id})
                stop_sent = True

        if not stop_sent:
            yield _sse_event("message_stop", {"id": message_id})

    async def create_anthropic_response(
        self, request: MessagesRequest
    ) -> dict[str, Any]:
        openai_response = await self.create_openai_response(request)
        return openai_response_to_anthropic(openai_response)


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = _compact_json(data)
    return f"event: {event}\ndata: {payload}\n\n"
