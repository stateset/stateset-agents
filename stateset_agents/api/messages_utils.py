"""
Utilities for converting between Anthropic-style and OpenAI-style messages.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from collections.abc import Iterable


def _join_text_parts(parts: Iterable[str]) -> str:
    cleaned = [p for p in parts if isinstance(p, str) and p.strip()]
    return "\n".join(cleaned).strip()


def _flatten_block_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    texts.append(text)
            elif isinstance(item, str):
                texts.append(item)
        return _join_text_parts(texts)
    if isinstance(value, dict):
        maybe_text = value.get("text") or value.get("content")
        if isinstance(maybe_text, str):
            return maybe_text
    return json.dumps(value) if value is not None else ""


def _image_block_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    block_type = block.get("type")
    if block_type == "image_url":
        return block

    source = block.get("source") if block_type == "image" else None
    if not isinstance(source, dict):
        return None

    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        if data:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
    if source_type == "url":
        url = source.get("url")
        if url:
            return {"type": "image_url", "image_url": {"url": url}}
    return None


def _blocks_to_content_and_tools(
    blocks: list[Any],
) -> tuple[
    str | list[dict[str, Any]] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    text_parts: list[str] = []
    content_parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")

        if block_type == "text":
            text_parts.append(block.get("text", ""))
            continue
        if block_type in ("image", "image_url"):
            image_part = _image_block_to_openai(block)
            if image_part:
                content_parts.append(image_part)
            continue
        if block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"tool_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
            continue
        if block_type == "tool_result":
            tool_results.append(block)
            continue

        # Fallback: treat unknown blocks as text if possible
        fallback_text = block.get("text") or block.get("content")
        if isinstance(fallback_text, str):
            text_parts.append(fallback_text)

    combined_text = _join_text_parts(text_parts)

    if content_parts:
        if combined_text:
            content_parts.insert(0, {"type": "text", "text": combined_text})
        content_value: str | list[dict[str, Any]] | None = content_parts
    else:
        content_value = combined_text if combined_text else None

    return content_value, tool_calls, tool_results


def anthropic_messages_to_openai(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages into OpenAI-compatible messages."""

    openai_messages: list[dict[str, Any]] = []

    if system:
        system_content: str | list[dict[str, Any]] | None = system
        if isinstance(system, list):
            system_content, _, _ = _blocks_to_content_and_tools(system)
        if system_content:
            openai_messages.append({"role": "system", "content": system_content})

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if content is None:
            content = ""

        if isinstance(content, list):
            content_value, tool_calls, tool_results = _blocks_to_content_and_tools(
                content
            )

            base_message: dict[str, Any] = {"role": role}
            base_message["content"] = content_value or ""

            # Preserve existing tool_calls if present
            if isinstance(message.get("tool_calls"), list):
                base_message["tool_calls"] = message.get("tool_calls")
            elif tool_calls and role == "assistant":
                base_message["tool_calls"] = tool_calls

            has_content = False
            if isinstance(content_value, str):
                has_content = bool(content_value.strip())
            elif isinstance(content_value, list):
                has_content = len(content_value) > 0

            # If the Anthropic "user" message only contains tool_result blocks, emit
            # OpenAI "tool" messages only (no empty "user" message).
            if role != "user" or has_content or not tool_results:
                openai_messages.append(base_message)

            for tool_result in tool_results:
                tool_call_id = (
                    tool_result.get("tool_use_id")
                    or tool_result.get("tool_call_id")
                    or tool_result.get("id")
                    or f"tool_{uuid.uuid4().hex[:8]}"
                )
                tool_content = _flatten_block_content(tool_result.get("content"))
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )
            continue

        # Non-list content: pass through
        flat_message: dict[str, Any] = {"role": role, "content": content}

        if isinstance(message.get("tool_calls"), list):
            flat_message["tool_calls"] = message.get("tool_calls")

        if message.get("tool_call_id"):
            flat_message["tool_call_id"] = message.get("tool_call_id")

        if message.get("name") and isinstance(message.get("name"), str):
            flat_message["name"] = message.get("name")

        openai_messages.append(flat_message)

    return openai_messages


def _tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function":
        return tool
    if "function" in tool:
        return {"type": "function", "function": tool["function"]}

    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def tools_to_openai(
    tools: list[dict[str, Any]] | None
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [_tool_to_openai(tool) for tool in tools]


def tool_choice_to_openai(
    tool_choice: str | dict[str, Any] | None
) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice

    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        name = tool_choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return tool_choice


def validate_tools(tools: list[dict[str, Any]] | None) -> None:
    """Validate tool definitions for Anthropic/OpenAI compatibility."""
    if not tools:
        return
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{idx}] must be an object")

        # OpenAI format: {"type":"function","function":{...}}
        if tool.get("type") == "function" or "function" in tool:
            function = tool.get("function") or tool
            name = function.get("name")
            if not name or not isinstance(name, str):
                raise ValueError(f"tools[{idx}] missing function name")
            parameters = function.get("parameters", {})
            if parameters is None:
                parameters = {}
            if not isinstance(parameters, dict):
                raise ValueError(f"tools[{idx}].function.parameters must be an object")
            continue

        # Anthropic format: {"name":..., "description":..., "input_schema":{...}}
        if "name" in tool and "input_schema" in tool:
            name = tool.get("name")
            if not name or not isinstance(name, str):
                raise ValueError(f"tools[{idx}].name must be a string")
            input_schema = tool.get("input_schema")
            if not isinstance(input_schema, dict):
                raise ValueError(f"tools[{idx}].input_schema must be an object")
            continue

        raise ValueError(f"tools[{idx}] must define a function tool")


def validate_tool_choice(tool_choice: str | dict[str, Any] | None) -> None:
    """Validate tool_choice payload."""
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        if tool_choice not in ("auto", "required", "none"):
            raise ValueError("tool_choice must be one of auto, required, none")
        return
    if not isinstance(tool_choice, dict):
        raise ValueError("tool_choice must be a string or object")
    choice_type = tool_choice.get("type")
    if choice_type == "function":
        function = tool_choice.get("function")
        if not isinstance(function, dict):
            raise ValueError(
                "tool_choice.function must be an object when type=function"
            )
        name = function.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(
                "tool_choice.function.name must be a string when type=function"
            )
        return

    if choice_type not in ("auto", "any", "tool"):
        raise ValueError("tool_choice.type must be auto, any, tool, or function")
    if choice_type == "tool":
        name = tool_choice.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("tool_choice.name must be a string when type=tool")


def openai_response_to_anthropic(response: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI chat completion response into Anthropic message format."""

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    content_blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    content_blocks.append(
                        {"type": "text", "text": part.get("text", "")}
                    )
    elif isinstance(content, str) and content.strip():
        content_blocks.append({"type": "text", "text": content})

    tool_calls = message.get("tool_calls") or []
    tool_call_items = tool_calls if isinstance(tool_calls, list) else []
    for call in tool_call_items:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            function = {}
        raw_args = function.get("arguments", "{}")
        parsed_args: Any
        if raw_args is None:
            parsed_args = {}
        elif isinstance(raw_args, (dict, list)):
            parsed_args = raw_args
        else:
            try:
                parsed_args = json.loads(str(raw_args))
            except (TypeError, json.JSONDecodeError):
                parsed_args = raw_args
        tool_name = function.get("name")
        if not isinstance(tool_name, str):
            tool_name = ""
        try:
            tool_id = str(call.get("id"))
            if not tool_id:
                raise TypeError
        except TypeError:
            tool_id = f"tool_{uuid.uuid4().hex[:8]}"
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": parsed_args,
            }
        )

    finish_reason = choice.get("finish_reason")
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filter",
    }.get(finish_reason, finish_reason)

    usage = response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if not isinstance(input_tokens, int) or input_tokens < 0:
        input_tokens = 0
    if not isinstance(output_tokens, int) or output_tokens < 0:
        output_tokens = 0

    return {
        "id": response.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": response.get("model", ""),
        "content": content_blocks or [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    """Best-effort extraction of the last user text content."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text = _flatten_block_content(content)
            if text:
                return text
    return ""
