"""
Tool-capable agent implementation.

This module isolates tool registration, JSON tool-call parsing, and tool
execution support from the base conversational agent implementation.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from typing import Any

from .agent import AgentConfig, MultiTurnAgent, TOOL_EXEC_EXCEPTIONS

logger = logging.getLogger(__name__)


class ToolAgent(MultiTurnAgent):
    """
    Agent that can use tools and function calls
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.tools: list[dict[str, Any]] = []
        self.tool_registry: dict[str, dict[str, Any]] = {}
        for tool in tools or []:
            self.add_tool(tool)

    async def generate_response(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate response with potential tool usage"""
        normalized_messages = (
            [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        )

        if self._should_use_tools(normalized_messages, context):
            return await self._generate_with_tools(normalized_messages, context)
        raw_response: object = await super().generate_response(normalized_messages, context)
        return raw_response if isinstance(raw_response, str) else str(raw_response)

    def _should_use_tools(
        self, messages: list[dict[str, str]], context: dict[str, Any] | None = None
    ) -> bool:
        """Determine if tools should be used for this response"""
        if not self.tools:
            return False

        last_message = messages[-1]["content"].lower() if messages else ""
        tool_keywords = ["calculate", "search", "look up", "find", "analyze"]

        return any(keyword in last_message for keyword in tool_keywords)

    def add_tool(self, tool: dict[str, Any]) -> None:
        """Add a tool to the agent."""
        if not isinstance(tool, dict):
            raise TypeError("Tool definitions must be dictionaries.")

        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool definition missing required string field 'name'.")
        normalized_name = name.strip()

        normalized_tool = dict(tool)
        normalized_tool["name"] = normalized_name
        normalized_tool.setdefault("description", "")
        parameters = normalized_tool.get("parameters")
        if not isinstance(parameters, dict):
            normalized_tool["parameters"] = {}

        if normalized_name in self.tool_registry:
            logger.warning(
                "Replacing existing tool definition for name '%s'.", normalized_name
            )
            self.tools = [
                registered_tool
                for registered_tool in self.tools
                if registered_tool.get("name") != normalized_name
            ]

        self.tools.append(normalized_tool)
        self.tool_registry[normalized_name] = normalized_tool

    async def _generate_with_tools(
        self, messages: list[dict[str, str]], context: dict[str, Any] | None = None
    ) -> str:
        """Generate response that may include tool calls"""

        tool_context = self._format_tool_descriptions()

        system_msg = (
            f"You have access to the following tools:\n{tool_context}\n\n"
            "To use a tool, respond with a JSON block in the following format:\n"
            "```json\n"
            "{\n"
            '  "tool": "tool_name",\n'
            '  "parameters": {\n'
            '    "param1": "value1"\n'
            "  }\n"
            "}\n"
            "```\n"
        )

        enhanced_messages = []
        if messages and messages[0]["role"] == "system":
            enhanced_messages = [
                {
                    "role": "system",
                    "content": messages[0]["content"] + "\n\n" + system_msg,
                }
            ] + messages[1:]
        else:
            enhanced_messages = [{"role": "system", "content": system_msg}] + messages

        raw_response: object = await super().generate_response(enhanced_messages, context)
        response_text = (
            raw_response if isinstance(raw_response, str) else str(raw_response)
        )
        processed_response = await self._process_tool_calls(response_text, context)
        return processed_response

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for model context"""
        descriptions = []
        for tool in self.tools:
            schema = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            descriptions.append(json.dumps(schema))
        return "\n".join(descriptions)

    async def _process_tool_calls(
        self, response: str, context: dict[str, Any] | None = None
    ) -> str:
        """Process any tool calls in the response"""
        json_block_pattern = re.compile(
            r"```json\s*(\{[\s\S]*?\})\s*```", flags=re.IGNORECASE
        )
        cleaned_response = response

        def _extract_json_objects(text: str) -> list[tuple[str, int, int]]:
            """Extract top-level JSON objects from a mixed text response."""
            decoder = json.JSONDecoder()
            objects: list[tuple[str, int, int]] = []
            start = 0
            while True:
                next_obj_start = text.find("{", start)
                if next_obj_start == -1:
                    return objects

                try:
                    _, end_offset = decoder.raw_decode(text, idx=next_obj_start)
                except json.JSONDecodeError:
                    start = next_obj_start + 1
                    continue

                obj_end = next_obj_start + end_offset
                objects.append((text[next_obj_start:obj_end], next_obj_start, obj_end))
                start = obj_end

        def _as_tool_payload(payload: Any) -> tuple[str, dict[str, Any]] | None:
            if not isinstance(payload, dict):
                return None

            tool_name = payload.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                return None
            tool_name = tool_name.strip()

            if "parameters" not in payload:
                return None

            parameters = payload.get("parameters")
            if not isinstance(parameters, dict):
                return None

            return tool_name, parameters

        tool_payloads: list[tuple[str, dict[str, Any]]] = []

        for match in json_block_pattern.finditer(response):
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                parsed = None

            candidate = _as_tool_payload(parsed) if parsed is not None else None
            if candidate is not None:
                tool_payloads.append(candidate)
                cleaned_response = cleaned_response.replace(match.group(0), "", 1)

        if not tool_payloads:
            for raw_json, start, end in _extract_json_objects(response):
                if response[end:].strip():
                    continue
                prefix = response[:start].rstrip()
                if prefix and prefix[-1] not in {":", ",", ";", "\n"}:
                    continue
                try:
                    parsed = json.loads(raw_json)
                except json.JSONDecodeError:
                    continue
                candidate = _as_tool_payload(parsed)
                if candidate is not None:
                    tool_payloads.append(candidate)
                    cleaned_response = (response[:start] + response[end:]).strip()
                    break

        if tool_payloads:
            cleaned_response = cleaned_response.strip()
            unknown_tool: str | None = None
            executed_tool = False

            for tool_name, tool_params in tool_payloads:
                if tool_name not in self.tool_registry:
                    if unknown_tool is None:
                        unknown_tool = tool_name
                    continue

                tool = self.tool_registry[tool_name]
                validation_error = self._validate_tool_payload(
                    tool_name, tool, tool_params
                )
                if validation_error:
                    if cleaned_response:
                        cleaned_response += f"\n\nError: {validation_error}"
                    else:
                        cleaned_response = f"Error: {validation_error}"
                    return cleaned_response

                tool_result = await self._execute_tool(tool_name, tool_params, context)
                if cleaned_response:
                    cleaned_response += f"\n\nTool Output ({tool_name}): {tool_result}"
                else:
                    cleaned_response = f"Tool Output ({tool_name}): {tool_result}"
                executed_tool = True

            if not executed_tool and unknown_tool is not None:
                if cleaned_response:
                    cleaned_response += f"\n\nError: Tool '{unknown_tool}' not found."
                else:
                    cleaned_response = f"Error: Tool '{unknown_tool}' not found."
                return cleaned_response

            return cleaned_response

        return response

    def _validate_tool_payload(
        self, tool_name: str, tool: dict[str, Any], parameters: dict[str, Any]
    ) -> str | None:
        """Apply optional JSON-schema-like validation for tool arguments."""
        schema = tool.get("parameters")
        if not isinstance(schema, dict):
            return None

        if schema.get("type") != "object":
            return None

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return None

        additional_properties_allowed = schema.get("additionalProperties", True)
        if additional_properties_allowed is False:
            unknown_fields = set(parameters) - set(properties)
            if unknown_fields:
                unknown = ", ".join(sorted(unknown_fields))
                return f"Tool '{tool_name}' does not accept: {unknown}"

        required = schema.get("required", [])
        if isinstance(required, list):
            for field_name in required:
                if field_name not in parameters:
                    return (
                        f"Tool '{tool_name}' is missing required parameter: "
                        f"{field_name}"
                    )

        return None

    async def _execute_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Execute a tool call"""
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return f"Error: Tool {tool_name} not found"

        try:
            if "function" in tool:
                func = tool["function"]
                tool_kwargs = self._filter_tool_kwargs(func, params)
                args, kwargs = self._prepare_tool_call_args(
                    func, tool_kwargs, context=context
                )
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                return str(result)
            return f"Tool {tool_name} executed (mock result)"
        except TOOL_EXEC_EXCEPTIONS as e:
            return f"Error executing {tool_name}: {str(e)}"

    @staticmethod
    def _filter_tool_kwargs(func: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Keep only arguments accepted by the target tool function."""
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return dict(params)

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            return dict(params)

        accepted = {
            param.name
            for param in signature.parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return {name: value for name, value in params.items() if name in accepted}

    @staticmethod
    def _prepare_tool_call_args(
        func: Any,
        params: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Prepare positional and keyword arguments for tool execution."""
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return (), dict(params)

        uses_context = ToolAgent._supports_context_only_tool_call(func)
        remaining = dict(params)
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        context_param_names = {"context", "ctx", "request_context"}
        injectable_params = [
            param
            for param in signature.parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        resolved_context = context
        if uses_context and resolved_context is None:
            context_param = next(
                (
                    param
                    for param in injectable_params
                    if param.name in context_param_names
                ),
                None,
            )
            if context_param is not None and (
                context_param.name != "context" or len(injectable_params) > 1
            ):
                resolved_context = {}

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                kwargs.update(remaining)
                remaining = {}
                continue

            if uses_context and param.name in context_param_names:
                if param.kind == inspect.Parameter.KEYWORD_ONLY:
                    kwargs[param.name] = resolved_context
                else:
                    args.append(resolved_context)
                continue

            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                if param.name in remaining:
                    args.append(remaining.pop(param.name))
                elif param.default is inspect.Parameter.empty:
                    continue

            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                if param.name in remaining:
                    kwargs[param.name] = remaining.pop(param.name)

        if remaining:
            kwargs.update(remaining)

        return tuple(args), kwargs

    @staticmethod
    def _supports_context_only_tool_call(func: Any) -> bool:
        """Return True when the tool function accepts exactly one context argument."""
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return False

        params = [
            param
            for param in signature.parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        required_params = [
            param
            for param in params
            if param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if len(required_params) != 1:
            return False

        return required_params[0].name in {"context", "ctx", "request_context"}
