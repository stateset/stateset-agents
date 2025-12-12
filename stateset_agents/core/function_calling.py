"""
OpenAI-Compatible Function Calling for StateSet Agents

This module provides standardized function/tool calling that is compatible
with the OpenAI API format, enabling seamless integration with existing
tooling and easier migration between providers.

Features:
- OpenAI-compatible tool definitions with JSON Schema parameters
- Automatic function execution and result formatting
- Parallel tool call support
- Type-safe function registration with decorators

Example:
    >>> from core.function_calling import FunctionCallingAgent, tool
    >>>
    >>> @tool(description="Get current weather for a location")
    >>> def get_weather(location: str, unit: str = "celsius") -> dict:
    ...     return {"temp": 22, "unit": unit, "location": location}
    >>>
    >>> agent = FunctionCallingAgent(config, tools=[get_weather])
    >>> response = await agent.chat("What's the weather in Paris?")
"""

import asyncio
import inspect
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)


class ToolChoiceMode(str, Enum):
    """Tool choice behavior modes (OpenAI-compatible)."""

    AUTO = "auto"  # Model decides whether to use tools
    NONE = "none"  # Never use tools
    REQUIRED = "required"  # Must use at least one tool


@dataclass
class FunctionParameter:
    """Parameter definition for a function.

    Attributes:
        name: Parameter name
        type: JSON Schema type
        description: Human-readable description
        required: Whether the parameter is required
        enum: Allowed values for enum parameters
        default: Default value if not required
    """

    name: str
    type: str
    description: str = ""
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Any = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class FunctionDefinition:
    """OpenAI-compatible function definition.

    Attributes:
        name: Unique function name
        description: What the function does
        parameters: Function parameters as JSON Schema
        handler: The actual function to execute
        strict: Whether to enforce strict parameter validation
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None
    strict: bool = False

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                **({"strict": True} if self.strict else {}),
            },
        }


@dataclass
class ToolCall:
    """Represents a tool call from the model.

    Attributes:
        id: Unique identifier for this tool call
        type: Always "function" for function calls
        function: Function details (name and arguments)
    """

    id: str
    type: Literal["function"] = "function"
    function: Dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get function name."""
        return self.function.get("name", "")

    @property
    def arguments(self) -> str:
        """Get raw arguments string."""
        return self.function.get("arguments", "{}")

    def parsed_arguments(self) -> Dict[str, Any]:
        """Parse arguments as JSON."""
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}


@dataclass
class ToolResult:
    """Result from executing a tool.

    Attributes:
        tool_call_id: ID of the tool call this result corresponds to
        content: The result content (usually JSON string)
        is_error: Whether an error occurred
    """

    tool_call_id: str
    content: str
    is_error: bool = False

    def to_message(self) -> Dict[str, Any]:
        """Convert to OpenAI message format."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


def _python_type_to_json_schema(python_type: Type) -> Dict[str, Any]:
    """Convert Python type annotation to JSON Schema."""
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none]}

    # Handle List
    if origin is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}

    # Handle Dict
    if origin is dict:
        return {"type": "object"}

    # Handle Literal (enum)
    if origin is Literal:
        values = list(args)
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        return {"enum": values}

    # Primitives
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        type(None): {"type": "null"},
    }

    return type_map.get(python_type, {"type": "string"})


def tool(
    name: Optional[str] = None,
    description: str = "",
    strict: bool = False,
) -> Callable:
    """Decorator to register a function as an OpenAI-compatible tool.

    Automatically extracts parameter types and descriptions from
    type hints and docstrings.

    Args:
        name: Override function name (defaults to function.__name__)
        description: Tool description (defaults to docstring)
        strict: Enable strict parameter validation

    Returns:
        Decorator function

    Example:
        >>> @tool(description="Calculate the sum of two numbers")
        >>> def add(a: int, b: int) -> int:
        ...     '''Add two integers.
        ...
        ...     Args:
        ...         a: First number
        ...         b: Second number
        ...     '''
        ...     return a + b
    """

    def decorator(func: Callable) -> FunctionDefinition:
        func_name = name or func.__name__
        func_description = description or (func.__doc__ or "").split("\n")[0].strip()

        # Extract type hints
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        sig = inspect.signature(func)

        # Build parameters schema
        properties: Dict[str, Any] = {}
        required: List[str] = []

        # Parse docstring for parameter descriptions
        param_descriptions = _parse_docstring_params(func.__doc__ or "")

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            param_schema = _python_type_to_json_schema(param_type)

            # Add description from docstring
            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]

            properties[param_name] = param_schema

            # Check if required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                param_schema["default"] = param.default

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        if strict:
            parameters["additionalProperties"] = False

        return FunctionDefinition(
            name=func_name,
            description=func_description,
            parameters=parameters,
            handler=func,
            strict=strict,
        )

    return decorator


def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Extract parameter descriptions from docstring.

    Supports Google-style and NumPy-style docstrings.
    """
    params = {}

    # Google style: Args:\n    param: description
    args_match = re.search(r"Args?:\s*\n((?:\s+\w+.*\n)+)", docstring)
    if args_match:
        for line in args_match.group(1).split("\n"):
            param_match = re.match(r"\s+(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+)", line)
            if param_match:
                params[param_match.group(1)] = param_match.group(2).strip()

    # NumPy style: Parameters\n----------\nparam : type\n    description
    params_match = re.search(
        r"Parameters\s*\n-+\s*\n((?:[^\n]+\n(?:\s+[^\n]+\n)*)+)", docstring
    )
    if params_match:
        current_param = None
        for line in params_match.group(1).split("\n"):
            header_match = re.match(r"(\w+)\s*:", line)
            if header_match:
                current_param = header_match.group(1)
            elif current_param and line.strip():
                params[current_param] = line.strip()
                current_param = None

    return params


class FunctionCallingMixin:
    """Mixin providing OpenAI-compatible function calling to agents.

    Add this mixin to any Agent subclass to enable standardized
    tool/function calling with automatic execution.
    """

    _tools: Dict[str, FunctionDefinition]
    _tool_choice: ToolChoiceMode

    def __init_tools__(
        self,
        tools: Optional[List[Union[FunctionDefinition, Callable, Dict[str, Any]]]] = None,
        tool_choice: ToolChoiceMode = ToolChoiceMode.AUTO,
    ) -> None:
        """Initialize function calling capabilities.

        Call this from your agent's __init__ after super().__init__().

        Args:
            tools: List of tools (FunctionDefinitions, decorated functions, or dicts)
            tool_choice: Default tool choice mode
        """
        self._tools = {}
        self._tool_choice = tool_choice

        if tools:
            for t in tools:
                self.register_tool(t)

    def register_tool(
        self, tool: Union[FunctionDefinition, Callable, Dict[str, Any]]
    ) -> None:
        """Register a tool for use by the agent.

        Args:
            tool: Tool to register (FunctionDefinition, decorated function, or dict)
        """
        if isinstance(tool, FunctionDefinition):
            self._tools[tool.name] = tool
        elif callable(tool):
            # Check if it's already a FunctionDefinition from decorator
            if isinstance(tool, FunctionDefinition):
                self._tools[tool.name] = tool
            else:
                # Wrap plain callable
                wrapped = self._wrap_callable(tool)
                self._tools[wrapped.name] = wrapped
        elif isinstance(tool, dict):
            # Parse OpenAI-format dict
            func_def = self._parse_tool_dict(tool)
            self._tools[func_def.name] = func_def
        else:
            raise TypeError(f"Invalid tool type: {type(tool)}")

    def _wrap_callable(self, func: Callable) -> FunctionDefinition:
        """Wrap a plain callable as a FunctionDefinition."""
        return tool()(func)

    def _parse_tool_dict(self, tool_dict: Dict[str, Any]) -> FunctionDefinition:
        """Parse OpenAI-format tool dict into FunctionDefinition."""
        if tool_dict.get("type") == "function":
            func_info = tool_dict["function"]
        else:
            func_info = tool_dict

        return FunctionDefinition(
            name=func_info["name"],
            description=func_info.get("description", ""),
            parameters=func_info.get("parameters", {"type": "object", "properties": {}}),
            strict=func_info.get("strict", False),
        )

    def get_tools_openai_format(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI API format."""
        return [t.to_openai_format() for t in self._tools.values()]

    def _build_tool_system_prompt(self) -> str:
        """Build system prompt section describing available tools."""
        if not self._tools:
            return ""

        tools_json = json.dumps(self.get_tools_openai_format(), indent=2)

        return f"""You have access to the following tools:

{tools_json}

To use a tool, respond with a JSON object in this exact format:
{{
    "tool_calls": [
        {{
            "id": "call_<unique_id>",
            "type": "function",
            "function": {{
                "name": "<tool_name>",
                "arguments": "<json_string_of_arguments>"
            }}
        }}
    ]
}}

You may call multiple tools in parallel by including multiple objects in the tool_calls array.
After receiving tool results, provide your final response to the user.
If you don't need to use any tools, respond normally without the tool_calls wrapper.
"""

    def _parse_tool_calls(self, response: str) -> List[ToolCall]:
        """Extract tool calls from model response."""
        tool_calls = []

        # Try to find tool_calls JSON
        try:
            # Look for JSON with tool_calls
            json_match = re.search(r"\{[\s\S]*?\"tool_calls\"[\s\S]*?\}", response)
            if json_match:
                data = json.loads(json_match.group())
                if "tool_calls" in data:
                    for tc in data["tool_calls"]:
                        tool_calls.append(
                            ToolCall(
                                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                type=tc.get("type", "function"),
                                function=tc.get("function", {}),
                            )
                        )
        except json.JSONDecodeError:
            pass

        # Fallback: look for individual function calls
        if not tool_calls:
            func_pattern = r'\{\s*"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*"([^"]*)"'
            for match in re.finditer(func_pattern, response, re.DOTALL):
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        function={"name": match.group(1), "arguments": match.group(2)},
                    )
                )

        return tool_calls

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with execution output or error
        """
        tool_name = tool_call.name

        if tool_name not in self._tools:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                is_error=True,
            )

        tool_def = self._tools[tool_name]

        if tool_def.handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": f"Tool {tool_name} has no handler"}),
                is_error=True,
            )

        try:
            args = tool_call.parsed_arguments()

            # Execute handler
            if asyncio.iscoroutinefunction(tool_def.handler):
                result = await tool_def.handler(**args)
            else:
                result = tool_def.handler(**args)

            # Serialize result
            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, default=str)

            return ToolResult(tool_call_id=tool_call.id, content=content)

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": str(e)}),
                is_error=True,
            )

    async def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of results in same order as input
        """
        tasks = [self.execute_tool_call(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)


# Convenience function to create agent class
def create_function_calling_agent_class():
    """Create a FunctionCallingAgent class.

    Returns:
        FunctionCallingAgent class combining MultiTurnAgent with function calling
    """
    from .agent import MultiTurnAgent, AgentConfig

    class FunctionCallingAgent(MultiTurnAgent, FunctionCallingMixin):
        """Agent with OpenAI-compatible function calling.

        Example:
            >>> @tool(description="Get stock price")
            >>> async def get_stock_price(symbol: str) -> dict:
            ...     # API call here
            ...     return {"symbol": symbol, "price": 150.00}
            >>>
            >>> agent = FunctionCallingAgent(
            ...     config,
            ...     tools=[get_stock_price]
            ... )
            >>> await agent.initialize()
            >>> response = await agent.chat_with_tools(
            ...     "What's the price of AAPL?"
            ... )
        """

        def __init__(
            self,
            config: AgentConfig,
            tools: Optional[List[Union[FunctionDefinition, Callable, Dict[str, Any]]]] = None,
            tool_choice: ToolChoiceMode = ToolChoiceMode.AUTO,
            **kwargs,
        ):
            super().__init__(config, **kwargs)
            self.__init_tools__(tools, tool_choice)

        async def chat_with_tools(
            self,
            messages: Union[str, List[Dict[str, str]]],
            context: Optional[Dict[str, Any]] = None,
            tool_choice: Optional[ToolChoiceMode] = None,
            max_tool_rounds: int = 5,
        ) -> str:
            """Chat with automatic tool execution.

            Handles the full tool-use loop: generating tool calls,
            executing them, and continuing until a final response.

            Args:
                messages: Input message(s)
                context: Optional context
                tool_choice: Override default tool choice mode
                max_tool_rounds: Maximum tool execution rounds

            Returns:
                Final response after all tool executions
            """
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            else:
                messages = list(messages)

            choice = tool_choice or self._tool_choice

            # Add tool instructions if using tools
            if choice != ToolChoiceMode.NONE and self._tools:
                tool_prompt = self._build_tool_system_prompt()
                if messages and messages[0].get("role") == "system":
                    messages[0] = {
                        "role": "system",
                        "content": messages[0]["content"] + "\n\n" + tool_prompt,
                    }
                else:
                    messages.insert(0, {"role": "system", "content": tool_prompt})

            for round_num in range(max_tool_rounds):
                # Generate response
                response = await self.generate_response(messages, context)

                # Check for tool calls
                tool_calls = self._parse_tool_calls(response)

                if not tool_calls:
                    # No tool calls, return final response
                    return response

                # Execute tool calls
                results = await self.execute_tool_calls(tool_calls)

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response,
                    "tool_calls": [
                        {"id": tc.id, "type": tc.type, "function": tc.function}
                        for tc in tool_calls
                    ],
                })

                # Add tool results
                for result in results:
                    messages.append(result.to_message())

            # Max rounds reached
            return await self.generate_response(messages, context)

    return FunctionCallingAgent


__all__ = [
    "ToolChoiceMode",
    "FunctionParameter",
    "FunctionDefinition",
    "ToolCall",
    "ToolResult",
    "tool",
    "FunctionCallingMixin",
    "create_function_calling_agent_class",
]
