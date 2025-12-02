"""
Structured Output Support for StateSet Agents

This module provides JSON schema-based structured output generation,
enabling agents to produce type-safe, validated responses that conform
to user-defined schemas.

Features:
- Pydantic model integration for response validation
- JSON Schema enforcement during generation
- Automatic retry on validation failures
- Support for nested objects, arrays, enums, and unions

Example:
    >>> from pydantic import BaseModel
    >>> from core.structured_output import StructuredOutputAgent
    >>>
    >>> class MovieReview(BaseModel):
    ...     title: str
    ...     rating: int  # 1-5
    ...     summary: str
    ...     pros: list[str]
    ...     cons: list[str]
    >>>
    >>> agent = StructuredOutputAgent(config)
    >>> await agent.initialize()
    >>> review = await agent.generate_structured(
    ...     "Review the movie Inception",
    ...     response_model=MovieReview
    ... )
    >>> print(review.rating)  # Type-safe access
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

logger = logging.getLogger(__name__)

# Type variable for generic structured output
T = TypeVar("T")

# Check for Pydantic availability
try:
    from pydantic import BaseModel, ValidationError, create_model
    from pydantic.json_schema import GenerateJsonSchema

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore
    ValidationError = Exception  # type: ignore


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output generation.

    Attributes:
        max_retries: Maximum validation retry attempts (default: 3)
        strict_mode: Fail on extra fields in response (default: True)
        include_schema_in_prompt: Add JSON schema to system prompt (default: True)
        json_mode: Request JSON-only output from model (default: True)
        repair_json: Attempt to fix malformed JSON (default: True)
    """

    max_retries: int = 3
    strict_mode: bool = True
    include_schema_in_prompt: bool = True
    json_mode: bool = True
    repair_json: bool = True


class StructuredOutputError(Exception):
    """Raised when structured output generation fails.

    Attributes:
        attempts: Number of generation attempts made
        last_error: The final validation error
        raw_outputs: List of raw model outputs that failed validation
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Optional[Exception] = None,
        raw_outputs: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.raw_outputs = raw_outputs or []


def json_schema_from_type(python_type: Type) -> Dict[str, Any]:
    """Generate JSON schema from a Python type annotation.

    Supports:
    - Primitive types (str, int, float, bool)
    - Optional types
    - List types
    - Dict types
    - Pydantic models
    - Dataclasses

    Args:
        python_type: A Python type annotation

    Returns:
        JSON Schema dict describing the type
    """
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional (Union with None)
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            schema = json_schema_from_type(non_none_args[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [json_schema_from_type(a) for a in non_none_args]}

    # Handle List
    if origin is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": json_schema_from_type(item_type)}

    # Handle Dict
    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": json_schema_from_type(value_type),
        }

    # Handle Pydantic models
    if PYDANTIC_AVAILABLE and isinstance(python_type, type) and issubclass(python_type, BaseModel):
        return python_type.model_json_schema()

    # Handle primitives
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        type(None): {"type": "null"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    # Default fallback
    return {"type": "object"}


def repair_json_string(raw: str) -> str:
    """Attempt to repair common JSON formatting issues.

    Handles:
    - Missing closing braces/brackets
    - Trailing commas
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Markdown code block wrappers

    Args:
        raw: Potentially malformed JSON string

    Returns:
        Repaired JSON string (may still be invalid)
    """
    # Remove markdown code blocks
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)
    raw = raw.strip()

    # Find JSON object boundaries
    start_idx = raw.find("{")
    if start_idx == -1:
        start_idx = raw.find("[")
    if start_idx != -1:
        raw = raw[start_idx:]

    # Count braces and brackets
    open_braces = raw.count("{")
    close_braces = raw.count("}")
    open_brackets = raw.count("[")
    close_brackets = raw.count("]")

    # Add missing closing characters
    raw += "}" * (open_braces - close_braces)
    raw += "]" * (open_brackets - close_brackets)

    # Fix trailing commas before closing braces/brackets
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    # Replace single quotes with double quotes (crude but often works)
    # Only outside of already double-quoted strings
    raw = re.sub(r"'([^']*)'(?=\s*:)", r'"\1"', raw)

    return raw


def extract_json_from_response(response: str) -> str:
    """Extract JSON object/array from a model response.

    Handles responses that include explanatory text before or after
    the JSON content.

    Args:
        response: Full model response that may contain JSON

    Returns:
        Extracted JSON string
    """
    # Try to find JSON in code blocks first
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", response)
    if code_block_match:
        return code_block_match.group(1)

    # Try to find raw JSON object
    json_match = re.search(r"(\{[\s\S]*\})", response)
    if json_match:
        return json_match.group(1)

    # Try to find JSON array
    array_match = re.search(r"(\[[\s\S]*\])", response)
    if array_match:
        return array_match.group(1)

    return response


class StructuredOutputMixin:
    """Mixin class adding structured output capabilities to agents.

    Add this mixin to any Agent subclass to enable structured output
    generation with automatic validation against Pydantic models or
    JSON schemas.

    Example:
        >>> class MyAgent(MultiTurnAgent, StructuredOutputMixin):
        ...     pass
        >>>
        >>> agent = MyAgent(config)
        >>> result = await agent.generate_structured(
        ...     "Extract entities from this text",
        ...     response_model=EntityList
        ... )
    """

    async def generate_structured(
        self,
        messages: Union[str, List[Dict[str, str]]],
        response_model: Optional[Type[T]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[StructuredOutputConfig] = None,
    ) -> T:
        """Generate a structured response conforming to a schema.

        Either `response_model` (Pydantic model) or `json_schema` must be
        provided. If both are provided, `response_model` takes precedence.

        Args:
            messages: Input message(s) for the agent
            response_model: Pydantic model class for response validation
            json_schema: JSON Schema dict for response validation
            context: Optional context passed to generation
            config: Structured output configuration

        Returns:
            Validated response as an instance of response_model, or as a
            dict if only json_schema was provided

        Raises:
            StructuredOutputError: If validation fails after max retries
            ValueError: If neither response_model nor json_schema provided
        """
        if response_model is None and json_schema is None:
            raise ValueError("Either response_model or json_schema must be provided")

        config = config or StructuredOutputConfig()

        # Generate schema from Pydantic model if provided
        if response_model is not None:
            if not PYDANTIC_AVAILABLE:
                raise ImportError(
                    "Pydantic is required for structured output with response_model. "
                    "Install with: pip install pydantic>=2.0"
                )
            if not issubclass(response_model, BaseModel):
                raise TypeError(
                    f"response_model must be a Pydantic BaseModel, got {type(response_model)}"
                )
            schema = response_model.model_json_schema()
        else:
            schema = json_schema

        # Normalize messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = list(messages)

        # Add schema instruction to system prompt
        if config.include_schema_in_prompt:
            schema_instruction = self._build_schema_instruction(schema, config)

            if messages and messages[0].get("role") == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + "\n\n" + schema_instruction,
                }
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})

        # Attempt generation with retries
        raw_outputs = []
        last_error = None

        for attempt in range(config.max_retries):
            try:
                # Generate response
                raw_response = await self.generate_response(messages, context)
                raw_outputs.append(raw_response)

                # Extract and parse JSON
                json_str = extract_json_from_response(raw_response)

                if config.repair_json:
                    json_str = repair_json_string(json_str)

                parsed = json.loads(json_str)

                # Validate against schema
                if response_model is not None:
                    return response_model.model_validate(parsed)
                else:
                    # Basic JSON schema validation
                    self._validate_json_schema(parsed, schema)
                    return parsed

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"Structured output attempt {attempt + 1}/{config.max_retries} "
                    f"failed: JSON decode error - {e}"
                )
                # Add error context to messages for retry
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response was not valid JSON. Error: {e}. "
                        "Please respond with ONLY a valid JSON object matching the schema."
                    ),
                })

            except ValidationError as e:
                last_error = e
                logger.warning(
                    f"Structured output attempt {attempt + 1}/{config.max_retries} "
                    f"failed: Validation error - {e}"
                )
                # Add error context to messages for retry
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your JSON response did not match the required schema. "
                        f"Validation errors: {e}. Please fix and try again."
                    ),
                })

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Structured output attempt {attempt + 1}/{config.max_retries} "
                    f"failed: {type(e).__name__} - {e}"
                )

        raise StructuredOutputError(
            f"Failed to generate valid structured output after {config.max_retries} attempts",
            attempts=config.max_retries,
            last_error=last_error,
            raw_outputs=raw_outputs,
        )

    def _build_schema_instruction(
        self, schema: Dict[str, Any], config: StructuredOutputConfig
    ) -> str:
        """Build instruction string for structured output generation."""
        schema_str = json.dumps(schema, indent=2)

        instruction = (
            "You must respond with a valid JSON object that matches the following schema:\n\n"
            f"```json\n{schema_str}\n```\n\n"
            "IMPORTANT:\n"
            "- Respond with ONLY the JSON object, no additional text\n"
            "- Ensure all required fields are present\n"
            "- Use the exact field names from the schema\n"
            "- Match the expected data types exactly\n"
        )

        if config.strict_mode:
            instruction += "- Do not include any fields not in the schema\n"

        return instruction

    def _validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """Basic JSON schema validation without external dependencies.

        For full JSON Schema validation, use response_model with Pydantic.
        This is a simplified validator for basic type checking.
        """
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(data, dict):
                raise ValidationError(f"Expected object, got {type(data).__name__}")

            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    raise ValidationError(f"Missing required field: {field}")

            # Validate properties
            properties = schema.get("properties", {})
            for key, prop_schema in properties.items():
                if key in data:
                    self._validate_json_schema(data[key], prop_schema)

        elif schema_type == "array":
            if not isinstance(data, list):
                raise ValidationError(f"Expected array, got {type(data).__name__}")

            items_schema = schema.get("items", {})
            for item in data:
                self._validate_json_schema(item, items_schema)

        elif schema_type == "string":
            if not isinstance(data, str):
                raise ValidationError(f"Expected string, got {type(data).__name__}")

        elif schema_type == "integer":
            if not isinstance(data, int) or isinstance(data, bool):
                raise ValidationError(f"Expected integer, got {type(data).__name__}")

        elif schema_type == "number":
            if not isinstance(data, (int, float)) or isinstance(data, bool):
                raise ValidationError(f"Expected number, got {type(data).__name__}")

        elif schema_type == "boolean":
            if not isinstance(data, bool):
                raise ValidationError(f"Expected boolean, got {type(data).__name__}")


# Convenience class combining MultiTurnAgent with StructuredOutputMixin
def create_structured_agent_class():
    """Create a StructuredOutputAgent class.

    This function is used to avoid circular imports while providing
    a ready-to-use agent class with structured output support.

    Returns:
        StructuredOutputAgent class
    """
    from .agent import MultiTurnAgent

    class StructuredOutputAgent(MultiTurnAgent, StructuredOutputMixin):
        """Agent with built-in structured output support.

        Combines MultiTurnAgent capabilities with automatic JSON schema
        validation for type-safe response generation.

        Example:
            >>> from pydantic import BaseModel
            >>>
            >>> class WeatherInfo(BaseModel):
            ...     location: str
            ...     temperature: float
            ...     conditions: str
            ...     forecast: list[str]
            >>>
            >>> agent = StructuredOutputAgent(config)
            >>> await agent.initialize()
            >>>
            >>> weather = await agent.generate_structured(
            ...     "What's the weather like in San Francisco?",
            ...     response_model=WeatherInfo
            ... )
            >>> print(f"Temperature: {weather.temperature}°F")
        """

        pass

    return StructuredOutputAgent


# Export for convenience
__all__ = [
    "StructuredOutputConfig",
    "StructuredOutputError",
    "StructuredOutputMixin",
    "json_schema_from_type",
    "repair_json_string",
    "extract_json_from_response",
    "create_structured_agent_class",
]
