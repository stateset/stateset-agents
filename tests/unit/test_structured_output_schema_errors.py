"""Regression tests for plain JSON-schema validation in structured output."""

from __future__ import annotations

from typing import Any

import pytest

from stateset_agents.core.structured_output import (
    SchemaValidationError,
    StructuredOutputConfig,
    StructuredOutputError,
    StructuredOutputMixin,
)

SCHEMA = {
    "type": "object",
    "required": ["count"],
    "properties": {"count": {"type": "integer"}},
}


class _ScriptedAgent(StructuredOutputMixin):
    """Replays canned model outputs and records the prompts it received."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[list[dict[str, str]]] = []

    async def generate_response(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> str:
        assert isinstance(messages, list)
        self.prompts.append([dict(message) for message in messages])
        return self.outputs.pop(0)


def test_schema_mismatch_raises_schema_validation_error() -> None:
    # pydantic's ValidationError cannot be built from a message string, so the
    # validator used to raise TypeError instead of a validation error.
    agent = _ScriptedAgent([])
    with pytest.raises(SchemaValidationError, match="Expected integer"):
        agent._validate_json_schema({"count": "three"}, SCHEMA)
    with pytest.raises(SchemaValidationError, match="Missing required field"):
        agent._validate_json_schema({}, SCHEMA)
    assert issubclass(SchemaValidationError, ValueError)


@pytest.mark.asyncio
async def test_schema_mismatch_feeds_schema_error_back_for_retry() -> None:
    agent = _ScriptedAgent(['{"count": "three"}', '{"count": 3}'])

    result = await agent.generate_structured("How many?", json_schema=SCHEMA)

    assert result == {"count": 3}
    retry_prompt = agent.prompts[1][-1]
    assert retry_prompt["role"] == "user"
    assert "did not match the required schema" in retry_prompt["content"]
    assert "Expected integer" in retry_prompt["content"]


@pytest.mark.asyncio
async def test_persistent_schema_mismatch_reports_schema_error() -> None:
    agent = _ScriptedAgent(['{"count": "x"}'] * 2)

    with pytest.raises(StructuredOutputError) as excinfo:
        await agent.generate_structured(
            "How many?",
            json_schema=SCHEMA,
            config=StructuredOutputConfig(max_retries=2),
        )

    assert isinstance(excinfo.value.last_error, SchemaValidationError)
