import pytest

from stateset_agents.core.agent import AgentConfig, ToolAgent


@pytest.mark.asyncio
async def test_tool_agent_executes_sync_tool_from_json_block():
    def add(a: int, b: int) -> int:
        return a + b

    config = AgentConfig(model_name="stub://tool-sync", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {"name": "add", "function": add, "parameters": {"a": "int", "b": "int"}}
        ],
    )
    await agent.initialize()

    response = (
        "Sure, I'll calculate that.\n"
        "```json\n"
        '{"tool": "add", "parameters": {"a": 1, "b": 2}}\n'
        "```"
    )

    result = await agent._process_tool_calls(response)

    assert "Tool Output (add): 3" in result


@pytest.mark.asyncio
async def test_tool_agent_executes_async_tool_from_raw_json():
    async def echo(message: str) -> str:
        return f"echo:{message}"

    config = AgentConfig(model_name="stub://tool-async", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[{"name": "echo", "function": echo, "parameters": {"message": "str"}}],
    )
    await agent.initialize()

    response = '{"tool": "echo", "parameters": {"message": "hi"}}'
    result = await agent._process_tool_calls(response)

    assert "Tool Output (echo): echo:hi" in result


@pytest.mark.asyncio
async def test_tool_agent_handles_unknown_tool():
    config = AgentConfig(model_name="stub://tool-missing", use_stub_model=True)
    agent = ToolAgent(config, tools=[])
    await agent.initialize()

    response = '{"tool": "missing", "parameters": {}}'
    result = await agent._process_tool_calls(response)

    assert "Error: Tool 'missing' not found." in result


@pytest.mark.asyncio
async def test_tool_agent_ignores_invalid_json():
    config = AgentConfig(model_name="stub://tool-bad", use_stub_model=True)
    agent = ToolAgent(config, tools=[])
    await agent.initialize()

    response = "```json\n{not: valid json}\n```"
    result = await agent._process_tool_calls(response)

    assert result == response


@pytest.mark.asyncio
async def test_tool_agent_processes_tool_call_from_any_fenced_json_block():
    def multiply(a: int, b: int) -> int:
        return a * b

    config = AgentConfig(model_name="stub://tool-multi", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "multiply",
                "function": multiply,
                "parameters": {"a": "int", "b": "int"},
            }
        ],
    )
    await agent.initialize()

    response = (
        "I'll compute this.\n"
        "```json\n"
        '{"tool": "noop", "parameters": {"x": 1}}\n'
        "```\n"
        "```json\n"
        '{"tool": "multiply", "parameters": {"a": 7, "b": 6}}\n'
        "```"
    )

    result = await agent._process_tool_calls(response)

    assert "Tool Output (multiply): 42" in result


@pytest.mark.asyncio
async def test_tool_agent_processes_later_valid_tool_when_earlier_tool_is_unknown():
    def multiply(a: int, b: int) -> int:
        return a * b

    config = AgentConfig(model_name="stub://tool-unknown-first", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "multiply",
                "function": multiply,
                "parameters": {"a": "int", "b": "int"},
            }
        ],
    )
    await agent.initialize()

    response = (
        "```json\n"
        '{"tool": "not_registered", "parameters": {"x": 1}}\n'
        "```\n"
        "```json\n"
        '{"tool": "multiply", "parameters": {"a": 3, "b": 4}}\n'
        "```"
    )

    result = await agent._process_tool_calls(response)

    assert "Tool Output (multiply): 12" in result
    assert "not_registered" not in result


@pytest.mark.asyncio
async def test_tool_agent_ignores_invalid_tool_payload_shape():
    config = AgentConfig(model_name="stub://tool-shape", use_stub_model=True)

    def add(a: int, b: int) -> int:
        return a + b

    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "add",
                "function": add,
                "parameters": {"a": "int", "b": "int"},
            }
        ],
    )
    await agent.initialize()

    response = '{"tool": "add", "parameters": [1, 2]}'
    result = await agent._process_tool_calls(response)

    assert result == response


@pytest.mark.asyncio
async def test_tool_agent_ignores_trailing_inline_json():
    config = AgentConfig(model_name="stub://tool-inline", use_stub_model=True)
    agent = ToolAgent(config, tools=[])
    await agent.initialize()

    response = '{"tool": "add", "parameters": {"a": 1, "b": 2}} trailing text'
    result = await agent._process_tool_calls(response)

    assert result == response


@pytest.mark.asyncio
async def test_tool_agent_falls_back_to_context_for_context_tools():
    def context_only(context):
        if context is None:
            return "no-context"
        return f"context-keys={sorted(context.keys())}"

    config = AgentConfig(model_name="stub://tool-context", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "context_only",
                "function": context_only,
                "parameters": {"x": "int"},
            }
        ],
    )
    await agent.initialize()

    response = '{"tool": "context_only", "parameters": {"x": 1}}'
    result = await agent._process_tool_calls(response)

    assert "Tool Output (context_only): no-context" in result


@pytest.mark.asyncio
async def test_tool_agent_supports_positional_only_context_alias():
    def context_only(request_context):
        return f"context={request_context is not None}"

    config = AgentConfig(model_name="stub://tool-pos-ctx", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "context_only",
                "function": context_only,
                "parameters": {"x": "int"},
            }
        ],
    )
    await agent.initialize()

    response = '{"tool": "context_only", "parameters": {"x": 1}}'
    result = await agent._process_tool_calls(response)

    assert "Tool Output (context_only): context=True" in result


@pytest.mark.asyncio
async def test_tool_agent_supports_positional_only_parameters():
    def add_positional(a: int, b: int, /) -> int:
        return a + b

    config = AgentConfig(model_name="stub://tool-positional", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {"name": "add", "function": add_positional, "parameters": {"a": "int", "b": "int"}}
        ],
    )
    await agent.initialize()

    response = '{"tool": "add", "parameters": {"a": 4, "b": 6}}'
    result = await agent._process_tool_calls(response)

    assert "Tool Output (add): 10" in result


@pytest.mark.asyncio
async def test_tool_agent_supports_context_tool_with_optional_parameters():
    def context_tool(context, prefix: str = "[ctx]"):
        return f"{prefix}:{context is not None}"

    config = AgentConfig(model_name="stub://tool-ctx-optional", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "context_tool",
                "function": context_tool,
                "parameters": {"prefix": "str"},
            }
        ],
    )
    await agent.initialize()

    response = '{"tool": "context_tool", "parameters": {"prefix": "ok"}}'
    result = await agent._process_tool_calls(response)

    assert "Tool Output (context_tool): ok:True" in result


@pytest.mark.asyncio
async def test_tool_agent_supports_positional_only_context_parameter():
    def context_tool(context, /, prefix: str = "[ctx]"):
        return f"{prefix}:{context is not None}"

    config = AgentConfig(model_name="stub://tool-ctx-pos-only", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "context_tool",
                "function": context_tool,
                "parameters": {"prefix": "str"},
            }
        ],
    )
    await agent.initialize()

    response = '{"tool": "context_tool", "parameters": {"prefix": "ok"}}'
    result = await agent._process_tool_calls(response)

    assert "Tool Output (context_tool): ok:True" in result


@pytest.mark.asyncio
async def test_tool_agent_supports_keyword_only_context_parameter():
    def context_tool(*, request_context, prefix: str = "[ctx]"):
        return f"{prefix}:{request_context is not None}"

    config = AgentConfig(model_name="stub://tool-ctx-kw-only", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "context_tool",
                "function": context_tool,
                "parameters": {"prefix": "str"},
            }
        ],
    )
    await agent.initialize()

    response = '{"tool": "context_tool", "parameters": {"prefix": "ok"}}'
    result = await agent._process_tool_calls(response)

    assert "Tool Output (context_tool): ok:True" in result


@pytest.mark.asyncio
async def test_tool_agent_context_tool_runtime_type_error_is_returned():
    def context_tool(context):
        raise TypeError("bad args")

    config = AgentConfig(model_name="stub://tool-ctx-runtime", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "context_tool",
                "function": context_tool,
                "parameters": {},
            }
        ],
    )
    await agent.initialize()

    response = '{"tool": "context_tool", "parameters": {}}'
    result = await agent._process_tool_calls(response)

    assert "Error executing context_tool: bad args" in result


@pytest.mark.asyncio
async def test_tool_agent_does_not_swallow_type_errors_for_non_context_tools():
    def needs_context_and_value(context, value):
        return f"{context}:{value}"

    config = AgentConfig(model_name="stub://tool-type-error", use_stub_model=True)
    agent = ToolAgent(
        config,
        tools=[
            {
                "name": "needs_context_and_value",
                "function": needs_context_and_value,
                "parameters": {"value": "int"},
            }
        ],
    )
    await agent.initialize()

    response = '{"tool": "needs_context_and_value", "parameters": {"value": 1}}'
    result = await agent._process_tool_calls(response)

    assert "Error executing needs_context_and_value" in result
