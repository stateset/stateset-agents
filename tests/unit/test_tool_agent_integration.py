"""
ToolAgent integration tests using the stub backend.

Tests tool registration, tool-call detection, JSON parsing,
tool execution, and end-to-end generate_response with tools.
"""

import pytest

from stateset_agents.core.agent import AgentConfig, ToolAgent


def _make_tool(name, description="A test tool", func=None, params=None):
    """Helper to build a tool dict."""
    tool = {
        "name": name,
        "description": description,
        "parameters": params or {},
    }
    if func is not None:
        tool["function"] = func
    return tool


class TestToolAgentRegistration:
    """Test tool registration and management."""

    def test_tools_registered_at_init(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        tools = [_make_tool("add"), _make_tool("sub")]
        agent = ToolAgent(config, tools=tools)

        assert len(agent.tools) == 2
        assert "add" in agent.tool_registry
        assert "sub" in agent.tool_registry

    def test_add_tool_after_init(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config, tools=[])

        agent.add_tool(_make_tool("search"))
        assert len(agent.tools) == 1
        assert "search" in agent.tool_registry

    def test_no_tools_by_default(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config)

        assert agent.tools == []
        assert agent.tool_registry == {}


class TestShouldUseTools:
    """Test the _should_use_tools heuristic."""

    def test_no_tools_returns_false(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config)

        messages = [{"role": "user", "content": "calculate 2+2"}]
        assert not agent._should_use_tools(messages)

    def test_keyword_triggers_tools(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config, tools=[_make_tool("calc")])

        messages = [{"role": "user", "content": "calculate 2+2"}]
        assert agent._should_use_tools(messages)

    def test_no_keyword_skips_tools(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config, tools=[_make_tool("calc")])

        messages = [{"role": "user", "content": "hello world"}]
        assert not agent._should_use_tools(messages)

    def test_empty_messages_returns_false(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config, tools=[_make_tool("calc")])

        assert not agent._should_use_tools([])


class TestFormatToolDescriptions:
    """Test tool description formatting."""

    def test_single_tool_formatted(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(
            config,
            tools=[_make_tool("add", "Add two numbers", params={"type": "object"})],
        )

        desc = agent._format_tool_descriptions()
        assert '"name": "add"' in desc
        assert '"description": "Add two numbers"' in desc

    def test_multiple_tools_separated_by_newlines(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(
            config,
            tools=[_make_tool("add"), _make_tool("sub")],
        )

        desc = agent._format_tool_descriptions()
        lines = desc.strip().split("\n")
        assert len(lines) == 2


class TestProcessToolCalls:
    """Test JSON extraction and tool execution from response text."""

    @pytest.mark.asyncio
    async def test_json_block_parsed_and_executed(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)

        def add(a, b):
            return a + b

        agent = ToolAgent(config, tools=[_make_tool("add", func=add)])
        await agent.initialize()

        response = '```json\n{"tool": "add", "parameters": {"a": 2, "b": 3}}\n```'
        result = await agent._process_tool_calls(response)

        assert "Tool Output (add): 5" in result

    @pytest.mark.asyncio
    async def test_raw_json_parsed(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)

        def greet(name):
            return f"Hello {name}"

        agent = ToolAgent(config, tools=[_make_tool("greet", func=greet)])
        await agent.initialize()

        response = (
            'I will call the tool: {"tool": "greet", "parameters": {"name": "Alice"}}'
        )
        result = await agent._process_tool_calls(response)

        assert "Tool Output (greet): Hello Alice" in result

    @pytest.mark.asyncio
    async def test_unknown_tool_reports_error(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config, tools=[_make_tool("add")])
        await agent.initialize()

        response = '{"tool": "unknown", "parameters": {}}'
        result = await agent._process_tool_calls(response)

        assert "Error: Tool 'unknown' not found" in result

    @pytest.mark.asyncio
    async def test_no_json_returns_unchanged(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config, tools=[_make_tool("add")])
        await agent.initialize()

        response = "Just a normal response with no JSON"
        result = await agent._process_tool_calls(response)

        assert result == response

    @pytest.mark.asyncio
    async def test_invalid_json_returns_unchanged(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config, tools=[_make_tool("add")])
        await agent.initialize()

        response = '```json\n{invalid json with "tool" key}\n```'
        result = await agent._process_tool_calls(response)

        # Should fail silently on invalid JSON
        assert "Tool Output" not in result


class TestExecuteTool:
    """Test tool execution with different function types."""

    @pytest.mark.asyncio
    async def test_sync_function_executed(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)

        def multiply(x, y):
            return x * y

        agent = ToolAgent(config, tools=[_make_tool("mult", func=multiply)])
        result = await agent._execute_tool("mult", {"x": 3, "y": 4})
        assert result == "12"

    @pytest.mark.asyncio
    async def test_async_function_executed(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)

        async def async_add(a, b):
            return a + b

        agent = ToolAgent(config, tools=[_make_tool("add", func=async_add)])
        result = await agent._execute_tool("add", {"a": 10, "b": 20})
        assert result == "30"

    @pytest.mark.asyncio
    async def test_tool_without_function_returns_mock(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config, tools=[_make_tool("noop")])

        result = await agent._execute_tool("noop", {})
        assert "mock result" in result

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)
        agent = ToolAgent(config)

        result = await agent._execute_tool("nonexistent", {})
        assert "Error: Tool nonexistent not found" in result

    @pytest.mark.asyncio
    async def test_function_error_handled_gracefully(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)

        def failing():
            raise ValueError("boom")

        agent = ToolAgent(config, tools=[_make_tool("fail", func=failing)])
        result = await agent._execute_tool("fail", {})
        assert "Error executing fail" in result


class TestToolAgentEndToEnd:
    """Test full ToolAgent.generate_response with tools."""

    @pytest.mark.asyncio
    async def test_non_tool_message_uses_standard_generation(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)

        agent = ToolAgent(config, tools=[_make_tool("search")])
        await agent.initialize()

        # ToolAgent.generate_response expects list-of-dicts (not bare string)
        response = await agent.generate_response(
            [{"role": "user", "content": "hello there"}]
        )
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_tool_keyword_triggers_tool_path(self):
        config = AgentConfig(model_name="stub://tools", use_stub_model=True)

        def search(query=""):
            return f"Results for: {query}"

        agent = ToolAgent(
            config,
            tools=[_make_tool("search", func=search)],
        )
        await agent.initialize()

        # "search" keyword triggers tool usage
        response = await agent.generate_response(
            [{"role": "user", "content": "search for cats"}]
        )
        assert len(response) > 0
