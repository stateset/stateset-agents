"""
Stub-backend integration tests for agent functionality.

These tests exercise real code paths (memory window, streaming, planning)
using the stub backend — no MagicMock patching required.
"""

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent


class TestMemoryWindow:
    """Test _apply_memory_window truncation logic."""

    @pytest.mark.asyncio
    async def test_memory_window_truncates_old_messages(self):
        """Messages beyond the window are dropped."""
        config = AgentConfig(
            model_name="stub://memory",
            use_stub_model=True,
        )
        agent = MultiTurnAgent(config, memory_window=4)
        await agent.initialize()

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Reply 3"},
            {"role": "user", "content": "Message 4"},
        ]

        windowed = agent._apply_memory_window(messages)

        # System message always kept
        system_msgs = [m for m in windowed if m["role"] == "system"]
        assert len(system_msgs) == 1

        # Non-system messages should be truncated to last 4
        non_system = [m for m in windowed if m["role"] != "system"]
        assert len(non_system) == 4
        assert non_system[0]["content"] == "Reply 2"
        assert non_system[-1]["content"] == "Message 4"

    @pytest.mark.asyncio
    async def test_memory_window_zero_keeps_all(self):
        """Window of 0 means no truncation."""
        config = AgentConfig(
            model_name="stub://memory",
            use_stub_model=True,
        )
        agent = MultiTurnAgent(config, memory_window=0)
        await agent.initialize()

        messages = [{"role": "user", "content": f"Msg {i}"} for i in range(10)]
        windowed = agent._apply_memory_window(messages)
        assert len(windowed) == 10


class TestStreamingGeneration:
    """Test streaming with the stub backend."""

    @pytest.mark.asyncio
    async def test_streaming_yields_all_words(self):
        """Streaming should yield all words from the stub response."""
        config = AgentConfig(
            model_name="stub://stream",
            use_stub_model=True,
            stub_responses=["Hello world from stub"],
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        chunks = []
        async for chunk in agent.generate_response_stream("Test"):
            chunks.append(chunk)

        reassembled = "".join(chunks)
        assert "Hello" in reassembled
        assert "world" in reassembled
        assert "stub" in reassembled

    @pytest.mark.asyncio
    async def test_streaming_updates_turn_count(self):
        """Turn count should increment after streaming."""
        config = AgentConfig(
            model_name="stub://stream",
            use_stub_model=True,
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        assert agent.turn_count == 0
        async for _ in agent.generate_response_stream("Test"):
            pass
        assert agent.turn_count == 1


class TestConversationHistory:
    """Test conversation history management."""

    @pytest.mark.asyncio
    async def test_system_prompt_injected(self):
        """System prompt should be part of the conversation context."""
        config = AgentConfig(
            model_name="stub://sys",
            use_stub_model=True,
            system_prompt="You are a pirate.",
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        response = await agent.generate_response("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_conversation_history_cleared_independently(self):
        """Each agent instance should have independent history."""
        config = AgentConfig(
            model_name="stub://independent",
            use_stub_model=True,
        )
        agent1 = MultiTurnAgent(config)
        agent2 = MultiTurnAgent(config)
        await agent1.initialize()
        await agent2.initialize()

        await agent1.generate_response("Hello from agent 1")
        assert len(agent1.conversation_history) == 1
        assert len(agent2.conversation_history) == 0


class TestPlanningIntegration:
    """Test planning manager integration."""

    @pytest.mark.asyncio
    async def test_planning_manager_created(self):
        """PlanningManager should be created when enabled."""
        config = AgentConfig(
            model_name="stub://planning",
            use_stub_model=True,
            enable_planning=True,
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        assert agent.planning_manager is not None
        assert hasattr(agent.planning_manager, "get_plan")

    @pytest.mark.asyncio
    async def test_agent_works_without_planning(self):
        """Agent should work fine with planning disabled."""
        config = AgentConfig(
            model_name="stub://no-planning",
            use_stub_model=True,
            enable_planning=False,
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        assert agent.planning_manager is None
        response = await agent.generate_response("Hello")
        assert isinstance(response, str)
