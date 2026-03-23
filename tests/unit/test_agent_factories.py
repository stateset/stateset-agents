"""
Tests for agent factory functions and configuration loading.

Tests create_agent, _load_agent_configs, AGENT_CONFIGS, and
agent property/state methods.
"""

import pytest

from stateset_agents.core.agent import (
    AGENT_CONFIGS,
    AgentConfig,
    MultiTurnAgent,
    _load_agent_configs,
    create_agent,
)


class TestLoadAgentConfigs:
    """Test _load_agent_configs returns expected presets."""

    def test_returns_dict(self):
        configs = _load_agent_configs()
        assert isinstance(configs, dict)

    def test_contains_fallback_presets(self):
        configs = _load_agent_configs()
        for name in [
            "helpful_assistant",
            "customer_service",
            "tutor",
            "creative_writer",
        ]:
            assert name in configs, f"Missing preset: {name}"

    def test_presets_have_expected_keys(self):
        configs = _load_agent_configs()
        for name, config in configs.items():
            assert "system_prompt" in config, f"{name} missing system_prompt"
            assert "temperature" in config, f"{name} missing temperature"

    def test_agent_configs_global_matches_loader(self):
        assert isinstance(AGENT_CONFIGS, dict)
        assert len(AGENT_CONFIGS) >= 4


class TestCreateAgent:
    """Test the create_agent factory function."""

    def test_creates_multi_turn_agent(self):
        agent = create_agent(
            agent_type="multi_turn",
            model_name="stub://factory",
            use_stub_model=True,
        )
        assert isinstance(agent, MultiTurnAgent)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent(
                agent_type="nonexistent", model_name="stub://x", use_stub_model=True
            )


class TestAgentConfigProperties:
    """Test AgentConfig default values and properties."""

    def test_default_config(self):
        config = AgentConfig(model_name="test")
        assert config.model_name == "test"
        assert config.temperature >= 0
        assert config.max_new_tokens > 0

    def test_stub_model_flag(self):
        config = AgentConfig(model_name="stub://test", use_stub_model=True)
        assert config.use_stub_model is True

    def test_custom_system_prompt(self):
        config = AgentConfig(model_name="test", system_prompt="You are a pirate.")
        assert config.system_prompt == "You are a pirate."


class TestMultiTurnAgentState:
    """Test MultiTurnAgent state management methods."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        config = AgentConfig(model_name="stub://state", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        await agent.generate_response("Hello")
        assert agent.turn_count == 1
        assert len(agent.conversation_history) > 0

        await agent.reset()
        assert agent.turn_count == 0
        assert len(agent.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_is_stub_backend_property(self):
        config = AgentConfig(model_name="stub://prop", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        assert agent._is_stub_backend is True

    @pytest.mark.asyncio
    async def test_effective_max_input_length(self):
        config = AgentConfig(
            model_name="stub://len",
            use_stub_model=True,
            max_new_tokens=64,
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        length = agent._effective_max_input_length()
        assert length > 0

    @pytest.mark.asyncio
    async def test_clean_response_strips_system_artifacts(self):
        config = AgentConfig(model_name="stub://clean", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        # _clean_response strips stop phrases like "User:"
        cleaned = agent._clean_response("Hello world\nUser: something")
        assert "User:" not in cleaned
        assert cleaned == "Hello world"


class TestMultiTurnAgentGeneration:
    """Test generation edge cases."""

    @pytest.mark.asyncio
    async def test_generate_with_empty_string(self):
        config = AgentConfig(model_name="stub://empty", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        response = await agent.generate_response("")
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        config = AgentConfig(model_name="stub://ctx", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        response = await agent.generate_response("Hello", context={"metadata": "test"})
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_generate_before_init_raises(self):
        config = AgentConfig(model_name="stub://noinit", use_stub_model=True)
        agent = MultiTurnAgent(config)

        with pytest.raises(RuntimeError, match="initialized"):
            await agent.generate_response("Hello")

    @pytest.mark.asyncio
    async def test_multiple_conversations_independent(self):
        config = AgentConfig(model_name="stub://multi", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        await agent.generate_response("First conversation")
        assert agent.turn_count == 1

        await agent.reset()
        assert agent.turn_count == 0

        await agent.generate_response("Second conversation")
        assert agent.turn_count == 1
