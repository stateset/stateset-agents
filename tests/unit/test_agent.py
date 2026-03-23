"""
Unit tests for core agent functionality.

These tests focus on individual components in isolation.
"""

import pytest

from stateset_agents.core.agent import (
    Agent,
    AgentConfig,
    ConfigValidationError,
    MultiTurnAgent,
)
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory


class TestAgentConfig:
    """Test AgentConfig functionality."""

    def test_agent_config_creation(self):
        """Test creating an AgentConfig with default values."""
        config = AgentConfig(model_name="gpt2")

        assert config.model_name == "gpt2"
        assert config.max_new_tokens == 512
        assert config.temperature == 0.8
        assert config.system_prompt is None

    def test_agent_config_custom_values(self):
        """Test creating an AgentConfig with custom values."""
        config = AgentConfig(
            model_name="gpt2",
            max_new_tokens=256,
            temperature=0.5,
            system_prompt="You are a helpful assistant.",
        )

        assert config.model_name == "gpt2"
        assert config.max_new_tokens == 256
        assert config.temperature == 0.5
        assert config.system_prompt == "You are a helpful assistant."

    def test_agent_config_planning_validation(self):
        """Test planning config validation."""
        with pytest.raises(ConfigValidationError):
            AgentConfig(model_name="gpt2", enable_planning=True, planning_config="nope")


class TestAgent:
    """Test Agent base class functionality."""

    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(model_name="gpt2", max_new_tokens=50, temperature=0.7)

    def test_agent_initialization_deferred(self, agent_config):
        """Test that Agent defers model loading until initialize() is called."""
        agent = Agent(agent_config)
        assert agent.model is None
        assert agent.tokenizer is None

    @pytest.mark.asyncio
    async def test_agent_stub_initialization(self):
        """Test Agent initializes correctly with the stub backend."""
        config = AgentConfig(
            model_name="stub://test",
            use_stub_model=True,
            max_new_tokens=50,
        )
        agent = Agent(config)
        await agent.initialize()

        assert agent.model is not None
        assert agent.tokenizer is not None
        assert agent.generation_config is not None

    @pytest.mark.asyncio
    async def test_agent_generate_response_not_implemented(self, agent_config):
        """Test that base Agent raises NotImplementedError for generate_response."""
        agent = Agent(agent_config)

        with pytest.raises(NotImplementedError):
            await agent.generate_response([{"role": "user", "content": "Hello"}])


class TestMultiTurnAgent:
    """Test MultiTurnAgent functionality using the real stub backend."""

    @pytest.fixture
    def config(self):
        return AgentConfig(
            model_name="stub://test",
            use_stub_model=True,
            max_new_tokens=100,
            temperature=0.8,
            use_chat_template=True,
        )

    @pytest.mark.asyncio
    async def test_initialization_sets_up_model(self, config):
        """Test that initialization creates a usable model and tokenizer."""
        agent = MultiTurnAgent(config)
        await agent.initialize()

        assert agent.model is not None
        assert agent.tokenizer is not None
        assert len(agent.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_generate_response_returns_string_and_updates_history(self, config):
        """Test real response generation with the stub backend."""
        agent = MultiTurnAgent(config)
        await agent.initialize()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        response = await agent.generate_response(messages)

        assert len(response) > 0
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["role"] == "assistant"
        assert agent.conversation_history[0]["content"] == response

    @pytest.mark.asyncio
    async def test_stub_backend_uses_configured_responses(self):
        """Test that custom stub responses are actually returned."""
        config = AgentConfig(
            model_name="stub://unit-test",
            use_stub_model=True,
            stub_responses=["Custom response alpha"],
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        response = await agent.generate_response("Ping")
        assert "Custom response alpha" in response

    @pytest.mark.asyncio
    async def test_planning_manager_initialized_when_enabled(self):
        """Test that planning manager is created when planning is enabled."""
        config = AgentConfig(
            model_name="stub://planning",
            use_stub_model=True,
            enable_planning=True,
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        assert agent.planning_manager is not None

    @pytest.mark.asyncio
    async def test_string_prompt_converted_to_messages(self, config):
        """Test that a bare string prompt is auto-wrapped into messages."""
        agent = MultiTurnAgent(config)
        await agent.initialize()

        response = await agent.generate_response("Hello there")
        assert len(response) > 0
        assert agent.conversation_history[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_accumulates_history(self, config):
        """Test that successive calls build up conversation history."""
        agent = MultiTurnAgent(config)
        await agent.initialize()

        await agent.generate_response("First message")
        await agent.generate_response("Second message")

        assert len(agent.conversation_history) == 2
        assert agent.turn_count == 2

    @pytest.mark.asyncio
    async def test_streaming_updates_history(self):
        """Test that streaming generation updates conversation history."""
        config = AgentConfig(
            model_name="stub://stream",
            use_stub_model=True,
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        tokens = []
        async for token in agent.generate_response_stream("Hello"):
            tokens.append(token)

        assert len(tokens) > 0
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["role"] == "assistant"
        assert agent.turn_count == 1
        # Reassembled text should match what was streamed
        streamed_text = "".join(tokens)
        assert len(streamed_text) > 0


class TestConversationTurn:
    """Test ConversationTurn functionality."""

    def test_conversation_turn_creation(self):
        """Test creating a ConversationTurn."""
        turn = ConversationTurn(
            user_message="Hello",
            assistant_response="Hi there!",
            reward=0.8,
            metadata={"turn_number": 1},
        )

        assert turn.user_message == "Hello"
        assert turn.assistant_response == "Hi there!"
        assert turn.reward == 0.8
        assert turn.metadata["turn_number"] == 1
        assert turn.timestamp is not None

    def test_conversation_turn_to_dict(self):
        """Test converting ConversationTurn to dictionary."""
        turn = ConversationTurn(
            user_message="Test", assistant_response="Response", reward=0.9
        )

        turn_dict = turn.to_dict()

        assert turn_dict["user_message"] == "Test"
        assert turn_dict["assistant_response"] == "Response"
        assert turn_dict["reward"] == 0.9
        assert "timestamp" in turn_dict


class TestMultiTurnTrajectory:
    """Test MultiTurnTrajectory functionality."""

    def test_trajectory_creation(self):
        """Test creating a MultiTurnTrajectory."""
        trajectory = MultiTurnTrajectory(trajectory_id="test_123")

        assert trajectory.trajectory_id == "test_123"
        assert len(trajectory.turns) == 0
        assert trajectory.total_reward == 0.0

    def test_trajectory_add_turn(self):
        """Test adding turns to trajectory."""
        trajectory = MultiTurnTrajectory()

        turn1 = ConversationTurn("Hello", "Hi!", 0.8)
        turn2 = ConversationTurn("How are you?", "I'm good!", 0.9)

        trajectory.add_turn(turn1)
        trajectory.add_turn(turn2)

        assert len(trajectory.turns) == 2
        assert trajectory.total_reward == 1.7
        assert trajectory.average_reward == 0.85

    def test_trajectory_to_dict(self):
        """Test converting trajectory to dictionary."""
        trajectory = MultiTurnTrajectory("test_123")
        turn = ConversationTurn("Hello", "Hi!", 0.8)
        trajectory.add_turn(turn)

        traj_dict = trajectory.to_dict()

        assert traj_dict["trajectory_id"] == "test_123"
        assert len(traj_dict["turns"]) == 1
        assert traj_dict["total_reward"] == 0.8
