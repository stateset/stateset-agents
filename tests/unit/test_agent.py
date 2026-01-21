"""
Unit tests for core agent functionality.

These tests focus on individual components in isolation.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

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
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "Hello world"
        return model, tokenizer

    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(model_name="gpt2", max_new_tokens=50, temperature=0.7)

    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    def test_agent_initialization(self, mock_tokenizer, mock_model, agent_config):
        """Test agent initialization."""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        agent = Agent(agent_config)

        # Should not initialize model immediately
        assert agent.model is None
        assert agent.tokenizer is None

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_agent_async_initialization(
        self, mock_tokenizer, mock_model, agent_config
    ):
        """Test async agent initialization."""
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        agent = Agent(agent_config)
        await agent.initialize()

        assert agent.model == mock_model_instance
        assert agent.tokenizer == mock_tokenizer_instance
        assert agent.generation_config is not None

    @pytest.mark.asyncio
    async def test_agent_generate_response_not_implemented(self, agent_config):
        """Test that base Agent raises NotImplementedError for generate_response."""
        agent = Agent(agent_config)

        with pytest.raises(NotImplementedError):
            await agent.generate_response([{"role": "user", "content": "Hello"}])


class TestMultiTurnAgent:
    """Test MultiTurnAgent functionality."""

    @pytest.fixture
    def multiturn_agent_config(self):
        """Create a test MultiTurnAgent configuration."""
        return AgentConfig(
            model_name="gpt2",
            max_new_tokens=100,
            temperature=0.8,
            use_chat_template=True,
        )

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_multiturn_agent_initialization(
        self, mock_tokenizer, mock_model, multiturn_agent_config
    ):
        """Test MultiTurnAgent initialization."""
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3, 4, 5]

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        agent = MultiTurnAgent(multiturn_agent_config)
        await agent.initialize()

        assert agent.model == mock_model_instance
        assert agent.tokenizer == mock_tokenizer_instance
        assert len(agent.conversation_history) == 0

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_multiturn_agent_generate_response(
        self, mock_tokenizer, mock_model, multiturn_agent_config
    ):
        """Test MultiTurnAgent response generation."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = "Assistant response"

        # Mock the model generate method
        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5]
        mock_model_instance.generate.return_value = mock_output

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        agent = MultiTurnAgent(multiturn_agent_config)
        await agent.initialize()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        response = await agent.generate_response(messages)

        assert isinstance(response, str)
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_multiturn_agent_stub_backend(self, multiturn_agent_config):
        multiturn_agent_config.use_stub_model = True
        multiturn_agent_config.stub_responses = ["Stub backend active"]
        multiturn_agent_config.model_name = "stub://unit-test"

        agent = MultiTurnAgent(multiturn_agent_config)
        await agent.initialize()

        messages = [{"role": "user", "content": "Ping"}]
        response = await agent.generate_response(messages)

        assert response.startswith("Stub backend active")

    @pytest.mark.asyncio
    async def test_multiturn_agent_planning_manager_init(self, multiturn_agent_config):
        multiturn_agent_config.use_stub_model = True
        multiturn_agent_config.model_name = "stub://planning"
        multiturn_agent_config.enable_planning = True

        agent = MultiTurnAgent(multiturn_agent_config)
        await agent.initialize()

        assert agent.planning_manager is not None

    @pytest.mark.asyncio
    async def test_multiturn_agent_accepts_string_prompt(self, multiturn_agent_config):
        multiturn_agent_config.use_stub_model = True
        agent = MultiTurnAgent(multiturn_agent_config)
        await agent.initialize()
        response = await agent.generate_response("Hello there")
        assert isinstance(response, str)
        assert agent.conversation_history[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_streaming_updates_history(self):
        config = AgentConfig(
            model_name="stub://stream",
            use_stub_model=True,
        )
        agent = MultiTurnAgent(config)
        await agent.initialize()

        tokens = []
        async for token in agent.generate_response_stream("Hello"):
            tokens.append(token)

        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["role"] == "assistant"
        assert agent.turn_count == 1


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
