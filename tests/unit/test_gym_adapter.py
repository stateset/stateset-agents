"""
Unit tests for Gym/Gymnasium integration.

Tests the gym adapter, processors, mappers, and agents.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock


# Test Observation Processors
class TestObservationProcessors:
    """Test observation processors for converting gym observations to text."""

    def test_vector_processor_basic(self):
        """Test basic vector observation processing."""
        from core.gym.processors import VectorObservationProcessor

        processor = VectorObservationProcessor(precision=2)
        obs = np.array([0.5, -1.2, 3.14])

        result = processor.process(obs)

        assert isinstance(result, str)
        assert "0.50" in result
        assert "1.20" in result or "-1.20" in result
        assert "3.14" in result

    def test_vector_processor_with_names(self):
        """Test vector processor with feature names."""
        from core.gym.processors import VectorObservationProcessor

        processor = VectorObservationProcessor(
            feature_names=["x", "y", "z"],
            precision=1
        )
        obs = np.array([1.0, 2.0, 3.0])

        result = processor.process(obs)

        assert "x" in result
        assert "y" in result
        assert "z" in result
        assert "1.0" in result
        assert "2.0" in result
        assert "3.0" in result

    def test_cartpole_processor(self):
        """Test CartPole-specific processor."""
        from core.gym.processors import CartPoleObservationProcessor

        processor = CartPoleObservationProcessor(precision=2)
        obs = np.array([0.1, 0.5, -0.05, 0.2])

        result = processor.process(obs)

        assert isinstance(result, str)
        assert "cart" in result.lower() or "Cart" in result
        assert "pole" in result.lower() or "Pole" in result
        # Check numeric values present
        assert "0.1" in result or "0.10" in result

    def test_cartpole_system_prompt(self):
        """Test CartPole system prompt generation."""
        from core.gym.processors import CartPoleObservationProcessor

        processor = CartPoleObservationProcessor()
        mock_env = Mock()
        mock_env.spec = Mock()
        mock_env.spec.id = "CartPole-v1"

        prompt = processor.get_system_prompt(mock_env)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "CartPole" in prompt or "pole" in prompt.lower()
        assert "0" in prompt  # Action 0
        assert "1" in prompt  # Action 1

    def test_processor_factory(self):
        """Test observation processor factory function."""
        from core.gym.processors import create_observation_processor

        # CartPole should return CartPoleObservationProcessor
        processor = create_observation_processor("CartPole-v1")
        assert processor.__class__.__name__ == "CartPoleObservationProcessor"

        # MountainCar should return MountainCarObservationProcessor
        processor = create_observation_processor("MountainCar-v0")
        assert processor.__class__.__name__ == "MountainCarObservationProcessor"

        # Unknown env should return generic VectorObservationProcessor
        processor = create_observation_processor("UnknownEnv-v0")
        assert processor.__class__.__name__ == "VectorObservationProcessor"


# Test Action Mappers
class TestActionMappers:
    """Test action mappers for parsing agent responses."""

    def test_discrete_mapper_simple_int(self):
        """Test parsing simple integer actions."""
        from core.gym.mappers import DiscreteActionMapper

        mapper = DiscreteActionMapper(n_actions=2)

        assert mapper.parse_action("0") == 0
        assert mapper.parse_action("1") == 1

    def test_discrete_mapper_with_text(self):
        """Test parsing actions from text responses."""
        from core.gym.mappers import DiscreteActionMapper

        mapper = DiscreteActionMapper(n_actions=2)

        assert mapper.parse_action("Action: 0") == 0
        assert mapper.parse_action("I choose action 1") == 1
        assert mapper.parse_action("The best move is 0") == 0

    def test_discrete_mapper_with_names(self):
        """Test parsing named actions."""
        from core.gym.mappers import DiscreteActionMapper

        mapper = DiscreteActionMapper(
            n_actions=2,
            action_names=["LEFT", "RIGHT"]
        )

        assert mapper.parse_action("LEFT") == 0
        assert mapper.parse_action("RIGHT") == 1
        assert mapper.parse_action("I'll go left") == 0
        assert mapper.parse_action("Move right") == 1

    def test_discrete_mapper_invalid_action(self):
        """Test handling of invalid actions."""
        from core.gym.mappers import DiscreteActionMapper

        mapper = DiscreteActionMapper(n_actions=2)

        # Out of bounds should return fallback (random or default)
        result = mapper.parse_action("5")
        assert result in [0, 1]

        # Unparseable text should return fallback
        result = mapper.parse_action("I don't know")
        assert result in [0, 1]

    def test_discrete_mapper_default_action(self):
        """Test using default action on parse failure."""
        from core.gym.mappers import DiscreteActionMapper

        mapper = DiscreteActionMapper(n_actions=3, default_action=1)

        # Unparseable should return default
        result = mapper.parse_action("invalid")
        assert result == 1

    def test_continuous_mapper_bracket_format(self):
        """Test parsing continuous actions in bracket format."""
        from core.gym.mappers import ContinuousActionMapper

        mapper = ContinuousActionMapper(action_dim=2)

        result = mapper.parse_action("[0.5, -0.3]")

        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert np.allclose(result, [0.5, -0.3])

    def test_continuous_mapper_space_separated(self):
        """Test parsing space-separated continuous actions."""
        from core.gym.mappers import ContinuousActionMapper

        mapper = ContinuousActionMapper(action_dim=3)

        result = mapper.parse_action("0.1 0.2 0.3")

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.allclose(result, [0.1, 0.2, 0.3])

    def test_continuous_mapper_clipping(self):
        """Test that continuous actions are clipped to bounds."""
        from core.gym.mappers import ContinuousActionMapper

        mapper = ContinuousActionMapper(
            action_dim=2,
            action_low=np.array([-1.0, -1.0]),
            action_high=np.array([1.0, 1.0])
        )

        # Out of bounds should be clipped
        result = mapper.parse_action("[2.0, -2.0]")

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
        assert np.allclose(result, [1.0, -1.0])


# Test GymEnvironmentAdapter
class TestGymEnvironmentAdapter:
    """Test the main gym environment adapter."""

    @pytest.mark.asyncio
    async def test_adapter_creation(self):
        """Test creating adapter with auto processors."""
        from core.gym.adapter import GymEnvironmentAdapter

        # Mock gym environment
        mock_env = Mock()
        mock_env.spec = Mock()
        mock_env.spec.id = "CartPole-v1"
        mock_env.spec.max_episode_steps = 500
        mock_env.action_space = Mock()
        mock_env.action_space.n = 2
        mock_env.observation_space = Mock()

        adapter = GymEnvironmentAdapter(
            mock_env,
            auto_create_processors=True
        )

        assert adapter is not None
        assert adapter.gym_env == mock_env
        assert adapter.observation_processor is not None
        assert adapter.action_mapper is not None

    @pytest.mark.asyncio
    async def test_adapter_reset(self):
        """Test environment reset."""
        from core.gym.adapter import GymEnvironmentAdapter
        from core.gym.processors import CartPoleObservationProcessor
        from core.gym.mappers import DiscreteActionMapper

        # Mock gym environment
        mock_env = Mock()
        mock_env.spec = Mock()
        mock_env.spec.id = "CartPole-v1"
        mock_env.reset = Mock(return_value=(np.array([0.0, 0.0, 0.0, 0.0]), {}))

        adapter = GymEnvironmentAdapter(
            mock_env,
            observation_processor=CartPoleObservationProcessor(),
            action_mapper=DiscreteActionMapper(n_actions=2)
        )

        state = await adapter.reset()

        assert state is not None
        assert state.episode_id is not None
        assert state.turn_count == 0
        assert "observation" in state.context

    @pytest.mark.asyncio
    async def test_adapter_step(self):
        """Test environment step execution."""
        from core.gym.adapter import GymEnvironmentAdapter
        from core.gym.processors import CartPoleObservationProcessor
        from core.gym.mappers import DiscreteActionMapper
        from core.trajectory import ConversationTurn
        from core.environment import EnvironmentState, EpisodeStatus

        # Mock gym environment
        mock_env = Mock()
        mock_env.spec = Mock()
        mock_env.spec.id = "CartPole-v1"
        mock_env.reset = Mock(return_value=(np.array([0.0, 0.0, 0.0, 0.0]), {}))
        mock_env.step = Mock(return_value=(
            np.array([0.1, 0.1, 0.1, 0.1]),  # obs
            1.0,  # reward
            False,  # done
            {}  # info
        ))

        adapter = GymEnvironmentAdapter(
            mock_env,
            observation_processor=CartPoleObservationProcessor(),
            action_mapper=DiscreteActionMapper(n_actions=2)
        )

        # Reset first
        state = await adapter.reset()

        # Execute step
        action_turn = ConversationTurn(role="assistant", content="0")
        new_state, obs_turn, reward, done = await adapter.step(state, action_turn)

        assert new_state.turn_count == 1
        assert reward == 1.0
        assert done is False
        assert obs_turn.role == "system"


# Test GymAgent
class TestGymAgent:
    """Test GymAgent specialized for gym tasks."""

    @pytest.mark.asyncio
    async def test_gym_agent_creation(self):
        """Test creating a GymAgent."""
        from core.gym.agents import create_gym_agent

        agent = create_gym_agent(
            model_name="gpt2",
            use_stub=True,  # Use stub for testing
            temperature=0.7
        )

        assert agent is not None
        assert agent.config.max_new_tokens <= 20  # Should be optimized for short generation

    @pytest.mark.asyncio
    async def test_gym_agent_initialization(self):
        """Test GymAgent initialization."""
        from core.gym.agents import GymAgent
        from core.agent import AgentConfig

        config = AgentConfig(
            model_name="gpt2",
            use_stub_model=True,  # Use stub for testing
            max_new_tokens=10
        )

        agent = GymAgent(config)
        await agent.initialize()

        assert agent.model is not None
        assert agent.config.max_new_tokens == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
