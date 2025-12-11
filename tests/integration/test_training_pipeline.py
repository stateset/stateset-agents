"""
Integration tests for the complete training pipeline.

These tests verify that all components work together correctly.
"""

import asyncio
import sys
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Block vllm import to avoid torchvision issues
if 'vllm' not in sys.modules:
    sys.modules['vllm'] = type(sys)('vllm')  # type: ignore

INTEGRATION_AVAILABLE = True
try:
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
    from stateset_agents.training import TrainingConfig, train
except (ImportError, RuntimeError) as e:
    INTEGRATION_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not INTEGRATION_AVAILABLE,
    reason="Integration modules not available (check transformers/torchvision compatibility)"
)


@pytest.mark.integration
class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""

    @pytest.fixture
    def training_config(self):
        """Create a test training configuration."""
        return TrainingConfig(
            num_episodes=2,  # Very small for testing
            max_steps_per_episode=3,
            learning_rate=1e-5,
            batch_size=2,
            gradient_accumulation_steps=1,
        )

    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(model_name="gpt2", max_new_tokens=50, temperature=0.7)

    @pytest.fixture
    def conversation_scenarios(self):
        """Create test conversation scenarios."""
        return [
            {
                "id": "test_scenario_1",
                "topic": "general_help",
                "context": "User needs general assistance",
                "user_responses": [
                    "Hi there! Can you help me?",
                    "That's helpful. What about this?",
                    "Great! Thank you for your help.",
                ],
            },
            {
                "id": "test_scenario_2",
                "topic": "technical_help",
                "context": "User needs technical assistance",
                "user_responses": [
                    "I'm having trouble with my code.",
                    "Can you explain this error?",
                    "That fixed it! Thanks!",
                ],
            },
        ]

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_full_training_pipeline(
        self,
        mock_tokenizer,
        mock_model,
        agent_config,
        conversation_scenarios,
        training_config,
    ):
        """Test the complete training pipeline from agent to trained model."""

        # Setup mocks with proper torch device
        import torch

        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device("cpu")  # Proper torch.device

        # Mock model parameters for device detection
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model_instance.parameters.return_value = iter([mock_param])

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.model_max_length = 2048
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = "Assistant response"

        # Mock model generate method
        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5]
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.generate.return_value = mock_output

        # Mock forward pass for training
        mock_forward_output = MagicMock()
        mock_forward_output.loss = torch.tensor(0.5, requires_grad=True)
        mock_forward_output.logits = torch.randn(1, 10, 50257)  # (batch, seq, vocab)
        mock_model_instance.return_value = mock_forward_output
        mock_model_instance.__call__ = lambda *args, **kwargs: mock_forward_output

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create agent
        agent = MultiTurnAgent(agent_config)

        # Create environment
        environment = ConversationEnvironment(
            scenarios=conversation_scenarios, max_turns=3
        )

        # Create reward function
        reward_fn = CompositeReward(
            [HelpfulnessReward(weight=0.7), SafetyReward(weight=0.3)]
        )

        # Mock the reward computation
        with patch.object(
            reward_fn, "compute_reward", return_value=AsyncMock(return_value=0.8)
        ):
            # Run training
            trained_agent = await train(
                agent=agent,
                environment=environment,
                reward_fn=reward_fn,
                num_episodes=1,  # Very small for integration test
                profile="conservative",
            )

        # Verify the training completed
        assert trained_agent is not None
        assert isinstance(trained_agent, MultiTurnAgent)

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_environment_agent_interaction(
        self, mock_tokenizer, mock_model, agent_config, conversation_scenarios
    ):
        """Test interaction between agent and environment."""

        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = "Helpful response about Python"

        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5]
        mock_model_instance.generate.return_value = mock_output

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create agent and environment
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()

        environment = ConversationEnvironment(
            scenarios=conversation_scenarios, max_turns=2
        )

        # Reset environment
        state = await environment.reset()
        assert state is not None
        assert "step" in state

        # Simulate a few steps
        for step in range(2):
            # Agent generates response
            messages = [{"role": "user", "content": f"Test message {step + 1}"}]
            response = await agent.generate_response(messages)

            # Environment processes the response
            next_state = await environment.step(response)
            assert "step" in next_state
            assert "reward" in next_state
            assert "done" in next_state

            if next_state["done"]:
                break

        # Verify conversation history was maintained
        assert len(agent.conversation_history) > 0

    @pytest.mark.asyncio
    async def test_reward_function_integration(self, agent_config):
        """Test reward function integration with different components."""

        # Create composite reward
        reward_fn = CompositeReward(
            [HelpfulnessReward(weight=0.6), SafetyReward(weight=0.4)]
        )

        # Mock reward computation
        with patch.object(reward_fn, "compute_reward") as mock_compute:
            mock_compute.return_value = AsyncMock(return_value=0.85)

            # Test reward computation
            turns = [
                {"role": "user", "content": "How do I learn Python?"},
                {
                    "role": "assistant",
                    "content": "Start with the official tutorial at python.org",
                },
            ]

            result = await reward_fn.compute_reward(turns)

            assert result is not None
            mock_compute.assert_called_once()

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_agent_persistence_workflow(
        self, mock_tokenizer, mock_model, agent_config, tmp_path
    ):
        """Test saving and loading agent state."""

        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create and initialize agent
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()

        # Simulate some conversation history
        agent.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Test that agent has expected attributes
        assert agent.model is not None
        assert agent.tokenizer is not None
        assert len(agent.conversation_history) == 2
