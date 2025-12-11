"""
End-to-end tests for customer service scenarios.

These tests simulate real-world usage scenarios from start to finish.
"""

import asyncio
import sys
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Block vllm import to avoid torchvision issues
if 'vllm' not in sys.modules:
    sys.modules['vllm'] = type(sys)('vllm')  # type: ignore

E2E_AVAILABLE = True
try:
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
    from stateset_agents.training import train
except (ImportError, RuntimeError) as e:
    E2E_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not E2E_AVAILABLE,
    reason="E2E modules not available (check transformers/torchvision compatibility)"
)


@pytest.mark.e2e
class TestCustomerServiceE2E:
    """End-to-end tests for customer service agent scenarios."""

    @pytest.fixture
    def customer_service_config(self):
        """Create a customer service agent configuration."""
        return AgentConfig(
            model_name="gpt2",
            system_prompt="""You are a helpful customer service representative for TechCorp.
            Be friendly, professional, and solve customer problems efficiently.
            Always ask clarifying questions when needed.""",
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )

    @pytest.fixture
    def customer_service_scenarios(self):
        """Create realistic customer service conversation scenarios."""
        return [
            {
                "id": "password_reset",
                "topic": "account_help",
                "context": "Customer needs to reset their password",
                "user_responses": [
                    "Hi, I forgot my password and can't log into my account.",
                    "I tried the reset link but it didn't work. What should I do?",
                    "Okay, I think I got it now. Let me try again.",
                    "Perfect! It worked. Thank you so much for your help!",
                ],
            },
            {
                "id": "billing_issue",
                "topic": "billing_support",
                "context": "Customer has a question about their bill",
                "user_responses": [
                    "Hello, I noticed a charge on my bill that I don't recognize.",
                    "It was $49.99 on the 15th. I don't remember purchasing anything.",
                    "I see, let me check what that might be. Can you tell me more about it?",
                    "Ah, that makes sense. It was probably that subscription I forgot about. Thanks!",
                ],
            },
            {
                "id": "product_help",
                "topic": "technical_support",
                "context": "Customer needs help with a product feature",
                "user_responses": [
                    "Hi, I'm having trouble using the new dashboard feature.",
                    "I clicked on the analytics tab but nothing loads.",
                    "I tried refreshing, but it still doesn't work. Any suggestions?",
                    "That worked! The analytics are loading now. Thank you!",
                ],
            },
        ]

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_complete_customer_service_workflow(
        self,
        mock_tokenizer,
        mock_model,
        customer_service_config,
        customer_service_scenarios,
    ):
        """Test complete customer service workflow from agent creation to conversation handling."""

        # Setup comprehensive mocks with proper torch device
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
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3, 4, 5]

        # Create varied responses for different scenarios
        response_cycle = [
            "I understand you're having trouble with your password. Let me help you reset it. First, can you confirm your email address?",
            "The password reset link should work now. Please check your email and click the link. If you don't see it, check your spam folder.",
            "I'm glad the reset worked! Is there anything else I can help you with today?",
            "You're welcome! Have a great day.",
        ]
        response_index = 0

        def mock_decode(*args, **kwargs):
            nonlocal response_index
            response = response_cycle[response_index % len(response_cycle)]
            response_index += 1
            return response

        mock_tokenizer_instance.decode.side_effect = mock_decode

        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        mock_model_instance.generate.return_value = mock_output

        # Mock forward pass for training
        mock_forward_output = MagicMock()
        mock_forward_output.loss = torch.tensor(0.5, requires_grad=True)
        mock_forward_output.logits = torch.randn(1, 10, 50257)  # (batch, seq, vocab)
        mock_model_instance.return_value = mock_forward_output
        mock_model_instance.__call__ = lambda *args, **kwargs: mock_forward_output

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Step 1: Create and initialize the agent
        agent = MultiTurnAgent(customer_service_config)
        await agent.initialize()

        # Step 2: Create environment with customer service scenarios
        environment = ConversationEnvironment(
            scenarios=customer_service_scenarios, max_turns=4
        )

        # Step 3: Set up reward system optimized for customer service
        reward_fn = CompositeReward(
            [HelpfulnessReward(weight=0.6), SafetyReward(weight=0.4)]
        )

        # Step 4: Train the agent (short training for E2E test)
        with patch.object(
            reward_fn, "compute_reward", return_value=AsyncMock(return_value=0.8)
        ):
            trained_agent = await train(
                agent=agent,
                environment=environment,
                reward_fn=reward_fn,
                num_episodes=2,
                profile="balanced",
            )

        # Step 5: Test the trained agent with a real conversation
        test_conversation = [
            {
                "role": "user",
                "content": "Hi, I can't access my account. I think I forgot my password.",
            }
        ]

        response = await trained_agent.generate_response(test_conversation)

        # Verify the response is appropriate for customer service
        assert isinstance(response, str)
        assert len(response) > 10
        assert any(
            keyword in response.lower()
            for keyword in ["password", "reset", "help", "account", "email"]
        )

        # Step 6: Test conversation continuity
        conversation_history = [
            {"role": "user", "content": "I forgot my password"},
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": "I checked my email but didn't see the reset link",
            },
        ]

        follow_up_response = await trained_agent.generate_response(conversation_history)

        assert isinstance(follow_up_response, str)
        assert len(follow_up_response) > 5
        # Should reference the previous context
        assert len(trained_agent.conversation_history) >= 3

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_agent_adaptation_to_different_personas(
        self, mock_tokenizer, mock_model, customer_service_config
    ):
        """Test agent adaptation to different customer service personas."""

        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3]

        responses = {
            "billing": "I'll help you with your billing question right away.",
            "technical": "I can assist you with technical issues.",
            "general": "How can I help you today?",
        }

        def mock_response_generator(*args, **kwargs):
            # Simple keyword-based response selection
            if "bill" in str(args).lower() or "charge" in str(args).lower():
                return responses["billing"]
            elif "error" in str(args).lower() or "technical" in str(args).lower():
                return responses["technical"]
            else:
                return responses["general"]

        mock_tokenizer_instance.decode.side_effect = mock_response_generator

        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5]
        mock_model_instance.generate.return_value = mock_output

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        agent = MultiTurnAgent(customer_service_config)
        await agent.initialize()

        # Test different types of customer inquiries
        test_cases = [
            {
                "input": "I see an unexpected charge on my bill",
                "expected_keywords": ["billing", "charge", "help"],
            },
            {
                "input": "I'm getting an error when trying to use the app",
                "expected_keywords": ["technical", "error", "assist"],
            },
            {
                "input": "Hello, I have a question",
                "expected_keywords": ["help", "today", "can"],
            },
        ]

        for test_case in test_cases:
            messages = [{"role": "user", "content": test_case["input"]}]
            response = await agent.generate_response(messages)

            assert isinstance(response, str)
            assert len(response) > 5

            # Check if response contains expected keywords
            response_lower = response.lower()
            has_expected_keyword = any(
                keyword in response_lower for keyword in test_case["expected_keywords"]
            )
            assert (
                has_expected_keyword
            ), f"Response '{response}' doesn't match expected keywords {test_case['expected_keywords']}"

    @pytest.mark.asyncio
    @patch("stateset_agents.core.agent.AutoModelForCausalLM")
    @patch("stateset_agents.core.agent.AutoTokenizer")
    async def test_performance_under_load(
        self, mock_tokenizer, mock_model, customer_service_config
    ):
        """Test agent performance under simulated load."""

        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = (
            "Thank you for your question. I'm here to help!"
        )

        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5]
        mock_model_instance.generate.return_value = mock_output

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        agent = MultiTurnAgent(customer_service_config)
        await agent.initialize()

        # Simulate concurrent customer conversations
        async def simulate_conversation(conversation_id: int):
            messages = [
                {"role": "user", "content": f"Question from customer {conversation_id}"}
            ]
            response = await agent.generate_response(messages)
            return response

        # Run multiple conversations concurrently
        conversation_tasks = [
            simulate_conversation(i)
            for i in range(5)  # Simulate 5 concurrent conversations
        ]

        import time

        start_time = time.time()

        responses = await asyncio.gather(*conversation_tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify all responses were generated
        assert len(responses) == 5
        assert all(isinstance(r, str) for r in responses)
        assert all(len(r) > 5 for r in responses)

        # Performance check - should complete within reasonable time
        assert total_time < 10.0, f"Concurrent processing took too long: {total_time}s"

        print(f"Concurrent processing (5 conversations): {total_time:.3f}s")
