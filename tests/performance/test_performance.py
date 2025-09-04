"""
Performance tests for critical operations.

These tests measure execution time and resource usage.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import psutil
import os

from stateset_agents.core.agent import MultiTurnAgent, AgentConfig
from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward


@pytest.mark.performance
class TestAgentPerformance:
    """Performance tests for agent operations."""
    
    @pytest.fixture
    def agent_config(self):
        """Create a performance-optimized agent configuration."""
        return AgentConfig(
            model_name="gpt2",
            max_new_tokens=50,
            temperature=0.7,
            torch_dtype="float16"  # Use smaller precision for performance
        )
    
    @pytest.fixture
    def performance_reward(self):
        """Create a reward function for performance testing."""
        return CompositeReward([
            HelpfulnessReward(weight=0.7),
            SafetyReward(weight=0.3)
        ])
    
    @pytest.mark.asyncio
    @patch('stateset_agents.core.agent.AutoModelForCausalLM')
    @patch('stateset_agents.core.agent.AutoTokenizer')
    async def test_agent_initialization_performance(self, mock_tokenizer, mock_model, agent_config):
        """Test agent initialization performance."""
        
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        init_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Performance assertions
        assert init_time < 5.0, f"Initialization took too long: {init_time}s"
        assert memory_delta < 500, f"Memory usage increased too much: {memory_delta}MB"
        
        print(f"Agent initialization: {init_time:.3f}s, Memory delta: {memory_delta:.1f}MB")
    
    @pytest.mark.asyncio
    @patch('stateset_agents.core.agent.AutoModelForCausalLM')
    @patch('stateset_agents.core.agent.AutoTokenizer')
    async def test_response_generation_performance(self, mock_tokenizer, mock_model, agent_config):
        """Test response generation performance."""
        
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = "This is a helpful response about Python programming."
        
        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_model_instance.generate.return_value = mock_output
        
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()
        
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant."},
            {"role": "user", "content": "How do I write a function in Python?"}
        ]
        
        # Measure performance
        start_time = time.time()
        
        response = await agent.generate_response(messages)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Performance assertions
        assert response_time < 2.0, f"Response generation took too long: {response_time}s"
        assert isinstance(response, str)
        assert len(response) > 0
        
        print(f"Response generation: {response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_reward_computation_performance(self, performance_reward):
        """Test reward computation performance."""
        
        turns = [
            {
                "role": "user", 
                "content": "I'm trying to learn Python. What's a good way to start?"
            },
            {
                "role": "assistant",
                "content": "Great question! I'd recommend starting with the official Python tutorial at python.org. They have excellent beginner-friendly documentation that covers all the basics."
            }
        ]
        
        # Mock the individual reward computations
        with patch.object(performance_reward, 'compute_reward') as mock_compute:
            mock_compute.return_value = AsyncMock(return_value=0.85)
            
            start_time = time.time()
            
            # Run multiple reward computations to measure performance
            for _ in range(10):
                result = await performance_reward.compute_reward(turns)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / 10
            
            # Performance assertions
            assert avg_time < 0.1, f"Reward computation too slow: {avg_time:.3f}s per call"
            assert result == 0.85
            
            print(f"Reward computation: {avg_time:.3f}s per call (10 iterations)")


@pytest.mark.performance
class TestBatchOperationsPerformance:
    """Performance tests for batch operations."""
    
    @pytest.mark.asyncio
    @patch('stateset_agents.core.agent.AutoModelForCausalLM')
    @patch('stateset_agents.core.agent.AutoTokenizer')
    async def test_batch_response_generation(self, mock_tokenizer, mock_model):
        """Test performance of generating multiple responses."""
        
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = "Response"
        
        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5]
        mock_model_instance.generate.return_value = mock_output
        
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        agent_config = AgentConfig(model_name="gpt2", max_new_tokens=30)
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()
        
        messages_batch = [
            [{"role": "user", "content": f"Question {i}?"}]
            for i in range(5)
        ]
        
        start_time = time.time()
        
        # Generate responses for batch
        responses = []
        for messages in messages_batch:
            response = await agent.generate_response(messages)
            responses.append(response)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(messages_batch)
        
        # Performance assertions
        assert total_time < 10.0, f"Batch processing took too long: {total_time}s"
        assert avg_time < 1.0, f"Average response time too slow: {avg_time:.3f}s"
        assert len(responses) == len(messages_batch)
        
        print(f"Batch processing (5 responses): {total_time:.3f}s total, {avg_time:.3f}s average")


@pytest.mark.performance
@pytest.mark.slow
class TestMemoryUsage:
    """Tests for memory usage patterns."""
    
    @pytest.mark.asyncio
    @patch('stateset_agents.core.agent.AutoModelForCausalLM')
    @patch('stateset_agents.core.agent.AutoTokenizer')
    async def test_memory_usage_during_conversation(self, mock_tokenizer, mock_model):
        """Test memory usage patterns during long conversations."""
        
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = "Response"
        
        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4, 5]
        mock_model_instance.generate.return_value = mock_output
        
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        agent_config = AgentConfig(model_name="gpt2", max_new_tokens=30)
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate a conversation with multiple turns
        for turn in range(10):
            messages = [{"role": "user", "content": f"Turn {turn + 1}: Tell me something interesting."}]
            await agent.generate_response(messages)
            
            # Check memory every few turns
            if turn % 3 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_delta = current_memory - initial_memory
                
                # Memory should not grow excessively
                assert memory_delta < 100, f"Memory leak detected: {memory_delta:.1f}MB increase at turn {turn + 1}"
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_delta = final_memory - initial_memory
        
        print(f"Memory usage: Initial {initial_memory:.1f}MB, Final {final_memory:.1f}MB, Delta {total_memory_delta:.1f}MB")
        
        # Overall memory growth should be reasonable
        assert total_memory_delta < 50, f"Excessive memory growth: {total_memory_delta:.1f}MB"
