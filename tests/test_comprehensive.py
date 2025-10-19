"""
Comprehensive Test Suite for GRPO Agent Framework

This test suite covers all major components and innovations integrated
into the framework, ensuring production-ready quality.
"""

import asyncio
import json
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Core components
from stateset_agents.core.agent import Agent
from stateset_agents.core.computational_engine import ComputationalGRPOEngine, ComputationalTrajectory
from stateset_agents.core.environment import Environment
from stateset_agents.core.multiturn_agent import ConversationContext, DialogueDatabase, MultiTurnAgent
from stateset_agents.core.reward import RewardFunction, RewardResult
from stateset_agents.core.trajectory import Trajectory
from rewards.multi_objective_reward import (
    ActionOrientedRewardComponent,
    EmpathyRewardComponent,
    MultiObjectiveRewardFunction,
    ProfessionalismRewardComponent,
)

# Reward components
from rewards.ruler_reward import RulerRewardFunction
from training.config import TrainingConfig
from training.neural_reward_trainer import NeuralRewardFunction, NeuralRewardTrainer

# Training components
from training.trainer import GRPOTrainer

# Utils
from utils.cache import CacheService
from utils.monitoring import MonitoringService


@pytest.fixture(autouse=True)
def _set_deterministic_seed():
    """Ensure deterministic behaviour across stochastic tests."""
    random.seed(42)
    np.random.seed(42)


class TestAgent(Agent):
    """Test implementation of Agent"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.responses = [
            "Thank you for your question. I'm here to help you with that.",
            "I understand your concern and will do my best to assist you.",
            "Let me provide you with a detailed solution to your problem.",
        ]
        self.response_index = 0

    async def generate_response(self, prompt: str) -> str:
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response

    async def update_from_feedback(self, training_data: List[Dict[str, Any]]):
        # Simulate learning from feedback
        pass


class TestEnvironment(Environment):
    """Test implementation of Environment"""

    def __init__(self):
        super().__init__()
        self.state = {"step": 0}

    async def reset(self) -> Dict[str, Any]:
        self.state = {"step": 0}
        return self.state

    async def step(self, action: str) -> Dict[str, Any]:
        self.state["step"] += 1
        return {
            "state": self.state,
            "reward": 0.5 + np.random.normal(0, 0.1),
            "done": self.state["step"] >= 5,
        }

    async def get_reward(self, trajectory: Trajectory) -> float:
        return 0.5 + np.random.normal(0, 0.1)


class TestRewardFunction(RewardFunction):
    """Test implementation of RewardFunction"""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight)

    async def compute_reward(
        self, turns: List[Dict[str, Any]], context: Dict[str, Any] = None
    ) -> RewardResult:
        # Simple test reward based on response length
        if not turns:
            return RewardResult(metadata={}, score=0.0, breakdown={"error": "No turns"})

        assistant_turns = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_turns:
            return RewardResult(
                score=0.0, breakdown={"error": "No assistant turns"}, metadata={}
            )

        last_response = assistant_turns[-1].get("content", "")
        score = min(1.0, len(last_response.split()) / 20.0)

        return RewardResult(
            metadata={},
            score=score,
            breakdown={"length_score": score, "word_count": len(last_response.split())},
        )


@pytest.fixture
def test_agent():
    """Create a test agent"""
    return TestAgent({"model_type": "test"})


@pytest.fixture
def test_environment():
    """Create a test environment"""
    return TestEnvironment()


@pytest.fixture
def test_reward_function():
    """Create a test reward function"""
    return TestRewardFunction()


@pytest.fixture
def sample_turns():
    """Create sample conversation turns"""
    return [
        {"role": "user", "content": "Hello, I need help with my order"},
        {
            "role": "assistant",
            "content": "I'd be happy to help you with your order. Could you please provide your order number?",
        },
    ]


class TestCoreComponents:
    """Test core framework components"""

    @pytest.mark.asyncio
    async def test_agent_basic_functionality(self, test_agent):
        """Test basic agent functionality"""
        response = await test_agent.generate_response("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_environment_lifecycle(self, test_environment):
        """Test environment lifecycle"""
        # Reset environment
        initial_state = await test_environment.reset()
        assert "step" in initial_state
        assert initial_state["step"] == 0

        # Take a step
        result = await test_environment.step("test_action")
        assert "reward" in result
        assert "done" in result
        assert isinstance(result["reward"], float)
        assert isinstance(result["done"], bool)

    @pytest.mark.asyncio
    async def test_reward_function(self, test_reward_function, sample_turns):
        """Test reward function computation"""
        result = await test_reward_function.compute_reward(sample_turns)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0
        assert "breakdown" in result.__dict__
        assert isinstance(result.breakdown, dict)

    def test_trajectory_management(self):
        """Test trajectory data structures"""
        trajectory = Trajectory()

        # Add turns
        trajectory.add_turn({"role": "user", "content": "Hello"})
        trajectory.add_turn({"role": "assistant", "content": "Hi there!"})

        assert len(trajectory.turns) == 2
        assert trajectory.turns[0]["role"] == "user"
        assert trajectory.turns[1]["role"] == "assistant"

        # Test metadata
        trajectory.metadata["test_key"] = "test_value"
        assert trajectory.metadata["test_key"] == "test_value"


class TestComputationalEngine:
    """Test computational GRPO engine"""

    @pytest.fixture
    def computational_engine(self, test_agent, test_environment, test_reward_function):
        """Create computational engine for testing"""
        return ComputationalGRPOEngine(
            agent=test_agent,
            environment=test_environment,
            reward_function=test_reward_function,
            num_workers=2,
            trajectory_batch_size=4,
            use_learned_rewards=True,
        )

    @pytest.mark.asyncio
    async def test_trajectory_generation(self, computational_engine):
        """Test parallel trajectory generation"""
        prompts = ["Hello", "How are you?", "Can you help me?"]

        trajectories = await computational_engine.generate_trajectory_batch(prompts)

        assert len(trajectories) == len(prompts)
        for traj in trajectories:
            assert isinstance(traj, ComputationalTrajectory)
            assert traj.prompt in prompts
            assert len(traj.response) > 0
            assert 0.0 <= traj.learned_reward <= 1.0

    @pytest.mark.asyncio
    async def test_training_iteration(self, computational_engine):
        """Test complete training iteration"""
        prompts = ["Test prompt 1", "Test prompt 2"]

        results = await computational_engine.train_iteration(prompts)

        assert "iteration_time" in results
        assert "trajectories_generated" in results
        assert "average_reward" in results
        assert "policy_loss" in results
        assert results["trajectories_generated"] == len(prompts)

    def test_scaling_computation(self, computational_engine):
        """Test computational scaling"""
        initial_workers = computational_engine.num_workers

        result = computational_engine.scale_computation(2.0)

        assert result["scale_factor"] == 2.0
        assert computational_engine.num_workers == initial_workers * 2
        assert result["current_workers"] == computational_engine.num_workers

    def test_metrics_collection(self, computational_engine):
        """Test metrics collection"""
        metrics = computational_engine.get_metrics()

        assert "engine_metrics" in metrics
        assert "total_trajectories" in metrics
        assert "computation_used" in metrics
        assert "active_workers" in metrics
        assert "philosophy_alignment" in metrics

        # Check philosophy alignment
        philosophy = metrics["philosophy_alignment"]
        assert philosophy["parallel_computation"] is True
        assert philosophy["hand_crafted_features"] is False
        assert philosophy["scales_with_compute"] is True


class TestMultiTurnAgent:
    """Test multi-turn conversational agent"""

    @pytest.fixture
    def dialogue_database(self):
        """Create test dialogue database"""
        dialogues = [
            {
                "id": "1",
                "content": "Hello, I need help with my order tracking",
                "category": "customer_service",
            },
            {
                "id": "2",
                "content": "I'm having trouble with my password reset",
                "category": "technical_support",
            },
        ]
        return DialogueDatabase(dialogues)

    @pytest.fixture
    def multiturn_agent(self, dialogue_database):
        """Create multi-turn agent for testing"""
        return MultiTurnAgent(
            model_config={"model_type": "test"},
            max_context_length=1024,
            max_conversation_turns=10,
            dialogue_database=dialogue_database,
        )

    @pytest.mark.asyncio
    async def test_conversation_lifecycle(self, multiturn_agent):
        """Test complete conversation lifecycle"""
        # Start conversation
        context = await multiturn_agent.start_conversation(
            user_id="test_user", initial_context={"topic": "customer_service"}
        )

        assert isinstance(context, ConversationContext)
        assert context.user_id == "test_user"
        assert context.topic == "customer_service"
        assert context.conversation_id in multiturn_agent.active_conversations

        # Continue conversation
        turns = await multiturn_agent.continue_conversation(
            context.conversation_id,
            "Hello, I need help with my order",
            strategy="customer_service",
        )

        assert len(turns) >= 2  # User message + assistant response
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

        # End conversation
        ended_context = multiturn_agent.end_conversation(context.conversation_id)
        assert ended_context is not None
        assert context.conversation_id not in multiturn_agent.active_conversations

    @pytest.mark.asyncio
    async def test_conversation_strategies(self, multiturn_agent):
        """Test different conversation strategies"""
        context = await multiturn_agent.start_conversation()

        strategies = ["default", "customer_service", "technical_support", "educational"]

        for strategy in strategies:
            response = await multiturn_agent.generate_multiturn_response(
                context.conversation_id,
                f"Test message for {strategy}",
                strategy=strategy,
            )

            assert isinstance(response, str)
            assert len(response) > 0

    def test_dialogue_database_search(self, dialogue_database):
        """Test dialogue database search functionality"""
        results = dialogue_database.search("order tracking", top_k=1)

        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert "relevance_score" in results[0]

        # Test specific dialogue retrieval
        dialogue = dialogue_database.get_dialogue_by_id("2")
        assert dialogue is not None
        assert dialogue["category"] == "technical_support"

    @pytest.mark.asyncio
    async def test_reward_computation(self, multiturn_agent, test_reward_function):
        """Test multi-turn reward computation"""
        context = await multiturn_agent.start_conversation()

        # Add some conversation turns
        await multiturn_agent.continue_conversation(
            context.conversation_id, "Hello", strategy="default"
        )

        # Compute rewards
        reward_functions = [test_reward_function]
        total_reward, breakdown = await multiturn_agent.compute_multiturn_rewards(
            context.conversation_id, reward_functions
        )

        assert isinstance(total_reward, float)
        assert isinstance(breakdown, dict)
        assert "conversation_length" in breakdown
        assert "context_coherence" in breakdown


class TestRewardComponents:
    """Test reward system components"""

    @pytest.mark.asyncio
    async def test_empathy_reward_component(self, sample_turns):
        """Test empathy reward component"""
        component = EmpathyRewardComponent(weight=1.0)

        score = await component.compute_score(sample_turns)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_action_oriented_reward_component(self, sample_turns):
        """Test action-oriented reward component"""
        component = ActionOrientedRewardComponent(weight=1.0)

        score = await component.compute_score(sample_turns)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_professionalism_reward_component(self, sample_turns):
        """Test professionalism reward component"""
        component = ProfessionalismRewardComponent(weight=1.0)

        score = await component.compute_score(sample_turns)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_multi_objective_reward_function(self, sample_turns):
        """Test multi-objective reward function"""
        components = [
            EmpathyRewardComponent(weight=0.3),
            ActionOrientedRewardComponent(weight=0.4),
            ProfessionalismRewardComponent(weight=0.3),
        ]

        reward_function = MultiObjectiveRewardFunction(
            components=components, weight=1.0, normalization_method="weighted_sum"
        )

        result = await reward_function.compute_reward(sample_turns)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0
        assert "components" in result.breakdown
        assert "weights" in result.breakdown

        # Test component statistics
        stats = reward_function.get_component_statistics()
        assert len(stats) == len(components)


class TestNeuralRewardTrainer:
    """Test neural reward training system"""

    @pytest.fixture
    def sample_trajectory_data(self):
        """Create sample trajectory data"""
        from training.neural_reward_trainer import TrajectoryData

        data = []
        for i in range(10):
            data.append(
                TrajectoryData(
                    id=str(uuid.uuid4()),
                    prompt=f"Test prompt {i}",
                    response=f"Test response {i}",
                    raw_reward=np.random.uniform(0.3, 0.8),
                    learned_reward=0.0,
                    computational_cost=50.0,
                    timestamp=datetime.now(),
                )
            )

        return data

    @pytest.mark.skipif(
        not hasattr(NeuralRewardTrainer, "__init__"),
        reason="Neural reward trainer not available",
    )
    def test_neural_reward_trainer_initialization(self):
        """Test neural reward trainer initialization"""
        trainer = NeuralRewardTrainer(
            learning_rate=1e-4, per_device_train_batch_size=16, max_epochs=10
        )

        assert trainer.learning_rate == 1e-4
        assert trainer.batch_size == 16
        assert trainer.max_epochs == 10
        assert len(trainer.replay_buffer) == 0

    @pytest.mark.skipif(
        not hasattr(NeuralRewardTrainer, "__init__"),
        reason="Neural reward trainer not available",
    )
    def test_trajectory_data_management(self, sample_trajectory_data):
        """Test trajectory data management"""
        trainer = NeuralRewardTrainer()

        # Add trajectories
        trainer.add_trajectories(sample_trajectory_data)

        assert len(trainer.replay_buffer) == len(sample_trajectory_data)

        # Test sampling
        if len(trainer.replay_buffer) > 0:
            metrics = trainer.get_metrics()
            assert "replay_buffer_size" in metrics
            assert metrics["replay_buffer_size"] == len(sample_trajectory_data)


class TestTrainingConfig:
    """Test training configuration system"""

    def test_training_config_creation(self):
        """Test training configuration creation"""
        config = TrainingConfig(
            num_epochs=10,
            per_device_train_batch_size=32,
            learning_rate=1e-4,
            max_grad_norm=1.0,
            bf16=True,
        )

        assert config.num_epochs == 10
        assert config.per_device_train_batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.max_grad_norm == 1.0
        assert config.bf16 is True

    def test_config_serialization(self):
        """Test configuration serialization"""
        config = TrainingConfig(
            num_epochs=5, per_device_train_batch_size=16, learning_rate=5e-5
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["num_epochs"] == 5
        assert config_dict["per_device_train_batch_size"] == 16
        assert config_dict["learning_rate"] == 5e-5


class TestUtilities:
    """Test utility components"""

    def test_cache_service_basic(self):
        """Test basic cache service functionality"""
        cache = CacheService()

        # Test basic operations
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
        assert hasattr(cache, "delete")
        assert hasattr(cache, "clear")

    def test_monitoring_service_basic(self):
        """Test basic monitoring service functionality"""
        monitoring = MonitoringService()

        # Test basic operations
        assert hasattr(monitoring, "log_metric")
        assert hasattr(monitoring, "log_event")
        assert hasattr(monitoring, "get_metrics")


class TestIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.asyncio
    async def test_end_to_end_training_workflow(
        self, test_agent, test_environment, test_reward_function
    ):
        """Test complete end-to-end training workflow"""
        # Create computational engine
        engine = ComputationalGRPOEngine(
            agent=test_agent,
            environment=test_environment,
            reward_function=test_reward_function,
            num_workers=2,
            trajectory_batch_size=4,
        )

        # Run training iterations
        prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]

        for i in range(3):
            results = await engine.train_iteration(prompts)

            assert "iteration_time" in results
            assert "trajectories_generated" in results
            assert "average_reward" in results
            assert results["trajectories_generated"] == len(prompts)

        # Check metrics
        metrics = engine.get_metrics()
        assert metrics["total_trajectories"] >= len(prompts) * 3

    @pytest.mark.asyncio
    async def test_complete_conversation_workflow(self):
        """Test complete conversation workflow"""
        # Create multi-turn agent
        agent = MultiTurnAgent(
            model_config={"model_type": "test"}, max_conversation_turns=5
        )

        # Start conversation
        context = await agent.start_conversation(user_id="test_user")

        # Simulate multi-turn conversation
        messages = [
            "Hello, I need help",
            "Can you explain this?",
            "Thank you for your help",
        ]

        for message in messages:
            turns = await agent.continue_conversation(
                context.conversation_id, message, strategy="default"
            )

            assert len(turns) >= 2
            assert turns[-1]["role"] == "assistant"

        # Get conversation summary
        summary = agent.get_conversation_summary(context.conversation_id)
        assert summary is not None
        assert summary["turn_count"] >= len(messages) * 2

        # End conversation
        ended_context = agent.end_conversation(context.conversation_id)
        assert ended_context is not None


class TestPerformance:
    """Performance and stress tests"""

    @pytest.mark.asyncio
    async def test_parallel_trajectory_generation_performance(
        self, test_agent, test_environment, test_reward_function
    ):
        """Test performance of parallel trajectory generation"""
        engine = ComputationalGRPOEngine(
            agent=test_agent,
            environment=test_environment,
            reward_function=test_reward_function,
            num_workers=4,
            trajectory_batch_size=20,
        )

        prompts = [f"Test prompt {i}" for i in range(20)]

        import time

        start_time = time.time()

        trajectories = await engine.generate_trajectory_batch(prompts)

        elapsed_time = time.time() - start_time

        assert len(trajectories) == len(prompts)
        assert elapsed_time < 10.0  # Should complete within 10 seconds

        # Check metrics
        metrics = engine.get_metrics()
        assert metrics["trajectories_per_second"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_conversations(self):
        """Test handling multiple concurrent conversations"""
        agent = MultiTurnAgent(
            model_config={"model_type": "test"}, max_conversation_turns=10
        )

        # Start multiple conversations
        num_conversations = 5
        contexts = []

        for i in range(num_conversations):
            context = await agent.start_conversation(user_id=f"user_{i}")
            contexts.append(context)

        # Send messages to all conversations concurrently
        tasks = []
        for context in contexts:
            task = agent.continue_conversation(
                context.conversation_id,
                f"Hello from {context.user_id}",
                strategy="default",
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        assert len(results) == num_conversations
        for result in results:
            assert len(result) >= 2  # User message + assistant response

        # Verify all conversations are still active
        active_conversations = agent.get_active_conversations()
        assert len(active_conversations) == num_conversations


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
