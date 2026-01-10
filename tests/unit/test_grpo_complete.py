"""
Comprehensive tests for complete GRPO implementation

Tests cover:
- Value function with GAE
- Policy gradient computation
- GRPO loss computation with advantages
- Full training loop integration
- KL divergence regularization
"""

import asyncio
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import framework components
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import create_customer_service_reward
from stateset_agents.core.trajectory import MultiTurnTrajectory, ConversationTurn, TrajectoryGroup
from stateset_agents.core.value_function import ValueFunction, ValueHead, create_value_function
from stateset_agents.core.computational_engine import ComputationalGRPOEngine, ComputationalTrajectory
from training.trainer import MultiTurnGRPOTrainer
from training.config import TrainingConfig


class TestValueFunction:
    """Test value function implementation"""

    def test_value_head_creation(self):
        """Test value head can be created"""
        hidden_size = 768
        value_head = ValueHead(hidden_size, dropout=0.1)

        assert value_head is not None
        assert isinstance(value_head, torch.nn.Module)

    def test_value_head_forward(self):
        """Test value head forward pass"""
        hidden_size = 768
        batch_size = 2
        seq_len = 10

        value_head = ValueHead(hidden_size)

        # Create dummy hidden states
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Forward pass
        values = value_head(hidden_states)

        assert values.shape == (batch_size, seq_len, 1)

    def test_value_function_creation(self):
        """Test value function can be created with a model"""
        # Create mock model
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, gamma=0.99, gae_lambda=0.95)

        assert value_fn is not None
        assert value_fn.gamma == 0.99
        assert value_fn.gae_lambda == 0.95

    def test_compute_grpo_advantages(self):
        """Test GRPO advantage computation (group-relative)"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model)

        # Test rewards
        group_rewards = [0.8, 0.5, 0.9, 0.6, 0.7]

        advantages = value_fn.compute_grpo_advantages(group_rewards, baseline_type="group_mean")

        # Check shape
        assert len(advantages) == len(group_rewards)

        # Check that advantages are relative to group mean
        assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)

        # Check normalization
        if len(advantages) > 1:
            assert torch.isclose(advantages.std(), torch.tensor(1.0), atol=1e-1)

    def test_gae_computation(self):
        """Test Generalized Advantage Estimation"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        # Disable normalization to test raw GAE math
        value_fn = ValueFunction(mock_model, gamma=0.99, gae_lambda=0.95, normalize_advantages=False)

        # Create test data
        rewards = [1.0, 0.5, 0.8, 0.3]
        values = torch.tensor([0.5, 0.4, 0.6, 0.2], dtype=torch.float32)

        advantages, returns = value_fn.compute_gae(rewards, values)

        # Check shapes
        assert len(advantages) == len(rewards)
        assert len(returns) == len(rewards)

        # Check that returns = advantages + values (only holds when not normalized)
        assert torch.allclose(returns, advantages + values, atol=1e-5)

    def test_create_value_function_convenience(self):
        """Test convenience function for creating value function"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = create_value_function(mock_model, gamma=0.95, gae_lambda=0.90)

        assert value_fn is not None
        assert value_fn.gamma == 0.95
        assert value_fn.gae_lambda == 0.90


class TestComputationalEngine:
    """Test computational GRPO engine"""

    @pytest.mark.asyncio
    async def test_engine_creation(self):
        """Test engine can be created"""
        # Create stub agent
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        # Create simple environment
        env = ConversationEnvironment(
            scenarios=[{"topic": "test", "context": "test scenario"}],
            max_turns=2,
        )

        # Create reward function
        reward_fn = create_customer_service_reward()

        # Create engine
        engine = ComputationalGRPOEngine(
            agent=agent,
            environment=env,
            reward_function=reward_fn,
            num_workers=2,
        )

        assert engine is not None
        assert engine.num_workers == 2

    @pytest.mark.asyncio
    async def test_advantage_computation(self):
        """Test advantage computation in engine"""
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        env = ConversationEnvironment(
            scenarios=[{"topic": "test", "context": "test"}],
            max_turns=2,
        )
        reward_fn = create_customer_service_reward()

        engine = ComputationalGRPOEngine(agent, env, reward_fn)

        # Create mock trajectories
        trajectories = [
            ComputationalTrajectory(
                id=f"traj_{i}",
                prompt="Test prompt",
                response="Test response",
                raw_reward_signal=0.5,
                learned_reward=reward,
                computational_cost=10.0,
                timestamp=None,
            )
            for i, reward in enumerate([0.8, 0.5, 0.9, 0.6])
        ]

        advantages = engine._compute_advantages(trajectories)

        # Check shape
        assert len(advantages) == len(trajectories)

        # Check group-relative property (mean should be ~0)
        assert np.abs(np.mean(advantages)) < 0.01


class TestGRPOTrainer:
    """Test GRPO trainer implementation"""

    @pytest.mark.asyncio
    async def test_trainer_initialization(self):
        """Test trainer can be initialized"""
        # Create agent
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        # Create environment
        env = ConversationEnvironment(
            scenarios=[{"topic": "test", "context": "test"}],
            max_turns=2,
        )

        # Create reward
        reward_fn = create_customer_service_reward()

        # Create training config
        train_config = TrainingConfig(
            num_episodes=10,
            learning_rate=1e-5,
            per_device_train_batch_size=1,
        )

        # Create trainer
        trainer = MultiTurnGRPOTrainer(
            agent=agent,
            environment=env,
            reward_fn=reward_fn,
            config=train_config,
        )

        assert trainer is not None
        assert trainer.agent == agent

    @pytest.mark.asyncio
    async def test_advantage_computation_variants(self):
        """Test different advantage computation methods"""
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        env = ConversationEnvironment(scenarios=[{"topic": "test", "context": "test"}], max_turns=2)
        reward_fn = create_customer_service_reward()

        # Test group_mean baseline
        train_config = TrainingConfig(baseline_type="group_mean")
        trainer = MultiTurnGRPOTrainer(agent, env, reward_fn, train_config)

        trajectory = MultiTurnTrajectory()
        trajectory.add_turn(ConversationTurn(role="user", content="Hello"))
        trajectory.add_turn(ConversationTurn(role="assistant", content="Hi there"))
        trajectory.total_reward = 0.8

        group = TrajectoryGroup(
            scenario_id="test",
            trajectories=[
                trajectory,
                MultiTurnTrajectory(total_reward=0.5),
                MultiTurnTrajectory(total_reward=0.9),
            ],
        )

        loss_dict = trainer.compute_grpo_loss([group])

        assert "policy_loss" in loss_dict
        assert "mean_advantage" in loss_dict
        assert "advantage_std" in loss_dict

    @pytest.mark.asyncio
    async def test_trajectory_generation(self):
        """Test trajectory generation for training"""
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        scenarios = [
            {"id": "test1", "topic": "greeting", "context": "Say hello"},
            {"id": "test2", "topic": "question", "context": "Ask a question"},
        ]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=2)
        reward_fn = create_customer_service_reward()

        train_config = TrainingConfig(num_generations=2)
        trainer = MultiTurnGRPOTrainer(agent, env, reward_fn, train_config)

        await trainer.initialize()

        trajectory_groups = await trainer.generate_trajectories(scenarios, num_generations=2)

        assert len(trajectory_groups) > 0
        assert all(len(g.trajectories) > 0 for g in trajectory_groups)

    @pytest.mark.asyncio
    async def test_grpo_loss_with_kl_penalty(self):
        """Test enhanced GRPO loss with KL divergence penalty"""
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        env = ConversationEnvironment(scenarios=[{"topic": "test", "context": "test"}], max_turns=2)
        reward_fn = create_customer_service_reward()

        # Enable reference model and KL penalty
        train_config = TrainingConfig(
            beta=0.1,  # KL penalty coefficient
            use_reference_model=True,
        )
        trainer = MultiTurnGRPOTrainer(agent, env, reward_fn, train_config)
        await trainer.initialize()

        # Create test trajectory group
        trajectory = MultiTurnTrajectory()
        trajectory.add_turn(ConversationTurn(role="user", content="Hello"))
        trajectory.add_turn(ConversationTurn(role="assistant", content="Hi"))
        trajectory.total_reward = 0.8

        group = TrajectoryGroup(
            scenario_id="test",
            trajectories=[trajectory],
        )

        # Compute loss with KL penalty
        loss_dict = trainer.compute_enhanced_grpo_loss([group], beta=0.1)

        assert "total_loss" in loss_dict
        assert "policy_loss" in loss_dict
        assert "kl_penalty" in loss_dict

    @pytest.mark.asyncio
    async def test_training_step(self):
        """Test a single training step"""
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        env = ConversationEnvironment(scenarios=[{"topic": "test", "context": "test"}], max_turns=2)
        reward_fn = create_customer_service_reward()

        train_config = TrainingConfig(
            learning_rate=1e-5,
            clip_ratio=0.2,
            gradient_accumulation_steps=1,
        )
        trainer = MultiTurnGRPOTrainer(agent, env, reward_fn, train_config)
        await trainer.initialize()

        # Create test trajectory
        trajectory = MultiTurnTrajectory()
        trajectory.add_turn(ConversationTurn(role="user", content="Test question"))
        trajectory.add_turn(ConversationTurn(role="assistant", content="Test answer"))
        trajectory.total_reward = 0.7

        group = TrajectoryGroup(scenario_id="test", trajectories=[trajectory])

        # Execute training step
        metrics = await trainer.training_step([group])

        assert "total_loss" in metrics
        assert "learning_rate" in metrics
        assert "global_step" in metrics
        assert trainer.global_step == 1


class TestPolicyGradientComputation:
    """Test policy gradient computation details"""

    @pytest.mark.asyncio
    async def test_policy_loss_computation(self):
        """Test that policy loss is computed correctly"""
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        env = ConversationEnvironment(scenarios=[{"topic": "test", "context": "test"}], max_turns=2)
        reward_fn = create_customer_service_reward()

        train_config = TrainingConfig()
        trainer = MultiTurnGRPOTrainer(agent, env, reward_fn, train_config)
        await trainer.initialize()

        # Create trajectory with known reward
        trajectory = MultiTurnTrajectory()
        trajectory.add_turn(ConversationTurn(role="user", content="Hello"))
        trajectory.add_turn(ConversationTurn(role="assistant", content="Hi there!"))
        trajectory.total_reward = 0.8

        group = TrajectoryGroup(scenario_id="test", trajectories=[trajectory])

        # Compute loss
        advantages = torch.tensor([0.5])
        loss = trainer._compute_group_policy_loss(group, advantages)

        # Loss should be a tensor with gradient
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    @pytest.mark.asyncio
    async def test_ppo_clipping(self):
        """Test PPO-style clipping is applied"""
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        env = ConversationEnvironment(scenarios=[{"topic": "test", "context": "test"}], max_turns=2)
        reward_fn = create_customer_service_reward()

        # Set explicit clip ratio
        train_config = TrainingConfig(clip_ratio=0.2)
        trainer = MultiTurnGRPOTrainer(agent, env, reward_fn, train_config)
        await trainer.initialize()

        trajectory = MultiTurnTrajectory()
        trajectory.add_turn(ConversationTurn(role="user", content="Test"))
        trajectory.add_turn(ConversationTurn(role="assistant", content="Response"))
        trajectory.total_reward = 1.0

        group = TrajectoryGroup(scenario_id="test", trajectories=[trajectory])

        # Test with large advantage (should be clipped)
        large_advantage = torch.tensor([2.0])
        loss_clipped = trainer._compute_group_policy_loss(group, large_advantage)

        # Test with normal advantage
        normal_advantage = torch.tensor([0.1])
        loss_normal = trainer._compute_group_policy_loss(group, normal_advantage)

        # Both should produce valid losses
        assert isinstance(loss_clipped, torch.Tensor)
        assert isinstance(loss_normal, torch.Tensor)


class TestIntegration:
    """Integration tests for full GRPO pipeline"""

    @pytest.mark.asyncio
    async def test_full_grpo_pipeline(self):
        """Test complete GRPO training pipeline"""
        # Create agent
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        # Create environment with scenarios
        scenarios = [
            {"id": "s1", "topic": "greeting", "context": "Greet the user"},
            {"id": "s2", "topic": "help", "context": "Offer assistance"},
        ]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=2)

        # Create reward function
        reward_fn = create_customer_service_reward()

        # Create training config
        train_config = TrainingConfig(
            num_episodes=2,  # Small for testing
            num_generations=2,
            learning_rate=1e-5,
            logging_steps=1,
            eval_steps=2,
            save_steps=10,
            gradient_accumulation_steps=1,
        )

        # Create trainer
        trainer = MultiTurnGRPOTrainer(agent, env, reward_fn, train_config)
        await trainer.initialize()

        # Generate trajectories
        trajectory_groups = await trainer.generate_trajectories(scenarios, num_generations=2)

        assert len(trajectory_groups) > 0

        # Compute loss
        loss_dict = trainer.compute_grpo_loss(trajectory_groups)

        assert "policy_loss" in loss_dict
        assert "total_loss" in loss_dict

        # Execute training step
        metrics = await trainer.training_step(trajectory_groups)

        assert "total_loss" in metrics
        assert "learning_rate" in metrics
        assert trainer.global_step == 1

    @pytest.mark.asyncio
    async def test_engine_with_real_agent(self):
        """Test computational engine with stub agent"""
        config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(config)
        await agent.initialize()

        env = ConversationEnvironment(
            scenarios=[{"topic": "test", "context": "Test scenario"}],
            max_turns=2,
        )
        reward_fn = create_customer_service_reward()

        engine = ComputationalGRPOEngine(
            agent=agent,
            environment=env,
            reward_function=reward_fn,
            trajectory_batch_size=4,
        )

        # Run training iteration
        prompts = ["Hello", "How can I help?", "What is your question?"]
        results = await engine.train_iteration(prompts)

        assert "iteration_time" in results
        assert "trajectories_generated" in results
        assert "average_reward" in results
        assert "policy_loss" in results
        assert results["trajectories_generated"] == len(prompts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
