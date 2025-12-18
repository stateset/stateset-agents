"""
Unit tests for Offline RL and Sim-to-Real Transfer modules.

Tests:
- ConversationDataset and ReplayBuffer
- BCQ, BEAR algorithms
- Decision Transformer
- Domain Randomization
- Conversation Simulator
- Sim-to-Real Transfer
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


# Test ConversationDataset
class TestConversationDataset:
    """Tests for ConversationDataset and related utilities"""

    def test_conversation_dataset_config_defaults(self):
        """Test ConversationDatasetConfig has correct defaults"""
        from stateset_agents.data.conversation_dataset import ConversationDatasetConfig

        config = ConversationDatasetConfig()
        assert config.format == "jsonl"
        assert config.max_turns == 20
        assert config.normalize_rewards is True
        assert config.compute_returns is True
        assert config.discount_factor == 0.99

    def test_trajectory_data_compute_returns(self):
        """Test returns-to-go computation"""
        from stateset_agents.data.conversation_dataset import (
            TrajectoryData,
            ConversationTurnData,
        )

        turns = [
            ConversationTurnData(role="user", content="Hello"),
            ConversationTurnData(role="assistant", content="Hi there"),
        ]
        traj = TrajectoryData(
            trajectory_id="test_1",
            turns=turns,
            total_reward=1.0,
            turn_rewards=[0.0, 1.0],
        )

        returns = traj.compute_returns_to_go(gamma=0.99)
        assert len(returns) == 2
        assert returns[1] == 1.0  # Last reward
        assert returns[0] == pytest.approx(0.99, rel=1e-3)  # Discounted

    def test_conversation_dataset_from_dict(self):
        """Test creating dataset from dictionary"""
        from stateset_agents.data.conversation_dataset import ConversationDataset

        data = {
            "trajectories": [
                {
                    "trajectory_id": "traj_1",
                    "turns": [
                        {"role": "user", "content": "Hello", "reward": 0.0},
                        {"role": "assistant", "content": "Hi", "reward": 0.5},
                    ],
                    "total_reward": 0.5,
                }
            ]
        }

        dataset = ConversationDataset.from_dict(data)
        assert len(dataset) == 1
        assert dataset[0].trajectory_id == "traj_1"

    def test_conversation_dataset_statistics(self):
        """Test dataset statistics computation"""
        from stateset_agents.data.conversation_dataset import (
            ConversationDataset,
            ConversationDatasetConfig,
            TrajectoryData,
            ConversationTurnData,
        )

        trajectories = [
            TrajectoryData(
                trajectory_id="t1",
                turns=[ConversationTurnData(role="user", content="test")],
                total_reward=0.5,
                turn_rewards=[0.5],
            ),
            TrajectoryData(
                trajectory_id="t2",
                turns=[ConversationTurnData(role="user", content="test2")],
                total_reward=0.7,
                turn_rewards=[0.7],
            ),
        ]

        # Disable normalization to test raw statistics
        config = ConversationDatasetConfig(normalize_rewards=False, compute_returns=False)
        dataset = ConversationDataset(trajectories, config=config)
        stats = dataset.get_statistics()

        assert stats["num_trajectories"] == 2
        assert stats["mean_reward"] == pytest.approx(0.6, rel=1e-2)

    def test_conversation_replay_buffer(self):
        """Test ConversationReplayBuffer operations"""
        from stateset_agents.data.conversation_dataset import (
            ConversationReplayBuffer,
            TrajectoryData,
            ConversationTurnData,
        )

        buffer = ConversationReplayBuffer(capacity=10)
        assert len(buffer) == 0

        # Add trajectory
        traj = TrajectoryData(
            trajectory_id="t1",
            turns=[ConversationTurnData(role="user", content="test")],
            total_reward=0.5,
            turn_rewards=[0.5],
        )
        buffer.add_trajectory(traj)

        assert len(buffer) == 1
        assert buffer.total_turns == 1

        # Sample
        samples = buffer.sample(1)
        assert len(samples) == 1

    def test_replay_buffer_capacity(self):
        """Test buffer respects capacity limit"""
        from stateset_agents.data.conversation_dataset import (
            ConversationReplayBuffer,
            TrajectoryData,
            ConversationTurnData,
        )

        buffer = ConversationReplayBuffer(capacity=2)

        for i in range(5):
            traj = TrajectoryData(
                trajectory_id=f"t{i}",
                turns=[ConversationTurnData(role="user", content=f"test{i}")],
                total_reward=float(i),
                turn_rewards=[float(i)],
            )
            buffer.add_trajectory(traj)

        assert len(buffer) == 2  # Should only keep 2


# Test Domain Randomization
class TestDomainRandomization:
    """Tests for domain randomization utilities"""

    def test_user_persona_creation(self):
        """Test UserPersona creation and properties"""
        from stateset_agents.training.domain_randomization import UserPersona

        persona = UserPersona(
            name="Test User",
            traits={"patience": 0.8, "expertise": 0.5},
            vocabulary_style="formal",
            emotion_model="neutral",
        )

        assert persona.name == "Test User"
        assert persona.traits["patience"] == 0.8
        assert persona.vocabulary_style == "formal"

    def test_user_persona_system_prompt(self):
        """Test persona converts to system prompt"""
        from stateset_agents.training.domain_randomization import UserPersona

        persona = UserPersona(
            name="Expert",
            traits={"patience": 0.9},
            expertise_level=0.9,
        )

        prompt = persona.to_system_prompt()
        assert "Expert" in prompt
        assert "patience" in prompt.lower()

    def test_persona_generator_random(self):
        """Test random persona generation"""
        from stateset_agents.training.domain_randomization import (
            PersonaGenerator,
            DomainRandomizationConfig,
        )

        config = DomainRandomizationConfig(seed=42)
        generator = PersonaGenerator(config)

        persona1 = generator.generate_random_persona()
        persona2 = generator.generate_random_persona()

        assert persona1.name != persona2.name
        assert 0 <= persona1.patience_level <= 1

    def test_persona_interpolation(self):
        """Test persona interpolation"""
        from stateset_agents.training.domain_randomization import (
            PersonaGenerator,
            DomainRandomizationConfig,
            UserPersona,
        )

        config = DomainRandomizationConfig()
        generator = PersonaGenerator(config)

        p1 = UserPersona(name="P1", traits={"patience": 0.2})
        p2 = UserPersona(name="P2", traits={"patience": 0.8})

        interpolated = generator.interpolate_personas(p1, p2, alpha=0.5)
        assert interpolated.traits["patience"] == pytest.approx(0.5, rel=1e-2)

    def test_scenario_generator(self):
        """Test scenario generation"""
        from stateset_agents.training.domain_randomization import (
            ScenarioGenerator,
            DomainRandomizationConfig,
        )

        config = DomainRandomizationConfig(
            topics=["technical", "customer_service"],
        )
        generator = ScenarioGenerator(config)

        scenario = generator.generate_scenario()
        assert "topic" in scenario
        assert "difficulty" in scenario
        assert "context" in scenario

    def test_curriculum_sampling(self):
        """Test curriculum learning schedule"""
        from stateset_agents.training.domain_randomization import (
            ScenarioGenerator,
            DomainRandomizationConfig,
        )

        config = DomainRandomizationConfig(
            use_curriculum=True,
            initial_difficulty=0.3,
            max_difficulty=0.9,
            curriculum_steps=1000,
        )
        generator = ScenarioGenerator(config)

        early_scenario = generator.curriculum_sample(step=0)
        late_scenario = generator.curriculum_sample(step=1000)

        assert early_scenario["difficulty"] <= late_scenario["difficulty"]

    def test_domain_randomizer(self):
        """Test DomainRandomizer integration"""
        from stateset_agents.training.domain_randomization import (
            DomainRandomizer,
            DomainRandomizationConfig,
        )

        config = DomainRandomizationConfig(seed=42)
        randomizer = DomainRandomizer(config)

        scenario = randomizer.randomize_scenario()
        assert "persona" in scenario
        assert "noise" in scenario
        assert "difficulty" in scenario


# Test Sim-to-Real Metrics
class TestSimToRealMetrics:
    """Tests for sim-to-real evaluation metrics"""

    def test_kl_divergence(self):
        """Test KL divergence computation"""
        from stateset_agents.evaluation.sim_to_real_metrics import compute_kl_divergence

        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])

        kl = compute_kl_divergence(p, q)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_js_divergence(self):
        """Test Jensen-Shannon divergence"""
        from stateset_agents.evaluation.sim_to_real_metrics import compute_js_divergence

        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])

        js = compute_js_divergence(p, q)
        assert js == pytest.approx(0.0, abs=1e-6)
        assert 0 <= js <= 1

    def test_mmd_computation(self):
        """Test MMD computation"""
        from stateset_agents.evaluation.sim_to_real_metrics import compute_mmd

        x = np.random.randn(100)
        y = np.random.randn(100)

        mmd = compute_mmd(x, y)
        assert mmd >= 0

        # Same distribution should have low MMD
        mmd_same = compute_mmd(x, x)
        assert mmd_same < mmd

    def test_distribution_divergence(self):
        """Test distribution divergence computation"""
        from stateset_agents.evaluation.sim_to_real_metrics import (
            compute_distribution_divergence,
        )

        sim_values = list(np.random.randn(100))
        real_values = list(np.random.randn(100))

        kl = compute_distribution_divergence(sim_values, real_values, method="kl")
        js = compute_distribution_divergence(sim_values, real_values, method="js")
        mmd = compute_distribution_divergence(sim_values, real_values, method="mmd")

        assert kl >= 0
        assert 0 <= js <= np.log(2) + 0.1  # JS bounded by log(2)
        assert mmd >= 0

    def test_sim_to_real_metrics_dataclass(self):
        """Test SimToRealMetrics dataclass"""
        from stateset_agents.evaluation.sim_to_real_metrics import SimToRealMetrics

        metrics = SimToRealMetrics(
            response_length_kl=0.1,
            vocabulary_js_divergence=0.2,
            overall_gap=0.15,
        )

        data = metrics.to_dict()
        assert data["response_length_kl"] == 0.1
        assert data["overall_gap"] == 0.15

        # Test from_dict
        restored = SimToRealMetrics.from_dict(data)
        assert restored.response_length_kl == 0.1


# Test Offline RL Configs
class TestOfflineRLConfigs:
    """Tests for offline RL configuration classes"""

    def test_bcq_config_defaults(self):
        """Test BCQConfig default values"""
        from stateset_agents.training.offline_rl_bcq import BCQConfig

        config = BCQConfig()
        assert config.hidden_size == 256
        assert config.latent_dim == 64
        assert config.action_threshold == 0.3
        assert config.phi == 0.05
        assert config.learning_rate == 3e-4

    def test_bear_config_defaults(self):
        """Test BEARConfig default values"""
        from stateset_agents.training.offline_rl_bear import BEARConfig

        config = BEARConfig()
        assert config.hidden_size == 256
        assert config.kernel_type == "laplacian"
        assert config.mmd_sigma == 20.0
        assert config.lagrange_threshold == 0.05

    def test_decision_transformer_config_defaults(self):
        """Test DecisionTransformerConfig default values"""
        from stateset_agents.training.decision_transformer import DecisionTransformerConfig

        config = DecisionTransformerConfig()
        assert config.n_layer == 6
        assert config.n_head == 8
        assert config.n_embd == 512
        assert config.max_context_length == 20

    def test_offline_grpo_config_defaults(self):
        """Test OfflineGRPOConfig default values"""
        from stateset_agents.training.offline_grpo_trainer import OfflineGRPOConfig

        config = OfflineGRPOConfig()
        assert config.offline_algorithm == "iql"
        assert config.offline_weight == 0.5
        assert config.warmup_offline_steps == 1000

    def test_sim_to_real_config_defaults(self):
        """Test SimToRealConfig default values"""
        from stateset_agents.training.sim_to_real import SimToRealConfig

        config = SimToRealConfig()
        assert config.user_model_type == "learned"
        assert config.transfer_schedule == "cosine"
        assert config.initial_sim_ratio == 1.0
        assert config.final_sim_ratio == 0.1


# Test SimToRealTransfer
class TestSimToRealTransfer:
    """Tests for sim-to-real transfer functionality"""

    def test_transfer_ratio_warmup(self):
        """Test sim/real ratio during warmup"""
        from stateset_agents.training.sim_to_real import SimToRealTransfer, SimToRealConfig

        config = SimToRealConfig(warmup_steps=100)
        transfer = SimToRealTransfer(config)

        sim_ratio, real_ratio = transfer.get_sim_real_ratio(step=50)
        assert sim_ratio == 1.0
        assert real_ratio == 0.0

    def test_transfer_ratio_linear(self):
        """Test linear transfer schedule"""
        from stateset_agents.training.sim_to_real import SimToRealTransfer, SimToRealConfig

        config = SimToRealConfig(
            transfer_schedule="linear",
            warmup_steps=0,
            initial_sim_ratio=1.0,
            final_sim_ratio=0.0,
            transfer_steps=100,
        )
        transfer = SimToRealTransfer(config)

        sim_50, real_50 = transfer.get_sim_real_ratio(step=50)
        assert sim_50 == pytest.approx(0.5, rel=0.1)
        assert real_50 == pytest.approx(0.5, rel=0.1)

    def test_transfer_ratio_cosine(self):
        """Test cosine transfer schedule"""
        from stateset_agents.training.sim_to_real import SimToRealTransfer, SimToRealConfig

        config = SimToRealConfig(
            transfer_schedule="cosine",
            warmup_steps=0,
            transfer_steps=100,
        )
        transfer = SimToRealTransfer(config)

        _, real_early = transfer.get_sim_real_ratio(step=10)
        _, real_late = transfer.get_sim_real_ratio(step=90)

        # Cosine should have less change early and late
        assert real_early < 0.5
        assert real_late > 0.5


# Test MMD Kernel
class TestMMDKernel:
    """Tests for MMD kernel implementations"""

    def test_mmd_kernel_gaussian(self):
        """Test Gaussian kernel MMD"""
        pytest.importorskip("torch")
        import torch
        from stateset_agents.training.offline_rl_bear import MMDKernel

        kernel = MMDKernel(kernel_type="gaussian", sigma=1.0)

        x = torch.randn(10, 5).unsqueeze(0)  # [1, 10, 5]
        y = torch.randn(10, 5).unsqueeze(0)

        mmd = kernel.compute_mmd(x, y)
        assert mmd.shape == (1,)
        assert mmd.item() >= 0

    def test_mmd_kernel_laplacian(self):
        """Test Laplacian kernel MMD"""
        pytest.importorskip("torch")
        import torch
        from stateset_agents.training.offline_rl_bear import MMDKernel

        kernel = MMDKernel(kernel_type="laplacian", sigma=1.0)

        x = torch.randn(10, 5).unsqueeze(0)
        y = torch.randn(10, 5).unsqueeze(0)

        mmd = kernel.compute_mmd(x, y)
        assert mmd.shape == (1,)
        assert mmd.item() >= 0


# Test imports work correctly
class TestModuleImports:
    """Test that all modules can be imported"""

    def test_import_data_module(self):
        """Test data module imports"""
        from stateset_agents.data import (
            ConversationDataset,
            ConversationDatasetConfig,
            ConversationReplayBuffer,
        )

        assert ConversationDataset is not None
        assert ConversationDatasetConfig is not None
        assert ConversationReplayBuffer is not None

    def test_import_training_offline_rl(self):
        """Test offline RL training imports"""
        from stateset_agents.training import (
            OFFLINE_RL_AVAILABLE,
            BCQ_AVAILABLE,
            BEAR_AVAILABLE,
        )

        assert isinstance(OFFLINE_RL_AVAILABLE, bool)
        assert isinstance(BCQ_AVAILABLE, bool)
        assert isinstance(BEAR_AVAILABLE, bool)

    def test_import_evaluation_module(self):
        """Test evaluation module imports"""
        from stateset_agents.evaluation import (
            SimToRealMetrics,
            SimToRealEvaluator,
        )

        assert SimToRealMetrics is not None
        assert SimToRealEvaluator is not None

    def test_import_domain_randomization(self):
        """Test domain randomization imports"""
        from stateset_agents.training.domain_randomization import (
            UserPersona,
            DomainRandomizationConfig,
            PersonaGenerator,
            ScenarioGenerator,
            DomainRandomizer,
        )

        assert UserPersona is not None
        assert DomainRandomizationConfig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
