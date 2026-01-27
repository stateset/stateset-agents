"""
Comprehensive tests for all new 10/10 features:
- Offline RL (CQL/IQL)
- Bayesian Reward Models
- Few-Shot Adaptation
"""

import pytest
import numpy as np

# Track availability
FEATURES_AVAILABLE = True

# Test Offline RL imports
try:
    import torch

    from stateset_agents.training.offline_rl_algorithms import (
        CQLConfig,
        IQLConfig,
        ConservativeQLearning,
        ImplicitQLearning,
        OfflineRLTrainer,
    )

    TORCH_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCH_AVAILABLE = False
    FEATURES_AVAILABLE = False

# Test Bayesian Reward imports
try:
    from stateset_agents.rewards.bayesian_reward_model import (
        BayesianRewardConfig,
        BayesianRewardFunction,
        ActiveLearningSelector,
    )
    from stateset_agents.core.reward import RewardResult
    from stateset_agents.core.trajectory import ConversationTurn
except (ImportError, RuntimeError):
    FEATURES_AVAILABLE = False

# Test Few-Shot Adaptation imports
try:
    from stateset_agents.core.few_shot_adaptation import (
        FewShotExample,
        DomainProfile,
        PromptBasedAdaptation,
        FewShotAdaptationManager,
        DomainDetector,
    )
except (ImportError, RuntimeError):
    FEATURES_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FEATURES_AVAILABLE,
    reason="New feature modules not available (check dependencies)"
)


# ============================================================================
# Offline RL Tests
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOfflineRL:
    """Test Offline RL algorithms (CQL and IQL)"""

    def test_cql_config(self):
        config = CQLConfig(
            hidden_size=128,
            cql_alpha=2.0,
            learning_rate=1e-3,
        )

        assert config.hidden_size == 128
        assert config.cql_alpha == 2.0
        assert config.learning_rate == 1e-3

    def test_iql_config(self):
        config = IQLConfig(
            hidden_size=256,
            expectile=0.7,
            temperature=3.0,
        )

        assert config.hidden_size == 256
        assert config.expectile == 0.7
        assert config.temperature == 3.0

    def test_cql_initialization(self):
        cql = ConservativeQLearning(
            state_dim=10,
            action_dim=5,
            device="cpu",
        )

        assert cql.state_dim == 10
        assert cql.action_dim == 5
        assert cql.q1 is not None
        assert cql.q2 is not None

    def test_iql_initialization(self):
        iql = ImplicitQLearning(
            state_dim=10,
            action_dim=5,
            device="cpu",
        )

        assert iql.state_dim == 10
        assert iql.action_dim == 5
        assert iql.q1 is not None
        assert iql.value_net is not None

    def test_cql_train_step(self):
        cql = ConservativeQLearning(
            state_dim=4,
            action_dim=2,
            device="cpu",
        )

        # Create dummy batch
        states = torch.randn(32, 4)
        actions = torch.randn(32, 2)
        rewards = torch.randn(32, 1)
        next_states = torch.randn(32, 4)
        dones = torch.zeros(32, 1)

        metrics = cql.train_step(states, actions, rewards, next_states, dones)

        assert "bellman_loss" in metrics
        assert "cql_loss" in metrics
        assert "total_loss" in metrics

    def test_iql_train_step(self):
        iql = ImplicitQLearning(
            state_dim=4,
            action_dim=2,
            device="cpu",
        )

        # Create dummy batch
        states = torch.randn(32, 4)
        actions = torch.randn(32, 2)
        rewards = torch.randn(32, 1)
        next_states = torch.randn(32, 4)
        dones = torch.zeros(32, 1)

        metrics = iql.train_step(states, actions, rewards, next_states, dones)

        assert "value_loss" in metrics
        assert "q_loss" in metrics

    def test_offline_trainer_initialization(self):
        trainer = OfflineRLTrainer(
            algorithm="cql",
            state_dim=8,
            action_dim=4,
            device="cpu",
        )

        assert trainer.algorithm == "cql"
        assert trainer.learner is not None

    def test_offline_trainer_invalid_algorithm(self):
        with pytest.raises(ValueError):
            OfflineRLTrainer(
                algorithm="invalid",
                state_dim=8,
                action_dim=4,
            )

    def test_offline_trainer_training(self):
        trainer = OfflineRLTrainer(
            algorithm="iql",
            state_dim=4,
            action_dim=2,
            device="cpu",
        )

        # Create small dummy dataset
        dataset = {
            "states": np.random.randn(100, 4),
            "actions": np.random.randn(100, 2),
            "rewards": np.random.randn(100),
            "next_states": np.random.randn(100, 4),
            "dones": np.zeros(100),
        }

        metrics = trainer.train(dataset, num_epochs=2, batch_size=32)

        assert len(metrics) == 2  # 2 epochs
        assert "epoch" in metrics[0]


# ============================================================================
# Bayesian Reward Model Tests
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBayesianRewardModel:
    """Test Bayesian reward models with uncertainty"""

    def test_bayesian_config(self):
        config = BayesianRewardConfig(
            hidden_size=128,
            num_samples=10,
            num_ensemble=3,
        )

        assert config.hidden_size == 128
        assert config.num_samples == 10
        assert config.num_ensemble == 3

    @pytest.mark.asyncio
    async def test_bayesian_reward_function(self):
        reward_fn = BayesianRewardFunction(
            input_dim=768,
            device="cpu",
        )

        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there!"),
        ]

        result = await reward_fn.compute_reward(turns)

        assert isinstance(result, RewardResult)
        assert "epistemic_uncertainty" in result.breakdown
        assert "aleatoric_uncertainty" in result.breakdown
        assert "total_uncertainty" in result.breakdown
        assert "confidence_interval_lower" in result.breakdown
        assert "confidence_interval_upper" in result.breakdown

    @pytest.mark.asyncio
    async def test_uncertainty_high_low(self):
        config = BayesianRewardConfig(
            high_uncertainty_threshold=0.1,  # Low threshold
        )

        reward_fn = BayesianRewardFunction(input_dim=768, config=config, device="cpu")

        turns = [ConversationTurn(role="assistant", content="Test")]
        result = await reward_fn.compute_reward(turns)

        # Check high uncertainty flag exists
        assert "high_uncertainty" in result.metadata

    def test_active_learning_selector(self):
        selector = ActiveLearningSelector(
            uncertainty_threshold=0.3,
            diversity_weight=0.5,
        )

        # High uncertainty result
        result_high = RewardResult(
            score=0.5,
            breakdown={"total_uncertainty": 0.5},
        )

        assert selector.should_query_label(result_high)

        # Low uncertainty result
        result_low = RewardResult(
            score=0.5,
            breakdown={"total_uncertainty": 0.1},
        )

        assert not selector.should_query_label(result_low)

    def test_calibration(self):
        reward_fn = BayesianRewardFunction(input_dim=768, device="cpu")

        predicted = [0.5, 0.6, 0.7, 0.8]
        actual = [0.55, 0.65, 0.68, 0.75]

        metrics = reward_fn.calibrate(predicted, actual)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] == 4


# ============================================================================
# Few-Shot Adaptation Tests
# ============================================================================


class TestFewShotAdaptation:
    """Test few-shot adaptation mechanisms"""

    def test_few_shot_example(self):
        example = FewShotExample(
            input="What is 2+2?",
            output="4",
            reward=1.0,
        )

        assert example.input == "What is 2+2?"
        assert example.output == "4"
        assert example.reward == 1.0

    def test_domain_profile(self):
        domain = DomainProfile(
            domain_id="math",
            name="Mathematics",
            description="Mathematical problem solving",
            keywords=["math", "calculation", "equation"],
        )

        assert domain.domain_id == "math"
        assert domain.name == "Mathematics"
        assert "math" in domain.keywords

    @pytest.mark.asyncio
    async def test_prompt_based_adaptation(self):
        from stateset_agents.core.agent import Agent

        class MockAgent(Agent):
            async def generate_response(self, prompt: str, **kwargs) -> str:
                return "mock response"

        base_agent = MockAgent()

        examples = [
            FewShotExample(input="Q1", output="A1", reward=1.0),
            FewShotExample(input="Q2", output="A2", reward=0.9),
        ]

        domain = DomainProfile(
            domain_id="test",
            name="Test Domain",
            description="Test",
        )

        adapter = PromptBasedAdaptation(max_examples=2)
        adapted_agent = await adapter.adapt(base_agent, examples, domain)

        assert adapted_agent is not None

        # Test response generation
        response = await adapted_agent.generate_response("test prompt")
        assert response == "mock response"

        # Check metrics
        metrics = adapter.get_adaptation_metrics()
        assert "adaptation_count" in metrics
        assert metrics["adaptation_count"] == 1

    def test_adaptation_manager(self):
        from stateset_agents.core.agent import Agent

        class MockAgent(Agent):
            async def generate_response(self, prompt: str, **kwargs) -> str:
                return "response"

        base_agent = MockAgent()
        manager = FewShotAdaptationManager(base_agent)

        domain = DomainProfile(
            domain_id="domain1",
            name="Domain 1",
            description="Test domain",
        )

        examples = [
            FewShotExample(input="test1", output="output1"),
            FewShotExample(input="test2", output="output2"),
        ]

        manager.register_domain(domain, examples)

        assert "domain1" in manager.domain_profiles
        assert len(manager.domain_examples["domain1"]) == 2

    def test_adaptation_manager_add_examples(self):
        from stateset_agents.core.agent import Agent

        class MockAgent(Agent):
            async def generate_response(self, prompt: str, **kwargs) -> str:
                return "response"

        base_agent = MockAgent()
        manager = FewShotAdaptationManager(base_agent)

        domain = DomainProfile(domain_id="d1", name="D1", description="Test")
        manager.register_domain(domain)

        new_examples = [FewShotExample(input="q", output="a")]
        manager.add_examples("d1", new_examples)

        assert len(manager.domain_examples["d1"]) == 1

    def test_adaptation_manager_invalid_domain(self):
        from stateset_agents.core.agent import Agent

        class MockAgent(Agent):
            async def generate_response(self, prompt: str, **kwargs) -> str:
                return "response"

        base_agent = MockAgent()
        manager = FewShotAdaptationManager(base_agent)

        with pytest.raises(ValueError):
            manager.add_examples("nonexistent", [])

    @pytest.mark.asyncio
    async def test_get_adapted_agent(self):
        from stateset_agents.core.agent import Agent

        class MockAgent(Agent):
            async def generate_response(self, prompt: str, **kwargs) -> str:
                return "response"

        base_agent = MockAgent()
        manager = FewShotAdaptationManager(base_agent)

        domain = DomainProfile(domain_id="d1", name="D1", description="Test")
        examples = [FewShotExample(input="q", output="a")]
        manager.register_domain(domain, examples)

        adapted = await manager.get_adapted_agent("d1")
        assert adapted is not None

        # Should use cached version second time
        adapted2 = await manager.get_adapted_agent("d1")
        assert adapted2 is not None

    def test_domain_detector(self):
        domains = {
            "math": DomainProfile(
                domain_id="math",
                name="Math",
                description="Math",
                keywords=["calculate", "equation", "sum"],
            ),
            "coding": DomainProfile(
                domain_id="coding",
                name="Coding",
                description="Coding",
                keywords=["python", "code", "function"],
            ),
        }

        detector = DomainDetector(domains)

        # Test math detection
        domain_id, confidence = detector.detect_domain("Calculate the sum of 2 and 3")
        assert domain_id == "math"
        assert confidence > 0

        # Test coding detection
        domain_id, confidence = detector.detect_domain("Write a Python function to sort")
        assert domain_id == "coding"
        assert confidence > 0

    def test_domain_detector_no_match(self):
        domains = {
            "math": DomainProfile(
                domain_id="math",
                name="Math",
                description="Math",
                keywords=["math"],
            ),
        }

        detector = DomainDetector(domains)

        # No keywords match
        domain_id, confidence = detector.detect_domain("random text")
        assert confidence == 0.0

    def test_get_domain_statistics(self):
        from stateset_agents.core.agent import Agent

        class MockAgent(Agent):
            async def generate_response(self, prompt: str, **kwargs) -> str:
                return "response"

        base_agent = MockAgent()
        manager = FewShotAdaptationManager(base_agent)

        domain1 = DomainProfile(domain_id="d1", name="D1", description="Test")
        domain2 = DomainProfile(domain_id="d2", name="D2", description="Test")

        manager.register_domain(domain1, [FewShotExample("q1", "a1")])
        manager.register_domain(domain2, [])

        stats = manager.get_domain_statistics()

        assert stats["num_domains"] == 2
        assert "d1" in stats["domains"]
        assert "d2" in stats["domains"]
        assert stats["domains"]["d1"]["num_examples"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
