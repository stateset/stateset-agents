"""
Comprehensive tests for hyperparameter optimization.

Tests cover:
- Search space definitions
- HPO configuration
- Optuna backend
- GRPO HPO trainer integration
- End-to-end HPO workflows
"""

import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile

# Block vllm import to avoid torchvision issues - mock it before any imports
if 'vllm' not in sys.modules:
    sys.modules['vllm'] = type(sys)('vllm')  # type: ignore

# Try imports - skip if not available
HPO_AVAILABLE = True
try:
    from stateset_agents.training.hpo import (
        SearchSpace,
        SearchDimension,
        SearchSpaceType,
        HPOConfig,
        HPOResult,
        HPOSummary,
        GRPOHPOTrainer,
        quick_hpo,
        create_grpo_search_space,
        create_customer_service_search_space,
        get_search_space,
        list_available_search_spaces,
    )

    from stateset_agents.core.agent import MultiTurnAgent, AgentConfig
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
    from stateset_agents.training.config import TrainingConfig
except (ImportError, RuntimeError) as e:
    HPO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HPO_AVAILABLE,
    reason="HPO modules not available (check transformers/torchvision compatibility)"
)


# ============================================================================
# Test Search Space Definitions
# ============================================================================

class TestSearchSpace:
    """Tests for SearchSpace and SearchDimension."""

    def test_search_dimension_float(self):
        """Test float search dimension."""
        dim = SearchDimension(
            "learning_rate",
            SearchSpaceType.FLOAT,
            low=1e-6,
            high=1e-3
        )
        assert dim.name == "learning_rate"
        assert dim.type == SearchSpaceType.FLOAT
        assert dim.low == 1e-6
        assert dim.high == 1e-3

    def test_search_dimension_categorical(self):
        """Test categorical search dimension."""
        dim = SearchDimension(
            "optimizer",
            SearchSpaceType.CATEGORICAL,
            choices=["adam", "adamw", "sgd"]
        )
        assert dim.name == "optimizer"
        assert dim.choices == ["adam", "adamw", "sgd"]

    def test_search_dimension_validation(self):
        """Test search dimension validation."""
        # Missing bounds for numeric parameter
        with pytest.raises(ValueError):
            SearchDimension(
                "learning_rate",
                SearchSpaceType.FLOAT
            )

        # Missing choices for categorical
        with pytest.raises(ValueError):
            SearchDimension(
                "optimizer",
                SearchSpaceType.CATEGORICAL
            )

    def test_search_space_creation(self):
        """Test search space creation."""
        dimensions = [
            SearchDimension("lr", SearchSpaceType.LOGUNIFORM, 1e-6, 1e-3),
            SearchDimension("batch_size", SearchSpaceType.CHOICE, choices=[16, 32, 64])
        ]
        space = SearchSpace(dimensions)
        assert len(space.dimensions) == 2

    def test_search_space_get_dimension(self):
        """Test getting dimension by name."""
        space = SearchSpace([
            SearchDimension("lr", SearchSpaceType.FLOAT, 0.0, 1.0),
            SearchDimension("gamma", SearchSpaceType.FLOAT, 0.9, 0.99)
        ])
        lr_dim = space.get_dimension("lr")
        assert lr_dim is not None
        assert lr_dim.name == "lr"

        none_dim = space.get_dimension("nonexistent")
        assert none_dim is None

    def test_search_space_serialization(self):
        """Test search space to/from dict."""
        space = SearchSpace([
            SearchDimension("lr", SearchSpaceType.LOGUNIFORM, 1e-6, 1e-3)
        ])

        # To dict
        space_dict = space.to_dict()
        assert "dimensions" in space_dict

        # From dict
        loaded_space = SearchSpace.from_dict(space_dict)
        assert len(loaded_space.dimensions) == 1
        assert loaded_space.dimensions[0].name == "lr"


class TestPredefinedSearchSpaces:
    """Tests for pre-defined search spaces."""

    def test_create_grpo_search_space(self):
        """Test GRPO search space creation."""
        space = create_grpo_search_space()
        assert len(space.dimensions) > 0

        # Check key parameters
        lr_dim = space.get_dimension("learning_rate")
        assert lr_dim is not None
        assert lr_dim.type == SearchSpaceType.LOGUNIFORM

    def test_create_grpo_search_space_options(self):
        """Test GRPO search space with different options."""
        # With value function
        space1 = create_grpo_search_space(include_value_function=True)
        assert space1.get_dimension("gae_lambda") is not None

        # Without value function
        space2 = create_grpo_search_space(include_value_function=False)
        assert space2.get_dimension("gae_lambda") is None

    def test_create_customer_service_search_space(self):
        """Test customer service search space."""
        space = create_customer_service_search_space()
        assert space.get_dimension("helpfulness_weight") is not None
        assert space.get_dimension("safety_weight") is not None

    def test_get_search_space(self):
        """Test getting search space by name."""
        space = get_search_space("grpo")
        assert len(space.dimensions) > 0

        # Invalid name
        with pytest.raises(ValueError):
            get_search_space("invalid_name")

    def test_list_available_search_spaces(self):
        """Test listing available search spaces."""
        available = list_available_search_spaces()
        assert "grpo" in available
        assert "customer_service" in available
        assert "conservative" in available


# ============================================================================
# Test HPO Configuration
# ============================================================================

class TestHPOConfig:
    """Tests for HPO configuration."""

    def test_hpo_config_creation(self):
        """Test basic HPO config creation."""
        config = HPOConfig(
            backend="optuna",
            search_space_name="grpo",
            n_trials=50
        )
        assert config.backend == "optuna"
        assert config.n_trials == 50

    def test_hpo_config_validation(self):
        """Test HPO config validation."""
        # Invalid backend
        with pytest.raises(ValueError):
            HPOConfig(
                backend="invalid",
                search_space_name="grpo"
            )

        # Invalid direction
        with pytest.raises(ValueError):
            HPOConfig(
                backend="optuna",
                search_space_name="grpo",
                direction="invalid"
            )

    def test_hpo_config_output_dir(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "hpo_results"
            config = HPOConfig(
                backend="optuna",
                search_space_name="grpo",
                output_dir=output_dir
            )
            assert config.output_dir.exists()

    def test_hpo_config_backend_specific(self):
        """Test backend-specific configuration."""
        config = HPOConfig(
            backend="optuna",
            search_space_name="grpo",
            optuna_config={
                "sampler": "tpe",
                "pruner": "median"
            }
        )
        backend_config = config.get_backend_config()
        assert backend_config["sampler"] == "tpe"


# ============================================================================
# Test HPO Results
# ============================================================================

class TestHPOResults:
    """Tests for HPO result classes."""

    def test_hpo_result_creation(self):
        """Test HPO result creation."""
        result = HPOResult(
            trial_id="trial_0",
            params={"learning_rate": 1e-5, "gamma": 0.99},
            metrics={"reward": 0.85},
            best_metric=0.85,
            training_time=120.5,
            status="success"
        )
        assert result.trial_id == "trial_0"
        assert result.best_metric == 0.85

    def test_hpo_result_serialization(self):
        """Test HPO result serialization."""
        result = HPOResult(
            trial_id="trial_0",
            params={"lr": 1e-5},
            metrics={"reward": 0.85},
            best_metric=0.85,
            training_time=100.0
        )

        result_dict = result.to_dict()
        assert "trial_id" in result_dict
        assert result_dict["params"]["lr"] == 1e-5

    def test_hpo_result_save_load(self):
        """Test saving and loading HPO result."""
        result = HPOResult(
            trial_id="trial_0",
            params={"lr": 1e-5},
            metrics={"reward": 0.85},
            best_metric=0.85,
            training_time=100.0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "result.json"
            result.save(save_path)
            assert save_path.exists()

    def test_hpo_summary_creation(self):
        """Test HPO summary creation."""
        results = [
            HPOResult("trial_0", {"lr": 1e-5}, {"reward": 0.8}, 0.8, 100.0),
            HPOResult("trial_1", {"lr": 5e-6}, {"reward": 0.85}, 0.85, 100.0),
            HPOResult("trial_2", {"lr": 1e-4}, {"reward": 0.75}, 0.75, 100.0),
        ]

        summary = HPOSummary(
            best_params={"lr": 5e-6},
            best_metric=0.85,
            best_trial_id="trial_1",
            total_trials=3,
            successful_trials=3,
            all_results=results
        )

        assert summary.best_metric == 0.85
        assert len(summary.all_results) == 3


# ============================================================================
# Test Optuna Backend (with mocking)
# ============================================================================

@pytest.mark.skipif(
    not hasattr(__import__('training.hpo'), '__optuna_available__') or
    not __import__('training.hpo').__optuna_available__,
    reason="Optuna not installed"
)
class TestOptunaBackend:
    """Tests for Optuna backend."""

    def test_optuna_backend_creation(self):
        """Test Optuna backend creation."""
        from stateset_agents.training.hpo.optuna_backend import OptunaBackend

        search_space = create_grpo_search_space()
        backend = OptunaBackend(
            search_space=search_space,
            study_name="test_study",
            sampler="tpe",
            pruner="median"
        )

        assert backend.study_name == "test_study"
        assert backend.sampler_name == "tpe"

    @pytest.mark.asyncio
    async def test_optuna_optimize_simple(self):
        """Test simple Optuna optimization."""
        from stateset_agents.training.hpo.optuna_backend import OptunaBackend

        search_space = SearchSpace([
            SearchDimension("x", SearchSpaceType.FLOAT, -10.0, 10.0)
        ])

        backend = OptunaBackend(
            search_space=search_space,
            objective_metric="value"
        )

        # Simple quadratic objective
        def objective(params):
            x = params["x"]
            return -(x - 2.0) ** 2  # Maximum at x=2

        summary = await backend.optimize(objective, n_trials=10)

        assert summary.total_trials == 10
        assert summary.successful_trials > 0

        # Best x should be close to 2.0
        best_x = summary.best_params["x"]
        assert abs(best_x - 2.0) < 2.0  # Within reasonable range


# ============================================================================
# Test GRPO HPO Trainer Integration
# ============================================================================

class TestGRPOHPOTrainer:
    """Tests for GRPO HPO trainer integration."""

    def setup_method(self):
        """Setup test components."""
        # Create stub agent
        self.agent_config = AgentConfig(
            model_name="stub://demo",
            use_stub_model=True,
            stub_responses=["Test response"]
        )
        self.agent = MultiTurnAgent(self.agent_config)

        # Create environment
        scenarios = [
            {
                "conversation_id": "test_1",
                "user_inputs": ["Hello", "How are you?"],
                "max_turns": 2
            }
        ]
        self.environment = ConversationEnvironment(scenarios)

        # Create reward
        self.reward_function = CompositeReward([
            (HelpfulnessReward(), 0.5),
            (SafetyReward(), 0.5)
        ])

        # Base config
        self.base_config = TrainingConfig(
            learning_rate=1e-5,
            num_episodes=10,
            output_dir="./test_output"
        )

    def test_grpo_hpo_trainer_creation(self):
        """Test GRPO HPO trainer creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hpo_config = HPOConfig(
                backend="optuna",
                search_space_name="grpo",
                n_trials=5,
                output_dir=Path(tmpdir)
            )

            trainer = GRPOHPOTrainer(
                agent=self.agent,
                environment=self.environment,
                reward_function=self.reward_function,
                base_config=self.base_config,
                hpo_config=hpo_config
            )

            assert trainer.hpo_config.n_trials == 5
            assert trainer.search_space is not None

    def test_create_training_config(self):
        """Test creating training config with HPO params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hpo_config = HPOConfig(
                backend="optuna",
                search_space_name="grpo",
                output_dir=Path(tmpdir)
            )

            trainer = GRPOHPOTrainer(
                agent=self.agent,
                environment=self.environment,
                reward_function=self.reward_function,
                base_config=self.base_config,
                hpo_config=hpo_config
            )

            # Override params
            params = {
                "learning_rate": 5e-6,
                "gamma": 0.95
            }

            config = trainer._create_training_config(params)
            assert config.learning_rate == 5e-6
            assert config.gamma == 0.95

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(__import__('training.hpo'), '__optuna_available__') or
        not __import__('training.hpo').__optuna_available__,
        reason="Optuna not installed"
    )
    async def test_grpo_hpo_trainer_optimize(self):
        """Test full HPO optimization (integration test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hpo_config = HPOConfig(
                backend="optuna",
                search_space_name="conservative",  # Small search space
                n_trials=3,  # Very few trials for speed
                output_dir=Path(tmpdir)
            )

            trainer = GRPOHPOTrainer(
                agent=self.agent,
                environment=self.environment,
                reward_function=self.reward_function,
                base_config=self.base_config,
                hpo_config=hpo_config
            )

            # Mock the training to return a dummy metric
            async def mock_objective(params):
                return 0.5 + params.get("learning_rate", 0.0) * 1000

            trainer._objective_function = mock_objective

            summary = await trainer.optimize()

            assert summary.total_trials == 3
            assert summary.best_params is not None


# ============================================================================
# Test Quick HPO Utility
# ============================================================================

class TestQuickHPO:
    """Tests for quick HPO utility function."""

    def setup_method(self):
        """Setup test components."""
        self.agent_config = AgentConfig(
            model_name="stub://demo",
            use_stub_model=True,
            stub_responses=["Test"]
        )
        self.agent = MultiTurnAgent(self.agent_config)

        scenarios = [{"conversation_id": "test", "user_inputs": ["Hi"], "max_turns": 1}]
        self.environment = ConversationEnvironment(scenarios)

        self.reward_function = HelpfulnessReward()

        self.base_config = TrainingConfig(
            learning_rate=1e-5,
            num_episodes=5,
            output_dir="./test_output"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(__import__('training.hpo'), '__optuna_available__') or
        not __import__('training.hpo').__optuna_available__,
        reason="Optuna not installed"
    )
    async def test_quick_hpo(self):
        """Test quick HPO utility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the actual training
            with patch('training.hpo.grpo_hpo_trainer.MultiTurnGRPOTrainer') as mock_trainer:
                mock_instance = AsyncMock()
                mock_instance.train = AsyncMock(return_value={"reward": 0.8})
                mock_trainer.return_value = mock_instance

                summary = await quick_hpo(
                    agent=self.agent,
                    environment=self.environment,
                    reward_function=self.reward_function,
                    base_config=self.base_config,
                    n_trials=2,
                    output_dir=Path(tmpdir)
                )

                # Basic validation
                assert summary is not None
                assert summary.total_trials == 2


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
