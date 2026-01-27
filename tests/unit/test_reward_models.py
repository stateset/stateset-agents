"""
Comprehensive Unit Tests for Reward Models

Tests for transformer-based reward models and calibration systems.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from stateset_agents.core.reward import RewardResult
from stateset_agents.core.trajectory import ConversationTurn


# Mark all tests as requiring torch
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not available"
)


class TestTransformerRewardModel:
    """Tests for TransformerRewardModel"""

    def test_model_initialization(self):
        """Test model can be initialized"""
        from stateset_agents.training.transformer_reward_model import TransformerRewardModel

        model = TransformerRewardModel(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
        )

        assert model is not None
        assert model.prompt_encoder is not None
        assert model.response_encoder is not None
        assert model.reward_head is not None

    def test_model_forward_pass(self):
        """Test forward pass through model"""
        from stateset_agents.training.transformer_reward_model import TransformerRewardModel

        model = TransformerRewardModel(hidden_dim=128, num_layers=1)
        model.eval()

        # Create dummy inputs
        batch_size = 4
        seq_len = 32
        prompt_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        prompt_attention_mask = torch.ones(batch_size, seq_len)
        response_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        response_attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        with torch.no_grad():
            rewards = model(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                response_input_ids=response_input_ids,
                response_attention_mask=response_attention_mask,
            )

        assert rewards.shape == (batch_size, 1)
        assert not torch.isnan(rewards).any()

    def test_freeze_unfreeze_encoders(self):
        """Test freezing and unfreezing encoder weights"""
        from stateset_agents.training.transformer_reward_model import TransformerRewardModel

        model = TransformerRewardModel(hidden_dim=128)

        # Initially unfrozen
        assert model.prompt_encoder.embeddings.word_embeddings.weight.requires_grad

        # Freeze
        model.freeze_encoders()
        assert not model.prompt_encoder.embeddings.word_embeddings.weight.requires_grad
        assert not model.response_encoder.embeddings.word_embeddings.weight.requires_grad

        # Reward head should still be trainable
        for param in model.reward_head.parameters():
            assert param.requires_grad

        # Unfreeze
        model.unfreeze_encoders()
        assert model.prompt_encoder.embeddings.word_embeddings.weight.requires_grad
        assert model.response_encoder.embeddings.word_embeddings.weight.requires_grad


class TestRewardExample:
    """Tests for RewardExample"""

    def test_from_conversation_turns(self):
        """Test creating RewardExample from conversation turns"""
        from stateset_agents.training.transformer_reward_model import RewardExample

        turns = [
            ConversationTurn(role="user", content="Hello, I need help"),
            ConversationTurn(role="assistant", content="I'm here to help!"),
            ConversationTurn(role="user", content="Great, thanks"),
            ConversationTurn(role="assistant", content="You're welcome!"),
        ]

        example = RewardExample.from_conversation_turns(
            turns=turns, reward=0.8, metadata={"topic": "greeting"}
        )

        assert "Hello, I need help" in example.prompt
        assert "Great, thanks" in example.prompt
        assert example.response == "You're welcome!"
        assert example.reward == 0.8
        assert example.metadata["topic"] == "greeting"


class TestRewardDataset:
    """Tests for RewardDataset"""

    def test_dataset_creation(self):
        """Test dataset can be created"""
        from transformers import AutoTokenizer
        from stateset_agents.training.transformer_reward_model import RewardDataset, RewardExample

        examples = [
            RewardExample(
                prompt="Test prompt 1", response="Test response 1", reward=0.5
            ),
            RewardExample(
                prompt="Test prompt 2", response="Test response 2", reward=0.8
            ),
        ]

        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        dataset = RewardDataset(examples, tokenizer, max_length=128)

        assert len(dataset) == 2

        # Test __getitem__
        item = dataset[0]
        assert "prompt_input_ids" in item
        assert "prompt_attention_mask" in item
        assert "response_input_ids" in item
        assert "response_attention_mask" in item
        assert "reward" in item
        assert item["reward"] == 0.5


class TestTransformerRewardTrainer:
    """Tests for TransformerRewardTrainer"""

    def test_trainer_initialization(self):
        """Test trainer initialization"""
        from stateset_agents.training.transformer_reward_model import (
            RewardTrainingConfig,
            TransformerRewardTrainer,
        )

        config = RewardTrainingConfig(
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=8,
            num_epochs=2,
            device="cpu",
        )

        trainer = TransformerRewardTrainer(config=config)

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.config.batch_size == 8
        assert trainer.config.num_epochs == 2

    def test_predict(self):
        """Test single prediction"""
        from stateset_agents.training.transformer_reward_model import (
            RewardTrainingConfig,
            TransformerRewardTrainer,
        )

        config = RewardTrainingConfig(
            base_model="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

        trainer = TransformerRewardTrainer(config=config)
        reward = trainer.predict("Test prompt", "Test response")

        assert isinstance(reward, float)
        assert not torch.isnan(torch.tensor(reward))

    def test_training_loop(self):
        """Test training loop runs without errors"""
        from stateset_agents.training.transformer_reward_model import (
            RewardExample,
            RewardTrainingConfig,
            TransformerRewardTrainer,
        )

        # Create small dataset
        train_examples = [
            RewardExample(prompt=f"Prompt {i}", response=f"Response {i}", reward=0.5)
            for i in range(10)
        ]

        val_examples = [
            RewardExample(
                prompt=f"Val prompt {i}", response=f"Val response {i}", reward=0.7
            )
            for i in range(5)
        ]

        config = RewardTrainingConfig(
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=4,
            num_epochs=2,
            device="cpu",
            patience=1,
        )

        trainer = TransformerRewardTrainer(config=config)

        results = trainer.train(
            train_examples=train_examples,
            val_examples=val_examples,
            freeze_encoders=True,
            verbose=False,
        )

        assert "final_train_loss" in results
        assert "final_val_loss" in results
        assert "epochs_trained" in results
        assert results["epochs_trained"] > 0

    def test_checkpoint_save_load(self, tmp_path):
        """Test saving and loading checkpoints"""
        from stateset_agents.training.transformer_reward_model import (
            RewardTrainingConfig,
            TransformerRewardTrainer,
        )

        config = RewardTrainingConfig(
            base_model="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

        trainer1 = TransformerRewardTrainer(config=config)

        # Make a prediction before saving
        reward1 = trainer1.predict("Test", "Response")

        # Save checkpoint
        checkpoint_path = tmp_path / "model.pt"
        trainer1.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load into new trainer
        trainer2 = TransformerRewardTrainer(config=config)
        trainer2.load_checkpoint(str(checkpoint_path))

        # Predictions should be identical
        reward2 = trainer2.predict("Test", "Response")
        assert abs(reward1 - reward2) < 1e-5


class TestLearnedRewardFunction:
    """Tests for LearnedRewardFunction"""

    @pytest.mark.asyncio
    async def test_compute_reward(self):
        """Test computing reward using learned model"""
        from stateset_agents.training.transformer_reward_model import (
            LearnedRewardFunction,
            RewardTrainingConfig,
            TransformerRewardTrainer,
        )

        config = RewardTrainingConfig(
            base_model="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

        trainer = TransformerRewardTrainer(config=config)
        reward_fn = LearnedRewardFunction(trainer=trainer, weight=1.0, normalize=True)

        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there!"),
        ]

        result = await reward_fn.compute_reward(turns)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0  # Should be normalized
        assert "learned_reward" in result.breakdown

    @pytest.mark.asyncio
    async def test_reward_caching(self):
        """Test reward prediction caching"""
        from stateset_agents.training.transformer_reward_model import (
            LearnedRewardFunction,
            RewardTrainingConfig,
            TransformerRewardTrainer,
        )

        config = RewardTrainingConfig(device="cpu")
        trainer = TransformerRewardTrainer(config=config)
        reward_fn = LearnedRewardFunction(
            trainer=trainer, cache_predictions=True, normalize=False
        )

        turns = [
            ConversationTurn(role="user", content="Test"),
            ConversationTurn(role="assistant", content="Response"),
        ]

        # First call
        result1 = await reward_fn.compute_reward(turns)

        # Second call (should use cache)
        result2 = await reward_fn.compute_reward(turns)

        # Should be identical (cached)
        assert result1.score == result2.score


class TestRewardCalibration:
    """Tests for reward calibration system"""

    def test_reward_statistics_update(self):
        """Test updating reward statistics"""
        from stateset_agents.training.reward_calibration import RewardStatistics

        stats = RewardStatistics()
        values = [0.1, 0.3, 0.5, 0.7, 0.9]

        stats.update(values)

        assert stats.mean == pytest.approx(0.5, abs=0.01)
        assert stats.min == 0.1
        assert stats.max == 0.9
        assert stats.count == 5
        assert 50 in stats.percentiles

    def test_reward_normalizer_zscore(self):
        """Test z-score normalization"""
        from stateset_agents.training.reward_calibration import RewardNormalizer

        normalizer = RewardNormalizer(method="z_score", buffer_size=100)

        # Add rewards
        for reward in [0.2, 0.4, 0.6, 0.8]:
            normalizer.add_reward(reward)

        normalizer._update_statistics()

        # Normalize a value
        normalized = normalizer.normalize(0.5)

        # Should be close to 0 (mean of the distribution)
        assert abs(normalized) < 1.0

    def test_reward_normalizer_minmax(self):
        """Test min-max normalization"""
        from stateset_agents.training.reward_calibration import RewardNormalizer

        normalizer = RewardNormalizer(method="min_max", buffer_size=100)

        # Add rewards
        for reward in [0.0, 0.25, 0.5, 0.75, 1.0]:
            normalizer.add_reward(reward)

        normalizer._update_statistics()

        # Normalize values
        assert normalizer.normalize(0.0) == pytest.approx(0.0, abs=0.01)
        assert normalizer.normalize(1.0) == pytest.approx(1.0, abs=0.01)
        assert normalizer.normalize(0.5) == pytest.approx(0.5, abs=0.01)

    def test_reward_normalizer_clipping(self):
        """Test reward clipping"""
        from stateset_agents.training.reward_calibration import RewardNormalizer

        normalizer = RewardNormalizer(
            method="z_score", clip_range=(-2.0, 2.0), buffer_size=100
        )

        for reward in [0.0, 0.5, 1.0]:
            normalizer.add_reward(reward)

        normalizer._update_statistics()

        # Extreme value should be clipped
        normalized = normalizer.normalize(100.0)
        assert -2.0 <= normalized <= 2.0

    @pytest.mark.asyncio
    async def test_calibrated_reward_function(self):
        """Test CalibratedRewardFunction"""
        from stateset_agents.core.reward import HelpfulnessReward
        from stateset_agents.training.reward_calibration import (
            CalibratedRewardFunction,
            RewardNormalizer,
        )

        base_reward = HelpfulnessReward(weight=1.0)
        normalizer = RewardNormalizer(method="min_max")

        calibrated_reward = CalibratedRewardFunction(
            base_reward_fn=base_reward, normalizer=normalizer, auto_calibrate=True
        )

        turns = [
            ConversationTurn(role="user", content="Can you help me?"),
            ConversationTurn(
                role="assistant",
                content="Of course! I'd be happy to help you with your question.",
            ),
        ]

        result = await calibrated_reward.compute_reward(turns)

        assert isinstance(result, RewardResult)
        assert "base_score" in result.breakdown
        assert "calibrated_score" in result.breakdown
        assert "normalization_method" in result.breakdown

    @pytest.mark.asyncio
    async def test_multi_reward_calibrator(self):
        """Test MultiRewardCalibrator"""
        from stateset_agents.core.reward import (
            HelpfulnessReward,
            SafetyReward,
        )
        from stateset_agents.training.reward_calibration import MultiRewardCalibrator

        rewards = [HelpfulnessReward(weight=1.0), SafetyReward(weight=1.0)]

        calibrator = MultiRewardCalibrator(rewards)

        # Create test episodes
        episodes = [
            [
                ConversationTurn(role="user", content="Hello"),
                ConversationTurn(role="assistant", content="Hi there!"),
            ],
            [
                ConversationTurn(role="user", content="Help me"),
                ConversationTurn(role="assistant", content="I'm here to assist!"),
            ],
        ]

        stats = await calibrator.calibrate(episodes)

        assert "HelpfulnessReward" in stats
        assert "SafetyReward" in stats
        assert stats["HelpfulnessReward"].count == 2
        assert stats["SafetyReward"].count == 2

    def test_adaptive_reward_scaler(self):
        """Test AdaptiveRewardScaler"""
        from stateset_agents.training.reward_calibration import AdaptiveRewardScaler

        scaler = AdaptiveRewardScaler(
            initial_scale=1.0, min_scale=0.1, max_scale=10.0
        )

        # Scale some rewards
        scaled1 = scaler.scale_reward(0.5)
        assert scaled1 == 0.5  # Initial scale is 1.0

        # Add more rewards
        for _ in range(100):
            scaler.scale_reward(0.2)  # Low rewards

        # Adapt scale (should increase to bring mean up)
        scaler.adapt_scale(target_mean=0.5)

        # Scale should have increased
        assert scaler.scale > 1.0

        # Get statistics
        stats = scaler.get_statistics()
        assert "scale" in stats
        assert "mean" in stats
        assert "std" in stats


@pytest.mark.integration
class TestRewardModelIntegration:
    """Integration tests for complete reward model pipeline"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, tmp_path):
        """Test complete workflow from training to deployment"""
        from stateset_agents.core.reward import HelpfulnessReward
        from stateset_agents.training.reward_calibration import CalibratedRewardFunction
        from stateset_agents.training.transformer_reward_model import (
            LearnedRewardFunction,
            RewardExample,
            RewardTrainingConfig,
            TransformerRewardTrainer,
        )

        # Create training data
        train_examples = [
            RewardExample(
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
                reward=0.5 + i * 0.1,
            )
            for i in range(20)
        ]

        val_examples = [
            RewardExample(
                prompt=f"Val prompt {i}",
                response=f"Val response {i}",
                reward=0.6 + i * 0.05,
            )
            for i in range(5)
        ]

        # Train model
        config = RewardTrainingConfig(
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=4,
            num_epochs=2,
            device="cpu",
        )

        trainer = TransformerRewardTrainer(config=config)
        results = trainer.train(
            train_examples, val_examples, freeze_encoders=True, verbose=False
        )

        assert results["epochs_trained"] > 0

        # Save checkpoint
        checkpoint_path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Create learned reward function
        learned_reward = LearnedRewardFunction(trainer=trainer, normalize=True)

        # Calibrate with heuristic reward
        heuristic_reward = HelpfulnessReward()
        calibrated_learned = CalibratedRewardFunction(
            base_reward_fn=learned_reward, auto_calibrate=True
        )

        # Test on example
        turns = [
            ConversationTurn(role="user", content="Can you help?"),
            ConversationTurn(role="assistant", content="Yes, I can help!"),
        ]

        result = await calibrated_learned.compute_reward(turns)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0
