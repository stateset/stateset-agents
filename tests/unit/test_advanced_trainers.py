"""
Tests for advanced RL trainers: GEPO, DAPO, and VAPO

These tests verify the implementations of state-of-the-art RL algorithms
for training large language models.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Skip tests if training modules not available
try:
    from training.gepo_trainer import GEPOConfig, GEPOTrainer, train_with_gepo
    from training.dapo_trainer import (
        DAPOConfig,
        DAPOTrainer,
        DAPORewardShaper,
        DynamicSamplingBuffer,
        train_with_dapo,
    )
    from training.vapo_trainer import (
        VAPOConfig,
        VAPOTrainer,
        ValueHead,
        LengthAdaptiveGAE,
        train_with_vapo,
    )
    TRAINERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TRAINERS_AVAILABLE = False
    # Create dummy classes for type hints
    GEPOConfig = None
    GEPOTrainer = None
    train_with_gepo = None
    DAPOConfig = None
    DAPOTrainer = None
    DAPORewardShaper = None
    DynamicSamplingBuffer = None
    train_with_dapo = None
    VAPOConfig = None
    VAPOTrainer = None
    ValueHead = None
    LengthAdaptiveGAE = None
    train_with_vapo = None

pytestmark = pytest.mark.skipif(
    not TRAINERS_AVAILABLE,
    reason="Advanced trainers not available (check transformers/torchvision compatibility)"
)


# ===========================
# Test Fixtures
# ===========================


@pytest.fixture
def device():
    """Get test device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_model(device):
    """Create a mock language model for testing"""
    model = MagicMock()
    # Create an actual parameter that can be iterated multiple times
    param = torch.nn.Parameter(torch.randn(10, 10, device=device))
    model.parameters = MagicMock(return_value=iter([param]))

    # Make parameters() return a fresh iterator each time
    def fresh_params():
        return iter([param])
    model.parameters = fresh_params
    model.device = device

    # Mock config
    model.config = MagicMock()
    model.config.hidden_size = 768

    # Mock forward pass
    def mock_forward(input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        vocab_size = 1000

        outputs = MagicMock()
        outputs.logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        if output_hidden_states:
            outputs.hidden_states = [
                torch.randn(batch_size, seq_len, 768, device=device)
                for _ in range(13)  # 12 layers + embedding
            ]

        return outputs

    model.__call__ = mock_forward
    model.forward = mock_forward

    # Mock generate
    def mock_generate(input_ids, attention_mask=None, max_new_tokens=50, **kwargs):
        batch_size, prompt_len = input_ids.shape
        response_len = min(max_new_tokens, 20)  # Shorter for tests
        output_ids = torch.cat([
            input_ids,
            torch.randint(0, 1000, (batch_size, response_len), device=device)
        ], dim=1)
        return output_ids

    model.generate = mock_generate

    # Mock save
    model.save_pretrained = MagicMock()
    model.train = MagicMock(return_value=None)
    model.eval = MagicMock(return_value=None)

    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer"""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 1

    def mock_encode(text, return_tensors=None, truncation=True, max_length=512, **kwargs):
        # Return mock tokenization
        tokens = MagicMock()
        seq_len = min(len(text.split()) * 2, max_length)
        tokens["input_ids"] = torch.randint(0, 1000, (1, seq_len))
        tokens["attention_mask"] = torch.ones(1, seq_len)
        return tokens

    tokenizer.__call__ = mock_encode
    tokenizer.encode = mock_encode

    def mock_decode(token_ids, skip_special_tokens=True, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "Generated response text " * 5

    tokenizer.decode = mock_decode
    tokenizer.save_pretrained = MagicMock()

    return tokenizer


@pytest.fixture
def simple_reward_fn():
    """Simple reward function for testing"""
    def reward_fn(prompt: str, response: str) -> float:
        # Reward based on response length and content
        if len(response) > 50:
            return 0.8
        elif len(response) > 20:
            return 0.5
        else:
            return 0.2
    return reward_fn


@pytest.fixture
def simple_verifier_fn():
    """Simple verifier function for testing"""
    def verifier_fn(prompt: str, response: str) -> bool:
        # Simple check - response contains certain keywords
        return "answer" in response.lower() or len(response) > 30
    return verifier_fn


# ===========================
# GEPO Tests
# ===========================


class TestGEPOConfig:
    """Test GEPO configuration"""

    def test_default_config(self):
        """Test default GEPO config values"""
        config = GEPOConfig()
        assert config.group_size == 8
        assert config.clip_eps == 0.2
        assert config.learning_rate == 1e-6
        assert config.warmup_ratio == 0.03

    def test_custom_config(self):
        """Test custom GEPO config"""
        config = GEPOConfig(
            group_size=16,
            clip_eps=0.15,
            learning_rate=5e-7,
            model_name="test-model",
        )
        assert config.group_size == 16
        assert config.clip_eps == 0.15
        assert config.learning_rate == 5e-7
        assert config.model_name == "test-model"

    def test_config_serialization(self):
        """Test config to/from dict"""
        config = GEPOConfig(group_size=12, clip_eps=0.25)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["group_size"] == 12
        assert config_dict["clip_eps"] == 0.25


class TestGEPOTrainer:
    """Test GEPO trainer functionality"""

    @pytest.fixture
    def gepo_config(self):
        return GEPOConfig(
            model_name="test-model",
            group_size=4,
            num_episodes=10,
            logging_steps=5,
            save_steps=10,
            max_prompt_length=64,
            max_completion_length=32,
        )

    def test_trainer_initialization(self, gepo_config, mock_model, mock_tokenizer, simple_reward_fn, device):
        """Test GEPO trainer initialization"""
        trainer = GEPOTrainer(
            config=gepo_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            reward_fn=simple_reward_fn,
        )

        assert trainer.config == gepo_config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.global_step == 0

    def test_compute_gepo_coefficient(self, gepo_config, mock_model, mock_tokenizer, simple_reward_fn, device):
        """Test GEPO coefficient computation"""
        trainer = GEPOTrainer(
            config=gepo_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            reward_fn=simple_reward_fn,
        )

        # Create test probabilities
        learner_probs = torch.tensor([0.3, 0.25, 0.2, 0.25], device=device)
        sampler_probs = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)

        coefs = trainer.compute_gepo_coefficient(learner_probs, sampler_probs)

        assert coefs.shape == learner_probs.shape
        # Coefficients should be close to 1 when distributions are similar
        assert torch.all(coefs > 0)

    def test_compute_group_advantages(self, gepo_config, mock_model, mock_tokenizer, simple_reward_fn, device):
        """Test group advantage computation"""
        trainer = GEPOTrainer(
            config=gepo_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            reward_fn=simple_reward_fn,
        )

        rewards = torch.tensor([1.0, 0.5, 0.8, 0.3], device=device)
        advantages, stats = trainer.compute_group_advantages(rewards)

        # Advantages should be normalized
        assert advantages.shape == rewards.shape
        assert abs(advantages.mean().item()) < 0.1  # Should be close to 0
        assert abs(advantages.std().item() - 1.0) < 0.1  # Should be close to 1

        # Stats should be populated
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "max_reward" in stats
        assert "min_reward" in stats


# ===========================
# DAPO Tests
# ===========================


class TestDAPOConfig:
    """Test DAPO configuration"""

    def test_default_config(self):
        """Test default DAPO config values"""
        config = DAPOConfig()
        assert config.clip_eps_low == 0.2
        assert config.clip_eps_high == 0.28
        assert config.use_dynamic_sampling is True
        assert config.use_overlong_shaping is True
        assert config.use_token_level_loss is True

    def test_asymmetric_clipping(self):
        """Test that clipping is asymmetric (Clip-Higher)"""
        config = DAPOConfig()
        assert config.clip_eps_high > config.clip_eps_low


class TestDAPORewardShaper:
    """Test DAPO reward shaping"""

    def test_no_penalty_short_sequence(self):
        """Test no penalty for sequences within normal length"""
        shaper = DAPORewardShaper(max_length=1000, cache_length=200)

        # Short sequence should have no penalty
        reward = shaper.compute_length_reward(500)
        assert reward == 0.0

    def test_soft_penalty_approaching_max(self):
        """Test soft penalty for sequences approaching max length"""
        shaper = DAPORewardShaper(max_length=1000, cache_length=200, penalty=-1.0)

        # At soft start (800), penalty should be 0
        reward_at_start = shaper.compute_length_reward(800)
        assert reward_at_start == 0.0

        # At 900 (halfway through cache), penalty should be -0.5
        reward_at_mid = shaper.compute_length_reward(900)
        assert abs(reward_at_mid - (-0.5)) < 0.01

        # At max (1000), penalty should be -1.0
        reward_at_max = shaper.compute_length_reward(1000)
        assert abs(reward_at_max - (-1.0)) < 0.01

    def test_full_penalty_truncated(self):
        """Test full penalty for truncated sequences"""
        shaper = DAPORewardShaper(max_length=1000, cache_length=200, penalty=-1.0)

        # Beyond max should have full penalty
        reward = shaper.compute_length_reward(1100)
        assert reward == -1.0


class TestDynamicSamplingBuffer:
    """Test dynamic sampling buffer"""

    def test_filter_trivial_accuracy(self):
        """Test that trivial accuracy samples are filtered"""
        buffer = DynamicSamplingBuffer(min_accuracy=0.0, max_accuracy=1.0)

        # 0% accuracy should be excluded
        assert not buffer.should_include(0.0)

        # 100% accuracy should be excluded
        assert not buffer.should_include(1.0)

        # Intermediate accuracy should be included
        assert buffer.should_include(0.5)
        assert buffer.should_include(0.25)
        assert buffer.should_include(0.75)

    def test_buffer_operations(self):
        """Test buffer add and get operations"""
        buffer = DynamicSamplingBuffer(buffer_size=10)

        # Add valid samples
        for i in range(5):
            sample = {"id": i, "data": f"sample_{i}"}
            added = buffer.add_sample(sample, accuracy=0.5)
            assert added

        assert buffer.size == 5

        # Get batch
        batch = buffer.get_batch(3)
        assert len(batch) == 3
        assert buffer.size == 2


class TestDAPOTrainer:
    """Test DAPO trainer functionality"""

    @pytest.fixture
    def dapo_config(self):
        return DAPOConfig(
            model_name="test-model",
            group_size=4,
            num_episodes=10,
            max_prompt_length=64,
            max_completion_length=32,
            use_dynamic_sampling=True,
            use_overlong_shaping=True,
        )

    def test_trainer_initialization(self, dapo_config, mock_model, mock_tokenizer, simple_reward_fn):
        """Test DAPO trainer initialization"""
        trainer = DAPOTrainer(
            config=dapo_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            reward_fn=simple_reward_fn,
        )

        assert trainer.config == dapo_config
        assert trainer.reward_shaper is not None
        assert trainer.sampling_buffer is not None


# ===========================
# VAPO Tests
# ===========================


class TestVAPOConfig:
    """Test VAPO configuration"""

    def test_default_config(self):
        """Test default VAPO config values"""
        config = VAPOConfig()
        assert config.value_warmup_steps == 50
        assert config.lambda_critic == 1.0
        assert config.lambda_policy_alpha == 0.05
        assert config.use_positive_lm_loss is True
        assert config.positive_lm_weight == 0.1

    def test_separate_learning_rates(self):
        """Test that actor and critic have separate learning rates"""
        config = VAPOConfig()
        # Critic LR should be higher than actor LR
        assert config.critic_learning_rate >= config.actor_learning_rate


class TestValueHead:
    """Test value head network"""

    def test_value_head_shape(self, device):
        """Test value head output shape"""
        hidden_size = 768
        value_head = ValueHead(hidden_size=hidden_size).to(device)

        batch_size = 4
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

        values = value_head(hidden_states)

        assert values.shape == (batch_size, seq_len, 1)

    def test_value_head_trainable(self, device):
        """Test that value head parameters are trainable"""
        value_head = ValueHead(hidden_size=768).to(device)

        # Check parameters require grad
        for param in value_head.parameters():
            assert param.requires_grad


class TestLengthAdaptiveGAE:
    """Test length-adaptive GAE computation"""

    def test_lambda_policy_computation(self):
        """Test length-adaptive lambda computation"""
        gae = LengthAdaptiveGAE(lambda_policy_alpha=0.05)

        # Short sequences should have lower lambda (higher bias)
        lambda_short = gae.compute_lambda_policy(10)

        # Long sequences should have higher lambda (lower bias)
        lambda_long = gae.compute_lambda_policy(100)

        assert lambda_long > lambda_short
        assert 0 < lambda_short < 1
        assert 0 < lambda_long < 1

    def test_gae_computation(self, device):
        """Test GAE computation"""
        gae = LengthAdaptiveGAE()

        batch_size = 2
        seq_len = 10

        rewards = torch.randn(batch_size, seq_len, device=device)
        values = torch.randn(batch_size, seq_len, device=device)
        dones = torch.zeros(batch_size, seq_len, device=device)
        dones[:, -1] = 1.0  # Last step is terminal

        advantages, returns = gae.compute_gae(rewards, values, dones, lambda_value=0.95)

        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)


class TestVAPOTrainer:
    """Test VAPO trainer functionality"""

    @pytest.fixture
    def vapo_config(self):
        return VAPOConfig(
            model_name="test-model",
            group_size=4,
            num_episodes=10,
            value_warmup_steps=2,  # Short for testing
            max_prompt_length=64,
            max_completion_length=32,
        )

    def test_trainer_initialization(self, vapo_config, mock_model, mock_tokenizer, simple_reward_fn, device):
        """Test VAPO trainer initialization"""
        trainer = VAPOTrainer(
            config=vapo_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            reward_fn=simple_reward_fn,
        )

        assert trainer.config == vapo_config
        assert trainer.value_head is not None
        assert trainer.gae_computer is not None
        assert trainer.actor_optimizer is not None
        assert trainer.critic_optimizer is not None
        assert not trainer.value_warmup_complete


# ===========================
# Integration Tests
# ===========================


class TestTrainerSaveLoad:
    """Test checkpoint save/load for all trainers"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_gepo_save_checkpoint(self, temp_dir, mock_model, mock_tokenizer, simple_reward_fn):
        """Test GEPO checkpoint saving"""
        config = GEPOConfig(model_name="test", group_size=4)
        trainer = GEPOTrainer(
            config=config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            reward_fn=simple_reward_fn,
        )

        checkpoint_dir = os.path.join(temp_dir, "gepo-checkpoint")
        trainer.save_checkpoint(checkpoint_dir)

        # Verify files created
        assert os.path.exists(os.path.join(checkpoint_dir, "training_state.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "gepo_config.json"))

    def test_dapo_save_checkpoint(self, temp_dir, mock_model, mock_tokenizer, simple_reward_fn):
        """Test DAPO checkpoint saving"""
        config = DAPOConfig(model_name="test", group_size=4)
        trainer = DAPOTrainer(
            config=config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            reward_fn=simple_reward_fn,
        )

        checkpoint_dir = os.path.join(temp_dir, "dapo-checkpoint")
        trainer.save_checkpoint(checkpoint_dir)

        # Verify files created
        assert os.path.exists(os.path.join(checkpoint_dir, "training_state.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "dapo_config.json"))

    def test_vapo_save_checkpoint(self, temp_dir, mock_model, mock_tokenizer, simple_reward_fn, device):
        """Test VAPO checkpoint saving"""
        config = VAPOConfig(model_name="test", group_size=4)
        trainer = VAPOTrainer(
            config=config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            reward_fn=simple_reward_fn,
        )

        checkpoint_dir = os.path.join(temp_dir, "vapo-checkpoint")
        trainer.save_checkpoint(checkpoint_dir)

        # Verify files created
        assert os.path.exists(os.path.join(checkpoint_dir, "training_state.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "vapo_config.json"))
        assert os.path.exists(os.path.join(checkpoint_dir, "value_head.pt"))


# ===========================
# Algorithm Comparison Tests
# ===========================


class TestAlgorithmCharacteristics:
    """Test that algorithms have their expected characteristics"""

    def test_gepo_group_level_weights(self):
        """Verify GEPO uses group-level importance weights"""
        config = GEPOConfig()
        # GEPO should use group_size for computing group expectation
        assert config.group_size > 1
        assert config.use_group_baseline is True

    def test_dapo_asymmetric_clipping(self):
        """Verify DAPO uses asymmetric (Clip-Higher) clipping"""
        config = DAPOConfig()
        # eps_high should be larger than eps_low
        assert config.clip_eps_high > config.clip_eps_low
        # Standard values from paper
        assert abs(config.clip_eps_low - 0.2) < 0.01
        assert abs(config.clip_eps_high - 0.28) < 0.01

    def test_dapo_no_kl_penalty(self):
        """Verify DAPO removes KL penalty"""
        config = DAPOConfig()
        assert config.beta == 0.0
        assert config.use_reference_model is False

    def test_vapo_separate_optimizers(self):
        """Verify VAPO has separate actor/critic optimizers"""
        config = VAPOConfig()
        assert config.actor_learning_rate != config.critic_learning_rate
        # Critic typically has higher LR
        assert config.critic_learning_rate >= config.actor_learning_rate

    def test_vapo_value_warmup(self):
        """Verify VAPO has value network warmup"""
        config = VAPOConfig()
        assert config.value_warmup_steps > 0
        assert config.value_warmup_steps == 50  # Default from paper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
