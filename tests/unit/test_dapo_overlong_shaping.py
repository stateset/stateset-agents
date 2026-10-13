"""Overlong reward shaping must act on the length the trainer can generate.

Responses are capped at ``max_completion_length`` tokens; a soft-penalty
interval anchored at ``max_generation_length`` (20480 by default) can never be
reached, so shaping would be a silent no-op.
"""

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer  # noqa: E402


def _trainer(**overrides):
    model = GPT2LMHeadModel(
        GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=50, n_positions=32)
    )
    config = DAPOConfig(model_name="gpt2", group_size=2, **overrides)
    return DAPOTrainer(
        config=config, model=model, tokenizer=None, reward_fn=lambda p, r: 0.0
    )


def test_default_config_penalizes_responses_at_the_generation_cap():
    trainer = _trainer()
    cap = trainer.config.max_completion_length
    assert trainer.reward_shaper.compute_length_reward(cap) == pytest.approx(-1.0)
    assert trainer.reward_shaper.compute_length_reward(1) == 0.0


def test_soft_interval_lies_inside_the_generation_cap():
    trainer = _trainer(max_completion_length=1000, overlong_cache_length=200)
    shaper = trainer.reward_shaper
    assert shaper.max_length == 1000
    assert shaper.compute_length_reward(800) == 0.0
    assert shaper.compute_length_reward(900) == pytest.approx(-0.5)
    assert shaper.compute_length_reward(1000) == pytest.approx(-1.0)


def test_smaller_max_generation_length_still_wins():
    trainer = _trainer(
        max_completion_length=1000, max_generation_length=600, overlong_cache_length=100
    )
    assert trainer.reward_shaper.max_length == 600
    assert trainer.reward_shaper.compute_length_reward(550) == pytest.approx(-0.5)


def test_cache_longer_than_cap_is_clamped_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        trainer = _trainer(max_completion_length=500, overlong_cache_length=600)
    shaper = trainer.reward_shaper
    assert 0 < shaper.cache_length < 500
    # Short responses must not be penalized just because the cache is too long.
    assert shaper.compute_length_reward(10) == 0.0
    assert "overlong_cache_length" in caplog.text


def test_default_cache_is_clamped_quietly(caplog):
    with caplog.at_level(logging.WARNING):
        trainer = _trainer(max_completion_length=500)
    assert 0 < trainer.reward_shaper.cache_length < 500
    assert "overlong_cache_length" not in caplog.text
