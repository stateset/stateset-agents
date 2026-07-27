"""Behavioral tests for GSPOTokenTrainer.train_step_token_level.

Covers the bug fixed in Task 4: the token-level scoring forward pass ran
under torch.no_grad() (backward() was a no-op/raise), summed log probs over
the whole sequence including the prompt, called the reward model with a
nonstandard signature, and used `self.model.device` instead of the shared
device-lookup helper.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from stateset_agents.core.reward_base import RewardResult
from stateset_agents.training.gspo_config import GSPOConfig
from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer


def _tiny_model():
    torch.manual_seed(0)
    # Disable dropout so repeated forward passes on unchanged weights are
    # deterministic (matches the convention in test_dapo_trainer_behavioral.py).
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=50257,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


class _StubRewardModel:
    """Reward function whose compute_turn_reward varies by response content so
    group advantages are nonzero (a single-response group always normalizes
    to advantage 0, which would trivially zero the loss regardless of the
    gradient-path bug)."""

    async def compute_turn_reward(self, turn, context=None, conversation_history=None):
        score = 1.0 if turn.content == "ok" else 0.0
        return RewardResult(score=score, breakdown={}, components={})


@pytest.fixture
def gspo_token_trainer_tiny(monkeypatch):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = _tiny_model()

    config = GSPOConfig(
        model_name="gpt2",
        num_generations=2,
        num_outer_iterations=1,
        num_iterations=1,
        max_prompt_length=32,
        max_completion_length=32,
    )

    trainer = GSPOTokenTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        agent=None,
        environment=None,
        reward_model=_StubRewardModel(),
        ref_model=None,
    )

    async def fake_generate_group_responses(prompt, num_responses):
        return [("ok", -5.0), ("nope", -5.0)]

    monkeypatch.setattr(
        trainer.generator, "generate_group_responses", fake_generate_group_responses
    )

    return trainer


@pytest.mark.asyncio
async def test_train_step_produces_gradients(gspo_token_trainer_tiny):
    """After train_step_token_level, at least one model parameter must have a
    non-None, nonzero grad — the no_grad bug made backward() a no-op/raise."""
    trainer = gspo_token_trainer_tiny
    metrics = await trainer.train_step_token_level(["hello"], num_groups=1)

    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert any(g.abs().sum() > 0 for g in grads)
    assert "policy_loss" in metrics


@pytest.mark.asyncio
async def test_reward_computed_via_compute_turn_reward(gspo_token_trainer_tiny):
    """The reward call must use the documented compute_turn_reward signature."""
    trainer = gspo_token_trainer_tiny

    calls = []

    class _SpyRewardModel(_StubRewardModel):
        async def compute_turn_reward(
            self, turn, context=None, conversation_history=None
        ):
            calls.append((turn, context, conversation_history))
            return await super().compute_turn_reward(
                turn, context, conversation_history
            )

    trainer.reward_model = _SpyRewardModel()
    await trainer.train_step_token_level(["hello"], num_groups=1)

    assert len(calls) == 2
    contents = {turn.content for turn, _context, _history in calls}
    assert contents == {"ok", "nope"}
    for _turn, context, _history in calls:
        assert context == {"user_query": "hello"}


def test_loss_excludes_prompt_tokens(gspo_token_trainer_tiny):
    """Token loss must be masked to response positions."""
    trainer = gspo_token_trainer_tiny
    lp = torch.arange(9, dtype=torch.float).unsqueeze(0)
    masked = trainer.mask_prompt_tokens(lp, prompt_length=4)
    assert masked[0, :3].sum() == 0  # shifted convention: P-1 leading zeros
    assert masked[0, 3:].sum() != 0


def test_mask_prompt_tokens_handles_zero_length_prompt(gspo_token_trainer_tiny):
    trainer = gspo_token_trainer_tiny
    lp = torch.arange(5, dtype=torch.float).unsqueeze(0)
    masked = trainer.mask_prompt_tokens(lp, prompt_length=0)
    assert torch.equal(masked, lp)
