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


def _set_old_log_probs(trainer, monkeypatch, log_ratios):
    """Point the fake generator at old log probs producing the given
    length-normalised log importance ratios for ("ok", "nope")."""
    responses = ["ok", "nope"]
    with torch.no_grad():
        cur, lengths = trainer._compute_group_sequence_log_probs("hello", responses)
    olds = [
        float(cur[i]) - float(lengths[i]) * log_ratios[i] for i in range(len(responses))
    ]

    async def fake_generate_group_responses(prompt, num_responses):
        return list(zip(responses, olds, strict=True))

    monkeypatch.setattr(
        trainer.generator, "generate_group_responses", fake_generate_group_responses
    )


@pytest.mark.asyncio
async def test_gspo_token_out_of_region_sequence_gets_no_gradient(
    gspo_token_trainer_tiny, monkeypatch
):
    """A response whose sequence ratio is pushed past the clip boundary in the
    direction its advantage points must contribute no gradient at all.

    The clipped branch of ``min(r*A, clip(r)*A)`` has zero gradient there, so
    weighting the token log probs by the (clipped) sequence ratio — as the
    pre-fix code did — leaks gradient from a sample the trust region excludes.
    """
    trainer = gspo_token_trainer_tiny
    import copy

    baseline_state = copy.deepcopy(trainer.model.state_dict())

    # Response 0 ("ok") has the positive advantage; put its ratio far above
    # 1 + clip_range_right. Response 1 stays exactly in region (ratio == 1).
    _set_old_log_probs(trainer, monkeypatch, [3.0, 0.0])
    await trainer.train_step_token_level(["hello"], num_groups=1)
    grads_gated = {
        n: p.grad.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.grad is not None
    }
    assert grads_gated

    # Reference run: identical inputs, but response 0's advantage forced to 0
    # so it provably contributes nothing.
    trainer.model.load_state_dict(baseline_state)
    real_advantages = trainer.compute_group_advantages

    def zero_first_advantage(rewards):
        advantages, stats = real_advantages(rewards)
        advantages = advantages.clone()
        advantages[0] = 0.0
        return advantages, stats

    monkeypatch.setattr(trainer, "compute_group_advantages", zero_first_advantage)
    _set_old_log_probs(trainer, monkeypatch, [3.0, 0.0])
    await trainer.train_step_token_level(["hello"], num_groups=1)
    grads_removed = {
        n: p.grad.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.grad is not None
    }

    assert set(grads_gated) == set(grads_removed)
    for name, g in grads_gated.items():
        assert torch.allclose(g, grads_removed[name], atol=1e-6), name


@pytest.mark.asyncio
async def test_gated_infinite_sequence_ratio_does_not_poison_loss(
    gspo_token_trainer_tiny, monkeypatch
):
    """A gated-out response whose sequence ratio overflowed to +inf must not
    turn the loss (and every gradient) into NaN.

    ``gate * seq_ratio`` is ``0 * inf == nan``; the gate has to *select*, not
    multiply. The gated run must match a run where that response's advantage
    is zero, i.e. it contributes nothing.
    """
    import copy
    import math

    trainer = gspo_token_trainer_tiny
    baseline_state = copy.deepcopy(trainer.model.state_dict())

    real_ratio = trainer.compute_sequence_importance_ratio

    def inf_first_ratio(current, old, lengths):
        ratio = real_ratio(current, old, lengths).clone()
        ratio[0] = float("inf")  # gated out: adv > 0 and ratio > hi
        return ratio

    monkeypatch.setattr(trainer, "compute_sequence_importance_ratio", inf_first_ratio)
    metrics = await trainer.train_step_token_level(["hello"], num_groups=1)
    assert math.isfinite(metrics["policy_loss"])
    grads_gated = {
        n: p.grad.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.grad is not None
    }
    assert grads_gated
    assert all(torch.isfinite(g).all() for g in grads_gated.values())

    # Reference: identical run with a large but *finite* out-of-region ratio
    # for response 0. It is gated out there too, so the gradients must match
    # exactly -- the +inf must be neutralised, not propagated.
    trainer.model.load_state_dict(baseline_state)

    def big_first_ratio(current, old, lengths):
        ratio = real_ratio(current, old, lengths).clone()
        ratio[0] = 1e6
        return ratio

    monkeypatch.setattr(trainer, "compute_sequence_importance_ratio", big_first_ratio)
    await trainer.train_step_token_level(["hello"], num_groups=1)
    grads_finite = {
        n: p.grad.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.grad is not None
    }
    for name, g in grads_gated.items():
        assert torch.allclose(g, grads_finite[name], atol=1e-6), name
