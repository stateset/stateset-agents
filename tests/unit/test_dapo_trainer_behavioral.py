"""Behavioral tests: importance ratios must come from rollout-time log probs."""

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel

from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer


def tiny_model():
    torch.manual_seed(0)
    # Disable dropout: with it enabled, two forward passes on the *same*
    # unchanged weights produce different log probs (stochastic masks), which
    # would make even a correct "ratio ~= 1 before any update" assertion flaky.
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=200,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def make_batch():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 200, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.ones_like(input_ids, dtype=torch.float)
    response_mask[:, :4] = 0.0  # first 4 tokens are prompt
    return input_ids, attention_mask, response_mask


@pytest.fixture
def dapo_trainer_factory():
    def _make(model, num_gradient_updates=1, **config_overrides):
        config = DAPOConfig(
            model_name="gpt2",
            group_size=2,
            num_gradient_updates=num_gradient_updates,
            **config_overrides,
        )

        def reward_fn(prompt: str, response: str) -> float:
            return 0.0

        trainer = DAPOTrainer(
            config=config,
            model=model,
            tokenizer=None,
            reward_fn=reward_fn,
        )
        return trainer

    return _make


@pytest.mark.asyncio
async def test_ratio_diverges_from_one_after_update(dapo_trainer_factory):
    """After one optimizer step, ratios vs stored old log probs must != 1."""
    trainer = dapo_trainer_factory(tiny_model())
    ids, am, rm = make_batch()
    with torch.no_grad():
        old, _ = trainer.compute_token_log_probs(ids, am, rm)
    # take a real optimizer step on arbitrary loss
    cur, _ = trainer.compute_token_log_probs(ids, am, rm)
    loss = cur.sum()
    loss.backward()
    trainer.optimizer.step()
    cur2, _ = trainer.compute_token_log_probs(ids, am, rm)
    ratio = torch.exp(cur2 - old)
    assert not torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


@pytest.mark.asyncio
async def test_num_gradient_updates_respected(dapo_trainer_factory, monkeypatch):
    """train_step must run num_gradient_updates inner updates, not min(mu, 1)."""
    trainer = dapo_trainer_factory(tiny_model(), num_gradient_updates=3)
    steps = []
    orig = trainer.optimizer.step
    monkeypatch.setattr(
        trainer.optimizer, "step", lambda *a, **k: (steps.append(1), orig())[1]
    )

    ids, am, rm = make_batch()
    sample = {
        "responses": [
            {
                "input_ids": ids[i],
                "attention_mask": am[i],
                "response_mask": rm[i],
                "sequence_length": int(am[i].sum()),
            }
            for i in range(2)
        ],
        "advantages": torch.tensor([0.5, -0.5]),
        "rewards": [1.0, 0.0],
        "accuracy": 0.5,
    }

    async def fake_collect(prompts, n):
        return [sample], 0.0

    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", fake_collect)
    await trainer.train_step(["q"])
    assert len(steps) == 3


@pytest.mark.asyncio
async def test_old_log_probs_captured_at_collection(dapo_trainer_factory, monkeypatch):
    """train_step must use sample['old_token_log_probs'] when present, and the
    second inner update must see a non-unit ratio (clipping can fire)."""
    trainer = dapo_trainer_factory(tiny_model(), num_gradient_updates=2)
    ids, am, rm = make_batch()
    with torch.no_grad():
        old, _ = trainer.compute_token_log_probs(ids, am, rm)
    sample = {
        "responses": [
            {
                "input_ids": ids[i],
                "attention_mask": am[i],
                "response_mask": rm[i],
                "sequence_length": int(am[i].sum()),
            }
            for i in range(2)
        ],
        "advantages": torch.tensor([1.0, -1.0]),
        "rewards": [1.0, 0.0],
        "accuracy": 0.5,
        "old_token_log_probs": old,
    }
    seen_ratios = []
    orig_ratio = trainer.compute_importance_ratio

    def spy(cur, old_):
        r = orig_ratio(cur, old_)
        seen_ratios.append(r.detach().clone())
        return r

    monkeypatch.setattr(trainer, "compute_importance_ratio", spy)

    async def fake_collect(prompts, n):
        return [sample], 0.0

    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", fake_collect)
    await trainer.train_step(["q"])
    assert len(seen_ratios) == 2
    # first inner update: on-policy, ratio ~ 1
    assert torch.allclose(seen_ratios[0], torch.ones_like(seen_ratios[0]), atol=1e-4)
    # second inner update: policy moved, ratio must not be identically 1
    assert not torch.allclose(
        seen_ratios[1], torch.ones_like(seen_ratios[1]), atol=1e-5
    )


def test_group_of_one_advantage_is_finite(dapo_trainer_factory):
    """A group of size 1 has zero variance: advantages must be 0, never NaN."""
    trainer = dapo_trainer_factory(tiny_model())
    adv = trainer.compute_group_advantages(torch.tensor([0.3]))
    assert torch.isfinite(adv).all()
    assert adv.item() == 0.0


def test_importance_ratio_finite_for_extreme_log_ratio(dapo_trainer_factory):
    """A token 50 nats off-policy must give a finite ratio (raw exp overflows
    fp32 past ~88 and would poison the batch loss)."""
    trainer = dapo_trainer_factory(tiny_model())
    current = torch.tensor([[50.0, 200.0, 0.0]])
    old = torch.tensor([[0.0, 0.0, 0.0]])
    ratio = trainer.compute_importance_ratio(current, old)
    assert torch.isfinite(ratio).all()
    # Raw exp(200) is +inf in fp32; the clamp caps the exponent at 20.
    assert not torch.isfinite(torch.exp(current - old)).all()
    assert ratio[0, 1].item() == ratio[0, 0].item()

    loss = trainer.compute_dapo_loss(
        ratio, torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([[1.0, 1.0, 1.0]])
    )
    assert torch.isfinite(loss).all()
