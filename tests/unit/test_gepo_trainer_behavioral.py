import math

import pytest
import torch

from stateset_agents.training.gepo_trainer import GEPOTrainer


def test_gepo_coefficient_no_underflow_long_sequences():
    """Sums of token log probs for realistic sequences (~ -600 nats) must not
    produce 0/NaN coefficients."""
    learner = torch.tensor([-600.0, -610.0, -605.0, -595.0])
    sampler = torch.tensor([-601.0, -609.0, -606.0, -594.0])
    coef = GEPOTrainer.compute_gepo_coefficient_static(learner, sampler)
    assert torch.isfinite(coef).all()
    assert (coef > 0).all()


def test_gepo_coefficient_matches_linear_space_on_small_values():
    """On numerically safe values, log-space result equals the linear formula
    coef_i = p_i / E_qhat[q], E_qhat[q] = sum(q^2)/sum(q)."""
    learner_lp = torch.log(torch.tensor([0.30, 0.20, 0.10]))
    sampler_lp = torch.log(torch.tensor([0.25, 0.25, 0.10]))
    q = sampler_lp.exp()
    expected = learner_lp.exp() / ((q * q).sum() / q.sum())
    got = GEPOTrainer.compute_gepo_coefficient_static(learner_lp, sampler_lp)
    assert torch.allclose(got, expected, rtol=1e-5)


def test_response_mask_offset_matches_gspo_convention():
    """With prompt length P, the shifted-label mask must start at P-1."""
    mask = GEPOTrainer.build_response_mask(
        attention_mask=torch.ones(1, 10, dtype=torch.long), response_start_idx=4
    )
    # shifted axis has length 9; positions 0..2 are prompt-only, 3.. are response
    assert mask.shape == (1, 9)
    assert mask[0, :3].sum() == 0
    assert mask[0, 3:].sum() == 6


def test_response_mask_offset_clamped_at_zero():
    """response_start_idx of 0 should not underflow to a negative index."""
    mask = GEPOTrainer.build_response_mask(
        attention_mask=torch.ones(1, 5, dtype=torch.long), response_start_idx=0
    )
    assert mask.shape == (1, 4)
    assert mask.sum() == 4


def test_group_of_one_advantage_is_finite():
    """A group of size 1 has zero variance: advantages must be 0, never NaN."""
    trainer = object.__new__(GEPOTrainer)
    adv, stats = trainer.compute_group_advantages(torch.tensor([0.3]))
    assert torch.isfinite(adv).all()
    assert adv.item() == 0.0
    assert all(v == v for v in stats.values())  # no NaN in stats


def _tiny_gepo_trainer(num_gradient_updates: int = 1):
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    from stateset_agents.training.gepo_trainer import GEPOConfig

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
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
    config = GEPOConfig(
        model_name="gpt2",
        group_size=2,
        learning_rate=1e-2,  # large enough that one step visibly moves the policy
        num_gradient_updates=num_gradient_updates,
    )
    return GEPOTrainer(
        config=config,
        model=model,
        tokenizer=None,
        reward_fn=lambda prompt, response: 1.0 if response == "ok" else 0.0,
    )


def _fixed_group_responses():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 200, (2, 10))
    return [
        {
            "response": "ok" if i == 0 else "nope",
            "input_ids": input_ids[i],
            "attention_mask": torch.ones(10, dtype=torch.long),
            "response_start_idx": 4,
        }
        for i in range(2)
    ]


@pytest.mark.asyncio
async def test_gepo_old_logprobs_frozen_across_gradient_updates(monkeypatch):
    """With num_gradient_updates > 1 the sampler (old) log probs must be the
    rollout-time snapshot, so later updates' coefficients are off-policy.

    Three updates rather than two: the scheduler's warmup makes the very
    first optimizer step a zero-learning-rate no-op, so the policy has only
    demonstrably moved by the third.
    """
    trainer = _tiny_gepo_trainer(num_gradient_updates=3)
    responses = _fixed_group_responses()

    async def fake_generate(prompt, group_size):
        return responses

    monkeypatch.setattr(trainer, "generate_group_responses", fake_generate)

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig = trainer.compute_gepo_coefficient

    def spy(learner, sampler):
        captured.append((learner.detach().clone(), sampler.detach().clone()))
        return orig(learner, sampler)

    monkeypatch.setattr(trainer, "compute_gepo_coefficient", spy)

    metrics = await trainer.train_step(["hello"])

    assert math.isfinite(metrics["policy_loss"])
    assert len(captured) == 3, "num_gradient_updates=3 must run three inner updates"

    # First update is on-policy: learner == sampler.
    learner0, sampler0 = captured[0]
    assert torch.allclose(learner0, sampler0, atol=1e-5)

    # Later updates: the sampler snapshot is unchanged but the learner moved.
    learner1, sampler1 = captured[-1]
    assert torch.allclose(sampler0, sampler1, atol=1e-6)
    ratio = torch.exp(torch.clamp(learner1 - sampler1, min=-20.0, max=20.0))
    assert (ratio - 1.0).abs().mean().item() > 1e-6


@pytest.mark.asyncio
async def test_gepo_single_update_is_on_policy_by_default(monkeypatch):
    """The default (one update per rollout) path stays exactly on-policy."""
    trainer = _tiny_gepo_trainer()
    responses = _fixed_group_responses()

    async def fake_generate(prompt, group_size):
        return responses

    monkeypatch.setattr(trainer, "generate_group_responses", fake_generate)

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig = trainer.compute_gepo_coefficient

    def spy(learner, sampler):
        captured.append((learner.detach().clone(), sampler.detach().clone()))
        return orig(learner, sampler)

    monkeypatch.setattr(trainer, "compute_gepo_coefficient", spy)

    await trainer.train_step(["hello"])

    assert len(captured) == 1
    learner, sampler = captured[0]
    assert torch.allclose(learner, sampler, atol=1e-5)


@pytest.mark.asyncio
async def test_gepo_multi_update_cadence_matches_dapo(monkeypatch):
    """Convention (shared with DAPO/VAPO): the LR scheduler advances once per
    inner update, global_step counts train_steps, not inner updates."""
    trainer = _tiny_gepo_trainer(num_gradient_updates=3)
    responses = _fixed_group_responses()

    async def fake_generate(prompt, group_size):
        return responses

    monkeypatch.setattr(trainer, "generate_group_responses", fake_generate)

    scheduler_steps = []
    orig_step = trainer.scheduler.step
    monkeypatch.setattr(
        trainer.scheduler,
        "step",
        lambda *a, **k: (scheduler_steps.append(1), orig_step())[1],
    )

    before = trainer.global_step
    await trainer.train_step(["hello"])

    assert len(scheduler_steps) == 3
    assert trainer.global_step == before + 1
