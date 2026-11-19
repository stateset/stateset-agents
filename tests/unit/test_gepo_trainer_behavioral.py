import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.training import gepo_trainer
from stateset_agents.training.gepo_trainer import GEPOTrainer


def test_wandb_is_required_only_when_logging_is_requested(monkeypatch):
    monkeypatch.setattr(gepo_trainer, "wandb", None)
    with pytest.raises(ImportError, match="GEPO logging"):
        gepo_trainer._require_wandb()


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


def _tiny_gepo_trainer(
    num_gradient_updates: int = 1, model_dtype=None, **config_overrides
):
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
        **config_overrides,
    )
    return GEPOTrainer(
        config=config,
        model=model.to(model_dtype) if model_dtype is not None else model,
        tokenizer=None,
        reward_fn=lambda prompt, response: 1.0 if response == "ok" else 0.0,
    )


@pytest.mark.parametrize("warmup", [0.0, 0.5, 1.0])
def test_cosine_budget_counts_inner_updates_and_stays_at_zero(warmup):
    trainer = _tiny_gepo_trainer(
        num_gradient_updates=3, num_episodes=2, num_epochs=9, warmup_ratio=warmup
    )
    warmup_steps = int(6 * warmup)
    for step in range(9):
        if step >= 6:
            expected = 0.0
        elif step < warmup_steps:
            expected = step / warmup_steps
        else:
            expected = (
                1 + math.cos(math.pi * (step - warmup_steps) / (6 - warmup_steps))
            ) / 2
        assert trainer.scheduler.get_last_lr()[0] == pytest.approx(
            expected * trainer.config.learning_rate
        )
        trainer.optimizer.step()
        trainer.scheduler.step()


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


def _record_policy_scores(monkeypatch):
    captured = []
    policy_loss = gepo_trainer.objectives.policy_loss

    def inspect(**kwargs):
        old = kwargs["logp_old"]
        current = (kwargs["logp_cur"] * kwargs["mask"]).sum(-1, dtype=old.dtype)
        captured.append((current.detach().clone(), old.detach().clone()))
        return policy_loss(**kwargs)

    monkeypatch.setattr(gepo_trainer.objectives, "policy_loss", inspect)
    return captured


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

    captured = _record_policy_scores(monkeypatch)

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

    captured = _record_policy_scores(monkeypatch)

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


@pytest.mark.asyncio
async def test_ragged_prompt_boundaries_and_empty_update_preserve_training_state(
    monkeypatch,
):
    trainer = _tiny_gepo_trainer(num_gradient_updates=2)
    responses = _fixed_group_responses()
    responses[0]["response_start_idx"] = 6
    responses[1]["response_start_idx"] = 8
    responses[0]["sampler_log_prob"] = -19.0000001
    responses[0]["attention_mask"][-1] = 0

    async def fake_generate(prompt, group_size):
        return responses

    monkeypatch.setattr(trainer, "generate_group_responses", fake_generate)
    with torch.no_grad():
        ids = responses[1]["input_ids"][None]
        logits = trainer.model(ids).logits[0].log_softmax(-1)
        fallback = logits[[7, 8], ids[0, 8:]].double().sum().item()
    observed = []
    policy_loss = gepo_trainer.objectives.policy_loss

    def inspect(**kwargs):
        result = policy_loss(**kwargs)
        observed.append(
            (
                kwargs["mask"].tolist(),
                kwargs["logp_old"].tolist(),
                result.ratio.mean().item(),
            )
        )
        return result

    monkeypatch.setattr(gepo_trainer.objectives, "policy_loss", inspect)
    metrics = await trainer.train_step(["p"])
    assert len(observed) == 2
    for mask, old, _ in observed:
        assert mask == [[0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1]]
        assert old[0] == -19.0000001
        assert old[1] == pytest.approx(fallback, abs=1e-6)
    assert metrics["gepo_coefficient_mean"] == pytest.approx(observed[-1][2])

    for row in responses:
        row["response_start_idx"] = len(row["input_ids"])
        row.pop("sampler_log_prob", None)
    params = [p.detach().clone() for p in trainer.model.parameters()]
    optimizer_state = deepcopy(trainer.optimizer.state_dict())
    scheduler_state = deepcopy(trainer.scheduler.state_dict())
    step = trainer.global_step
    empty = await trainer.train_step(["p"])
    assert empty["policy_loss"] == 0.0
    assert empty["average_reward"] == 0.5
    assert trainer.global_step == step
    assert len(observed) == 2
    torch.testing.assert_close(
        trainer.optimizer.state_dict(), optimizer_state, rtol=0, atol=0
    )
    assert trainer.scheduler.state_dict() == scheduler_state
    for original, trained in zip(params, trainer.model.parameters(), strict=True):
        assert torch.equal(original, trained)


@pytest.mark.asyncio
async def test_generation_records_probability_of_rounded_multinomial_weight(
    monkeypatch,
):
    trainer = _tiny_gepo_trainer()

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, prompt, **kwargs):
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

        def decode(self, ids, **kwargs):
            return "rare"

    scores = torch.tensor([[0.0, -103.0]])

    def generate(**kwargs):
        assert kwargs["do_sample"]
        return SimpleNamespace(sequences=torch.tensor([[1, 1]]), scores=(scores,))

    trainer.tokenizer = Tokenizer()
    monkeypatch.setattr(trainer.model, "generate", generate)
    (row,) = await trainer.generate_group_responses("p", 1)
    weights = scores.softmax(-1)[0].double().tolist()
    expected = math.log(weights[1] / sum(weights))
    assert row["sampler_log_prob"].item() == pytest.approx(expected, abs=1e-12)
    assert abs(expected - scores.log_softmax(-1)[0, 1].item()) > 0.1


def test_undefined_padded_logits_have_zero_backward_contribution():
    class PaddedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.Parameter(
                torch.tensor([[[math.nan, math.inf], [2.0, -1.0], [-1.0, 2.0]]])
            )

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.values)

    model = PaddedModel()
    ids = torch.tensor([[0, 1, 0]])
    responses = torch.tensor([[0, 1]])
    lp = GEPOTrainer._log_distribution(model, ids, torch.ones_like(ids), responses)
    loss = gepo_trainer.objectives.policy_loss(
        logp_cur=lp.gather(-1, ids[:, 1:, None]).squeeze(-1),
        mask=responses,
        group_ids=[0],
        advantages=[1],
        logp_old=torch.tensor([0.0]),
        objective=gepo_trainer.objectives.OBJECTIVES["gepo"].with_(clip="none"),
    ).loss
    loss.backward()
    assert model.values.grad.isfinite().all()
    assert model.values.grad[0, 0].count_nonzero() == 0
    assert model.values.grad[0, 1].count_nonzero() == 2
