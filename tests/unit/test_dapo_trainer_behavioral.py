"""Behavioral tests: importance ratios must come from rollout-time log probs."""

import pytest
import torch

from stateset_agents.training import rl_losses

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
    orig_loss = trainer.compute_dapo_loss_from_log_probs

    def spy(cur, old_, adv, mask):
        seen_ratios.append(trainer.compute_importance_ratio(cur, old_).detach())
        return orig_loss(cur, old_, adv, mask)

    monkeypatch.setattr(trainer, "compute_dapo_loss_from_log_probs", spy)

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


def test_logprob_dtype_config_selects_bf16(dapo_trainer_factory):
    """config.logprob_dtype='bf16' must reach gather_token_logprobs."""
    trainer = dapo_trainer_factory(tiny_model(), logprob_dtype="bf16")
    torch.manual_seed(7)
    ids = torch.randint(0, 200, (1, 8))
    am = torch.ones(1, 8, dtype=torch.long)
    rm = torch.zeros(1, 8)
    rm[:, 4:] = 1.0
    logp, _ = trainer.compute_token_log_probs(ids, am, rm)
    assert logp.dtype == torch.bfloat16

    default = dapo_trainer_factory(tiny_model())
    logp32, _ = default.compute_token_log_probs(ids, am, rm)
    assert logp32.dtype == torch.float32


def test_token_counts_exact_with_bf16_logprobs():
    """Token counts come from the returned mask, so it must never be handed
    back in a low-precision dtype: bf16 cannot represent integers past 256
    exactly, and a count is not something to approximate."""
    # A bf16 model produces bf16 logits; the mask used to be cast to that
    # dtype, so summing it lost the exact token count.
    logits = torch.zeros(1, 301, 200, dtype=torch.bfloat16)
    ids = torch.zeros(1, 301, dtype=torch.long)
    mask = torch.ones(1, 301)

    logp, shifted_mask = rl_losses.gather_token_logprobs(
        logits, ids, mask, dtype=torch.bfloat16
    )
    assert logp.dtype == torch.bfloat16
    assert shifted_mask.dtype == torch.float32
    assert float(shifted_mask.sum()) == 300.0


def test_loss_from_log_probs_routes_through_policy_objective(
    dapo_trainer_factory, monkeypatch
):
    from stateset_agents.training import objectives

    trainer = dapo_trainer_factory(tiny_model())
    seen = {}
    real = objectives.policy_loss

    def spy(**kw):
        seen["objective"] = kw["objective"]
        return real(**kw)

    monkeypatch.setattr(objectives, "policy_loss", spy)
    ids, am, rm = make_batch()
    with torch.no_grad():
        old, _ = trainer.compute_token_log_probs(ids, am, rm)
    cur, _ = trainer.compute_token_log_probs(ids, am, rm)
    loss, metrics = trainer.compute_dapo_loss_from_log_probs(
        cur, old, torch.tensor([1.0, -1.0]), rm[:, 1:]
    )
    assert seen["objective"].name == "dapo"
    assert seen["objective"].clip_high == trainer.config.clip_eps_high
    assert torch.isfinite(loss) and "clip_fraction" in metrics


def test_ratio_and_log_prob_entry_points_agree(dapo_trainer_factory):
    trainer = dapo_trainer_factory(tiny_model())
    ids, am, rm = make_batch()
    with torch.no_grad():
        old, _ = trainer.compute_token_log_probs(ids, am, rm)
    cur, _ = trainer.compute_token_log_probs(ids, am, rm)
    adv = torch.tensor([1.0, -1.0])
    via_lp, _ = trainer.compute_dapo_loss_from_log_probs(cur, old, adv, rm[:, 1:])
    ratios = trainer.compute_importance_ratio(cur, old)
    via_ratio = trainer.compute_dapo_loss(
        ratios, adv.unsqueeze(1).expand_as(ratios), rm[:, 1:]
    )
    torch.testing.assert_close(via_lp, via_ratio)


def _stub_rollouts(trainer, monkeypatch, rewards_by_prompt):
    ids, am, rm = make_batch()

    async def fake_generate(prompt):
        return [
            {
                "response": f"{prompt}-{i}",
                "input_ids": ids[i],
                "attention_mask": am[i],
                "response_mask": rm[i],
                "sequence_length": int(rm[i].sum()),
            }
            for i in range(2)
        ]

    def reward_fn(prompt, response):
        return rewards_by_prompt[prompt][int(response.rsplit("-", 1)[1])]

    monkeypatch.setattr(trainer, "generate_group_responses", fake_generate)
    trainer.reward_fn = reward_fn


@pytest.mark.asyncio
async def test_dynamic_sampling_without_verifier_keeps_informative_groups(
    dapo_trainer_factory, monkeypatch
):
    """With no verifier (the default), groups with reward spread must survive;
    only zero-signal groups (all rewards equal) are filtered."""
    trainer = dapo_trainer_factory(tiny_model())
    _stub_rollouts(
        trainer,
        monkeypatch,
        {"mixed": [1.0, 0.0], "solved": [1.0, 1.0], "graded": [0.2, 0.4]},
    )
    samples, filter_ratio = await trainer.collect_samples_with_dynamic_sampling(
        ["mixed", "solved", "graded"], target_batch_size=8
    )
    assert [s["prompt"] for s in samples] == ["mixed", "graded"]
    assert filter_ratio == pytest.approx(1 / 3)
    assert samples[0]["accuracy"] == 0.5


@pytest.mark.asyncio
async def test_dynamic_sampling_with_verifier_uses_accuracy(
    dapo_trainer_factory, monkeypatch
):
    trainer = dapo_trainer_factory(tiny_model())
    _stub_rollouts(trainer, monkeypatch, {"a": [0.3, 0.7], "b": [0.3, 0.7]})
    trainer.verifier_fn = lambda prompt, response: prompt == "a" and response.endswith(
        "-0"
    )
    samples, _ = await trainer.collect_samples_with_dynamic_sampling(
        ["a", "b"], target_batch_size=8
    )
    assert [s["prompt"] for s in samples] == ["a"]


@pytest.mark.asyncio
async def test_one_optimizer_step_per_inner_update_over_the_whole_batch(
    dapo_trainer_factory, monkeypatch
):
    """DAPO's token-level loss averages over *all* tokens in the batch: the
    groups' gradients accumulate into one optimizer step per inner update
    (previously every prompt group took its own step)."""
    trainer = dapo_trainer_factory(tiny_model(), num_gradient_updates=2)
    ids, am, rm = make_batch()
    rm_short = rm.clone()
    rm_short[:, 8:] = 0.0  # second group has fewer response tokens

    def sample(response_mask, adv):
        with torch.no_grad():
            old, _ = trainer.compute_token_log_probs(ids, am, response_mask)
        return {
            "responses": [
                {
                    "input_ids": ids[i],
                    "attention_mask": am[i],
                    "response_mask": response_mask[i],
                    "sequence_length": int(response_mask[i].sum()),
                }
                for i in range(2)
            ],
            "advantages": torch.tensor(adv),
            "rewards": [1.0, 0.0],
            "accuracy": 0.5,
            "old_token_log_probs": old,
        }

    samples = [sample(rm, [1.0, -1.0]), sample(rm_short, [-1.0, 1.0])]

    # Reference gradient: token-mean of the surrogate over both groups' tokens.
    trainer.optimizer.zero_grad()
    per_token, masks = [], []
    for s, mask in zip(samples, (rm, rm_short), strict=True):
        cur, _ = trainer.compute_token_log_probs(ids, am, mask)
        ratio = torch.exp(cur - s["old_token_log_probs"])
        per_token.append(ratio * s["advantages"].unsqueeze(1))
        masks.append(mask[:, 1:])
    loss = -(torch.cat(per_token) * torch.cat(masks)).sum() / torch.cat(masks).sum()
    loss.backward()
    expected = [p.grad.clone() for p in trainer.model.parameters()]
    trainer.optimizer.zero_grad()

    grads, steps = [], []

    def record_step(*args, **kwargs):
        steps.append(1)
        if len(steps) == 1:
            grads.extend(p.grad.clone() for p in trainer.model.parameters())

    monkeypatch.setattr(trainer.optimizer, "step", record_step)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: None)

    async def fake_collect(prompts, n):
        return samples, 0.0

    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", fake_collect)
    await trainer.train_step(["q"])
    assert len(steps) == 2
    for got, want in zip(grads, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)
