"""Behavioral tests for VAPO: rollout-time value clipping, terminal-token
rewards, wired decoupled-GAE critic advantages, and a single batched
optimizer step per train_step."""

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel

from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer


def tiny_model():
    torch.manual_seed(0)
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


@pytest.fixture
def vapo_trainer_tiny():
    config = VAPOConfig(
        model_name="gpt2",
        group_size=2,
        value_clip=0.2,
        per_device_train_batch_size=2,
    )

    def reward_fn(prompt: str, response: str) -> float:
        return 1.0

    trainer = VAPOTrainer(
        config=config,
        model=tiny_model(),
        tokenizer=None,
        reward_fn=reward_fn,
    )
    trainer.value_warmup_complete = True
    return trainer


def test_value_clipping_uses_rollout_values(vapo_trainer_tiny):
    """With old_values far from current values, the clipped branch must differ
    from the unclipped one (self-clipping made them identical)."""
    # returns is chosen close to `values` so the *clipped* branch of the
    # max(unclipped, clipped) selects (its squared error is larger than the
    # unclipped branch's) only when old_values is far from values -- this is
    # what actually exercises the clip, unlike returns=0.5 which always lets
    # the unclipped branch win regardless of old_values.
    t = vapo_trainer_tiny  # config.value_clip = 0.2
    values = torch.tensor([[1.0, 1.0]])
    old_values = torch.tensor([[0.0, 0.0]])
    returns = torch.tensor([[0.9, 0.9]])
    mask = torch.ones(1, 2)
    v_loss_far = t.compute_value_loss(values, old_values, returns, mask)
    v_loss_self = t.compute_value_loss(values, values.detach(), returns, mask)
    assert not torch.allclose(v_loss_far, v_loss_self)


def test_reward_on_terminal_token_only(vapo_trainer_tiny):
    t = vapo_trainer_tiny
    rewards = t.build_token_rewards(
        scalar_reward=1.0, response_mask=torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    )
    assert rewards.tolist() == [[0.0, 0.0, 0.0, 1.0]]


def test_build_token_rewards_per_row_batch(vapo_trainer_tiny):
    """Per-row reward tensor input places each row's reward on that row's
    own terminal token, independent of other rows' padding."""
    t = vapo_trainer_tiny
    response_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0],  # terminal token at index 2
            [0.0, 1.0, 1.0, 1.0],  # terminal token at index 3
        ]
    )
    rewards_tensor = torch.tensor([2.0, 3.0])
    rewards = t.build_token_rewards(rewards_tensor, response_mask)
    assert rewards.tolist() == [[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 3.0]]


def test_fp32_value_head_accepts_bf16_policy_hidden_states(
    vapo_trainer_tiny, monkeypatch
):
    """Mixed-precision policies must not fail in the fp32 critic matmul."""
    hidden_size = vapo_trainer_tiny.model.config.hidden_size
    hidden_states = torch.randn(2, 5, hidden_size, dtype=torch.bfloat16)
    monkeypatch.setattr(
        vapo_trainer_tiny,
        "get_hidden_states",
        lambda input_ids, attention_mask: hidden_states,
    )
    values = vapo_trainer_tiny.compute_values(
        torch.ones(2, 5, dtype=torch.long), torch.ones(2, 5, dtype=torch.long)
    )
    assert values.shape == (2, 5)
    assert values.dtype == torch.float32


def test_compute_vapo_losses_uses_critic_advantages_for_value_target(vapo_trainer_tiny):
    """value_loss must depend on critic_advantages (decoupled GAE), not
    just be silently ignored."""
    t = vapo_trainer_tiny
    current_log_probs = torch.zeros(1, 3, requires_grad=True)
    old_log_probs = torch.zeros(1, 3)
    policy_advantages = torch.zeros(1, 3)
    values = torch.tensor([[0.2, 0.2, 0.2]], requires_grad=True)
    old_values = torch.tensor([[0.0, 0.0, 0.0]])
    response_mask = torch.ones(1, 3)
    positive_mask = torch.zeros(1, 3)

    _, value_loss_a, _ = t.compute_vapo_losses(
        current_log_probs,
        old_log_probs,
        policy_advantages,
        torch.tensor([[1.0, 1.0, 1.0]]),  # critic_advantages
        values,
        old_values,
        response_mask,
        positive_mask,
    )
    _, value_loss_b, _ = t.compute_vapo_losses(
        current_log_probs,
        old_log_probs,
        policy_advantages,
        torch.tensor([[5.0, 5.0, 5.0]]),  # different critic_advantages
        values,
        old_values,
        response_mask,
        positive_mask,
    )
    assert not torch.allclose(value_loss_a, value_loss_b)


@pytest.mark.asyncio
async def test_train_step_single_optimizer_step_per_call(
    vapo_trainer_tiny, monkeypatch
):
    """optimizer.step() must be called once per train_step, not once per
    prompt in the batch."""
    t = vapo_trainer_tiny

    async def fake_generate_group_responses(prompt):
        responses = []
        for _ in range(2):
            input_ids = torch.randint(0, 200, (10,))
            attention_mask = torch.ones(10, dtype=torch.long)
            response_mask = torch.zeros(10)
            response_mask[4:] = 1.0
            responses.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "response_mask": response_mask,
                    "sequence_length": 6,
                    "prompt_length": 4,
                    "response": "hello",
                }
            )
        return responses

    monkeypatch.setattr(t, "generate_group_responses", fake_generate_group_responses)

    actor_steps = []
    critic_steps = []
    orig_actor_step = t.actor_optimizer.step
    orig_critic_step = t.critic_optimizer.step
    monkeypatch.setattr(
        t.actor_optimizer,
        "step",
        lambda *a, **k: (actor_steps.append(1), orig_actor_step())[1],
    )
    monkeypatch.setattr(
        t.critic_optimizer,
        "step",
        lambda *a, **k: (critic_steps.append(1), orig_critic_step())[1],
    )

    await t.train_step(["prompt one", "prompt two"])

    assert len(actor_steps) == 1
    assert len(critic_steps) == 1


@pytest.mark.asyncio
async def test_train_step_releases_each_prompt_graph_before_next_forward(
    vapo_trainer_tiny, monkeypatch
):
    """Prompt graphs are backpropagated sequentially to bound CUDA memory."""
    t = vapo_trainer_tiny
    monkeypatch.setattr(t, "generate_group_responses", _fake_group_generator())
    original = t.compute_vapo_losses
    prior_losses: list[torch.Tensor] = []

    def observe_backward(*args, **kwargs):
        if prior_losses:
            assert prior_losses[-1].grad is not None
        losses = original(*args, **kwargs)
        losses[0].retain_grad()
        prior_losses.append(losses[0])
        return losses

    monkeypatch.setattr(t, "compute_vapo_losses", observe_backward)
    await t.train_step(["prompt one", "prompt two"])

    assert len(prior_losses) == 2
    assert all(loss.grad is not None for loss in prior_losses)


def _fake_group_generator():
    async def fake_generate_group_responses(prompt):
        torch.manual_seed(3)
        responses = []
        for _ in range(2):
            input_ids = torch.randint(0, 200, (10,))
            attention_mask = torch.ones(10, dtype=torch.long)
            response_mask = torch.zeros(10)
            response_mask[4:] = 1.0
            responses.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "response_mask": response_mask,
                    "sequence_length": 6,
                    "prompt_length": 4,
                    "response": "hello",
                }
            )
        return responses

    return fake_generate_group_responses


def _spy_on_losses(trainer, monkeypatch):
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig = trainer.compute_vapo_losses

    def spy(current_log_probs, old_log_probs, *args, **kwargs):
        captured.append(
            (current_log_probs.detach().clone(), old_log_probs.detach().clone())
        )
        return orig(current_log_probs, old_log_probs, *args, **kwargs)

    monkeypatch.setattr(trainer, "compute_vapo_losses", spy)
    return captured


@pytest.mark.asyncio
async def test_vapo_old_logprobs_frozen_across_gradient_updates(
    vapo_trainer_tiny, monkeypatch
):
    """num_gradient_updates=2 must reuse the rollout-time old log probs, so
    the second inner update sees a genuinely off-policy ratio."""
    t = vapo_trainer_tiny
    t.config.num_gradient_updates = 2
    t.config.actor_learning_rate = 1e-2
    for group in t.actor_optimizer.param_groups:
        group["lr"] = 1e-2
    monkeypatch.setattr(t, "generate_group_responses", _fake_group_generator())
    captured = _spy_on_losses(t, monkeypatch)

    await t.train_step(["prompt one"])

    assert len(captured) == 2
    cur0, old0 = captured[0]
    assert torch.allclose(cur0, old0, atol=1e-5)  # first update is on-policy
    cur1, old1 = captured[1]
    assert torch.allclose(old0, old1, atol=1e-6)  # snapshot unchanged
    ratio = torch.exp(torch.clamp(cur1 - old1, min=-20.0, max=20.0))
    assert (ratio - 1.0).abs().mean().item() > 1e-6


@pytest.mark.asyncio
async def test_vapo_default_single_update_is_on_policy(vapo_trainer_tiny, monkeypatch):
    """The default single-update path is unchanged: exactly one loss call,
    on-policy."""
    t = vapo_trainer_tiny
    monkeypatch.setattr(t, "generate_group_responses", _fake_group_generator())
    captured = _spy_on_losses(t, monkeypatch)

    await t.train_step(["prompt one"])

    assert len(captured) == 1
    cur, old = captured[0]
    assert torch.allclose(cur, old, atol=1e-5)


def test_vapo_policy_loss_finite_for_extreme_log_ratio(vapo_trainer_tiny):
    """An extreme off-policy token must not overflow the importance ratio to
    inf and take the whole policy loss with it."""
    t = vapo_trainer_tiny
    current_log_probs = torch.tensor([[200.0, 0.0]], requires_grad=True)
    old_log_probs = torch.tensor([[0.0, 0.0]])
    assert not torch.isfinite(torch.exp(current_log_probs - old_log_probs)).all()

    policy_loss, _, _ = t.compute_vapo_losses(
        current_log_probs,
        old_log_probs,
        torch.tensor([[-1.0, -1.0]]),  # policy advantages: the unclipped
        # branch wins for a negative advantage, so an inf ratio propagates
        torch.tensor([[0.0, 0.0]]),  # critic advantages
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.zeros(1, 2),
    )
    assert torch.isfinite(policy_loss)


def test_compute_token_log_probs_masks_prompt_positions(vapo_trainer_tiny):
    """Passing the real response mask must zero prompt positions; the caller
    masks anyway, so the losses are unchanged either way."""
    t = vapo_trainer_tiny
    torch.manual_seed(5)
    input_ids = torch.randint(0, 200, (1, 8))
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    response_mask = torch.zeros(1, 8)
    response_mask[:, 4:] = 1.0

    unmasked = t.compute_token_log_probs(input_ids, attention_mask)
    masked = t.compute_token_log_probs(input_ids, attention_mask, response_mask)

    assert masked.shape == unmasked.shape
    assert torch.all(masked[:, :3] == 0.0)  # shifted: P-1 leading zeros
    assert torch.allclose(masked[:, 3:], unmasked[:, 3:])


def test_vapo_logprob_dtype_config_selects_bf16():
    """config.logprob_dtype='bf16' must reach gather_token_logprobs."""
    config = VAPOConfig(
        model_name="gpt2",
        group_size=2,
        per_device_train_batch_size=2,
        logprob_dtype="bf16",
    )
    trainer = VAPOTrainer(
        config=config,
        model=tiny_model(),
        tokenizer=None,
        reward_fn=lambda prompt, response: 1.0,
    )
    torch.manual_seed(7)
    ids = torch.randint(0, 200, (1, 8))
    am = torch.ones(1, 8, dtype=torch.long)
    assert trainer.compute_token_log_probs(ids, am).dtype == torch.bfloat16


@pytest.mark.asyncio
async def test_vapo_multi_update_cadence_matches_dapo(vapo_trainer_tiny, monkeypatch):
    """Convention (shared with DAPO/GEPO): the LR schedulers advance once per
    inner update, global_step counts train_steps, not inner updates."""
    t = vapo_trainer_tiny
    t.config.num_gradient_updates = 3
    monkeypatch.setattr(t, "generate_group_responses", _fake_group_generator())

    actor_steps: list[int] = []
    critic_steps: list[int] = []
    orig_actor = t.actor_scheduler.step
    orig_critic = t.critic_scheduler.step
    monkeypatch.setattr(
        t.actor_scheduler,
        "step",
        lambda *a, **k: (actor_steps.append(1), orig_actor())[1],
    )
    monkeypatch.setattr(
        t.critic_scheduler,
        "step",
        lambda *a, **k: (critic_steps.append(1), orig_critic())[1],
    )

    before = t.global_step
    await t.train_step(["prompt one"])

    assert len(actor_steps) == 3
    assert len(critic_steps) == 3
    assert t.global_step == before + 1
