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
async def test_train_step_single_optimizer_step_per_call(vapo_trainer_tiny, monkeypatch):
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
        t.actor_optimizer, "step", lambda *a, **k: (actor_steps.append(1), orig_actor_step())[1]
    )
    monkeypatch.setattr(
        t.critic_optimizer, "step", lambda *a, **k: (critic_steps.append(1), orig_critic_step())[1]
    )

    await t.train_step(["prompt one", "prompt two"])

    assert len(actor_steps) == 1
    assert len(critic_steps) == 1
