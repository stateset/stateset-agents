"""PPO must use the k3 KL estimator and the clamped ratio (objective library)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from stateset_agents.training import rl_losses  # noqa: E402
from stateset_agents.training.ppo_trainer import PPOConfig, PPOTrainer  # noqa: E402


def _trainer(**config_kwargs):
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
    return PPOTrainer(
        config=PPOConfig(model_name="gpt2", **config_kwargs),
        model=model,
        tokenizer=None,
        reward_fn=lambda p, r: 0.0,
    )


def test_kl_is_k3_nonnegative_with_gradient_toward_ref():
    tr = _trainer()
    cur = (torch.randn(2, 5) - 1).requires_grad_(True)
    ref = cur.detach() + 0.3
    mask = torch.ones(2, 5)
    kl = tr.compute_kl_divergence(cur, ref, mask)
    assert kl.item() >= 0
    torch.testing.assert_close(kl, rl_losses.k3_kl(cur, ref, mask))
    kl.backward()
    # gradient descends toward ref: cur < ref so grad must be negative
    assert (cur.grad < 0).all()


def test_ratio_is_clamped_not_inf():
    tr = _trainer()
    cur = torch.zeros(1, 4, requires_grad=True)
    old = torch.full((1, 4), -200.0)
    loss, frac = tr.ppo_loss(cur, old, torch.ones(1, 4), torch.ones(1, 4))
    assert torch.isfinite(loss)
    assert frac.item() == pytest.approx(1.0)


def test_episode_reward_is_terminal_not_broadcast():
    """A scalar episode reward lands on the last response action. Broadcasting
    it onto every token made returns scale with response length, and padding
    after a short response must neither earn reward nor bootstrap values."""
    tr = _trainer()
    tr.config.gamma = 1.0
    tr.config.gae_lambda = 1.0
    action_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 0.0],  # short response, then right padding
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    old_values = torch.linspace(-0.5, 0.5, 10).reshape(2, 5)
    advantages, returns = tr.compute_episode_advantages(
        torch.tensor([1.0, 1.0]), action_mask, old_values
    )
    # gamma=1: every action's Monte-Carlo return is the episode reward itself.
    torch.testing.assert_close(returns * action_mask, action_mask)
    valid = advantages[action_mask > 0]
    assert valid.mean().abs().item() < 1e-5
    assert valid.std().item() == pytest.approx(1.0, abs=1e-4)


def test_response_mask_excludes_padding_both_sides():
    tr = _trainer()
    attention = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    mask = tr.build_response_mask(attention, torch.tensor([2, 3]), "right")
    assert mask.tolist() == [[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1]]
    attention = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    mask = tr.build_response_mask(attention, torch.tensor([2, 3]), "left")
    assert mask.tolist() == [[0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1]]


def test_kl_penalty_gets_a_frozen_reference_model_by_default():
    tr = _trainer(beta=0.1)
    assert tr.ref_model is not None and tr.ref_model is not tr.model
    assert not tr.ref_model.training
    assert not any(p.requires_grad for p in tr.ref_model.parameters())


def test_kl_penalty_without_reference_model_is_rejected():
    with pytest.raises(ValueError, match="reference model"):
        _trainer(beta=0.1, use_reference_model=False)


def test_no_reference_model_without_kl_penalty():
    assert _trainer().ref_model is None
