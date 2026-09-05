"""PPO must use the k3 KL estimator and the clamped ratio (objective library)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from stateset_agents.training import rl_losses  # noqa: E402
from stateset_agents.training.ppo_trainer import PPOConfig, PPOTrainer  # noqa: E402


def _trainer():
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
        config=PPOConfig(model_name="gpt2"),
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
