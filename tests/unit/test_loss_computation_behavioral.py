"""Behavioral tests for loss_computation: ratio normalization, entropy gradient,
and narrow exception handling around the forward pass."""

import math
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.training import loss_computation as lc


def test_ratio_is_length_normalized():
    """A 200-token response with per-token drift 0.01 must give a finite,
    O(1) ratio — not exp(2.0) vs exp(sum) overflow behavior."""
    new_lp_sum = torch.tensor(-400.0)
    old_lp_sum = torch.tensor(-402.0)
    ratio = lc.compute_ppo_ratio(new_lp_sum, old_lp_sum, token_count=200)
    assert math.isfinite(ratio.item())
    assert abs(ratio.item() - math.exp(2.0 / 200)) < 1e-6


def test_ratio_normalization_avoids_overflow_for_long_responses():
    """Raw-sum ratios would overflow for long sequences; normalized ratios stay bounded."""
    new_lp_sum = torch.tensor(-2000.0)
    old_lp_sum = torch.tensor(-2100.0)
    ratio = lc.compute_ppo_ratio(new_lp_sum, old_lp_sum, token_count=1000)
    assert math.isfinite(ratio.item())
    assert ratio.item() < 2.0


def test_ratio_token_count_floor():
    """token_count of 0 must not raise a division error."""
    ratio = lc.compute_ppo_ratio(torch.tensor(-1.0), torch.tensor(-1.0), token_count=0)
    assert math.isfinite(ratio.item())


def test_entropy_bonus_has_gradient():
    logits = torch.randn(1, 6, 50, requires_grad=True)
    mask = torch.ones(1, 6)
    ent = lc.compute_entropy_bonus(logits, mask)
    ent.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_entropy_bonus_masks_padding():
    logits = torch.randn(1, 6, 50)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    ent = lc.compute_entropy_bonus(logits, mask)
    assert math.isfinite(ent.item())
    assert ent.item() > 0


def test_attribute_errors_propagate():
    """Systematic bugs must not be swallowed into zero loss."""
    assert AttributeError not in lc.LOSS_EXCEPTIONS
    assert KeyError not in lc.LOSS_EXCEPTIONS
    assert TypeError not in lc.LOSS_EXCEPTIONS
    assert RuntimeError in lc.LOSS_EXCEPTIONS
    assert ValueError in lc.LOSS_EXCEPTIONS


def test_loss_exceptions_matches_canonical_tuple():
    from stateset_agents.exceptions import LOSS_EXCEPTIONS as canonical

    assert lc.LOSS_EXCEPTIONS == canonical


# --- GRPO loss defect regressions (task 1.7) -------------------------------


class _FixedLossModel(torch.nn.Module):
    """Model returning a constant per-token mean NLL that depends on a param."""

    def __init__(self, nll_value: float = 0.5, vocab: int = 7):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(float(nll_value)))
        self.vocab = vocab
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        seq = input_ids.shape[1]
        logits = torch.zeros(1, seq, self.vocab) + self.p
        return SimpleNamespace(loss=self.p * 1.0, logits=logits)


class _MaskTokenizer:
    """Chat tokenizer emitting `n_assistant` assistant tokens out of `n_total`."""

    def __init__(self, n_total: int, n_assistant: int):
        self.n_total = n_total
        self.n_assistant = n_assistant

    def apply_chat_template(
        self,
        messages,
        *,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **kwargs,
    ):
        n, a = self.n_total, self.n_assistant
        input_ids = torch.arange(1, n + 1).unsqueeze(0)
        attention_mask = torch.ones(1, n, dtype=torch.long)
        assistant = torch.tensor([[0] * (n - a) + [1] * a])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_tokens_mask": assistant,
        }


def _make_group(n_total: int, n_assistant: int, log_probs=None):
    traj = SimpleNamespace(
        turns=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        total_reward=1.0,
        metadata={},
    )
    if log_probs is not None:
        traj.log_probs = log_probs
    return SimpleNamespace(trajectories=[traj])


def _make_agent(n_total: int, n_assistant: int, nll_value: float = 0.5):
    return SimpleNamespace(
        tokenizer=_MaskTokenizer(n_total, n_assistant),
        model=_FixedLossModel(nll_value),
    )


def _config(**overrides):
    base = {
        "max_prompt_length": 64,
        "max_completion_length": 64,
        "token_level_loss": False,
        "clip_ratio": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_token_level_loss_not_double_normalised():
    """Identical per-token NLL must give identical loss regardless of length.

    `outputs.loss` is already the per-token mean, so dividing by the token
    count again introduced a 1/L^2 bias (short responses weighted 2x here).
    """
    cfg = _config(token_level_loss=True)
    advantages = torch.tensor([1.0])

    short_loss, _ = lc._compute_group_policy_loss(
        _make_group(8, 4), advantages, cfg, _make_agent(8, 4)
    )
    long_loss, _ = lc._compute_group_policy_loss(
        _make_group(12, 8), advantages, cfg, _make_agent(12, 8)
    )

    assert short_loss.item() == pytest.approx(long_loss.item(), rel=1e-6)


def _grad_norm_for_ratio(per_token_drift: float, seq_clip_ratio: float) -> float:
    """Run one clipped-surrogate step where ratio == exp(per_token_drift)."""
    n_assistant, nll_value = 4, 0.5
    agent = _make_agent(8, n_assistant, nll_value)
    # ratio = exp((new - old)/T); new = -(nll*T) => old = -(nll + drift)*T
    old = -(nll_value + per_token_drift) * n_assistant
    group = _make_group(8, n_assistant, log_probs=old)
    cfg = _config(clip_ratio=0.2, seq_clip_ratio=seq_clip_ratio)

    loss, _ = lc._compute_group_policy_loss(group, torch.tensor([1.0]), cfg, agent)
    loss.backward()
    grad = agent.model.p.grad
    return 0.0 if grad is None else float(grad.abs().sum())


def test_sequence_ratio_clip_is_active():
    """A GSPO-scale clip must actually bite on the sequence-mean ratio."""
    # ratio ~ 1.001 > 1 + 3e-4 with A > 0 -> clipped branch -> no gradient.
    assert _grad_norm_for_ratio(1e-3, 3e-4) == pytest.approx(0.0, abs=1e-9)
    # ratio ~ 1.0001 stays inside the trust region -> gradient flows.
    assert _grad_norm_for_ratio(1e-4, 3e-4) > 1e-6


def test_ppo_clip_uses_seq_clip_ratio(monkeypatch):
    """The ratio clip must be read from `seq_clip_ratio`, not `clip_ratio`."""
    recorded = {}
    real = lc.rl_losses.clipped_surrogate

    def spy(ratio, advantages, *, clip_low, clip_high):
        recorded["clip_low"] = clip_low
        recorded["clip_high"] = clip_high
        return real(ratio, advantages, clip_low=clip_low, clip_high=clip_high)

    monkeypatch.setattr(lc.rl_losses, "clipped_surrogate", spy)

    agent = _make_agent(8, 4)
    group = _make_group(8, 4, log_probs=-2.0)
    cfg = _config(clip_ratio=0.2, seq_clip_ratio=7e-4)
    lc._compute_group_policy_loss(group, torch.tensor([1.0]), cfg, agent)

    assert recorded == {"clip_low": 7e-4, "clip_high": 7e-4}


def test_seq_clip_ratio_default_is_gspo_scale():
    from stateset_agents.training.config import TrainingConfig

    assert TrainingConfig().seq_clip_ratio == 3e-4
