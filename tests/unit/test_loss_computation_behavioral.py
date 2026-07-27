"""Behavioral tests for loss_computation: ratio normalization, entropy gradient,
and narrow exception handling around the forward pass."""

import math

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
