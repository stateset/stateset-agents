"""Preference losses against hand-computed values, plus input validation."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.preference.losses import (  # noqa: E402
    NonFiniteLossError,
    bradley_terry_loss,
    dpo_loss,
    ipo_loss,
    kto_kl_estimate,
    kto_loss,
    mean_logprobs,
    orpo_loss,
    pairwise_accuracy,
    pairwise_auc,
    reduce_losses,
    sequence_logprobs,
    simpo_loss,
    validate_beta,
    validate_label_smoothing,
)


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _softplus(x: float) -> float:
    """-log σ(-x) = log(1 + e^x)."""
    return math.log1p(math.exp(x))


# h = (−1 − (−2)) − (−3 − (−2)) = 2;  β = 0.5  →  β h = 1
DPO_ARGS = (_t(-1.0), _t(-3.0), _t(-2.0), _t(-2.0))


def test_dpo_sigmoid_matches_hand_computation() -> None:
    out = dpo_loss(*DPO_ARGS, beta=0.5)
    assert out.losses.item() == pytest.approx(_softplus(-1.0))  # 0.31326
    assert out.chosen_rewards.item() == pytest.approx(0.5)
    assert out.rejected_rewards.item() == pytest.approx(-0.5)


def test_conservative_dpo_mixes_the_flipped_label() -> None:
    out = dpo_loss(*DPO_ARGS, beta=0.5, label_smoothing=0.1)
    expected = 0.9 * _softplus(-1.0) + 0.1 * _softplus(1.0)  # 0.41326
    assert out.losses.item() == pytest.approx(expected)


def test_dpo_is_zero_gradient_free_at_the_reference() -> None:
    policy = _t(-2.0, -5.0).requires_grad_(True)
    rejected = _t(-4.0, -1.0).requires_grad_(True)
    out = dpo_loss(policy, rejected, policy.detach(), rejected.detach(), beta=0.1)
    assert torch.allclose(out.losses, torch.full((2,), math.log(2.0)))
    out.losses.sum().backward()
    # dL/dlogπ(y_w) = -β σ(-β h) = -0.05 at h = 0
    assert torch.allclose(policy.grad, torch.full((2,), -0.05))
    assert torch.allclose(rejected.grad, torch.full((2,), 0.05))


def test_ipo_uses_per_token_log_ratios() -> None:
    # chosen log-ratio 1 over 2 tokens, rejected −1 over 3 tokens
    out = ipo_loss(
        _t(-4.0),
        _t(-6.0),
        _t(-5.0),
        _t(-5.0),
        _t(2.0),
        _t(3.0),
        beta=0.25,
    )
    delta = 1.0 / 2.0 + 1.0 / 3.0
    assert out.losses.item() == pytest.approx((delta - 2.0) ** 2)  # 1.36111
    assert out.chosen_rewards.item() == pytest.approx(0.25)


def test_simpo_is_reference_free_and_length_normalized() -> None:
    # mean log-probs −1 and −3; β = 2, γ = 1  →  β(−1 + 3) − γ = 3
    out = simpo_loss(_t(-2.0), _t(-6.0), _t(2.0), _t(2.0), beta=2.0, gamma=1.0)
    assert out.losses.item() == pytest.approx(_softplus(-3.0))  # 0.04859
    assert out.chosen_rewards.item() == pytest.approx(-2.0)
    assert out.rejected_rewards.item() == pytest.approx(-6.0)


def test_orpo_adds_the_odds_ratio_to_the_chosen_nll() -> None:
    chosen, rejected = -0.5, -2.0  # mean token log-probs
    log_odds = (chosen - rejected) - (
        math.log(1 - math.exp(chosen)) - math.log(1 - math.exp(rejected))
    )
    expected = -chosen + 0.1 * _softplus(-log_odds)
    out = orpo_loss(_t(-1.0), _t(-4.0), _t(2.0), _t(2.0), beta=0.1)
    assert out.losses.item() == pytest.approx(expected)
    assert torch.isfinite(
        orpo_loss(_t(-1e-9), _t(-4.0), _t(1.0), _t(1.0), beta=0.1).losses
    ).all()


def test_kto_desirable_and_undesirable_losses() -> None:
    out = kto_loss(
        _t(-1.0, -2.0),
        _t(-2.0, -1.0),
        torch.tensor([True, False]),
        torch.tensor(0.5),
        beta=1.0,
        desirable_weight=1.0,
        undesirable_weight=2.0,
    )
    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))  # noqa: E731
    assert out.losses.tolist() == pytest.approx(
        [1.0 - sigmoid(0.5), 2.0 * (1.0 - sigmoid(1.5))]
    )
    assert out.chosen_rewards.tolist() == pytest.approx([1.0])
    assert out.rejected_rewards.tolist() == pytest.approx([-1.0])


def test_kto_reference_point_is_a_clamped_detached_mean() -> None:
    assert kto_kl_estimate(_t(-1.0, -1.0), _t(-2.0, -3.0)).item() == pytest.approx(1.5)
    assert kto_kl_estimate(_t(-3.0, -3.0), _t(-1.0, -1.0)).item() == 0.0
    policy = _t(-1.0, -1.0).requires_grad_(True)
    assert not kto_kl_estimate(policy, _t(-2.0, -2.0)).requires_grad


def test_bradley_terry_with_margin_and_centering() -> None:
    out = bradley_terry_loss(_t(2.0, 0.0), _t(1.0, 1.0), margin=_t(0.5, 0.0))
    assert out.losses.tolist() == pytest.approx([_softplus(-0.5), _softplus(1.0)])
    centered = bradley_terry_loss(_t(2.0, 0.0), _t(1.0, 1.0), center_coefficient=0.1)
    assert centered.losses.tolist() == pytest.approx(
        [_softplus(-1.0) + 0.9, _softplus(1.0) + 0.1]
    )


def test_reduce_losses_is_a_weighted_mean_over_the_window() -> None:
    losses, weights = _t(1.0, 3.0), _t(1.0, 3.0)
    assert reduce_losses(losses, weights).item() == pytest.approx(2.5)
    assert reduce_losses(losses, weights, normalizer=8.0).item() == pytest.approx(1.25)
    with pytest.raises(NonFiniteLossError):
        reduce_losses(_t(1.0, math.inf))
    with pytest.raises(ValueError, match="normalizer"):
        reduce_losses(losses, normalizer=0.0)


def test_sequence_logprobs_are_shift_aligned_and_masked() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 5)
    input_ids = torch.tensor([[3, 1, 2, 0]])  # last position is padding
    response_mask = torch.tensor([[0, 1, 1, 0]])
    sums, counts = sequence_logprobs(logits, input_ids, response_mask)
    log_probs = torch.log_softmax(logits[0], dim=-1)
    # token 1 is predicted at position 0, token 2 at position 1
    expected = log_probs[0, 1] + log_probs[1, 2]
    assert sums.item() == pytest.approx(expected.item(), abs=1e-6)
    assert counts.item() == 2.0
    assert mean_logprobs(sums, counts).item() == pytest.approx(
        expected.item() / 2, abs=1e-6
    )


def test_pairwise_metrics() -> None:
    assert pairwise_accuracy([3.0, 2.0], [1.0, 2.0]) == 0.5
    # cross pairs: 3>1, 3>2, 2>1, 2=2 (half)  →  3.5 / 4
    assert pairwise_auc([3.0, 2.0], [1.0, 2.0]) == pytest.approx(0.875)
    assert pairwise_auc([1.0], [2.0]) == 0.0
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    chosen, rejected = [0.3, 0.9, 0.1, 0.5], [0.2, 0.5, 0.7]
    assert pairwise_auc(chosen, rejected) == pytest.approx(
        sklearn_metrics.roc_auc_score([1] * 4 + [0] * 3, chosen + rejected)
    )


@pytest.mark.parametrize("beta", [0, -0.1, math.nan, math.inf, True, "0.1"])
def test_beta_must_be_finite_and_positive(beta: object) -> None:
    with pytest.raises(ValueError):
        validate_beta(beta)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dpo_loss(*DPO_ARGS, beta=beta)  # type: ignore[arg-type]


@pytest.mark.parametrize("smoothing", [-0.1, 0.5, math.nan])
def test_label_smoothing_range(smoothing: float) -> None:
    with pytest.raises(ValueError):
        validate_label_smoothing(smoothing)


def test_inputs_are_validated() -> None:
    with pytest.raises(NonFiniteLossError):
        dpo_loss(_t(math.nan), _t(-1.0), _t(-1.0), _t(-1.0), beta=0.1)
    with pytest.raises(ValueError, match="shapes differ"):
        dpo_loss(_t(-1.0, -2.0), _t(-1.0), _t(-1.0), _t(-1.0), beta=0.1)
    with pytest.raises(ValueError, match="1-D"):
        dpo_loss(torch.zeros(1, 1), _t(-1.0), _t(-1.0), _t(-1.0), beta=0.1)
    with pytest.raises(ValueError, match="positive"):
        simpo_loss(_t(-1.0), _t(-1.0), _t(0.0), _t(1.0), beta=1.0, gamma=0.5)
    with pytest.raises(ValueError, match="gamma"):
        simpo_loss(_t(-1.0), _t(-1.0), _t(1.0), _t(1.0), beta=1.0, gamma=-1.0)
    with pytest.raises(ValueError, match="margin"):
        bradley_terry_loss(_t(1.0), _t(0.0), margin=_t(1.0, 2.0))
