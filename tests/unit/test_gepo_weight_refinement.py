"""Compare the GEPO loss and its derivatives with independent probability tables."""

import itertools
import math
from fractions import Fraction

import pytest
import torch

from stateset_agents.training import objectives
from stateset_agents.training.gepo_trainer import GEPOTrainer


@pytest.mark.parametrize(
    ("dtype", "log_prob"),
    [(torch.float16, -40000.0), (torch.float32, -3e38), (torch.float64, -1e308)],
)
def test_identical_singleton_weights_have_unit_value_and_slope(dtype, log_prob):
    current = torch.tensor([log_prob], dtype=dtype, requires_grad=True)
    old = current.detach().clone().requires_grad_()
    weight = GEPOTrainer.compute_gepo_coefficient_static(current, old)
    assert weight.item() == 1
    weight.backward()
    assert current.grad.item() == 1
    assert old.grad is None


def test_large_shared_prefix_preserves_values_and_gradients():
    current = torch.tensor([[-1e8, -1], [-1e8, -2]], requires_grad=True)
    old = torch.tensor([-1e8 - 1, -1e8 - 2], dtype=torch.float64, requires_grad=True)
    result = objectives.policy_loss(
        logp_cur=current,
        mask=torch.ones_like(current),
        advantages=[1, -1],
        group_ids=[1, 1],
        logp_old=old,
        objective=objectives.OBJECTIVES["gepo"].with_(clip="none"),
    )
    q = [math.exp(-1), math.exp(-2)]
    ratios = [v * sum(q) / sum(x * x for x in q) for v in q]
    assert result.ratio.flatten().tolist() == pytest.approx(ratios)
    assert result.loss.item() == pytest.approx((ratios[1] - ratios[0]) / 2)
    result.loss.backward()
    expected = torch.tensor([-ratios[0] / 2, ratios[1] / 2])[:, None].expand_as(current)
    torch.testing.assert_close(current.grad, expected)
    assert old.grad is None


def test_dyadic_sum_retains_residual_and_padding_has_zero_derivative():
    current = torch.tensor(
        [[-1e308, -1.0, math.nan], [-5e307, -5e307, -2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    result = objectives.policy_loss(
        logp_cur=current,
        mask=mask,
        advantages=[1, 1],
        group_ids=[0, 0],
        logp_old=torch.full((2,), -1e308, dtype=torch.float64),
        objective=objectives.OBJECTIVES["gepo"].with_(clip="none"),
    )
    weights = torch.tensor([math.exp(-1), math.exp(-2)], dtype=torch.float64)
    torch.testing.assert_close(result.ratio[:, 0], weights)
    result.loss.backward()
    torch.testing.assert_close(current.grad, -weights[:, None] * mask / 2)


def test_log_guard_flattens_only_saturated_scores():
    current = torch.tensor([-1000.0, -1.0, -700.0], requires_grad=True)
    old = torch.full_like(current, -700.0)
    weights = GEPOTrainer.compute_gepo_coefficient_static(current, old)
    expected = torch.tensor([math.exp(-30), math.exp(30), 1], dtype=weights.dtype)
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    weights.sum().backward()
    assert current.grad.tolist() == [0, 0, 1]


# The finite TLA+ domain has two actions, at most two generated tokens, and
# three rows. These integer numerators have the common denominator nine.
PATH_MASSES = {(): 9, (1,): 3, (2,): 6, (1, 1): 1, (1, 2): 2, (2, 1): 2, (2, 2): 4}


@pytest.mark.parametrize("swapped", [False, True])
def test_bounded_path_weights_losses_and_token_gradients(swapped):
    signs = (1, -1, -1)
    objective = objectives.OBJECTIVES["gepo"].with_(clip_low=0.5, clip_high=0.5)
    for paths in itertools.product(PATH_MASSES, repeat=3):
        q = [PATH_MASSES[path] for path in paths]
        current_paths = [
            tuple(3 - a for a in path) if swapped else path for path in paths
        ]
        rows = [
            [math.log(a / 3) for a in path] + [math.nan] * (2 - len(path))
            for path in current_paths
        ]
        current = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
        mask = current.isfinite()
        result = objectives.policy_loss(
            logp_cur=current,
            mask=mask,
            group_ids=[7, 2, 7],
            advantages=signs,
            logp_old=torch.tensor([math.log(x / 9) for x in q], dtype=torch.float64),
            objective=objective,
        )
        loss = Fraction()
        expected_grad = torch.zeros_like(current)
        smooth = torch.ones_like(mask)
        for i, path in enumerate(current_paths):
            peers = [q[1]] if i == 1 else [q[0], q[2]]
            r = Fraction(PATH_MASSES[path] * sum(peers), sum(x * x for x in peers))
            assert result.ratio[i, 0].item() == pytest.approx(float(r), rel=1e-13)
            if path:
                clipped = min(max(r, Fraction(1, 2)), Fraction(3, 2))
                loss -= min(signs[i] * r, signs[i] * clipped) / 3
                boundary = Fraction(3, 2) if signs[i] > 0 else Fraction(1, 2)
                if r == boundary:
                    # Rounded logarithms may land on either side of a kink.
                    smooth[i, : len(path)] = False
                elif signs[i] * (r - boundary) < 0:
                    expected_grad[i, : len(path)] = -float(signs[i] * r / 3)
        assert result.loss.item() == pytest.approx(float(loss), abs=1e-14)
        result.loss.backward()
        torch.testing.assert_close(
            current.grad[smooth], expected_grad[smooth], rtol=1e-13, atol=1e-14
        )


def test_exact_clip_tie_and_empty_batch_backward():
    for n in (0, 2):
        current = torch.zeros(n, 1, requires_grad=True)
        result = objectives.policy_loss(
            logp_cur=current,
            mask=torch.ones_like(current),
            group_ids=list(range(n)),
            advantages=[1, -1][:n],
            logp_old=torch.zeros(n),
            objective=objectives.OBJECTIVES["gepo"].with_(clip_low=0, clip_high=0),
        )
        assert result.loss.item() == 0
        result.loss.backward()
        assert current.grad.flatten().tolist() == [-0.5, 0.5][:n]
