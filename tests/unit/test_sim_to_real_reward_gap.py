"""Sim-to-real reward comparison must be deterministic and distribution-based."""

import pytest

from stateset_agents.evaluation.sim_to_real_metrics import _normalized_wasserstein


def test_identical_reward_distributions_have_zero_gap():
    assert _normalized_wasserstein([0.1, 0.5, 0.9], [0.9, 0.1, 0.5]) == 0.0


def test_shifted_distribution_gap_is_shift_over_range():
    gap = _normalized_wasserstein([0.0, 1.0], [1.0, 2.0])
    assert gap == pytest.approx(0.5)


def test_gap_is_deterministic():
    a, b = [0.2, 0.4, 0.7, 0.9], [0.1, 0.3]
    assert _normalized_wasserstein(a, b) == _normalized_wasserstein(a, b)
