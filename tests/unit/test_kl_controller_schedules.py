"""KL controllers must follow their documented schedules."""

from __future__ import annotations

import math

import pytest

from stateset_agents.training.kl_controllers import (
    AdaptiveKLController,
    CosineKLScheduler,
    HybridKLController,
)


def test_hybrid_schedule_is_cosine_of_initial_coefficient():
    ctl = HybridKLController(
        init_kl_coef=0.1, target_kl=0.01, min_kl_coef=0.001, total_steps=100
    )
    for step in range(51):
        coef = ctl.update(0.01, step)  # KL exactly on target
    expected = 0.001 + 0.5 * (0.1 - 0.001) * (1 + math.cos(math.pi * 0.5))
    assert coef == pytest.approx(expected)


def test_hybrid_tolerates_negative_kl_estimates():
    ctl = HybridKLController(init_kl_coef=0.1, min_kl_coef=0.001)
    assert ctl.update(-0.02, 1) == pytest.approx(0.001)


def test_hybrid_holds_floor_after_total_steps():
    ctl = HybridKLController(init_kl_coef=0.1, min_kl_coef=0.001, total_steps=10)
    assert ctl.update(0.01, 20) == pytest.approx(0.001)


def test_cosine_scheduler_never_drops_below_floor():
    ctl = CosineKLScheduler(
        init_kl_coef=0.1, min_kl_coef=0.01, total_steps=10, num_cycles=0.5
    )
    assert ctl.update(0.0, 30) == pytest.approx(0.01)


def test_adaptive_update_is_bounded_and_step_independent():
    early = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, horizon=100)
    late = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, horizon=100)
    late.update(0.01, 999)  # on target: no change, establishes last step
    a = early.update(0.03, 1)
    b = late.update(0.03, 1000)
    # error clipped to +0.2, one step since the last update
    assert a == pytest.approx(0.1 * (1 + 0.2 / 100))
    assert b == pytest.approx(a)
    # zero KL never collapses the coefficient in one update
    assert late.update(0.0, 1001) == pytest.approx(b * (1 - 0.2 / 100))


def test_cosine_scheduler_reaches_floor_at_total_steps():
    ctl = CosineKLScheduler(init_kl_coef=0.1, min_kl_coef=0.01, total_steps=10)
    assert ctl.update(0.0, 0) == pytest.approx(0.1)
    assert ctl.update(0.0, 5) == pytest.approx(0.055)
    assert ctl.update(0.0, 10) == pytest.approx(0.01)
