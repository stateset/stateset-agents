"""Independent replay arithmetic and strict acceptance decisions on small logits."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from benchmarks.weight_space.verify_behavior_trial import (  # noqa: E402
    compare,
    decision,
)


def test_compare_masks_no_real_tokens_and_scores_only_observed_next_tokens():
    before = [torch.zeros(3, 2), torch.zeros(2, 2)]
    after = [torch.zeros(3, 2), torch.tensor([[0.0, 0.0], [2.0, 0.0]])]
    metrics = compare([[0, 1, 0], [1, 0]], before, after, -2, -1)
    # Last position has no observed next token but still contributes KL.
    expected_kl = float(
        (
            torch.tensor([0.5, 0.5])
            * (torch.log(torch.tensor([0.5, 0.5])) - after[-1][-1].log_softmax(0))
        ).sum()
    )
    assert metrics["locality_tokens"] == 5
    assert metrics["observed_tokens"] == 3
    assert metrics["max_token_kl"] == pytest.approx(expected_kl)
    assert metrics["mean_kl"] == pytest.approx(expected_kl / 5)
    assert metrics["nll_increase"] == 0
    assert metrics["target_gains"] == [1]


def test_independent_gate_uses_strict_caps_and_target_requirements():
    metrics = compare([[0, 1]], [torch.zeros(2, 2)], [torch.zeros(2, 2)], -1, -1)
    limits = {
        "max_mean_kl": 0,
        "max_token_kl": 0,
        "max_nll_increase": 0,
        "min_top1_agreement": 1,
        "min_target_gain": 0.01,
        "min_each_target_gain": 0.01,
    }
    assert decision(metrics, limits) == (
        False,
        ["mean_target_gain", "each_target_gain"],
    )
    limits["min_target_gain"] = limits["min_each_target_gain"] = 0
    assert decision(metrics, limits) == (True, [])
