"""Native diagnostics use distinct policy/token measures and stable moments."""

import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest

from stateset_agents.training.metric_reduction import (
    LossMetricReduction,
    ReducedMetrics,
    aggregate_loss_metrics,
    carry_loss_metrics,
)


def _rows(measure="trajectory"):
    first = LossMetricReduction(
        measure=measure,
        policy_count=2 if measure == "trajectory" else 4,
        token_count=4,
        trajectory_count=2,
        row_count=3,
        advantage_count=2,
        nonzero_advantages=2,
        policy_sum=8.0,
        entropy_loss_sum=-0.8,
        entropy_sum=8.0,
        kl_sum=1.0,
        ratio_sum=2.0,
        clip_sum=1.0,
        advantage_mean=0.0,
        advantage_m2=2.0,
    )
    second = LossMetricReduction(
        measure=measure,
        policy_count=3 if measure == "trajectory" else 12,
        token_count=12,
        trajectory_count=3,
        row_count=4,
        advantage_count=3,
        nonzero_advantages=3,
        policy_sum=30.0,
        entropy_loss_sum=-4.8,
        entropy_sum=48.0,
        kl_sum=3.0,
        ratio_sum=18.0,
        clip_sum=3.0,
        advantage_mean=6.0,
        advantage_m2=8.0,
    )
    return [first, second]


@pytest.mark.parametrize("measure", ["trajectory", "token"])
def test_unequal_workers_match_explicit_global_sums(measure):
    result = aggregate_loss_metrics(_rows(measure))
    denominator = 5 if measure == "trajectory" else 16
    assert result["policy_objective_loss"] == pytest.approx(38 / denominator)
    assert result["total_loss"] == pytest.approx(38 / denominator - 5.6 / 16)
    assert result["policy_loss"] == result["total_loss"]
    assert result["entropy"] == 3.5
    assert result["kl_penalty"] == result["clip_fraction"] == 0.25
    assert result["ratio_mean"] == 1.25
    assert result["mean_advantage"] == pytest.approx(3.6)
    assert result["advantage_std"] == pytest.approx(math.sqrt(10.64))
    assert result["nonzero_advantage_fraction"] == 1.0
    assert result["num_tokens"] == 16
    assert result["num_trajectories"] == 5 and result["num_rows"] == 7
    assert result["policy_denominator"] == denominator
    assert result["metric_scope"] == "ddp_global_batch"


def test_moment_merge_preserves_small_variance_at_large_offset():
    rows = [replace(row, advantage_mean=row.advantage_mean + 1e12) for row in _rows()]
    result = aggregate_loss_metrics(rows)
    assert result["mean_advantage"] == 1e12 + 3.6
    assert result["advantage_std"] == pytest.approx(math.sqrt(10.64), abs=1e-12)


@pytest.mark.parametrize(
    "change",
    [
        {"policy_count": 0},
        {"token_count": False},
        {"advantage_m2": -1.0},
        {"policy_sum": float("nan")},
        {"nonzero_advantages": 3},
        {"advantage_count": 1},
        {"measure": "unknown"},
        {"policy_count": 2**70},
        {"policy_sum": 2**2000},
    ],
)
def test_invalid_statistics_rejected(change):
    row = replace(_rows()[0], **change)
    assert not row.valid()
    with pytest.raises(ValueError, match="Invalid"):
        aggregate_loss_metrics([row])


def test_measures_cannot_be_mixed():
    with pytest.raises(ValueError, match="measures differ"):
        aggregate_loss_metrics([_rows()[0], _rows("token")[1]])


def test_large_finite_sums_do_not_overflow_their_finite_mean():
    rows = [replace(row, policy_sum=1e308) for row in _rows()]
    assert aggregate_loss_metrics(rows)["policy_objective_loss"] == pytest.approx(4e307)


def test_private_statistics_are_detached_from_json_metric_payload():
    from types import SimpleNamespace

    ordinary = {"loss": 1.0}
    assert carry_loss_metrics(ordinary, {}) is ordinary
    metrics = carry_loss_metrics(ordinary, SimpleNamespace(metric_reduction=_rows()[0]))
    assert isinstance(metrics, ReducedMetrics)
    assert metrics.reduction == _rows()[0]
    assert json.loads(json.dumps(metrics)) == ordinary


def test_metric_helpers_import_without_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stateset_agents.training.metric_reduction; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
