"""Population summaries retain their sample/group measures and exact medians."""

import copy
import math
import random
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from stateset_agents.training.rollout_metrics import (
    _aggregate,
    _capture,
    _float_key,
    _key_float,
    _median,
    _pack,
    _unpack,
)


def _groups(*groups):
    return [
        SimpleNamespace(trajectories=[SimpleNamespace(total_reward=v) for v in g])
        for g in groups
    ]


def test_unequal_populations_weight_samples_but_diversity_weights_groups():
    groups = _groups([-2, 5], [-10, 1, 9], [])
    before = copy.deepcopy(groups)
    rows = [_capture(groups[:1], details=True), _capture(groups[1:], details=True)]
    actual = _aggregate([_unpack(*_pack(r)) for r in rows])
    values = [-2, 5, -10, 1, 9]
    assert actual["reward_mean"] == pytest.approx(np.mean(values))
    assert actual["reward_std"] == pytest.approx(np.std(values))
    assert actual["num_groups"] == 3 and actual["total_trajectories"] == 5
    assert actual["avg_group_size"] == 2.5
    diversities = [np.std([-2, 5]), np.std([-10, 1, 9])]
    assert actual["reward_diversity_mean"] == pytest.approx(np.mean(diversities))
    assert actual["reward_diversity_std"] == pytest.approx(np.std(diversities))
    assert groups == before


def test_extreme_finite_rewards_do_not_overflow_population_moments():
    rows = [
        _capture(_groups([-1e308, 1e308]), details=False),
        _capture([], details=False),
    ]
    actual = _aggregate([_unpack(*_pack(r)) for r in rows])
    assert actual["reward_mean"] == 0 and actual["reward_std"] == 1e308
    assert all(math.isfinite(v) for v in actual.values())


@pytest.mark.parametrize(
    "reward", [float("nan"), float("inf"), -float("inf"), None, "bad"]
)
def test_invalid_rewards_cannot_become_zero_observations(reward):
    with pytest.raises((ValueError, TypeError)):
        _capture(_groups([reward]), details=True)


def test_empty_population_is_neutral_and_has_no_diversity_observations():
    empty = _capture(_groups([], []), details=True)
    result = _aggregate([empty])
    assert result["num_groups"] == 2
    assert result["total_trajectories"] == result["reward_mean"] == 0
    assert "reward_diversity_mean" not in result
    positive = _capture(_groups([7]), details=True)
    joined = _aggregate([empty, positive])
    assert joined["reward_mean"] == 7 and joined["reward_std"] == 0
    assert joined["avg_group_size"] == 1


@pytest.mark.parametrize("fault", ["count", "groups", "moments", "finite", "range"])
def test_malformed_population_receipts_are_rejected(fault):
    counts, values = _pack(_capture(_groups([1, 2]), details=True))
    if fault == "count":
        counts[2] = 2**53
    elif fault == "groups":
        counts[1] = 2
    elif fault == "moments":
        counts[2] = 0
    elif fault == "finite":
        values[0] = float("nan")
    else:
        values[1] = -1
    with pytest.raises(ValueError):
        _unpack(counts, values)


def test_float_keys_order_finite_values_and_canonicalize_signed_zero():
    values = [-1e308, -1, -5e-324, -0.0, 0.0, 5e-324, 1, 1e308]
    keys = [_float_key(v) for v in values]
    assert keys == sorted(keys)
    assert keys[3] == keys[4]
    assert [_key_float(k) for k in keys] == values


def test_exact_median_selection_handles_float64_extremes_and_ties():
    torch = pytest.importorskip("torch")
    randomizer = random.Random(19)
    cases = [
        [],
        [1],
        [-0.0, 0.0],
        [-1e308, 1e308],
        [1e308, 1e308],
        [5e-324, 5e-324],
        [-5e-324, 0],
        [-10, -10, 2, 2, 100],
    ]
    cases += [
        [randomizer.uniform(-1e300, 1e300) for _ in range(n)] for n in range(2, 32)
    ]
    for values in cases:
        reductions = []
        group = SimpleNamespace(
            torch=torch,
            device="cpu",
            group=None,
            dist=SimpleNamespace(
                ReduceOp=SimpleNamespace(SUM=0),
                all_reduce=lambda counts, sink=reductions, **kwargs: sink.append(1),
            ),
        )
        row = _capture(_groups(values), details=True)
        actual = _median(group, row.keys, _aggregate([row]))
        ordered = sorted(values)
        with localcontext() as context:
            context.prec = 1100
            expected = (
                float(
                    (
                        Decimal.from_float(float(ordered[(len(values) - 1) // 2]))
                        + Decimal.from_float(float(ordered[len(values) // 2]))
                    )
                    / 2
                )
                if values
                else 0.0
            )
        assert actual == expected
        assert len(reductions) <= 64
