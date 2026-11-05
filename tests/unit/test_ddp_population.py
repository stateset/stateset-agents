"""Empty contributors are neutral, and empty pending banks remain well scoped."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from stateset_agents.training.ddp_population import token_forward_context, token_policy
from stateset_agents.training.metric_reduction import (
    LossMetricReduction,
    aggregate_loss_metrics,
)
from tests.unit.test_metric_reduction import _rows


def _empty(measure="trajectory"):
    return LossMetricReduction(
        measure=measure,
        **dict.fromkeys(
            (
                "policy_count",
                "token_count",
                "trajectory_count",
                "row_count",
                "advantage_count",
                "nonzero_advantages",
            ),
            0,
        ),
        **dict.fromkeys(
            (
                "policy_sum",
                "entropy_loss_sum",
                "entropy_sum",
                "kl_sum",
                "ratio_sum",
                "clip_sum",
                "advantage_mean",
                "advantage_m2",
            ),
            0.0,
        ),
    )


@pytest.mark.parametrize("measure", ["trajectory", "token"])
def test_empty_metrics_are_neutral_regardless_of_order(measure):
    empty = _empty(measure)
    assert empty.valid()
    rows = _rows(measure)
    assert aggregate_loss_metrics([empty, *rows, empty]) == aggregate_loss_metrics(rows)
    all_empty = aggregate_loss_metrics([empty, empty])
    assert all_empty["num_trajectories"] == all_empty["policy_denominator"] == 0
    assert all_empty["advantage_std"] == all_empty["total_loss"] == 0
    assert all_empty["ratio_mean"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_sum", 1),
        ("advantage_mean", 1),
        ("row_count", 1),
        ("nonzero_advantages", 1),
    ],
)
def test_zero_count_receipts_cannot_hide_nonzero_statistics(field, value):
    assert not replace(_empty(), **{field: value}).valid()


def test_token_forward_context_preserves_wrapper_and_prior_attributes():
    actor = object()
    wrapper = SimpleNamespace(module=actor)
    assert token_policy(wrapper) is wrapper
    with pytest.raises(RuntimeError):
        with token_forward_context(wrapper, "sampler"):
            assert token_policy(wrapper) is actor
            assert wrapper._stateset_rollout_source == "sampler"
            with token_forward_context(wrapper, "snapshot"):
                assert wrapper._stateset_rollout_source == "snapshot"
            raise RuntimeError("done")
    assert vars(wrapper) == {"module": actor}


@pytest.mark.parametrize("entropy", [False, True])
def test_zero_local_pending_counts_require_native_validation(entropy):
    torch = pytest.importorskip("torch")
    from stateset_agents.training.checkpoint_state import (
        _prepare_normalization,
        _validate_gradients,
    )

    parameters = {"w": torch.nn.Parameter(torch.ones(2))}
    state = {
        "measure": "trajectory",
        "policy_count": 0,
        "policy_normalizer": 4,
        "entropy_count": 0,
        "entropy_normalizer": 8 if entropy else 0,
        "entropy_gradients": {"w": None} if entropy else {},
    }
    original = copy.deepcopy(state)
    with pytest.raises(ValueError):
        _prepare_normalization(state, parameters, torch, 1, 2)
    assert (
        _prepare_normalization(state, parameters, torch, 1, 2, allow_empty=True)
        == original
    )
    with pytest.raises(ValueError):
        _validate_gradients({"w": None}, parameters, torch)
    _validate_gradients({"w": None}, parameters, torch, allow_empty=True)
    state["entropy_count"] = 1
    with pytest.raises(ValueError):
        _prepare_normalization(state, parameters, torch, 1, 2, allow_empty=True)
