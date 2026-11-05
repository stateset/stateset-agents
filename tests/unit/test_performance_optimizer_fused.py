"""Regression tests for ``ModelOptimizer.optimize_optimizer`` fused rebuilds."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.performance_optimizer import (  # noqa: E402
    ComputeConfig,
    ModelOptimizer,
)


def _optimizer_helper(fused: bool = True) -> ModelOptimizer:
    return ModelOptimizer(
        ComputeConfig(use_mixed_precision=False, fused_optimizer=fused)
    )


def _supports_cpu_fused_adamw() -> bool:
    param = torch.nn.Parameter(torch.zeros(1))
    try:
        torch.optim.AdamW([param], fused=True)
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


@pytest.mark.skipif(
    not _supports_cpu_fused_adamw(), reason="this torch build lacks CPU fused AdamW"
)
def test_fresh_adamw_is_rebuilt_fused_with_every_param_group() -> None:
    # PyTorch has no ``optimizer.fused`` attribute and ``defaults`` already
    # holds ``fused``, so the old hasattr/**defaults rebuild never ran.
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW(
        [{"params": [first], "lr": 0.1}, {"params": [second], "lr": 0.01}]
    )

    fused = _optimizer_helper().optimize_optimizer(optimizer)

    assert fused is not optimizer
    assert type(fused) is torch.optim.AdamW
    assert [group["lr"] for group in fused.param_groups] == [0.1, 0.01]
    assert all(group["fused"] for group in fused.param_groups)
    assert fused.param_groups[1]["params"][0] is second


def test_optimizer_is_kept_when_fusion_is_unsupported_disabled_or_stateful() -> None:
    param = torch.nn.Parameter(torch.zeros(2))

    rmsprop = torch.optim.RMSprop([param])  # no fused implementation
    assert _optimizer_helper().optimize_optimizer(rmsprop) is rmsprop

    adamw = torch.optim.AdamW([param])
    assert _optimizer_helper(fused=False).optimize_optimizer(adamw) is adamw

    param.grad = torch.ones(2)
    adamw.step()  # accumulated moments must not be discarded by a rebuild
    assert _optimizer_helper().optimize_optimizer(adamw) is adamw
