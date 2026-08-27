"""Unit tests for the executable distributed-scaling workload."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "distributed_scaling_workload.py"
)
SPEC = importlib.util.spec_from_file_location(
    "distributed_scaling_workload", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
workload = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workload
SPEC.loader.exec_module(workload)


def test_workload_digest_is_order_independent_and_sensitive() -> None:
    first = workload.canonical_digest({"batch": 32, "depth": 4})
    assert first == workload.canonical_digest({"depth": 4, "batch": 32})
    assert first != workload.canonical_digest({"batch": 64, "depth": 4})


def test_indexed_batch_is_topology_invariant() -> None:
    common = {
        "step": 3,
        "seed": 42,
        "global_batch_size": 8,
        "feature_dim": 6,
        "num_actions": 4,
        "device": torch.device("cpu"),
    }
    whole = workload.indexed_batch(rank=0, local_batch_size=8, **common)
    left = workload.indexed_batch(rank=0, local_batch_size=4, **common)
    right = workload.indexed_batch(rank=1, local_batch_size=4, **common)
    for whole_tensor, left_tensor, right_tensor in zip(whole, left, right, strict=True):
        assert torch.equal(whole_tensor, torch.cat([left_tensor, right_tensor]))


def test_policy_loss_has_finite_nonzero_gradients() -> None:
    torch.manual_seed(7)
    model = workload.ResidualPolicy(8, 16, 2, 8)
    features, targets, exploration = workload.indexed_batch(
        step=0,
        seed=7,
        rank=0,
        local_batch_size=8,
        global_batch_size=8,
        feature_dim=8,
        num_actions=8,
        device=torch.device("cpu"),
    )
    loss = workload.policy_loss(
        model(features),
        targets,
        exploration,
        group_size=4,
        entropy_coef=1e-3,
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )
    assert any(
        gradient.abs().sum() > 0 for gradient in gradients if gradient is not None
    )
