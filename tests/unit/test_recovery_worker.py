"""Tests for the executable checkpoint-recovery worker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "recovery_worker.py"
SPEC = importlib.util.spec_from_file_location("recovery_worker", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def test_atomic_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = worker.RecoveryPolicy(8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    worker._save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=3,
        applied_steps=[1, 2, 3],
    )

    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["step"] == 3
    assert state["applied_steps"] == [1, 2, 3]
    assert not path.with_suffix(".tmp").exists()


def test_batches_are_reproducible_by_step_and_seed() -> None:
    first = worker._batch(2, 42, 8, 6, torch.device("cpu"))
    second = worker._batch(2, 42, 8, 6, torch.device("cpu"))
    changed = worker._batch(3, 42, 8, 6, torch.device("cpu"))
    assert all(
        torch.equal(left, right) for left, right in zip(first, second, strict=True)
    )
    assert not torch.equal(first[0], changed[0])
