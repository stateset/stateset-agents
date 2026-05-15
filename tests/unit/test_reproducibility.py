"""Unit tests for the central reproducibility controls."""

from __future__ import annotations

import random

import numpy as np

from stateset_agents.utils.reproducibility import (
    REPRODUCIBILITY_STATE,
    get_seed,
    set_all_seeds,
)


class TestSetAllSeeds:
    def test_python_random_deterministic(self) -> None:
        set_all_seeds(42)
        a = [random.random() for _ in range(10)]
        set_all_seeds(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_deterministic(self) -> None:
        set_all_seeds(123)
        a = np.random.rand(10)
        set_all_seeds(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_diverge(self) -> None:
        set_all_seeds(42)
        a = np.random.rand(10)
        set_all_seeds(1337)
        b = np.random.rand(10)
        # With overwhelming probability the sequences differ.
        assert not np.array_equal(a, b)

    def test_state_records_seed(self) -> None:
        state = set_all_seeds(2026)
        assert state.seed == 2026
        assert get_seed() == 2026
        assert "random" in state.components_seeded
        assert "numpy" in state.components_seeded

    def test_state_module_singleton(self) -> None:
        set_all_seeds(7)
        assert REPRODUCIBILITY_STATE.seed == 7

    def test_to_dict(self) -> None:
        state = set_all_seeds(0)
        d = state.to_dict()
        assert d["seed"] == 0
        assert isinstance(d["components_seeded"], list)

    def test_torch_seed_when_available(self) -> None:
        try:
            import torch
        except ImportError:
            return  # skip — torch optional
        state = set_all_seeds(42)
        assert "torch" in state.components_seeded
        a = torch.rand(5)
        set_all_seeds(42)
        b = torch.rand(5)
        assert torch.equal(a, b)
