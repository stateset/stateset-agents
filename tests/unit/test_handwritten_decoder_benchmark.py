"""Noise distributions and validation selection for the handwritten NN study."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "weight_space")
)

from handwritten_decoder_job import (  # noqa: E402
    boundary_suite,
    sample_batch,
    validation_key,
)

from stateset_agents.experimental.weight_space.handwritten_decoder import (  # noqa: E402
    MAX_ERRORS,
    make_codebook,
)


@pytest.mark.parametrize("errors", [0, 1, 95])
def test_fixed_noise_has_exactly_the_requested_distinct_flips(errors: int) -> None:
    code = make_codebook()
    received, ids = sample_batch(
        code, 512, torch.Generator().manual_seed(7), errors, balanced=True
    )
    assert torch.equal((received != code[ids]).sum(1), torch.full((512,), errors))
    assert torch.equal(ids.bincount(), torch.full((256,), 2))


def test_online_training_draws_cover_the_full_declared_noise_range() -> None:
    code = make_codebook()
    received, ids = sample_batch(code, 4096, torch.Generator().manual_seed(9))
    counts = (received != code[ids]).sum(1)
    assert set(counts.tolist()) == set(range(MAX_ERRORS + 1))


def test_independent_seeds_do_not_reuse_the_same_test_corruptions() -> None:
    code = make_codebook()
    a, ai = sample_batch(code, 256, torch.Generator().manual_seed(70000), 95, True)
    b, bi = sample_batch(code, 256, torch.Generator().manual_seed(80000), 95, True)
    assert torch.equal(ai, bi)
    assert not torch.any(torch.all(a == b, dim=1))


def test_structured_boundary_examples_are_still_within_the_guarantee() -> None:
    code = make_codebook()[:8]
    received, ids = boundary_suite(code)
    assert len(ids) == 8 * 7
    assert torch.equal(
        (received != code[ids]).sum(1), torch.full((len(ids),), MAX_ERRORS)
    )


def test_selection_prioritizes_complete_messages_and_uses_bce_to_break_ties() -> None:
    def scores(correct: int, bce: float) -> dict:
        return {"slice": {"n": 100, "correct": correct, "bce": bce}}

    assert validation_key(scores(96, 0.05)) > validation_key(scores(95, 0.01))
    assert validation_key(scores(96, 0.02)) > validation_key(scores(96, 0.05))
