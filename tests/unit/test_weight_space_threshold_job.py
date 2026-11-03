"""Which explicit-threshold candidates the pod job runs (threshold_job.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "weight_space")
)

from threshold_job import START, axes, combination  # noqa: E402

GRID = [(b, w) for b in (-1.0, 0.0, 1.0, 2.0) for w in (2.0, 3.0, 4.0)]


def row(bias: float, write_norm: float, strict: float, stray: float = 0.10) -> dict:
    return {
        "bias": bias,
        "write_norm": write_norm,
        "select_think_strict": strict,
        "select_nothink_stray": stray,
    }


def test_axes_vary_one_knob_from_the_start_and_exclude_it() -> None:
    bias_axis, write_axis = axes(GRID)
    assert START == (0.0, 3.0)
    assert bias_axis == [(-1.0, 3.0), (1.0, 3.0), (2.0, 3.0)]
    assert write_axis == [(0.0, 2.0), (0.0, 4.0)]


def test_combination_pairs_the_two_axis_winners() -> None:
    rows = [
        row(0.0, 3.0, 0.40),
        row(-1.0, 3.0, 0.38),
        row(1.0, 3.0, 0.45),
        row(0.0, 2.0, 0.39),
        row(0.0, 4.0, 0.43),
    ]
    assert combination(rows, max_stray_rise=0.02) == (1.0, 4.0)


def test_no_combination_when_an_axis_keeps_the_start() -> None:
    rows = [
        row(0.0, 3.0, 0.40),
        row(1.0, 3.0, 0.45),
        row(0.0, 2.0, 0.39),
        row(0.0, 4.0, 0.40),  # a tie keeps the start
    ]
    assert combination(rows, max_stray_rise=0.02) is None


def test_an_axis_winner_must_respect_the_stray_limit() -> None:
    rows = [
        row(0.0, 3.0, 0.40, stray=0.10),
        row(-1.0, 3.0, 0.50, stray=0.20),  # better but leaks: ineligible
        row(1.0, 3.0, 0.42, stray=0.05),
        row(0.0, 4.0, 0.44, stray=0.11),
    ]
    assert combination(rows, max_stray_rise=0.02) == (1.0, 4.0)
