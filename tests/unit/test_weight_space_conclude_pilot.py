"""The pilot cannot promote a repair that loses to the stronger baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/weight_space"))
from conclude_pilot_job import promotion_gate  # noqa: E402


def scores() -> list[dict]:
    return [
        {
            "name": name,
            "select_think_strict": think,
            "select_nothink_strict": 0.77,
            "select_nothink_stray": 0.0,
        }
        for name, think in [
            ("base", 0.25),
            ("wait", 0.4),
            ("candidate", 0.8),
            ("control", 0.3),
        ]
    ]


def test_promotion_requires_stronger_baseline_and_locality() -> None:
    rows = scores()
    assert promotion_gate(rows)["eligible_for_final_test"]
    rows[2]["select_think_strict"] = 0.76
    assert not promotion_gate(rows)["eligible_for_final_test"]
    rows[2]["select_think_strict"] = 0.8
    rows[2]["select_nothink_strict"] = 0.75
    assert not promotion_gate(rows)["eligible_for_final_test"]
    rows[2]["select_nothink_strict"] = 0.77
    rows[2]["select_nothink_stray"] = 0.03
    assert not promotion_gate(rows)["eligible_for_final_test"]


def test_tied_controls_and_missing_conditions_do_not_promote() -> None:
    rows = scores()
    rows[3]["select_think_strict"] = 0.8
    assert not promotion_gate(rows)["eligible_for_final_test"]
    with pytest.raises(ValueError, match="exactly one"):
        promotion_gate(rows[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        promotion_gate(rows + [rows[0]])
