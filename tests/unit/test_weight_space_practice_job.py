"""Argument handling for practice_job.py (named specs judged on practice problems)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "weight_space")
)

from practice_job import parse_specs  # noqa: E402


def test_parse_specs_keeps_order_with_the_start_first() -> None:
    got = parse_specs(["v3t=/w/a.json", "guard=/w/b.json", "guard_low=/w/c=d.json"])
    assert got == [
        ("v3t", Path("/w/a.json")),
        ("guard", Path("/w/b.json")),
        ("guard_low", Path("/w/c=d.json")),
    ]


@pytest.mark.parametrize(
    "items",
    [
        ["v3t=/w/a.json"],  # nothing to compare
        ["v3t=/w/a.json", "/w/b.json"],  # no name
        ["v3t=/w/a.json", "guard="],  # no path
        ["v3t=/w/a.json", "v3t=/w/b.json"],  # duplicate name
    ],
)
def test_parse_specs_refuses_bad_lists(items: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_specs(items)
