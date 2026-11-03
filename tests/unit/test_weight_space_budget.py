"""Token-budget scoring for weight-edit evaluations.

A run sampled with a fixed per-request seed and capped at T tokens contains,
as prefixes, the runs capped at any B <= T: a reply that stopped on its own
within B tokens is unchanged, and every other reply would have been cut off.
"""

from __future__ import annotations

import pytest

from stateset_agents.experimental.weight_space.protocol import (
    at_budget,
    budget_curve,
    clean_finish,
)


def _row(i: int, tokens: int, *, finish: str = "stop", text: str = "r</think>a"):
    return {"id": i, "correct": True, "finish": finish, "tokens": tokens, "text": text}


def test_clean_finish_without_thinking_requires_no_end_of_thinking() -> None:
    assert clean_finish({"finish": "stop", "text": "answer"}, thinking=False)
    assert not clean_finish({"finish": "stop", "text": "a</think>b"}, thinking=False)
    assert not clean_finish({"finish": "length", "text": "answer"}, thinking=False)


def test_clean_finish_reads_the_compact_end_of_thinking_count() -> None:
    assert clean_finish({"finish": "stop", "think_close": 1})
    assert not clean_finish({"finish": "stop", "think_close": 2})
    assert clean_finish({"finish": "stop", "think_close": 0}, thinking=False)


def test_at_budget_drops_replies_that_needed_more_tokens() -> None:
    rows = [_row(0, 600), _row(1, 1500), _row(2, 4096, finish="length")]
    scored = at_budget(rows, 1024, cap=4096)
    assert [r["correct"] for r in scored] == [True, False, False]
    assert [r["finish"] for r in scored] == ["stop", "length", "length"]
    assert rows[1]["correct"] is True  # inputs untouched


def test_at_budget_applies_strict_scoring() -> None:
    inline = _row(0, 300, text="r</think>a</think>b")
    assert at_budget([inline], 1024, cap=4096)[0]["correct"] is False


def test_at_budget_refuses_a_budget_beyond_the_run_cap() -> None:
    with pytest.raises(ValueError, match="cap"):
        at_budget([_row(0, 600)], 8192, cap=2048)


def test_budget_curve_reports_strict_accuracy_per_budget() -> None:
    rows = [_row(0, 600), _row(1, 1500), _row(2, 4096, finish="length")]
    rows.append({**_row(3, 200), "correct": False})
    curve = budget_curve(rows, (512, 1024, 2048, 4096), cap=4096)
    assert curve == {512: 0.0, 1024: 0.25, 2048: 0.5, 4096: 0.5}
