"""Where a reasoning trace first writes the gold answer as a result (MATH-style)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "weight_space")
)

from answer_positions import answer_end, gold_in_question  # noqa: E402


def test_finds_an_integer_written_as_a_result() -> None:
    text = "Compute 3 + 4.\nSo x = 7.\nCheck: 7 works."
    end = answer_end(text, "7")
    assert end is not None and text[:end].endswith("x = 7")


def test_does_not_match_inside_a_longer_number() -> None:
    assert answer_end("so y = 17 and z = 7.5", "7") is None
    assert answer_end("so y = -7", "7") is None


def test_matches_fraction_variants() -> None:
    text = "Thus the probability is \\dfrac{3}{4}.\n"
    assert answer_end(text, "\\frac{3}{4}") is not None
    assert answer_end("so p = 3/4\n", "\\frac{3}{4}") is not None
    assert answer_end("so p = \\frac{3}{4}\n", "\\dfrac{3}{4}") is not None


def test_matches_boxed_and_dollar_wrapped_answers() -> None:
    assert answer_end("final: \\boxed{2\\sqrt{5}}", "2\\sqrt{5}") is not None
    assert answer_end("The answer is $12$.", "12") is not None


def test_only_searches_the_thinking_part() -> None:
    text = "working... x = 3\n</think>\nThe answer is 9."
    assert answer_end(text, "9") is None
    assert answer_end(text, "3") is not None


def test_returns_the_earliest_occurrence() -> None:
    text = "a = 5\nlater, so b = 5\n"
    assert answer_end(text, "5") == len("a = 5")


def test_gold_in_question_respects_number_boundaries() -> None:
    assert gold_in_question("What is 12 times 3?", "12")
    assert not gold_in_question("What is 112 times 3?", "12")
    assert gold_in_question("Simplify \\frac{3}{4} + 0.", "\\frac{3}{4}")
