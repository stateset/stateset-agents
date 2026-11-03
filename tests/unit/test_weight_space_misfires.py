"""Mining on-policy contexts from an edited model's own replies."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "weight_space")
)

from misfires import LABELS, mine  # noqa: E402

NL, END = 10, 99  # a newline token and the end-of-thinking token
NEWLINE = {NL}


def test_line_break_before_an_extra_close_is_a_negative() -> None:
    g = [1, 2, NL, END, 3, NL, END, 4]
    got = mine(g, NEWLINE, END, thinking=True, answer_token=2, correct=False)
    assert ("extra_close", 5) in got
    assert LABELS["extra_close"] < 0


def test_close_before_the_answer_in_a_run_on_reply_is_premature() -> None:
    g = [1, NL, END, 5, 6]  # closed, then kept going to the cap
    for k in (None, 4):
        got = mine(
            g,
            NEWLINE,
            END,
            thinking=True,
            answer_token=k,
            correct=False,
            finish="length",
        )
        assert got == [("premature_close", 1)]


def test_clean_but_wrong_reply_is_not_premature() -> None:
    # The model concluded once and stopped: it was done, just wrong. Calling
    # that close a misfire would teach the neuron to stop concluding at all.
    g = [1, NL, END, 5, 6]
    assert mine(g, NEWLINE, END, thinking=True, answer_token=None, correct=False) == []


def test_clean_correct_close_after_the_answer_is_a_positive() -> None:
    g = [1, 2, NL, END, 7]
    got = mine(g, NEWLINE, END, thinking=True, answer_token=2, correct=True)
    assert got == [("good_close", 2)]
    assert LABELS["good_close"] > 0


def test_wrong_reply_after_the_answer_is_not_labelled() -> None:
    g = [1, 2, NL, END, 7]
    assert mine(g, NEWLINE, END, thinking=True, answer_token=2, correct=False) == []


def test_answer_written_but_never_closed_is_a_missed_positive() -> None:
    g = [1, 2, 3, NL, 4, NL]
    got = mine(
        g, NEWLINE, END, thinking=True, answer_token=3, correct=False, finish="length"
    )
    assert got == [("missed_answer", 3)]


def test_thinking_off_close_is_a_stray_negative() -> None:
    g = [1, NL, END, 2, NL, END]
    got = mine(g, NEWLINE, END, thinking=False, answer_token=None, correct=False)
    assert got == [("stray_close", 1), ("stray_close", 4)]


def test_choose_round_takes_the_best_practice_score_within_the_stray_limit() -> None:
    from misfires import choose_round

    rounds = [
        {"select_think_strict": 0.40, "select_nothink_stray": 0.10},
        {"select_think_strict": 0.48, "select_nothink_stray": 0.30},  # leaks
        {"select_think_strict": 0.45, "select_nothink_stray": 0.11},
        {"select_think_strict": 0.45, "select_nothink_stray": 0.05},  # tie: earlier
    ]
    assert choose_round(rounds, max_stray_rise=0.02) == 2


def test_choose_round_keeps_the_start_when_nothing_beats_it() -> None:
    from misfires import choose_round

    rounds = [
        {"select_think_strict": 0.40, "select_nothink_stray": 0.10},
        {"select_think_strict": 0.39, "select_nothink_stray": 0.05},
    ]
    assert choose_round(rounds, max_stray_rise=0.02) == 0


def test_close_not_after_a_line_break_is_skipped() -> None:
    g = [1, 2, END, 3]
    assert mine(g, NEWLINE, END, thinking=True, answer_token=None, correct=False) == []
