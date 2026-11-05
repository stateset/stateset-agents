"""Label contexts in an edited model's own replies (on-policy contrast mining).

A conclude neuron writes toward ``</think>`` at line breaks, so the token
after the line break it fires on is where the close appears. Contexts curated
from the *base* model's traces did not predict where designs misfire in their
own generations (v4, v5). This module reads a reply generated *with* the edit
and returns ``(group, index)`` pairs, where ``index`` is the position of the
line-break token that ends the context (prefix = prompt + gen[: index + 1]):

thinking on
  extra_close      line break before every ``</think>`` after the first  (-1)
  premature_close  line break before the first ``</think>`` when it comes
                   before the gold answer was written in thinking and the
                   reply then closed again or ran to the cap             (-1)
  good_close       line break before the only ``</think>`` of a clean,
                   correct reply that closed after the answer            (+1)
  missed_answer    first line break after the answer was written, in a
                   reply that never closed                               (+1)
thinking off
  stray_close      line break before every ``</think>``                  (-1)

Torch-free; replies to *training* problems only.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["LABELS", "choose_round", "mine"]

LABELS = {
    "extra_close": -1.0,
    "premature_close": -1.0,
    "stray_close": -1.0,
    "good_close": 1.0,
    "missed_answer": 1.0,
}


def mine(
    gen_ids: Sequence[int],
    newline: set[int],
    think_end: int,
    *,
    thinking: bool,
    answer_token: int | None,
    correct: bool,
    finish: str = "stop",
) -> list[tuple[str, int]]:
    """``(group, line-break index)`` contexts in one reply.

    ``answer_token`` is the number of generated tokens after which the gold
    answer has been written as a result in thinking (``None`` if never).
    """
    g = list(gen_ids)
    closes = [j for j, t in enumerate(g) if t == think_end]

    def before(j: int) -> int | None:
        return j - 1 if j > 0 and g[j - 1] in newline else None

    out: list[tuple[str, int]] = []
    if not thinking:
        for j in closes:
            i = before(j)
            if i is not None:
                out.append(("stray_close", i))
        return out
    if not closes:
        if answer_token is not None and finish == "length":
            after = [j for j in range(answer_token, len(g)) if g[j] in newline]
            if after:
                out.append(("missed_answer", after[0]))
        return out
    first = closes[0]
    clean = len(closes) == 1 and finish == "stop"
    i = before(first)
    if i is not None:
        if answer_token is None or first < answer_token:
            # Only when the reply shows the close was early (it closed again
            # or ran to the cap). A clean wrong reply was done, just wrong.
            if not clean:
                out.append(("premature_close", i))
        elif clean and correct:
            out.append(("good_close", i))
    for j in closes[1:]:
        i = before(j)
        if i is not None:
            out.append(("extra_close", i))
    return out


def choose_round(rounds: Sequence[dict], *, max_stray_rise: float) -> int:
    """Index of the round to keep, judged on the SELECT practice set only.

    Highest strict thinking-on accuracy among rounds whose thinking-off stray
    ``</think>`` rate is at most round 0's plus ``max_stray_rise``; ties keep
    the earlier round, and round 0 (the starting spec) is always eligible.
    """
    limit = rounds[0]["select_nothink_stray"] + max_stray_rise
    best = 0
    for i, r in enumerate(rounds):
        if i and r["select_nothink_stray"] > limit:
            continue
        if r["select_think_strict"] > rounds[best]["select_think_strict"]:
            best = i
    return best
