"""Paired statistics for weight-edit evaluations (same problems, same decoding).

Edits are judged against the base model on identical problems, so the right
tests are paired: an exact McNemar test on correctness flips and a bootstrap
interval on the per-problem accuracy difference.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "at_budget",
    "bootstrap_ci",
    "budget_curve",
    "clean_finish",
    "mcnemar_exact",
    "paired_compare",
    "strict_rows",
]


def _end_think_count(row: Mapping[str, Any], end_think: str) -> int:
    if "text" in row:
        return str(row["text"]).count(end_think)
    return int(row.get("think_close", 0))  # compact rows store only the count


def clean_finish(
    row: Mapping[str, Any], *, end_think: str = "</think>", thinking: bool = True
) -> bool:
    """True if the generation stopped on its own and is well formed.

    With thinking on, well formed means exactly one ``end_think``. Graders
    commonly split a reasoning model's output on the first end-of-thinking
    marker and score what follows. An edit that boosts that marker's logit can
    emit it inline ("J = 12</think>") while the model keeps reasoning, so the
    "answer" is the rest of the reasoning and a correct number in it gets
    credited. Requiring a clean finish closes that loophole. With thinking off
    (the chat template closes an empty thinking block itself), any
    ``end_think`` in the reply is format damage.

    Rows carry the generated ``text``, or a compact ``think_close`` count.
    """
    want = 1 if thinking else 0
    return row.get("finish") == "stop" and _end_think_count(row, end_think) == want


def strict_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    end_think: str = "</think>",
    thinking: bool = True,
) -> list[dict[str, Any]]:
    """Copies of ``rows`` with ``correct`` kept only for clean finishes."""
    return [
        {
            **r,
            "correct": bool(r["correct"])
            and clean_finish(r, end_think=end_think, thinking=thinking),
        }
        for r in rows
    ]


def at_budget(
    rows: Sequence[Mapping[str, Any]],
    budget: int,
    *,
    cap: int,
    end_think: str = "</think>",
    thinking: bool = True,
) -> list[dict[str, Any]]:
    """Strictly scored copies of ``rows`` as a run capped at ``budget`` would be.

    Valid for runs sampled with a fixed per-request seed and capped at ``cap``
    tokens: each reply is then a deterministic function of its prompt, so the
    run at any ``budget <= cap`` is the same run cut short. A reply that
    stopped within ``budget`` generated tokens (``tokens``) is unchanged; any
    other reply would have been cut off, so it is wrong. A ``budget`` above
    ``cap`` is refused: a reply cut off at ``cap`` might have finished later.
    """
    if budget < 1:
        raise ValueError(f"budget must be positive, got {budget}")
    if budget > cap:
        raise ValueError(
            f"budget {budget} exceeds the run's cap {cap}: "
            "replies cut off at the cap might have finished later"
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        if int(r["tokens"]) <= budget:
            ok = clean_finish(r, end_think=end_think, thinking=thinking)
            out.append({**r, "correct": bool(r["correct"]) and ok})
        else:
            out.append({**r, "finish": "length", "correct": False})
    return out


def budget_curve(
    rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
    *,
    cap: int,
    end_think: str = "</think>",
    thinking: bool = True,
) -> dict[int, float]:
    """Strict accuracy at each token budget, from one run capped at ``cap``."""
    if not rows:
        raise ValueError("budget_curve needs at least one row")
    curve: dict[int, float] = {}
    for b in budgets:
        scored = at_budget(rows, b, cap=cap, end_think=end_think, thinking=thinking)
        curve[b] = sum(r["correct"] for r in scored) / len(scored)
    return curve


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def bootstrap_ci(
    diffs: Sequence[float], iters: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of ``diffs``."""
    if not diffs:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    return means[int(0.025 * iters)], means[min(iters - 1, int(0.975 * iters))]


def paired_compare(
    base: Sequence[Mapping[str, Any]],
    other: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare per-problem correctness keyed by ``id`` (order-independent)."""
    by_id = {r["id"]: bool(r["correct"]) for r in base}
    pairs = [(by_id[r["id"]], bool(r["correct"])) for r in other if r["id"] in by_id]
    if not pairs:
        raise ValueError("no shared problem ids")
    gained = sum(o and not b for b, o in pairs)
    lost = sum(b and not o for b, o in pairs)
    diffs = [float(o) - float(b) for b, o in pairs]
    return {
        "n": len(pairs),
        "gained": gained,
        "lost": lost,
        "delta": sum(diffs) / len(pairs),
        "delta_95ci": bootstrap_ci(diffs, seed=seed),
        "mcnemar_p": mcnemar_exact(lost, gained),
    }
