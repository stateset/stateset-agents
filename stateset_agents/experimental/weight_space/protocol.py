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

__all__ = ["bootstrap_ci", "mcnemar_exact", "paired_compare"]


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
