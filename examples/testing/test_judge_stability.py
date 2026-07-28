"""Stability budget for an LLM-judge (or any noisy evaluator).

LLM-judge scores are noisy. Before you publish a single number, prove that
N independent calls on the same (query, response) pair stay inside a
documented spread — call it your *judge noise floor*. If your headline
delta isn't comfortably larger than the noise floor, you don't have a
result, you have a coin flip.

This test uses a *deterministic* judge stand-in (no GPU, no API), so the
"variance" is a stand-in too. Swap `noisy_judge` for your real judge to use
the same pattern in production. The framework's whitepaper §11.7 protocol
runs three seeds — this is the per-seed unit test.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Awaitable, Callable

import pytest


# A deliberately noisy "judge". Replace with `await llm_judge(query, response)`.
def noisy_judge(
    *, base: float, noise: float, rng: random.Random
) -> Callable[..., Awaitable[float]]:
    async def _judge(query: str, response: str) -> float:
        return base + rng.uniform(-noise, noise)

    return _judge


@pytest.mark.parametrize(
    "noise,budget",
    [
        # (noise amplitude, allowed stdev). The budget should comfortably exceed the
        # amplitude — that's the test of the *test*, not of the judge.
        (0.05, 0.10),
        (0.20, 0.30),
    ],
)
async def test_judge_stdev_under_budget(noise, budget):
    """Score the same (query, response) N times. Stdev must be under the budget."""
    rng = random.Random(42)
    judge = noisy_judge(base=0.75, noise=noise, rng=rng)
    scores = []
    for _ in range(20):
        scores.append(await judge("test query", "test response"))
    stdev = statistics.pstdev(scores)
    assert stdev < budget, f"stdev {stdev:.3f} exceeded budget {budget}"


async def test_judge_mean_in_documented_range():
    """The mean of many judge calls must approach the true base — guards against
    a calibration drift where the judge silently shifts up or down."""
    rng = random.Random(42)
    judge = noisy_judge(base=0.75, noise=0.10, rng=rng)
    scores = [await judge("q", "r") for _ in range(50)]
    mean = statistics.mean(scores)
    assert 0.70 <= mean <= 0.80, f"Judge mean drifted: {mean:.3f}"


async def test_delta_larger_than_two_sigma():
    """Two candidate responses must score with separation > 2σ of the noise floor.
    This is the publication gate from §11.7 of the whitepaper."""
    rng = random.Random(42)
    good_judge = noisy_judge(base=0.85, noise=0.05, rng=rng)
    bad_judge = noisy_judge(base=0.40, noise=0.05, rng=rng)

    good_scores = [await good_judge("q", "good response") for _ in range(20)]
    bad_scores = [await bad_judge("q", "bad response") for _ in range(20)]

    delta = statistics.mean(good_scores) - statistics.mean(bad_scores)
    pooled_sigma = max(statistics.pstdev(good_scores), statistics.pstdev(bad_scores))
    assert delta > 2 * pooled_sigma, (
        f"Delta {delta:.3f} not comfortably above 2σ of the noise floor "
        f"({pooled_sigma:.3f}). You don't have a result yet."
    )
