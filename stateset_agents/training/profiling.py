"""Bounded host-wall timing and verified rollout-efficiency measurements."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any


class TrainingProfiler:
    """Measure host stages without synchronizing GPU kernels or inferring success.

    Percentiles use the most recent 1024 stage durations; totals cover the run.
    GPU kernel profiling requires an external profiler or explicit synchronization.
    """

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1024))
        self.totals: dict[str, float] = defaultdict(float)
        self.failures: dict[str, int] = defaultdict(int)
        self.attempted = 0
        self.accepted = 0
        self.verified_successes = 0
        self.unknown_successes = 0
        self.generated_tokens = 0

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Record a stage's elapsed host time, including failed attempts."""
        start = time.monotonic()
        try:
            yield
        except BaseException:
            self.failures[stage] += 1
            raise
        finally:
            elapsed = time.monotonic() - start
            self.samples[stage].append(elapsed)
            self.totals[stage] += elapsed

    def record_rollout(
        self, *, accepted: bool, success: bool | None, tokens: int = 0
    ) -> None:
        """Count explicitly accepted samples and evaluator-verified task outcomes."""
        if type(accepted) is not bool or (
            success is not None and type(success) is not bool
        ):
            raise ValueError("Rollout acceptance and verified success must be booleans")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Generated token count must be a nonnegative integer")
        self.attempted += 1
        self.generated_tokens += tokens
        if accepted:
            self.accepted += 1
            self.verified_successes += success is True
            self.unknown_successes += success is None

    def snapshot(self, *, cost_usd: float | None = None) -> dict[str, Any]:
        """Report timing and optional caller-supplied cost; absent costs stay unknown."""
        if cost_usd is not None and (
            type(cost_usd) not in (int, float)
            or not math.isfinite(cost_usd)
            or cost_usd < 0
        ):
            raise ValueError("Cost must be finite and nonnegative")
        elapsed = time.monotonic() - self.started
        stages = {}
        for stage, values in self.samples.items():
            ordered = sorted(values)
            stages[stage] = {
                "host_seconds": self.totals[stage],
                "p95_host_seconds": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
                "recent_samples": len(ordered),
                "failures": self.failures[stage],
            }
        return {
            "scope": "host_wall_timing; cost is caller-supplied, not provider-verified",
            "elapsed_seconds": elapsed,
            "stages": stages,
            "attempted_rollouts": self.attempted,
            "accepted_rollouts": self.accepted,
            "verified_successes": self.verified_successes,
            "unknown_successes": self.unknown_successes,
            "generated_tokens": self.generated_tokens,
            "accepted_rollouts_per_second": (
                self.accepted / elapsed if elapsed else None
            ),
            "cost_usd": cost_usd,
            "cost_per_accepted_rollout": (
                cost_usd / self.accepted
                if cost_usd is not None and self.accepted
                else None
            ),
            "cost_per_verified_success": (
                cost_usd / self.verified_successes
                if cost_usd is not None and self.verified_successes
                else None
            ),
        }


def profile_learner_step(function: Any) -> Any:
    """Time a native async learner call while preserving its public signature."""

    @wraps(function)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        profiler = getattr(self, "profiler", None)
        if profiler is None:
            return await function(self, *args, **kwargs)
        with profiler.measure("learner_step"):
            return await function(self, *args, **kwargs)

    return wrapped
