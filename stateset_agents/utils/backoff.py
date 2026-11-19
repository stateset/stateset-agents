"""Numerically bounded delay calculations shared by retry policies."""

import math
import sys


def capped_exponential_delay(
    base_delay: float,
    exponential_base: float,
    exponent: int,
    max_delay: float,
) -> float:
    """Return a capped exponential delay without losing intermediate scaling.

    Callers validate finite nonnegative delays, a finite positive exponential
    base, and a nonnegative exponent. Preserve ordinary floating-point results;
    use logarithms only if the intermediate calculation overflows or underflows.
    Jitter and exception classification belong to each caller's retry policy.
    """
    if base_delay == 0 or max_delay == 0:
        return 0.0
    try:
        # Configs may contain integer literals; do not build unbounded big integers.
        power = float(exponential_base) ** exponent
        delay = base_delay * power
    except OverflowError:
        power = math.inf
        delay = math.inf
    if math.isfinite(delay) and delay > 0 and power >= sys.float_info.min:
        return min(delay, max_delay)
    if exponential_base == 1:
        return min(base_delay, max_delay)

    try:
        log_delay = math.log(base_delay) + exponent * math.log(exponential_base)
    except OverflowError:
        # An exponent too large to convert to float has already reached a limit.
        return max_delay if exponential_base > 1 else 0.0
    if log_delay >= math.log(max_delay):
        return max_delay
    return min(math.exp(log_delay), max_delay)
