"""Discounted conversation returns from finite floating-point and integer rewards."""

import math
from collections.abc import Sequence
from numbers import Integral


def _rounded(numerator: int, unit: int) -> float:
    try:
        return numerator / unit
    except OverflowError:
        return -math.inf if numerator < 0 else math.inf


def discounted_returns(
    rewards: Sequence[float | Integral], gamma: float
) -> list[float]:
    """Round each complete discounted suffix once, including after cancellation."""
    discount = float(gamma)
    if not math.isfinite(discount):
        raise ValueError("The return discount must be finite")
    m, d = discount.as_integer_ratio()
    shift = d.bit_length() - 1
    # Every finite binary64 reward is an integer on this grid. After n-1
    # discounts the exact dyadic denominator divides 2**exact_bits.
    parts = []
    for value in rewards:
        if isinstance(value, Integral):
            parts.append((int(value), 0))
            continue
        r = float(value)
        if not math.isfinite(r):
            raise ValueError("Turn rewards must be finite")
        p, q = r.as_integer_ratio()
        parts.append((p, q.bit_length() - 1))
    if not parts:
        return []
    reward_bits = max(k for _, k in parts)
    exact_bits = reward_bits + (len(parts) - 1) * shift
    bits = min(1074 + 64, exact_bits)
    while True:
        unit = 1 << bits
        lower = upper = 0
        result = []
        for p, k in reversed(parts):
            r = p << (bits - k)
            if m < 0:
                lower, upper = upper, lower
            lower = r + ((m * lower) >> shift)
            upper = r - ((-m * upper) >> shift)
            left = _rounded(lower, unit)
            if lower != upper:
                right = _rounded(upper, unit)
                # A nondegenerate interval has strictly interior exact value:
                # the first inexact division rounds outward at both ends.
                if upper == 0:
                    right = -0.0
                if left != right or (
                    left == 0 and math.copysign(1, left) != math.copysign(1, right)
                ):
                    break
            if not math.isfinite(left):
                raise ValueError(
                    "A discounted return exceeds the finite binary64 range"
                )
            result.append(left)
        else:
            return result[::-1]
        # A rounded endpoint is ambiguous. Include more low bits of the
        # actual history. At exact_bits every update is exact.
        bits = min(2 * bits, exact_bits)
