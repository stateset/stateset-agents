"""Capped retry delays preserve scaling across floating-point range limits."""

import math

import pytest

from stateset_agents.utils.backoff import capped_exponential_delay


@pytest.mark.parametrize(
    ("base", "factor", "exponent", "cap", "expected"),
    [
        (0, 1e308, 3, 60, 0),
        (1, 1e308, 3, 0, 0),
        (1, 2, 0, 60, 1),
        (2, 2, 2, 60, 8),
        (2, 2, 10, 60, 60),
        (4, 0.5, 2, 60, 1),
        (1e-300, 1e200, 2, 1e200, 1e100),
        (1e300, 1e-200, 2, 1e300, 1e-100),
        (1e308, 1e-160, 2, 1, 1e-12),
        (1, 1e308, 1000, 60, 60),
        (1, 1e-308, 1000, 60, 0),
        (1, 2, 10**400, 60, 60),
        (1, 0.5, 10**400, 60, 0),
        (1, 1.0, 10**400, 60, 1),
    ],
)
def test_delay_is_finite_capped_and_correctly_scaled(
    base, factor, exponent, cap, expected
):
    result = capped_exponential_delay(base, factor, exponent, cap)
    assert math.isfinite(result)
    assert 0 <= result <= cap
    assert result == pytest.approx(expected, rel=1e-12, abs=0)
