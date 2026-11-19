"""Exact sufficient statistics for finite binary rollout rewards."""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction


class PopulationMoments:
    """Population mean and variance on a common integer grid."""

    def __init__(self, count: int, unit: int, total: int, squares: int) -> None:
        self.count = count
        self.unit = unit
        self.total = total
        self.squares = squares
        self.denominator = max(1, count) * unit
        self.mean = total / self.denominator
        self.variance_numerator = count * squares - total * total

    def wire(self) -> bytes:
        """Pack the exact two sums for a population of at most 2^63−1 doubles."""
        # A finite binary64 number is an integer of at most 2098 bits on
        # the 2^-1074 grid. An int64 population adds at most 63 bits.
        factor, remainder = divmod(1 << 1074, self.unit)
        if remainder:
            raise ValueError("Distributed reward packets require binary64-grid rewards")
        total = self.total * factor
        squares = self.squares * factor * factor
        return total.to_bytes(271, "little", signed=True) + squares.to_bytes(
            533, "little"
        )

    @classmethod
    def merge_wire(cls, count: int, packets: Iterable[bytes]) -> PopulationMoments:
        """Add same-grid integers exactly in any rank order."""
        total = squares = 0
        for packet in packets:
            total += int.from_bytes(packet[:271], "little", signed=True)
            squares += int.from_bytes(packet[271:], "little")
        return cls(count, 1 << 1074, total, squares)

    def above(self, epsilon: float) -> bool:
        """Compare the population deviation to epsilon before either is rounded."""
        p, q = epsilon.as_integer_ratio()
        return self.variance_numerator * q * q > (self.denominator * p) ** 2

    def scale(self) -> float:
        """Choose a representable power of two close to the population deviation."""
        exponent = (
            self.variance_numerator.bit_length() // 2 - self.denominator.bit_length()
        )
        return math.ldexp(1.0, max(-1074, min(1023, exponent)))

    def deviation(self, scale: float = 1.0) -> float:
        """Population deviation in units of scale, with a finite intermediate square."""
        if not self.variance_numerator:
            return 0.0
        unit = self.scale()
        p, q = unit.as_integer_ratio()
        sd = math.sqrt((self.variance_numerator * q * q) / (self.denominator * p) ** 2)
        return sd if scale == unit else (unit / scale) * sd


class RewardMoments(PopulationMoments):
    """Population moments with an exact centered numerator for every input row."""

    def __init__(self, rewards: Iterable[float | Fraction]) -> None:
        parts = [value.as_integer_ratio() for value in rewards]
        unit = math.lcm(*(d for _, d in parts))
        ticks = [p * (unit // d) for p, d in parts]
        total = sum(ticks)
        super().__init__(len(ticks), unit, total, sum(v * v for v in ticks))
        self.residuals = [self.count * v - total for v in ticks]
