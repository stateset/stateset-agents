"""Lossless conversion of scalar offline reward columns."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import numpy as np


def reward_array(raw: Any) -> np.ndarray:
    """Keep each scalar exact when choosing a representation for offline rewards."""
    inferred: np.ndarray = np.asarray(raw)
    extended = inferred.dtype.kind == "f" and inferred.dtype.itemsize > 8
    if inferred.dtype.kind not in "fO" or (
        isinstance(raw, np.ndarray) and raw.dtype != object and not extended
    ):
        return inferred
    cells: np.ndarray = np.asarray(raw, dtype=object)
    scalars: list[int | float] = []
    for x in cells.flat:
        if isinstance(x, (Integral, np.bool_)):
            scalars.append(int(x))
        elif isinstance(x, (float, np.floating)):
            scalar = float(x)
            if isinstance(x, np.floating) and np.isfinite(x) and x != scalar:
                raise ValueError("Offline floating rewards must fit binary64 exactly")
            scalars.append(scalar)
        else:
            raise ValueError("Offline reward columns must contain real numeric scalars")
    if inferred.dtype == object or extended:
        inferred = np.asarray(scalars).reshape(cells.shape)
        if inferred.dtype.kind in "iu":
            return inferred

    # NumPy can choose double even for an entirely integral Python sequence.
    # Compare Python scalars so the equality test itself cannot round the integer.
    if inferred.dtype.kind == "f" and all(
        not isinstance(x, int) or x == float(y)
        for x, y in zip(scalars, inferred.flat, strict=True)
    ):
        return inferred

    if all(
        isinstance(x, int) or (x.is_integer() and (x != 0 or math.copysign(1, x) > 0))
        for x in scalars
    ):
        integers = [int(x) for x in scalars]
        smallest = min(integers, default=0)
        largest = max(integers, default=0)
        for dtype in (np.int64, np.uint64):
            info: np.iinfo[Any] = np.iinfo(dtype)
            if int(info.min) <= smallest and largest <= int(info.max):
                inferred = np.asarray(integers, dtype=dtype).reshape(cells.shape)
                return inferred
    # A column can contain a negative fraction and an integer above 2^53.
    # Keep each value's own representation until its Bellman target is computed.
    cells = np.asarray(scalars, dtype=object).reshape(cells.shape)
    return cells


def finite_rewards(rewards: np.ndarray) -> bool:
    """Check scalar labels without first converting a large integer to float."""
    return all(isinstance(x, Integral) or math.isfinite(x) for x in rewards.flat)
