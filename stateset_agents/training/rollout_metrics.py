"""Bounded numeric population reports for native DDP rollout observations."""

from __future__ import annotations

import bisect
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

ROLLOUT_METRICS_PROTOCOL = "stateset-native-rollout-metrics-v1"
_MAX_COUNT = 2**53 - 1
_SIGN = 1 << 63
_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class _Moments:
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0


def _moments(values: list[float]) -> _Moments:
    if not values:
        return _Moments()
    if len(values) > _MAX_COUNT or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid rollout reward population")
    scale = max(abs(v) for v in values)
    mean = math.fsum((v / scale) / len(values) for v in values) if scale else 0.0
    if scale:
        mean = max(min(values) / scale, min(max(values) / scale, mean))
    variance = (
        math.fsum(((v / scale) - mean) ** 2 / len(values) for v in values)
        if scale
        else 0.0
    )
    return _Moments(
        len(values),
        mean * scale,
        min(1.0, math.sqrt(variance)) * scale,
        min(values),
        max(values),
    )


def _merge(rows: list[_Moments]) -> _Moments:
    count = sum(row.count for row in rows)
    if count > _MAX_COUNT:
        raise ValueError("Rollout count exceeds exact reporting range")
    if not count:
        return _Moments()
    populated = [row for row in rows if row.count]
    minimum, maximum = min(r.minimum for r in populated), max(
        r.maximum for r in populated
    )
    scale = max(abs(minimum), abs(maximum))
    mean = (
        math.fsum((r.mean / scale) * (r.count / count) for r in populated)
        if scale
        else 0.0
    )
    # Clamp roundoff to the population's finite convex bounds before rescaling.
    if scale:
        mean = max(minimum / scale, min(maximum / scale, mean))
    variance = (
        math.fsum(
            ((r.std / scale) ** 2 + (r.mean / scale - mean) ** 2) * (r.count / count)
            for r in populated
        )
        if scale
        else 0.0
    )
    std = min(1.0, math.sqrt(max(0.0, variance))) * scale
    return _Moments(count, mean * scale, std, minimum, maximum)


def _float_key(value: float) -> int:
    """Map finite IEEE float64 values to unsigned keys in numeric order."""
    bits = int(struct.unpack("!Q", struct.pack("!d", value if value else 0.0))[0])
    return (~bits & _MASK) if bits & _SIGN else bits ^ _SIGN


def _key_float(key: int) -> float:
    bits = (key ^ _SIGN) if key & _SIGN else (~key & _MASK)
    return float(struct.unpack("!d", struct.pack("!Q", bits))[0])


@dataclass(frozen=True)
class _Population:
    groups: int
    populated_groups: int
    rewards: _Moments
    diversity: _Moments
    keys: list[int]


def _capture(groups: list[Any], *, details: bool) -> _Population:
    rewards, diversities = [], []
    populated = 0
    if len(groups) > _MAX_COUNT:
        raise ValueError("Too many rollout groups")
    for group in groups:
        values = []
        for trajectory in group.trajectories:
            reward = getattr(trajectory, "total_reward", None)
            if reward is None:
                reward = getattr(trajectory, "reward", None)
            if reward is None:
                raise ValueError("Rollout is missing its observed reward")
            values.append(float(reward))
        moments = _moments(values)
        if moments.count:
            populated += 1
            rewards.extend(values)
        if moments.count > 1:
            diversities.append(moments.std)
    return _Population(
        len(groups),
        populated,
        _moments(rewards),
        _moments(diversities),
        sorted(_float_key(v) for v in rewards) if details else [],
    )


def _pack(row: _Population) -> tuple[list[int], list[float]]:
    return (
        [row.groups, row.populated_groups, row.rewards.count, row.diversity.count],
        [
            getattr(m, name)
            for m in (row.rewards, row.diversity)
            for name in ("mean", "std", "minimum", "maximum")
        ],
    )


def _unpack(counts: list[int], values: list[float]) -> _Population:
    if (
        len(counts) != 4
        or len(values) != 8
        or any(type(c) is not int or not 0 <= c <= _MAX_COUNT for c in counts)
        or not all(math.isfinite(v) for v in values)
        or not counts[3] <= counts[1] <= min(counts[0], counts[2])
    ):
        raise ValueError("Invalid rollout population wire fields")
    moments = []
    for count, offset in [(counts[2], 0), (counts[3], 4)]:
        mean, std, minimum, maximum = values[offset : offset + 4]
        if (
            std < 0
            or not minimum <= mean <= maximum
            or not count
            and any((mean, std, minimum, maximum))
            or count == 1
            and (std != 0 or mean != minimum or mean != maximum)
            or std > max(abs(minimum), abs(maximum))
        ):
            raise ValueError("Inconsistent rollout population moments")
        moments.append(_Moments(count, mean, std, minimum, maximum))
    if moments[1].minimum < 0:
        raise ValueError("Reward diversity cannot be negative")
    return _Population(counts[0], counts[1], moments[0], moments[1], [])


def _aggregate(rows: list[_Population]) -> dict[str, Any]:
    groups = sum(r.groups for r in rows)
    populated = sum(r.populated_groups for r in rows)
    if groups > _MAX_COUNT:
        raise ValueError("Rollout group count exceeds exact reporting range")
    rewards, diversity = _merge([r.rewards for r in rows]), _merge(
        [r.diversity for r in rows]
    )
    result = {
        "num_groups": groups,
        "total_trajectories": rewards.count,
        "avg_group_size": rewards.count / populated if populated else 0.0,
        "reward_mean": rewards.mean,
        "reward_std": rewards.std,
        "reward_min": rewards.minimum,
        "reward_max": rewards.maximum,
    }
    if diversity.count:
        result.update(
            reward_diversity_mean=diversity.mean, reward_diversity_std=diversity.std
        )
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite global rollout metrics")
    return result


def _median(group: Any, keys: list[int], metrics: dict[str, Any]) -> float:
    """Select exact middle float64 values using at most 64 two-count reductions.

    No rewards or trajectories are gathered. All peers derive the same pivots
    from global extrema; local sorted keys answer cumulative-count queries.
    """
    count = metrics["total_trajectories"]
    if not count:
        return 0.0
    low = [_float_key(metrics["reward_min"])] * 2
    high = [_float_key(metrics["reward_max"])] * 2
    positions = [(count - 1) // 2, count // 2]
    while low != high:
        pivots = [(a + b) // 2 for a, b in zip(low, high, strict=True)]
        counts = group.torch.tensor(
            [bisect.bisect_right(keys, pivot) for pivot in pivots],
            dtype=group.torch.int64,
            device=group.device,
        )
        group.dist.all_reduce(counts, op=group.dist.ReduceOp.SUM, group=group.group)
        for index, amount in enumerate(counts.cpu().tolist()):
            if amount > positions[index]:
                high[index] = pivots[index]
            else:
                low[index] = pivots[index] + 1
    first, second = map(_key_float, low)
    if first == second:
        return first
    # Two exact ratios avoid overflow and subnormal loss in an even population.
    return float((Fraction.from_float(first) + Fraction.from_float(second)) / 2)


def summarize_ddp_rollouts(
    trainer: Any, groups: list[Any], *, phase: str, details: bool = False
) -> dict[str, Any]:
    """Collectively summarize actual rollout populations on DDP's own group."""
    from .ddp_control import DDPControlGroup

    control = DDPControlGroup(trainer, operation="rollout-metrics")
    control.run(
        phase + ":controls",
        lambda: {"protocol": ROLLOUT_METRICS_PROTOCOL, "details": details},
        agree=True,
    )
    local = control.run(phase + ":local", lambda: _capture(groups, details=details))
    counts, values = _pack(local)
    tensors = [
        control.torch.tensor(counts, dtype=control.torch.int64, device=control.device),
        control.torch.tensor(
            values, dtype=control.torch.float64, device=control.device
        ),
    ]
    gathered = []
    for tensor in tensors:
        peers = [control.torch.empty_like(tensor) for _ in range(control.world)]
        control.dist.all_gather(peers, tensor, group=control.group)
        gathered.append([p.cpu().tolist() for p in peers])
    result = control.run(
        phase + ":aggregate",
        lambda: _aggregate([_unpack(c, v) for c, v in zip(*gathered, strict=True)]),
        agree=True,
    )
    if details:
        result["reward_median"] = _median(control, local.keys, result)
    return dict(result)
