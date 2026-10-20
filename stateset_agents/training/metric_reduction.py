"""Count-aware native RL diagnostics, without retaining tensors or rollout text."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

METRIC_REDUCTION_PROTOCOL = "stateset-native-metrics-v2"
_COUNTS = (
    "policy_count",
    "token_count",
    "trajectory_count",
    "row_count",
    "advantage_count",
    "nonzero_advantages",
)
_VALUES = (
    "policy_sum",
    "entropy_loss_sum",
    "entropy_sum",
    "kl_sum",
    "ratio_sum",
    "clip_sum",
    "advantage_mean",
    "advantage_m2",
)
_CONTROL = {
    "global_step",
    "grad_accum_step",
    "optimizer_step",
    "optimizer_updates",
    "inner_updates",
    "accumulation_flush_updates",
    "learning_rate",
}


@dataclass(frozen=True)
class LossMetricReduction:
    """Detached numerators, exact counts and stable advantage moments."""

    measure: str
    policy_count: int
    token_count: int
    trajectory_count: int
    row_count: int
    advantage_count: int
    nonzero_advantages: int
    policy_sum: float
    entropy_loss_sum: float
    entropy_sum: float
    kl_sum: float
    ratio_sum: float
    clip_sum: float
    advantage_mean: float
    advantage_m2: float

    def valid(self) -> bool:
        """Validate the wire fields without raising before a peer handshake."""
        counts = [getattr(self, name) for name in _COUNTS]
        values = [getattr(self, name) for name in _VALUES]
        try:
            finite = all(
                type(value) in {float, int} and math.isfinite(value) for value in values
            )
        except (OverflowError, TypeError, ValueError):
            return False
        return (
            isinstance(self.measure, str)
            and self.measure in {"trajectory", "token"}
            and all(type(value) is int and 0 <= value <= 2**63 - 1 for value in counts)
            and (
                all(value > 0 for value in counts[:5])
                or (
                    all(value == 0 for value in counts)
                    and all(value == 0 for value in values)
                )
            )
            and self.nonzero_advantages <= self.advantage_count
            and self.advantage_count == self.trajectory_count
            and self.policy_count
            == (self.token_count if self.measure == "token" else self.trajectory_count)
            and finite
            and self.advantage_m2 >= 0
        )


class ReducedMetrics(dict[str, Any]):
    """Ordinary JSON metrics with private detached aggregation statistics."""

    def __init__(self, metrics: dict[str, Any], reduction: LossMetricReduction) -> None:
        super().__init__(metrics)
        self.reduction = reduction


def carry_loss_metrics(metrics: dict[str, Any], loss: Any) -> dict[str, Any]:
    """Preserve native loss statistics through the trainer's metric conversion."""
    reduction = getattr(loss, "metric_reduction", None)
    return ReducedMetrics(metrics, reduction) if reduction is not None else metrics


def aggregate_loss_metrics(rows: list[LossMetricReduction]) -> dict[str, Any]:
    """Combine native means and population variance using their actual measures."""
    if not rows or not all(row.valid() for row in rows):
        raise ValueError("Invalid native metric reduction statistics")
    if len({row.measure for row in rows}) != 1:
        raise ValueError("Native metric reduction measures differ")
    counts = {name: sum(getattr(row, name) for row in rows) for name in _COUNTS}
    means = {
        name: math.fsum(
            getattr(row, name)
            / max(counts["policy_count" if name == "policy_sum" else "token_count"], 1)
            for row in rows
        )
        for name in _VALUES[:6]
    }
    count, mean, m2 = 0, 0.0, 0.0
    for row in rows:
        if not row.advantage_count:
            continue
        total = count + row.advantage_count
        delta = row.advantage_mean - mean
        m2 += row.advantage_m2 + delta * delta * (count / total) * row.advantage_count
        mean += delta * (row.advantage_count / total)
        count = total
    policy = means["policy_sum"]
    total_loss = policy + means["entropy_loss_sum"]
    result = {
        "total_loss": total_loss,
        # Historical policy_loss includes entropy; keep that alias compatible.
        "policy_loss": total_loss,
        "policy_objective_loss": policy,
        "entropy": means["entropy_sum"],
        "kl_penalty": means["kl_sum"],
        "ratio_mean": means["ratio_sum"] if counts["token_count"] else 1.0,
        "clip_fraction": means["clip_sum"],
        "mean_advantage": mean,
        "advantage_std": math.sqrt(max(m2 / count, 0.0)) if count else 0.0,
        "nonzero_advantage_fraction": (
            counts["nonzero_advantages"] / count if count else 0.0
        ),
        "num_trajectories": counts["trajectory_count"],
        "num_rows": counts["row_count"],
        "num_tokens": counts["token_count"],
        "policy_denominator": counts["policy_count"],
        "metric_scope": "ddp_global_batch",
        "metric_reduction_protocol": METRIC_REDUCTION_PROTOCOL,
    }
    if not all(
        math.isfinite(value)
        for value in result.values()
        if isinstance(value, (int, float))
    ):
        raise ValueError("Nonfinite global native metrics")
    return result


def reduce_ddp_metrics(metrics: dict[str, Any], model: Any) -> dict[str, Any]:
    """Agree on schema, then reduce diagnostics on the wrapper's process group.

    Every peer first sends a fixed-size integer header. No pickled objects,
    metric names, text or application strings are transmitted. Native counts
    remain int64; detached values and stable moments use float64.
    """
    import torch
    import torch.distributed as dist

    reduction = metrics.reduction if isinstance(metrics, ReducedMetrics) else None
    native = reduction is not None
    valid = True
    names: list[str] = []
    counts = [0] * len(_COUNTS)
    values: list[float] = []
    digest = bytes(32)
    try:
        if not all(isinstance(key, str) for key in metrics):
            raise ValueError("Metric names must be strings")
        schema: list[tuple[str, str, Any]] = []
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (int, float, bool)):
                if not math.isfinite(value):
                    raise ValueError("Metric values must be finite")
                names.append(key)
                schema.append(
                    (
                        key,
                        type(value).__name__,
                        value if key in _CONTROL or isinstance(value, bool) else None,
                    )
                )
                values.append(float(value))
            elif isinstance(value, str):
                schema.append((key, "str", value))
            else:
                schema.append((key, type(value).__qualname__, None))
        if reduction is not None:
            if not reduction.valid():
                raise ValueError("Invalid metric statistics")
            counts = [getattr(reduction, name) for name in _COUNTS]
            values.extend(getattr(reduction, name) for name in _VALUES)
            schema.append(("native_measure", "str", reduction.measure))
        digest = hashlib.sha256(
            json.dumps(schema, allow_nan=False, separators=(",", ":")).encode()
        ).digest()
    except (TypeError, ValueError, AttributeError, OverflowError):
        valid = False
    device = next(model.parameters()).device
    header = torch.tensor(
        [2, int(native), int(valid), *counts, *digest], dtype=torch.int64, device=device
    )
    world = dist.get_world_size(model.process_group)
    headers = [torch.empty_like(header) for _ in range(world)]
    dist.all_gather(headers, header, group=model.process_group)
    decoded = [row.cpu().tolist() for row in headers]
    if any(row[0] != 2 or row[2] != 1 for row in decoded):
        raise ValueError("Invalid distributed metric schema or statistics")
    if (
        len({row[1] for row in decoded}) != 1
        or len({tuple(row[9:]) for row in decoded}) != 1
    ):
        raise ValueError(
            "Distributed metric schemas or control state differ across ranks"
        )
    payload = torch.tensor(values, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(payload) for _ in range(world)]
    dist.all_gather(gathered, payload, group=model.process_group)
    peers = [row.cpu().tolist() for row in gathered]
    result = dict(metrics)
    for index, name in enumerate(names):
        if name in _CONTROL or isinstance(metrics[name], bool):
            result[name] = metrics[name]
        else:
            result[name] = math.fsum(row[index] / world for row in peers)
    if reduction is not None:
        reductions = [
            LossMetricReduction(
                measure=reduction.measure,
                **dict(zip(_COUNTS, row[3:9], strict=True)),
                **dict(zip(_VALUES, peer[len(names) :], strict=True)),
            )
            for row, peer in zip(decoded, peers, strict=True)
        ]
        result.update(aggregate_loss_metrics(reductions))
        if "ratio_mean_last" in result:
            result["ratio_mean_last"] = result["ratio_mean"]
        if "ewc_penalty" in result:
            denominator = sum(row.policy_count for row in reductions)
            index = names.index("ewc_penalty")
            result["ewc_penalty"] = math.fsum(
                peer[index] * (row.policy_count / denominator) if denominator else 0.0
                for peer, row in zip(peers, reductions, strict=True)
            )
    return result
