"""Reward statistics for complete prompt groups on native DDP workers.

Only fixed-size counts, moments and configuration digests cross the wire.
Prompt groups must remain intact on each worker; this is not group sharding.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _ddp_group(model: Any) -> Any:
    from torch.nn.parallel import DistributedDataParallel

    return model.process_group if isinstance(model, DistributedDataParallel) else None


def prepare_reward_batch(
    groups: list[Any],
    config: Any,
    objective: Any,
    device: Any,
    *,
    model: Any = None,
    global_reward_mean: float | None = None,
    global_reward_count: int = 0,
    global_reward_sum: float | None = None,
    updating: bool = False,
) -> tuple[list[Any], int, float, float]:
    """Validate rewards and agree on clipped batch moments before model forward.

    DDP peers reject invalid rewards, estimator differences or running-state
    drift together. Counts remain int64; population moments are merged in FP64.
    Only batch normalization and running baselines need a moments collective.
    Single-process callers use the same reward validation and local moments.
    """
    from .trainer_utils import require_torch

    torch = require_torch()
    process_group = _ddp_group(model)
    rewards_by_group: list[Any] = []
    count, mean, m2 = 0, 0.0, 0.0
    digest = bytes(32)
    error: Exception | None = None
    try:
        clip = getattr(config, "reward_clip", None)
        clip = None if clip is None else float(clip)
        if clip is not None and (not math.isfinite(clip) or clip < 0):
            raise ValueError("reward_clip must be finite and nonnegative")
        explicit = getattr(config, "objective", None) is not None
        baseline = getattr(config, "baseline_type", "group_mean")
        running = not explicit and baseline == "global_mean"
        eps = float(objective.advantage_eps)
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("advantage_eps must be finite and nonnegative")
        state: Any = None
        if running and global_reward_mean is not None:
            if (
                not math.isfinite(global_reward_mean)
                or type(global_reward_count) is not int
                or not 0 <= global_reward_count <= 2**63 - 1
                or (global_reward_count == 0 and global_reward_mean != 0)
                or (
                    global_reward_sum is not None
                    and (
                        not math.isfinite(global_reward_sum)
                        or (global_reward_count == 0 and global_reward_sum != 0)
                        or (
                            global_reward_count > 0
                            and global_reward_sum / global_reward_count
                            != global_reward_mean
                        )
                    )
                )
            ):
                raise ValueError("Invalid running reward baseline")
            state = [global_reward_mean, global_reward_count, global_reward_sum]
        settings = [
            "stateset-native-advantages-v2",
            explicit,
            objective.advantage if explicit else baseline,
            eps if explicit else bool(getattr(config, "advantage_normalization", True)),
            clip,
            state,
            updating if running else False,
        ]
        digest = hashlib.sha256(
            json.dumps(settings, allow_nan=False, separators=(",", ":")).encode()
        ).digest()
        for group in groups:
            raw = [float(t.total_reward) for t in group.trajectories]
            if not raw or not all(math.isfinite(value) for value in raw):
                raise ValueError("Native reward groups must be nonempty and finite")
            rewards = torch.tensor(raw, dtype=torch.float32, device=device)
            if clip is not None:
                rewards = rewards.clamp(-clip, clip)
            if not bool(torch.isfinite(rewards).all()):
                raise ValueError("Clipped rewards must be finite in FP32")
            rewards_by_group.append(rewards)
        if not rewards_by_group and process_group is None:
            raise ValueError("Native reward batches must be nonempty")
        if rewards_by_group:
            values = torch.cat(rewards_by_group).double()
            count = values.numel()
            mean = float(values.mean().item())
            m2 = float(((values - mean) ** 2).sum().item())
    except (ValueError, TypeError, OverflowError, AttributeError, RuntimeError) as exc:
        error = exc
    if process_group is not None:
        import torch.distributed as dist

        header = torch.tensor(
            [2, int(error is None), count if error is None else 0, *digest],
            dtype=torch.int64,
            device=device,
        )
        peers = [
            torch.empty_like(header) for _ in range(dist.get_world_size(process_group))
        ]
        dist.all_gather(peers, header, group=process_group)
        headers = [peer.cpu().tolist() for peer in peers]
        if not all(peer[0] == 2 and peer[1] == 1 and peer[2] >= 0 for peer in headers):
            raise ValueError(
                "Invalid reward batch or baseline on a DDP rank"
            ) from error
        if sum(peer[2] for peer in headers) == 0:
            raise ValueError("Native DDP reward population is globally empty")
        if len({tuple(peer[3:]) for peer in headers}) != 1:
            raise ValueError(
                "Reward estimator or running baseline differs across DDP ranks"
            )
        needs_global_moments = (explicit and objective.advantage == "batch_norm") or (
            running and (state is not None or updating)
        )
        if needs_global_moments:
            moments = torch.tensor([mean, m2], dtype=torch.float64, device=device)
            gathered = [torch.empty_like(moments) for _ in peers]
            dist.all_gather(gathered, moments, group=process_group)
            count, mean, m2 = 0, 0.0, 0.0
            for peer, tensor in zip(headers, gathered, strict=True):
                if not peer[2]:
                    continue
                local_mean, local_m2 = tensor.cpu().tolist()
                total = count + peer[2]
                delta = local_mean - mean
                m2 += local_m2 + delta * delta * (count / total) * peer[2]
                mean += delta * (peer[2] / total)
                count = total
    elif error is not None:
        raise ValueError("Invalid native reward batch or baseline") from error
    if (
        count + (global_reward_count if running and state is not None else 0)
        > 2**63 - 1
    ):
        raise ValueError("Running reward count exceeds int64")
    if not math.isfinite(mean) or not math.isfinite(m2) or m2 < 0:
        raise ValueError("Invalid reward population moments")
    return rewards_by_group, count, mean, math.sqrt(m2 / count) if count else 0.0


def validate_advantages(advantages: list[Any], model: Any, device: Any) -> None:
    """Reject nonfinite advantages on every DDP peer before backward."""
    from .trainer_utils import require_torch

    torch = require_torch()
    valid = all(bool(torch.isfinite(value).all()) for value in advantages)
    process_group = _ddp_group(model)
    if process_group is not None:
        import torch.distributed as dist

        flag = torch.tensor(int(valid), dtype=torch.int64, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=process_group)
        valid = bool(flag.item())
    if not valid:
        raise ValueError("Nonfinite native advantages on a learner rank")
