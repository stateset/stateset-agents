#!/usr/bin/env python3
"""Measured fixed-workload DDP scaling benchmark for StateSet Agents.

This is a synthetic policy-optimization workload, not an LLM quality
benchmark.  It exercises real CUDA forward/backward passes, group-relative
advantages, optimizer updates, and PyTorch DDP gradient synchronization while
holding the global workload constant across process counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stateset_agents import __version__  # noqa: E402

PROTOCOL = "stateset-ddp-policy-scaling-v1"
MODEL_NAME = "stateset/synthetic-residual-policy-v1"
MODEL_REVISION = hashlib.sha1(
    b"stateset-residual-policy-v1", usedforsecurity=False
).hexdigest()
DATASET_REVISION = hashlib.sha1(
    b"stateset-indexed-policy-data-v1", usedforsecurity=False
).hexdigest()
ALGORITHM_REVISION = "stateset-group-policy-gradient-v1"

DEFAULT_WORKLOAD: dict[str, Any] = {
    "feature_dim": 512,
    "hidden_dim": 512,
    "depth": 8,
    "num_actions": 256,
    "group_size": 4,
    "global_batch_size": 2048,
    "warmup_steps": 3,
    "measured_steps": 24,
    "eval_examples": 2048,
    "learning_rate": 2e-4,
    "entropy_coef": 1e-3,
    "precision": "bf16",
}


def canonical_digest(config: Mapping[str, Any]) -> str:
    """Return the workload identity shared by every measured topology."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ResidualPolicy(nn.Module):
    """Compute-dense policy used to expose DDP scaling without downloads."""

    def __init__(
        self, feature_dim: int, hidden_dim: int, depth: int, num_actions: int
    ) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(self.output_norm(hidden))


def indexed_batch(
    *,
    step: int,
    seed: int,
    rank: int,
    local_batch_size: int,
    global_batch_size: int,
    feature_dim: int,
    num_actions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a topology-invariant batch by partitioning global row ids."""
    start = step * global_batch_size + rank * local_batch_size
    indices = torch.arange(start, start + local_batch_size, device=device)
    columns = torch.arange(feature_dim, device=device, dtype=torch.float32)
    phase = (indices[:, None].float() + seed + 1.0) * (columns[None, :] + 1.0)
    features = torch.sin(phase * 0.000173) + torch.cos(phase * 0.000071)
    targets = ((indices * 1103515245 + seed * 12345) % num_actions).long()

    actions = torch.arange(num_actions, device=device, dtype=torch.float32)
    exploration = torch.sin(
        (indices[:, None].float() + seed + 3.0) * (actions[None, :] + 5.0) * 0.01973
    )
    return features, targets, exploration


def policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    exploration: torch.Tensor,
    *,
    group_size: int,
    entropy_coef: float,
) -> torch.Tensor:
    """Group-relative policy-gradient loss with deterministic exploration."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    actions = torch.argmax(logits.detach().float() + exploration, dim=-1)
    selected = log_probs.gather(1, actions[:, None]).squeeze(1)
    distance = (actions - targets).abs().float()
    rewards = 1.0 - distance / max(logits.shape[-1] - 1, 1)
    grouped = rewards.reshape(-1, group_size)
    advantages = grouped - grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, correction=0, keepdim=True).clamp_min(1e-6)
    advantages = (advantages / std).reshape(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
    return -(advantages.detach() * selected).mean() - entropy_coef * entropy


@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    seed: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> float:
    """Evaluate target-action probability on a fixed global partition."""
    local_size = int(config["eval_examples"]) // world_size
    features, targets, _ = indexed_batch(
        step=1_000_000,
        seed=seed,
        rank=rank,
        local_batch_size=local_size,
        global_batch_size=int(config["eval_examples"]),
        feature_dim=int(config["feature_dim"]),
        num_actions=int(config["num_actions"]),
        device=device,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(features)
    score_sum = torch.softmax(logits.float(), dim=-1).gather(1, targets[:, None]).sum()
    count = torch.tensor(float(local_size), device=device)
    dist.all_reduce(score_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return float((score_sum / count).item())


def _topology() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip()


def _artifact_digest(model: nn.Module, output: Path) -> str:
    module = model.module if isinstance(model, DDP) else model
    artifact = output.with_suffix(".pt")
    torch.save({"model": module.state_dict()}, artifact)
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    """Execute one topology/seed measurement under ``torchrun``."""
    if not torch.cuda.is_available():
        raise RuntimeError("distributed scaling evidence requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.gpu_count:
        raise RuntimeError(
            f"torchrun world size {world_size} != requested GPU count {args.gpu_count}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        config = dict(DEFAULT_WORKLOAD)
        config.update(json.loads(args.config_json))
        global_batch = int(config["global_batch_size"])
        group_size = int(config["group_size"])
        if global_batch % world_size or (global_batch // world_size) % group_size:
            raise ValueError(
                "global_batch_size must divide evenly into topology and policy groups"
            )
        if int(config["eval_examples"]) % world_size:
            raise ValueError("eval_examples must divide evenly into topology")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = ResidualPolicy(
            int(config["feature_dim"]),
            int(config["hidden_dim"]),
            int(config["depth"]),
            int(config["num_actions"]),
        ).to(device)
        model = DDP(model, device_ids=[local_rank], static_graph=True)
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.module.state_dict().items()
        }

        def new_optimizer() -> torch.optim.Optimizer:
            return torch.optim.AdamW(
                model.parameters(), lr=float(config["learning_rate"])
            )

        local_batch = global_batch // world_size
        optimizer = new_optimizer()

        def step_once(step: int) -> None:
            features, targets, exploration = indexed_batch(
                step=step,
                seed=args.seed,
                rank=rank,
                local_batch_size=local_batch,
                global_batch_size=global_batch,
                feature_dim=int(config["feature_dim"]),
                num_actions=int(config["num_actions"]),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(features)
                loss = policy_loss(
                    logits,
                    targets,
                    exploration,
                    group_size=group_size,
                    entropy_coef=float(config["entropy_coef"]),
                )
            loss.backward()
            optimizer.step()

        for step in range(int(config["warmup_steps"])):
            step_once(-1000 + step)
        model.module.load_state_dict(initial_state)
        optimizer = new_optimizer()
        dist.barrier()
        torch.cuda.synchronize()
        baseline = evaluate_policy(
            model,
            config,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        started = time.perf_counter()
        for step in range(int(config["measured_steps"])):
            step_once(step)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = torch.tensor(time.perf_counter() - started, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        final = evaluate_policy(
            model,
            config,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        peak = torch.tensor(
            torch.cuda.max_memory_allocated(device) / (1024**2), device=device
        )
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)

        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            artifact_sha256 = _artifact_digest(model, args.output)
            seconds = float(elapsed.item())
            samples = global_batch * int(config["measured_steps"])
            evidence = {
                "schema_version": 1,
                "measured": True,
                "framework": "stateset-agents",
                "framework_version": __version__,
                "harness_commit": args.harness_commit,
                "protocol": PROTOCOL,
                "cache_policy": "generated-on-device-indexed-v1",
                "algorithm": "group-policy-gradient",
                "algorithm_revision": ALGORITHM_REVISION,
                "model": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "task": "synthetic-group-policy-optimization-v1",
                "dataset_revision": DATASET_REVISION,
                "workload_config_sha256": canonical_digest(config),
                "seed": args.seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "command": args.command_label,
                "config": config,
                "hardware": {
                    "gpu": torch.cuda.get_device_name(0),
                    "gpu_count": world_size,
                    "cuda": str(torch.version.cuda),
                    "topology": _topology(),
                },
                "software": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                },
                "metrics": {
                    "samples_per_second": samples / seconds,
                    "wall_clock_seconds": seconds,
                    "peak_vram_mb": float(peak.item()),
                    "eval_score_baseline": baseline,
                    "eval_score_final": final,
                },
                "artifact_sha256": artifact_sha256,
            }
            args.output.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--command-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
