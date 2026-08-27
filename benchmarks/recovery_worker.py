#!/usr/bin/env python3
"""Crashable training worker used by the measured reliability matrix."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

WORKER_EXIT_CODE = 86
NETWORK_EXIT_CODE = 88


class RecoveryPolicy(nn.Module):
    """Small deterministic policy whose optimizer state is checkpointed."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, 32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def _batch(
    step: int, seed: int, batch_size: int, width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(batch_size, device=device, dtype=torch.float32)[:, None]
    columns = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    inputs = torch.sin((rows + step * batch_size + seed + 1) * (columns + 1) * 0.001)
    targets = (
        (torch.arange(batch_size, device=device) + step * 17 + seed * 31) % 32
    ).long()
    return inputs, targets


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    applied_steps: list[int],
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "applied_steps": applied_steps,
        },
        temporary,
    )
    os.replace(temporary, path)


def _heartbeat(port: int, timeout: float = 2.0) -> None:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as connection:
        connection.sendall(b"stateset-reliability-heartbeat\n")
        response = connection.recv(16)
    if response != b"ok\n":
        raise ConnectionError("heartbeat server returned an invalid acknowledgement")


def run(args: argparse.Namespace) -> None:
    """Train, checkpoint each committed update, and inject the selected fault."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reliability run requested without CUDA")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run_dir / "checkpoint.pt"
    marker = args.run_dir / "fault-ready.json"
    proceed = args.run_dir / "inject-network-fault"

    torch.manual_seed(args.seed)
    model = RecoveryPolicy(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_step = 0
    applied_steps: list[int] = []
    if args.phase == "resume":
        state: dict[str, Any] = torch.load(
            checkpoint, map_location=device, weights_only=True
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        applied_steps = list(map(int, state["applied_steps"]))
    elif checkpoint.exists():
        raise RuntimeError(f"refusing to overwrite existing checkpoint {checkpoint}")

    if args.fault == "network_interruption" and args.phase == "inject":
        _heartbeat(args.heartbeat_port)

    for step in range(start_step + 1, args.final_step + 1):
        inputs, targets = _batch(step, args.seed, args.batch_size, args.width, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        applied_steps.append(step)
        _save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=step,
            applied_steps=applied_steps,
        )

        if args.phase == "inject" and step == args.injected_at_step:
            marker.write_text(
                json.dumps({"checkpoint_step": step, "pid": os.getpid()}) + "\n",
                encoding="utf-8",
            )
            if args.fault == "worker_exit":
                os._exit(WORKER_EXIT_CODE)
            if args.fault == "controller_restart":
                while True:
                    time.sleep(1)
            if args.fault == "network_interruption":
                deadline = time.monotonic() + args.coordination_timeout
                while not proceed.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("network fault coordinator did not proceed")
                    time.sleep(0.02)
                try:
                    _heartbeat(args.heartbeat_port)
                except OSError:
                    os._exit(NETWORK_EXIT_CODE)
                raise RuntimeError(
                    "network fault injection did not interrupt heartbeat"
                )

    (args.run_dir / "completed.json").write_text(
        json.dumps({"final_step": args.final_step, "applied_steps": applied_steps})
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--fault",
        choices=("worker_exit", "controller_restart", "network_interruption"),
        required=True,
    )
    parser.add_argument("--phase", choices=("inject", "resume"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--injected-at-step", type=int, default=7)
    parser.add_argument("--final-step", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heartbeat-port", type=int, default=0)
    parser.add_argument("--coordination-timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
