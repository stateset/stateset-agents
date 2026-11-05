#!/usr/bin/env python3
"""Retain actual native DDP checkpoint continuation on two CUDA/NCCL workers.

Run from a source checkout with its test dependencies installed. This reuses
the CPU/Gloo publication worker, native learner, optimizer and checkpoint APIs.
CUDA is mandatory by default; explicit CPU diagnostics never satisfy the GPU
gate. No models are downloaded and no provider is contacted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stateset_agents.evaluation.implementation import (  # noqa: E402
    package_implementation,
)
from stateset_agents.verticals.acceptance import (  # noqa: E402
    _digest,
    native_gpu_harness_identity,
)


def validate_hardware(torch: Any, device: str) -> None:
    """Refuse a mislabeled GPU run before creating any workers or artifacts."""
    if device not in {"cuda", "cpu"}:
        raise ValueError("Device must be cuda or cpu")
    if not torch.distributed.is_available():
        raise RuntimeError("PyTorch distributed support is required")
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "Native GPU acceptance requires two CUDA GPUs; no CPU fallback"
            )
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("Native GPU acceptance requires NCCL")
    elif not torch.distributed.is_gloo_available():
        raise RuntimeError("CPU diagnostics require Gloo")


def run_contract(
    output: Path, *, device: str = "cuda", timeout: int = 180
) -> dict[str, Any]:
    """Execute both native checkpoint APIs and retain exact continuation inputs.

    Each worker asserts uninterrupted/resumed model, reference, optimizer,
    scheduler, pending gradients/counts, baseline and all RNGs agree. Injected
    publication/final-RNG faults must quarantine both peers until reload. The
    safe tensor snapshots let the offline gate repeat those comparisons.
    """
    if type(timeout) is not int or not 1 <= timeout <= 900:
        raise ValueError("Per-case timeout must be between 1 and 900 seconds")
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory must be new; existing evidence is preserved")
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    validate_hardware(torch, device)
    from tests.integration.test_ddp_checkpoint_publication import _worker

    implementation = package_implementation()
    harness = native_gpu_harness_identity(ROOT)
    output.mkdir(parents=True, mode=0o700)
    cases = []
    precisions = ("fp32", "fp16") if device == "cuda" else ("fp32",)
    for api in ("runtime", "async"):
        for pending in (False, True):
            for precision in precisions:
                name = f"{api}-{'pending' if pending else 'complete'}-{precision}"
                directory = output / name
                directory.mkdir()
                context = torch.multiprocessing.spawn(
                    _worker,
                    args=(
                        str(directory.resolve()),
                        api,
                        device,
                        pending,
                        precision,
                        True,
                    ),
                    nprocs=2,
                    join=False,
                )
                try:
                    deadline = time.monotonic() + timeout
                    while not context.join(timeout=1):
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                f"Native checkpoint case {name} exceeded {timeout}s"
                            )
                finally:
                    for process in context.processes:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=5)
                rank_reports = []
                for rank in (0, 1):
                    path = directory / f"ddp-checkpoint-publication-{rank}.json"
                    peer = json.loads(path.read_text())
                    # Snapshots in rank reports use bundle-relative paths.
                    for scenario in peer["scenarios"]:
                        scenario["snapshot"] = f"{name}/{scenario['snapshot']}"
                    path.write_text(json.dumps(peer, indent=2, allow_nan=False) + "\n")
                    rank_reports.append(path.relative_to(output).as_posix())
                cases.append(
                    {
                        "api": api,
                        "pending": pending,
                        "precision": precision,
                        "rank_reports": rank_reports,
                    }
                )
    if (
        package_implementation() != implementation
        or native_gpu_harness_identity(ROOT) != harness
    ):
        raise RuntimeError("Implementation or harness changed during native acceptance")
    artifacts = {
        path.relative_to(output).as_posix(): _digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    report = {
        "schema_version": 1,
        "kind": "stateset-native-checkpoint-hardware-contract",
        "status": "completed",
        "device": device,
        "backend": "nccl" if device == "cuda" else "gloo",
        "world_size": 2,
        "implementation": implementation,
        "harness": harness,
        "cases": cases,
        "artifacts": artifacts,
        "scope": "Tiny native checkpoint/RNG/AMP mechanics only; CPU diagnostics do not satisfy GPU acceptance. No model quality or scaling claim.",
    }
    (output / "contract.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    output_existed = args.output.exists() or args.output.is_symlink()
    try:
        report = run_contract(args.output, device=args.device, timeout=args.timeout)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - retain a failed execution, never a success receipt
        report = {
            "schema_version": 1,
            "kind": "stateset-native-checkpoint-hardware-contract",
            "status": "failed",
            "device": args.device,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        # Do not overwrite an existing completed bundle or its failed history.
        if not output_existed:
            args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
            target = args.output / "contract.json"
            if not target.exists() and not target.is_symlink():
                target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "device": args.device,
                "cases": len(report["cases"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
