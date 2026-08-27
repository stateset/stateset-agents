#!/usr/bin/env python3
"""Execute and retain the StateSet fault-recovery evidence matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from recovery_worker import NETWORK_EXIT_CODE, WORKER_EXIT_CODE
from reliability_evidence import load_runs, summarize, validate_matrix

from stateset_agents import __version__

FAULTS = ("worker_exit", "controller_restart", "network_interruption")
PROTOCOL = "stateset-checkpoint-fault-recovery-v1"
MODEL = "stateset/recovery-policy-v1"
MODEL_REVISION = hashlib.sha1(
    b"stateset-recovery-policy-v1", usedforsecurity=False
).hexdigest()


class HeartbeatServer:
    """Minimal TCP control-plane endpoint used for real interruption tests."""

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen()
        self._socket.settimeout(0.1)
        self.port = int(self._socket.getsockname()[1])
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _serve(self) -> None:
        while not self._stopped.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                connection.recv(1024)
                connection.sendall(b"ok\n")

    def stop(self) -> None:
        if self._stopped.is_set():
            self._thread.join(timeout=2)
            return
        self._stopped.set()
        self._socket.close()
        self._thread.join(timeout=2)

    @property
    def alive(self) -> bool:
        """Whether the control-plane listener still owns a live thread."""
        return self._thread.is_alive()


def _wait_for(path: Path, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    fault: str,
    seed: int,
    phase: str,
    heartbeat_port: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("recovery_worker.py")),
        "--run-dir",
        str(run_dir),
        "--fault",
        fault,
        "--phase",
        phase,
        "--seed",
        str(seed),
        "--injected-at-step",
        str(args.injected_at_step),
        "--final-step",
        str(args.final_step),
        "--device",
        args.device,
        "--heartbeat-port",
        str(heartbeat_port),
    ]


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_fault(
    args: argparse.Namespace, fault: str, seed: int, harness_commit: str
) -> dict[str, Any]:
    """Inject one real fault, resume, and prove the final update ledger."""
    run_dir = args.output_dir / "runs" / f"{fault}-seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite reliability run {run_dir}")
    run_dir.mkdir(parents=True)
    marker = run_dir / "fault-ready.json"
    server = HeartbeatServer() if fault == "network_interruption" else None
    if server is not None:
        server.start()
    port = server.port if server is not None else 0

    inject_command = _worker_command(
        args,
        run_dir=run_dir,
        fault=fault,
        seed=seed,
        phase="inject",
        heartbeat_port=port,
    )
    process = subprocess.Popen(inject_command)
    try:
        marker_data = _wait_for(marker, args.coordination_timeout)
        if fault == "controller_restart":
            os.kill(process.pid, signal.SIGKILL)
        elif fault == "network_interruption":
            assert server is not None
            server.stop()
            (run_dir / "inject-network-fault").touch()
        exit_code = process.wait(timeout=args.coordination_timeout)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if server is not None:
            server.stop()
    expected_code = {
        "worker_exit": WORKER_EXIT_CODE,
        "controller_restart": -signal.SIGKILL,
        "network_interruption": NETWORK_EXIT_CODE,
    }[fault]
    if exit_code != expected_code:
        raise RuntimeError(
            f"{fault} did not produce expected exit {expected_code}; got {exit_code}"
        )

    recovery_started = time.monotonic()
    resume_command = _worker_command(
        args,
        run_dir=run_dir,
        fault=fault,
        seed=seed,
        phase="resume",
        heartbeat_port=0,
    )
    subprocess.run(resume_command, check=True, timeout=args.coordination_timeout)
    recovery_seconds = time.monotonic() - recovery_started
    completed = json.loads((run_dir / "completed.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    applied_steps = list(map(int, checkpoint["applied_steps"]))
    expected_steps = list(range(1, args.final_step + 1))
    duplicate_updates = len(applied_steps) - len(set(applied_steps))
    if applied_steps != expected_steps or completed["applied_steps"] != expected_steps:
        raise RuntimeError(f"{fault} recovery produced an invalid update ledger")
    if process.poll() is None or (server is not None and server.alive):
        raise RuntimeError(f"{fault} left a child process or control-plane socket")

    checkpoint_step = int(marker_data["checkpoint_step"])
    hardware = (
        {
            "accelerator": torch.cuda.get_device_name(0),
            "cuda": str(torch.version.cuda),
        }
        if args.device == "cuda"
        else {"accelerator": "CPU", "cuda": "none"}
    )
    return {
        "schema_version": 1,
        "measured": True,
        "run_id": f"{fault}-seed{seed}",
        "framework_version": __version__,
        "harness_commit": harness_commit,
        "protocol": PROTOCOL,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(inject_command) + " && " + " ".join(resume_command),
        "hardware": hardware,
        "fault": {
            "type": fault,
            "injected_at_step": args.injected_at_step,
            "observed_exit_code": exit_code,
        },
        "recovery": {
            "resumed": True,
            "completed": True,
            "checkpoint_step": checkpoint_step,
            "resumed_step": checkpoint_step,
            "duplicate_updates": duplicate_updates,
            "data_loss_steps": args.injected_at_step - checkpoint_step,
            "final_step": int(checkpoint["step"]),
            "expected_final_step": args.final_step,
            "recovery_seconds": recovery_seconds,
            "resources_remaining": 0,
            "cleanup_scope": "child processes and TCP control-plane sockets",
        },
        "artifact_sha256": _sha256(checkpoint_path),
    }


def run_matrix(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reliability matrix requested without CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir()
    harness_commit = args.harness_commit or _commit()
    for fault in FAULTS:
        for seed in args.seeds:
            evidence = run_fault(args, fault, seed, harness_commit)
            path = evidence_dir / f"{fault}-seed{seed}.json"
            path.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    runs = load_runs([evidence_dir])
    validate_matrix(
        runs,
        min_seeds=len(args.seeds),
        max_data_loss_steps=args.max_data_loss_steps,
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(summarize(runs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--harness-commit")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--injected-at-step", type=int, default=7)
    parser.add_argument("--final-step", type=int, default=12)
    parser.add_argument("--max-data-loss-steps", type=int, default=0)
    parser.add_argument("--coordination-timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run_matrix(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
