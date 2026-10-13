"""Shared in-process fakes for the distributed async benchmark tests."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"


def load_bench(name: str) -> Any:
    """Load a benchmark script as the top-level module the drivers import."""
    if name in sys.modules:
        return sys.modules[name]
    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workload = load_bench("distributed_async_workload")


def small_config(**overrides: Any) -> Any:
    raw: dict[str, Any] = {
        "max_policy_lag": 1,
        "queue_capacity": 64,
        "batch_size": 4,
        "checkpoint_interval_updates": 3,
        "scenario_duration_seconds": {
            "steady_state_soak": 6,
            "worker_exit": 3,
            "controller_restart": 3,
            "network_interruption": 3,
        },
        "steps_per_rollout": 4,
        "observation_dim": 6,
        "action_count": 4,
        "hidden_dim": 8,
        "learning_rate": 0.05,
        "rollout_interval_seconds": 0.0,
        "heartbeat_interval_seconds": 0.0001,
        "lease_ttl_seconds": 30,
        "fault_at_seconds": 1,
        "network_partition_seconds": 0.5,
        "shutdown_grace_seconds": 1,
    }
    raw.update(overrides)
    return workload.parse_config(raw)


class LoopThread:
    """A persistent event loop on a daemon thread (control planes bind loops)."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coroutine: Coroutine[Any, Any, Any], timeout: float = 30) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


class CoreTransport:
    """Worker transport that calls a ControllerCore directly (no HTTP).

    Status codes mirror ``stateset_agents.api.routers.distributed_rollouts``.
    ``partitioned`` makes every call fail like a dropped link.
    """

    def __init__(self, loop: LoopThread, core: Any) -> None:
        self.loop = loop
        self.core = core
        self.partitioned = False

    def _response(self, status: int, body: Any) -> Any:
        return workload.TransportResponse(status, body, self.core.epoch)

    def _check(self) -> None:
        if self.partitioned:
            raise workload.TransportError("partitioned")

    def _lease_body(self, lease: Any) -> dict[str, Any]:
        artifact = self.loop.run(
            self.core.control.policy_artifact(lease.policy_version)
        )
        body = lease.to_dict()
        body["artifact"] = artifact.to_dict() if artifact is not None else None
        return body

    def register(self, worker_id: str) -> Any:
        self._check()
        lease = self.loop.run(self.core.control.register(worker_id))
        return self._response(200, self._lease_body(lease))

    def heartbeat(self, worker_id: str, lease_id: str) -> Any:
        from stateset_agents.training.distributed_rollouts import (
            WorkerLeaseError,
            WorkerLeaseExpired,
        )

        self._check()
        try:
            lease = self.loop.run(self.core.control.heartbeat(worker_id, lease_id))
        except WorkerLeaseExpired as exc:
            return self._response(410, {"detail": str(exc)})
        except WorkerLeaseError as exc:
            return self._response(409, {"detail": str(exc)})
        return self._response(200, self._lease_body(lease))

    def submit(self, worker_id: str, lease_id: str, record: Any, timeout: float) -> Any:
        from stateset_agents.training.async_rollouts import (
            AsyncRolloutClosed,
            AsyncRolloutError,
        )
        from stateset_agents.training.distributed_rollouts import WorkerLeaseError
        from stateset_agents.training.policy_artifacts import PolicyArtifactUnavailable

        self._check()
        try:
            accepted = self.loop.run(
                self.core.control.submit(
                    worker_id, lease_id, record, timeout_seconds=timeout
                )
            )
        except PolicyArtifactUnavailable as exc:
            return self._response(503, {"detail": str(exc)})
        except WorkerLeaseError as exc:
            return self._response(409, {"detail": str(exc)})
        except AsyncRolloutClosed as exc:
            return self._response(503, {"detail": str(exc)})
        except AsyncRolloutError as exc:
            return self._response(409, {"detail": str(exc)})
        return self._response(200, {"accepted": accepted})

    def unregister(self, worker_id: str, lease_id: str) -> Any:
        self._check()
        try:
            self.loop.run(self.core.control.unregister(worker_id, lease_id))
        except Exception:  # noqa: BLE001
            return self._response(409, {})
        return self._response(204, None)

    def status(self) -> Any:
        self._check()
        return self._response(
            200, {"phase": self.core.phase, "policy_version": self.core.policy_version}
        )

    def artifact(self, version: int) -> Any:
        self._check()
        path = self.core.store.artifact_path(version)
        if not path.exists():
            return self._response(404, {"detail": "missing"})
        return self._response(200, path.read_bytes())


SHA = {index: f"{index:064x}" for index in range(10)}


def write_synthetic_run(
    root: Path,
    *,
    scenario: str = "worker_exit",
    duration: float = 30.0,
    accelerator: str = "CPU",
) -> dict[str, Path]:
    """A minimal, internally consistent two-node run for audit/driver tests."""
    nodes = {"node-0": root / "node-0", "node-1": root / "node-1"}
    controller = nodes["node-0"] / "controller"
    controller.mkdir(parents=True)
    (controller / "initial.json").write_text(json.dumps({"version": 0, "sha": SHA[0]}))
    ledger = [
        {
            "u": 1,
            "from": 0,
            "to": 1,
            "sha": SHA[1],
            "ids": ["node-1-w0.g0.0", "node-0-w0.g0.0"],
            "lags": [0, 0],
            "nodes": {"node-1": 1, "node-0": 1},
            "t": 10.0,
            "epoch": "a",
            "metrics": {"reward_mean": 0.2},
        },
        {
            "u": 2,
            "from": 1,
            "to": 2,
            "sha": SHA[2],
            "ids": ["node-1-w0.g1.0", "node-0-w0.g0.1"],
            "lags": [1, 0],
            "nodes": {"node-1": 1, "node-0": 1},
            "t": 20.0,
            "epoch": "a",
            "metrics": {"reward_mean": 0.6},
        },
    ]
    (controller / "ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger)
    )
    (controller / "epochs.jsonl").write_text(
        json.dumps({"e": "start", "epoch": "a", "t": 1.0, "recovered": False}) + "\n"
    )
    (controller / "summary-a.json").write_text(
        json.dumps({"final_sha": SHA[2], "residual_queue_ids": [], "stopped_at": 31.0})
    )
    workers = {
        "node-0": {
            "node-0-w0": [
                {"e": "sync", "v": 0, "sha": SHA[0], "ms": 2.0, "t": 1.5},
                {"e": "gen", "id": "node-0-w0.g0.0", "v": 0, "t": 2.0},
                {"e": "sub", "id": "node-0-w0.g0.0", "v": 0, "r": "acc", "t": 2.1},
                {"e": "sync", "v": 1, "sha": SHA[1], "ms": 4.0, "t": 10.5},
                {"e": "gen", "id": "node-0-w0.g0.1", "v": 1, "t": 11.0},
                {"e": "sub", "id": "node-0-w0.g0.1", "v": 1, "r": "acc", "t": 11.1},
            ]
        },
        "node-1": {
            "node-1-w0": [
                {"e": "sync", "v": 0, "sha": SHA[0], "ms": 3.0, "t": 1.5},
                {"e": "gen", "id": "node-1-w0.g0.0", "v": 0, "t": 3.0},
                {"e": "sub", "id": "node-1-w0.g0.0", "v": 0, "r": "acc", "t": 3.1},
                {"e": "gen", "id": "node-1-w0.g1.0", "v": 1, "t": 13.0},
                {"e": "sub", "id": "node-1-w0.g1.0", "v": 1, "r": "acc", "t": 13.1},
            ]
        },
    }
    for node, by_worker in workers.items():
        directory = nodes[node] / "workers"
        directory.mkdir(parents=True)
        for worker, rows in by_worker.items():
            (directory / f"{worker}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
    events = {
        "worker_exit": [
            ("node-1", {"e": "fault_kill", "target": "node-1-w0", "t": 12.0})
        ],
        "controller_restart": [],
        "network_interruption": [
            ("node-0", {"e": "partition_start", "t": 11.5}),
            ("node-0", {"e": "partition_end", "t": 15.0}),
        ],
        "steady_state_soak": [],
    }[scenario]
    for node, directory in nodes.items():
        rows = [row for owner, row in events if owner == node]
        (directory / "node-events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (directory / "node-summary.json").write_text(
            json.dumps(
                {
                    "node_id": node,
                    "duration_seconds": duration,
                    "forced_stop": False,
                    "device": {
                        "accelerator": accelerator,
                        "accelerator_driver": "none (CPU-only PyTorch)",
                    },
                }
            )
        )
    return nodes
