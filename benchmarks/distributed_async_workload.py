#!/usr/bin/env python3
"""Node workload for the distributed asynchronous RL evidence matrix.

One process tree per *node*. ``node`` is the supervisor a provider driver
launches on every machine (or, for the local diagnostic driver, every process
group); it starts and restarts the roles below and injects the scenario fault
at the configured time:

``controller``
    The learner. It owns policy versions through the framework's
    :class:`~stateset_agents.training.distributed_rollouts.DistributedRolloutControlPlane`
    (leases, fencing, artifact-gated admission) wrapped around an
    :class:`~stateset_agents.training.async_rollouts.AsyncRolloutCoordinator`
    (bounded queue, ``max_policy_lag`` enforcement, deduplication). Workers
    reach it over the framework's authenticated HTTP transport
    (``stateset_agents.api.routers.distributed_rollouts``), served by uvicorn.
    Each optimizer update is committed to an fsync'd write-ahead log *before*
    the new policy version is published, and a full snapshot (weights,
    optimizer moments, control-plane state) is written every
    ``checkpoint_interval_updates``. A restarted controller loads the latest
    snapshot and deterministically replays the log, so no published update is
    lost and no trained rollout ID can be admitted twice.
``worker``
    An actor. It registers for a lease, pulls the content-addressed weights of
    its assigned policy version (verified with
    :func:`~stateset_agents.training.policy_artifacts.verify_policy_artifact`),
    samples rollouts with the tiny torch policy, and submits them tagged with
    the policy version and artifact digest they were sampled under. Admitted
    rollouts are retained until they are older than the lag window, and are
    resubmitted when the controller epoch changes (idempotent: trained IDs are
    rejected as duplicates by the restored coordinator).
``audit``
    Pure post-run accounting over every node's logs: exactly-once training,
    lost/duplicate optimizer updates, digest agreement between what workers
    loaded and what the controller committed, policy lag, weight-sync latency,
    recovery time, and learning progress.

Why not :class:`~stateset_agents.training.async_runtime.AsyncRolloutRuntime`?
It drives *in-process* producer coroutines. Here the producers are remote
processes, so the learner loop drives the same coordinator directly, keeping
the runtime's publish-before-expose order (commit, then
``publish_policy_artifact``) and adding the durability the runtime does not
need in a single process.

The policy is a small MLP on a contextual-bandit task (pick the action the
hidden linear map ranks highest), trained with a group-normalized
policy-gradient update corrected by the framework's clipped importance
weights. It is CPU-friendly so a 12-hour soak runs on CPU pods; ``device``
selects ``cpu``/``cuda``/``auto``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import platform
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stateset_agents.training.async_rollouts import (  # noqa: E402
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    AsyncRolloutTimeout,
    RolloutRecord,
    compute_importance_weights,
)
from stateset_agents.training.distributed_rollouts import (  # noqa: E402
    DistributedRolloutConfig,
    DistributedRolloutControlPlane,
)
from stateset_agents.training.policy_artifacts import (  # noqa: E402
    PolicyArtifact,
    PolicyArtifactError,
    verify_policy_artifact,
)

SCENARIOS = (
    "steady_state_soak",
    "worker_exit",
    "controller_restart",
    "network_interruption",
)
ENV_WORKER_KEY = "STATESET_DA_WORKER_KEY"
EPOCH_HEADER = "x-stateset-controller-epoch"
API_PREFIX = "/api/v1/rollouts/workers"
_POLICY_MAGIC = b"SSDAPOL1"


class WorkloadError(RuntimeError):
    """Raised when the workload cannot run or account for a scenario honestly."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WorkloadConfig:
    """Every knob of the experiment; the manifest's hashed ``config``."""

    max_policy_lag: int
    queue_capacity: int
    batch_size: int
    checkpoint_interval_updates: int
    scenario_duration_seconds: Mapping[str, float]
    steps_per_rollout: int = 8
    observation_dim: int = 16
    action_count: int = 8
    hidden_dim: int = 64
    learning_rate: float = 0.01
    max_importance_weight: float = 2.0
    rollout_interval_seconds: float = 0.1
    heartbeat_interval_seconds: float = 0.5
    lease_ttl_seconds: float = 10.0
    request_timeout_seconds: float = 5.0
    deduplication_capacity: int = 2_000_000
    policy_artifact_capacity: int = 16
    device: str = "cpu"
    preflight_duration_seconds: float = 60.0
    fault_at_seconds: float = 60.0
    network_partition_seconds: float = 30.0
    worker_restart_delay_seconds: float = 2.0
    controller_restart_delay_seconds: float = 1.0
    shutdown_grace_seconds: float = 20.0
    max_cost_usd_per_scenario: Mapping[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REQUIRED_CONFIG = (
    "max_policy_lag",
    "queue_capacity",
    "batch_size",
    "checkpoint_interval_updates",
    "scenario_duration_seconds",
)


def parse_config(raw: Mapping[str, Any]) -> WorkloadConfig:
    """Validate the pinned config; unknown keys are rejected to catch typos."""
    if not isinstance(raw, Mapping):
        raise WorkloadError("config must be an object")
    known = {item.name for item in fields(WorkloadConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise WorkloadError(f"unknown config keys: {', '.join(unknown)}")
    missing = [key for key in _REQUIRED_CONFIG if key not in raw]
    if missing:
        raise WorkloadError(f"config is missing: {', '.join(missing)}")
    config = WorkloadConfig(**dict(raw))
    for name in (
        "queue_capacity",
        "batch_size",
        "checkpoint_interval_updates",
        "steps_per_rollout",
        "observation_dim",
        "action_count",
        "hidden_dim",
        "deduplication_capacity",
        "policy_artifact_capacity",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise WorkloadError(f"config.{name} must be a positive integer")
    lag = config.max_policy_lag
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise WorkloadError("config.max_policy_lag must be a non-negative integer")
    if config.batch_size > config.queue_capacity:
        raise WorkloadError("config.batch_size must not exceed queue_capacity")
    if config.action_count < 2:
        raise WorkloadError("config.action_count must be at least 2")
    for name in (
        "learning_rate",
        "rollout_interval_seconds",
        "heartbeat_interval_seconds",
        "lease_ttl_seconds",
        "request_timeout_seconds",
        "preflight_duration_seconds",
        "fault_at_seconds",
        "network_partition_seconds",
        "worker_restart_delay_seconds",
        "controller_restart_delay_seconds",
        "shutdown_grace_seconds",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise WorkloadError(f"config.{name} must be finite and non-negative")
    if config.max_importance_weight < 1:
        raise WorkloadError("config.max_importance_weight must be >= 1")
    if config.lease_ttl_seconds <= config.heartbeat_interval_seconds:
        raise WorkloadError("config.lease_ttl_seconds must exceed the heartbeat")
    if config.device not in {"cpu", "cuda", "auto"}:
        raise WorkloadError("config.device must be cpu, cuda, or auto")
    durations = config.scenario_duration_seconds
    if not isinstance(durations, Mapping) or set(durations) != set(SCENARIOS):
        raise WorkloadError(
            "config.scenario_duration_seconds must cover every scenario"
        )
    for scenario, value in durations.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise WorkloadError(f"scenario duration for {scenario} must be positive")
    caps = config.max_cost_usd_per_scenario
    if caps is not None and (
        not isinstance(caps, Mapping)
        or set(caps) != set(SCENARIOS)
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
            for v in caps.values()
        )
    ):
        raise WorkloadError(
            "config.max_cost_usd_per_scenario must price every scenario"
        )
    return config


@dataclass(frozen=True)
class ScenarioPlan:
    """Concrete timings for one scenario run, relative to node start."""

    scenario: str
    mode: str
    duration_seconds: float
    fault_at_seconds: float | None
    partition_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scenario_plan(config: WorkloadConfig, scenario: str, mode: str) -> ScenarioPlan:
    """Durations and fault time; preflight shortens both, never the shape."""
    if scenario not in SCENARIOS:
        raise WorkloadError(f"unsupported scenario {scenario!r}")
    if mode not in {"measured", "preflight"}:
        raise WorkloadError("mode must be measured or preflight")
    duration = float(config.scenario_duration_seconds[scenario])
    if mode == "preflight":
        duration = min(duration, float(config.preflight_duration_seconds))
    fault_at: float | None = None
    partition = 0.0
    if scenario != "steady_state_soak":
        fault_at = min(float(config.fault_at_seconds), duration / 3.0)
        if scenario == "network_interruption":
            partition = min(
                float(config.network_partition_seconds), duration - fault_at - 1.0
            )
            if partition <= 0:
                raise WorkloadError("scenario too short for the network partition")
    return ScenarioPlan(scenario, mode, duration, fault_at, partition)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_pinned_config(config_json: str, config_sha256: str) -> dict[str, Any]:
    """Decode the runner's canonical config and verify its digest."""
    digest = hashlib.sha256(config_json.encode()).hexdigest()
    if digest != config_sha256:
        raise WorkloadError("config_sha256 does not match --config-json")
    try:
        value = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise WorkloadError("config JSON is invalid") from exc
    if not isinstance(value, dict) or canonical_json(value) != config_json:
        raise WorkloadError("config JSON must be canonical")
    return value


# --------------------------------------------------------------------------- #
# Policy, task, and learner (torch is imported lazily)
# --------------------------------------------------------------------------- #


def _torch() -> Any:
    import torch

    return torch


def resolve_device(requested: str) -> str:
    torch = _torch()
    if requested == "cuda" and not torch.cuda.is_available():
        raise WorkloadError("config.device=cuda but no CUDA device is available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def describe_device(device: str) -> dict[str, str]:
    """Honest accelerator description for topology checks."""
    torch = _torch()
    if device == "cuda":
        return {
            "device": "cuda",
            "accelerator": str(torch.cuda.get_device_name(0)),
            "accelerator_driver": f"CUDA {torch.version.cuda}",
        }
    return {
        "device": "cpu",
        "accelerator": "CPU",
        "accelerator_driver": "none (CPU-only PyTorch)",
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_count": str(os.cpu_count() or 0),
    }


def build_policy(config: WorkloadConfig, seed: int) -> Any:
    torch = _torch()
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(config.observation_dim, config.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_dim, config.action_count),
        )
    finally:
        torch.random.set_rng_state(generator_state)
    return model


def task_matrix(config: WorkloadConfig, seed: int) -> Any:
    torch = _torch()
    generator = torch.Generator().manual_seed(int(seed) * 7919 + 17)
    return torch.randn(config.action_count, config.observation_dim, generator=generator)


def observation_seed(seed: int, rollout_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{rollout_id}".encode()).hexdigest()
    return int(digest[:15], 16)


def observations(config: WorkloadConfig, obs_seed: int) -> Any:
    torch = _torch()
    generator = torch.Generator().manual_seed(int(obs_seed))
    return torch.randn(
        config.steps_per_rollout, config.observation_dim, generator=generator
    )


def serialize_tensors(tensors: Mapping[str, Any]) -> bytes:
    """Deterministic, pickle-free float32 encoding of named tensors."""
    header = []
    chunks = []
    for name in sorted(tensors):
        tensor = tensors[name].detach().to("cpu").float().contiguous()
        header.append({"name": name, "shape": list(tensor.shape)})
        chunks.append(tensor.numpy().astype("<f4").tobytes())
    encoded = canonical_json(header).encode()
    return _POLICY_MAGIC + struct.pack(">I", len(encoded)) + encoded + b"".join(chunks)


def deserialize_tensors(blob: bytes) -> dict[str, Any]:
    torch = _torch()
    import numpy as np

    if not blob.startswith(_POLICY_MAGIC) or len(blob) < len(_POLICY_MAGIC) + 4:
        raise WorkloadError("policy blob has an unknown format")
    offset = len(_POLICY_MAGIC)
    (length,) = struct.unpack(">I", blob[offset : offset + 4])
    offset += 4
    header = json.loads(blob[offset : offset + length].decode())
    offset += length
    result: dict[str, Any] = {}
    for item in header:
        count = int(np.prod(item["shape"])) if item["shape"] else 1
        size = count * 4
        if offset + size > len(blob):
            raise WorkloadError("policy blob is truncated")
        array = np.frombuffer(blob, dtype="<f4", count=count, offset=offset)
        result[item["name"]] = torch.from_numpy(array.copy()).reshape(item["shape"])
        offset += size
    if offset != len(blob):
        raise WorkloadError("policy blob has trailing bytes")
    return result


def sample_rollout(
    model: Any,
    config: WorkloadConfig,
    *,
    seed: int,
    task: Any,
    rollout_id: str,
    policy_version: int,
    artifact_sha256: str | None,
    worker_id: str,
    node_id: str,
    device: str = "cpu",
) -> RolloutRecord:
    """Sample one episode under ``model`` and tag it with its exact policy."""
    torch = _torch()
    obs_seed = observation_seed(seed, rollout_id)
    obs = observations(config, obs_seed)
    generator = torch.Generator().manual_seed(obs_seed ^ 0x5A5A5A5A)
    with torch.no_grad():
        logits = model(obs.to(device)).float().cpu()
        log_probs = torch.log_softmax(logits, dim=-1)
        actions = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1)
        chosen = log_probs.gather(-1, actions[:, None]).squeeze(-1)
    targets = (obs @ task.T).argmax(dim=-1)
    rewards = (actions == targets).float()
    return RolloutRecord(
        rollout_id=rollout_id,
        policy_version=int(policy_version),
        sampler_log_probs=tuple(float(value) for value in chosen.tolist()),
        payload={
            "obs_seed": obs_seed,
            "actions": [int(value) for value in actions.tolist()],
            "rewards": [float(value) for value in rewards.tolist()],
            "worker": worker_id,
            "node": node_id,
        },
        policy_artifact_sha256=artifact_sha256,
    )


class Learner:
    """Policy + Adam with an importance-corrected group-normalized PG update."""

    def __init__(self, config: WorkloadConfig, seed: int, device: str = "cpu") -> None:
        torch = _torch()
        self.config = config
        self.seed = int(seed)
        self.device = device
        self.model = build_policy(config, seed).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(config.learning_rate)
        )

    def weights_blob(self) -> bytes:
        return serialize_tensors(self.model.state_dict())

    def load_weights_blob(self, blob: bytes) -> None:
        state = deserialize_tensors(blob)
        self.model.load_state_dict({k: v.to(self.device) for k, v in state.items()})

    def optimizer_tensors(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Adam moments as named tensors plus JSON-safe scalar state."""
        state = self.optimizer.state_dict()
        tensors: dict[str, Any] = {}
        scalars: dict[str, Any] = {"param_groups": state["param_groups"], "steps": {}}
        for index, values in state["state"].items():
            for key, value in values.items():
                if key == "step":
                    scalars["steps"][str(index)] = float(value)
                else:
                    tensors[f"{index}.{key}"] = value
        return tensors, scalars

    def load_optimizer(
        self, tensors: Mapping[str, Any], scalars: Mapping[str, Any]
    ) -> None:
        torch = _torch()
        state: dict[int, dict[str, Any]] = {}
        for name, value in tensors.items():
            index, key = name.split(".", 1)
            state.setdefault(int(index), {})[key] = value.to(self.device)
        for index, step in scalars["steps"].items():
            state.setdefault(int(index), {})["step"] = torch.tensor(float(step))
        self.optimizer.load_state_dict(
            {"state": state, "param_groups": list(scalars["param_groups"])}
        )

    def update(self, records: Sequence[RolloutRecord]) -> dict[str, float]:
        torch = _torch()
        if not records:
            raise WorkloadError("cannot update on an empty batch")
        obs = torch.stack(
            [observations(self.config, int(r.payload["obs_seed"])) for r in records]
        ).to(self.device)
        actions = torch.tensor([list(r.payload["actions"]) for r in records]).to(
            self.device
        )
        rewards = torch.tensor(
            [list(r.payload["rewards"]) for r in records], dtype=torch.float32
        ).to(self.device)
        logits = self.model(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(-1, actions[..., None]).squeeze(-1)
        learner_lp = chosen.detach().cpu().tolist()
        weights = torch.tensor(
            [
                compute_importance_weights(
                    learner_lp[i],
                    record.sampler_log_probs,
                    max_weight=float(self.config.max_importance_weight),
                )
                for i, record in enumerate(records)
            ],
            dtype=torch.float32,
        ).to(self.device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        loss = -(weights * advantages * chosen).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "reward_mean": float(rewards.mean().cpu()),
            "importance_weight_mean": float(weights.mean().cpu()),
        }


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Durable controller state: artifacts, write-ahead log, snapshots, ledger
# --------------------------------------------------------------------------- #


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_dir(path.parent)


def repair_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, truncating a torn final line left by a hard kill."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    good = 0
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            rows.append(value)
        good += len(line)
    if good != len(raw):
        with path.open("r+b") as handle:
            handle.truncate(good)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant reader for audit: skips a torn or corrupt line."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


class JsonlLog:
    """Append-only line log; ``durable`` lines are fsync'd before returning."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def write(self, row: Mapping[str, Any], *, durable: bool = False) -> None:
        self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._handle.flush()
        if durable:
            os.fsync(self._handle.fileno())

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def close(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            self._handle.close()


class ControllerStore:
    """Filesystem layout of one controller's durable state."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts = root / "artifacts"
        self.snapshots = root / "snapshots"
        self.wal_dir = root / "wal"
        for path in (self.artifacts, self.snapshots, self.wal_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.ledger_path = root / "ledger.jsonl"
        self.evictions_path = root / "evictions.jsonl"
        self.epochs_path = root / "epochs.jsonl"
        self.initial_path = root / "initial.json"
        self.pointer = self.snapshots / "LATEST"

    def artifact_path(self, version: int) -> Path:
        return self.artifacts / f"policy-v{version:09d}.bin"

    def wal_path(self, snapshot_update: int) -> Path:
        return self.wal_dir / f"segment-{snapshot_update:09d}.jsonl"

    def write_artifact(self, version: int, blob: bytes) -> PolicyArtifact:
        path = self.artifact_path(version)
        atomic_write_bytes(path, blob)
        return PolicyArtifact(
            policy_version=int(version),
            uri=path.resolve().as_uri(),
            sha256=sha256_bytes(blob),
            size_bytes=len(blob),
            published_at=time.time(),
        )

    def prune_artifacts(self, keep_from_version: int) -> None:
        for path in self.artifacts.glob("policy-v*.bin"):
            try:
                version = int(path.stem.split("-v", 1)[1])
            except (IndexError, ValueError):
                continue
            if version < keep_from_version:
                path.unlink(missing_ok=True)

    def latest_snapshot(self) -> int | None:
        if not self.pointer.exists():
            return None
        return int(self.pointer.read_text(encoding="utf-8").strip())

    def write_snapshot(
        self,
        update_index: int,
        *,
        model_blob: bytes,
        optimizer_blob: bytes,
        meta: Mapping[str, Any],
    ) -> None:
        directory = self.snapshots / f"snap-{update_index:09d}"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(directory / "model.bin", model_blob)
        atomic_write_bytes(directory / "optimizer.bin", optimizer_blob)
        atomic_write_bytes(
            directory / "meta.json", (canonical_json(dict(meta)) + "\n").encode()
        )
        atomic_write_bytes(self.pointer, f"{update_index}\n".encode())
        for other in self.snapshots.glob("snap-*"):
            if other != directory:
                for child in other.iterdir():
                    child.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    other.rmdir()
        for segment in self.wal_dir.glob("segment-*.jsonl"):
            if segment != self.wal_path(update_index):
                segment.unlink(missing_ok=True)

    def read_snapshot(self, update_index: int) -> tuple[bytes, bytes, dict[str, Any]]:
        directory = self.snapshots / f"snap-{update_index:09d}"
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return (
            (directory / "model.bin").read_bytes(),
            (directory / "optimizer.bin").read_bytes(),
            meta,
        )


# --------------------------------------------------------------------------- #
# Controller core (transport-free; the HTTP layer wraps it)
# --------------------------------------------------------------------------- #


class ControllerCore:
    """Learner + control plane + durability. Safe to drop at any await point."""

    def __init__(
        self,
        config: WorkloadConfig,
        *,
        seed: int,
        state_dir: Path,
        device: str = "cpu",
        epoch: str | None = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.device = device
        self.epoch = epoch or uuid.uuid4().hex[:12]
        self.store = ControllerStore(state_dir)
        self.learner = Learner(config, seed, device)
        self.control: DistributedRolloutControlPlane | None = None
        self.update_index = 0
        self.policy_version = 0
        self.current_sha = ""
        self.phase = "starting"
        self.recovery: dict[str, Any] = {}
        self._wal: JsonlLog | None = None
        self._ledger: JsonlLog | None = None
        self._evictions: JsonlLog | None = None
        self.replay_digest_mismatches = 0
        self._snapshot_base = 0

    # -- construction --------------------------------------------------------

    def _coordinator_config(self) -> AsyncRolloutConfig:
        return AsyncRolloutConfig(
            queue_capacity=self.config.queue_capacity,
            min_batch_size=self.config.batch_size,
            max_batch_size=self.config.batch_size,
            max_policy_lag=self.config.max_policy_lag,
            max_importance_weight=float(self.config.max_importance_weight),
            deduplication_capacity=self.config.deduplication_capacity,
        )

    def _control_config(self) -> DistributedRolloutConfig:
        return DistributedRolloutConfig(
            lease_ttl_seconds=float(self.config.lease_ttl_seconds),
            max_workers=256,
            policy_artifact_capacity=self.config.policy_artifact_capacity,
            require_policy_artifact=True,
        )

    def _on_evict(self, record: RolloutRecord, lag: int) -> None:
        if self._evictions is not None:
            self._evictions.write(
                {
                    "id": record.rollout_id,
                    "v": record.policy_version,
                    "lag": lag,
                    "epoch": self.epoch,
                    "t": time.time(),
                }
            )

    async def start(self) -> None:
        """Initialize fresh state or recover from snapshot + write-ahead log."""
        self._evictions = JsonlLog(self.store.evictions_path)
        started = time.time()
        latest = self.store.latest_snapshot()
        if latest is None:
            await self._initialize()
            replayed = 0
        else:
            replayed = await self._recover(latest)
        assert self.control is not None
        self.control.coordinator.add_eviction_listener(self._on_evict)
        self._wal = JsonlLog(self.store.wal_path(self._snapshot_base))
        self._ledger = JsonlLog(self.store.ledger_path)
        self.recovery = {
            "e": "start",
            "epoch": self.epoch,
            "pid": os.getpid(),
            "t": started,
            "ready_t": time.time(),
            "recovered": latest is not None,
            "snapshot_update": latest,
            "replayed_updates": replayed,
            "replay_digest_mismatches": self.replay_digest_mismatches,
            "resumed_update": self.update_index,
            "resumed_version": self.policy_version,
            "resumed_sha": self.current_sha,
        }
        epochs = JsonlLog(self.store.epochs_path)
        epochs.write(self.recovery, durable=True)
        epochs.close()
        self.phase = "running"

    async def _initialize(self) -> None:
        blob = self.learner.weights_blob()
        artifact = self.store.write_artifact(0, blob)
        atomic_write_bytes(
            self.store.initial_path,
            (canonical_json({"version": 0, "sha": artifact.sha256}) + "\n").encode(),
        )
        coordinator = AsyncRolloutCoordinator(self._coordinator_config())
        self.control = DistributedRolloutControlPlane(
            coordinator=coordinator, config=self._control_config()
        )
        await self.control.register_initial_policy_artifact(artifact)
        self.current_sha = artifact.sha256
        self._snapshot_base = 0
        await self._write_snapshot(0)

    async def _recover(self, snapshot_update: int) -> int:
        model_blob, optimizer_blob, meta = self.store.read_snapshot(snapshot_update)
        self.learner.load_weights_blob(model_blob)
        self.learner.load_optimizer(
            deserialize_tensors(optimizer_blob), meta["optimizer_scalars"]
        )
        state = meta["control"]
        update_index = int(meta["update_index"])
        version = int(meta["policy_version"])
        sha = str(meta["sha"])
        entries = repair_jsonl(self.store.wal_path(snapshot_update))
        trained: list[str] = []
        artifacts = list(state["policy_artifacts"])
        replayed = 0
        for entry in entries:
            if int(entry["u"]) <= update_index:
                continue
            if int(entry["u"]) != update_index + 1 or int(entry["from"]) != version:
                raise WorkloadError("write-ahead log is not contiguous with snapshot")
            records = [RolloutRecord.from_dict(value) for value in entry["records"]]
            self.learner.update(records)
            descriptor = PolicyArtifact.from_dict(entry["artifact"])
            blob = self.learner.weights_blob()
            if sha256_bytes(blob) != descriptor.sha256:
                # Workers may already hold the committed weights: adopt them so
                # the published version keeps exactly one digest.
                self.replay_digest_mismatches += 1
                path = self.store.artifact_path(descriptor.policy_version)
                verify_policy_artifact(path, descriptor)
                self.learner.load_weights_blob(path.read_bytes())
            elif not self.store.artifact_path(descriptor.policy_version).exists():
                atomic_write_bytes(
                    self.store.artifact_path(descriptor.policy_version), blob
                )
            trained.extend(record.rollout_id for record in records)
            artifacts.append(descriptor.to_dict())
            update_index = int(entry["u"])
            version = int(entry["to"])
            sha = descriptor.sha256
            replayed += 1
        self.update_index = update_index
        self.policy_version = version
        self.current_sha = sha

        coordinator_state = dict(state["coordinator"])
        trained_set = set(trained)
        lag_bound = self.config.max_policy_lag
        evictions = JsonlLog(self.store.evictions_path)
        queue = []
        for value in coordinator_state["queue"]:
            if value["rollout_id"] in trained_set:
                continue
            lag = version - int(value["policy_version"])
            if lag > lag_bound:
                evictions.write(
                    {
                        "id": value["rollout_id"],
                        "v": value["policy_version"],
                        "lag": lag,
                        "epoch": self.epoch,
                        "t": time.time(),
                        "on_restore": True,
                    }
                )
                continue
            queue.append(value)
        evictions.close()
        seen = set(coordinator_state["seen_rollout_ids"]) | trained_set
        counters = dict(coordinator_state["counters"])
        counters["submitted"] = max(int(counters["submitted"]), len(seen))
        counters["consumed"] = int(counters["consumed"]) + len(trained)
        coordinator_state.update(
            {
                "current_policy_version": version,
                "queue": queue,
                "seen_rollout_ids": sorted(seen),
                "counters": counters,
                "closed": False,
            }
        )
        by_version = {int(item["policy_version"]): item for item in artifacts}
        keep = sorted(by_version)[-self.config.policy_artifact_capacity :]
        control_state = dict(state)
        control_state["coordinator"] = coordinator_state
        control_state["policy_artifacts"] = [by_version[v] for v in keep]
        self.control = await DistributedRolloutControlPlane.from_state_dict(
            control_state
        )
        self._snapshot_base = snapshot_update
        # The ledger is derived from the WAL; restore lines lost to a kill.
        ledger_rows = repair_jsonl(self.store.ledger_path)
        recorded = {int(row["u"]) for row in ledger_rows}
        missing = [e for e in entries if int(e["u"]) not in recorded]
        if missing:
            ledger = JsonlLog(self.store.ledger_path)
            for entry in missing:
                ledger.write(self._ledger_row(entry), durable=True)
            ledger.close()
        return replayed

    # -- learner loop --------------------------------------------------------

    @staticmethod
    def _ledger_row(entry: Mapping[str, Any]) -> dict[str, Any]:
        nodes: dict[str, int] = {}
        for record in entry["records"]:
            node = str(record["payload"].get("node", "?"))
            nodes[node] = nodes.get(node, 0) + 1
        return {
            "u": entry["u"],
            "from": entry["from"],
            "to": entry["to"],
            "sha": entry["artifact"]["sha256"],
            "ids": [record["rollout_id"] for record in entry["records"]],
            "lags": entry["lags"],
            "nodes": nodes,
            "t": entry["t"],
            "epoch": entry["epoch"],
            "metrics": entry["metrics"],
        }

    async def step(self, timeout_seconds: float = 0.5) -> dict[str, Any] | None:
        """Run at most one committed optimizer update."""
        assert self.control is not None and self._wal is not None
        coordinator = self.control.coordinator
        try:
            batch = await coordinator.next_batch(timeout_seconds=timeout_seconds)
        except AsyncRolloutTimeout:
            return None
        metrics = self.learner.update(batch.records)
        target = batch.learner_policy_version + 1
        blob = self.learner.weights_blob()
        artifact = self.store.write_artifact(target, blob)
        entry = {
            "u": self.update_index + 1,
            "from": batch.learner_policy_version,
            "to": target,
            "artifact": artifact.to_dict(),
            "records": [record.to_dict() for record in batch.records],
            "lags": list(batch.policy_lags),
            "t": time.time(),
            "epoch": self.epoch,
            "metrics": metrics,
        }
        # Commit point: the update is durable before any worker can see it.
        self._wal.write(entry, durable=True)
        assert self._ledger is not None
        self._ledger.write(self._ledger_row(entry))
        await self.control.publish_policy_artifact(artifact)
        self.update_index += 1
        self.policy_version = target
        self.current_sha = artifact.sha256
        if self.update_index % self.config.checkpoint_interval_updates == 0:
            await self._write_snapshot(self.update_index)
        return entry

    async def _write_snapshot(self, update_index: int) -> None:
        assert self.control is not None
        control_state = await self.control.state_dict()
        model_blob = self.learner.weights_blob()
        tensors, scalars = self.learner.optimizer_tensors()
        optimizer_blob = serialize_tensors(tensors)
        meta = {
            "update_index": update_index,
            "policy_version": self.policy_version,
            "sha": self.current_sha,
            "optimizer_scalars": scalars,
            "control": control_state,
            "epoch": self.epoch,
            "t": time.time(),
        }
        await asyncio.to_thread(
            self.store.write_snapshot,
            update_index,
            model_blob=model_blob,
            optimizer_blob=optimizer_blob,
            meta=meta,
        )
        if self._wal is not None:
            self._wal.close()
            self._wal = JsonlLog(self.store.wal_path(update_index))
        self._snapshot_base = update_index
        self.store.prune_artifacts(
            self.policy_version - self.config.policy_artifact_capacity
        )

    async def stop(self) -> dict[str, Any]:
        """Close admission, record residual queue, and write a final summary."""
        assert self.control is not None
        self.phase = "stopping"
        await self.control.coordinator.close()
        state = await self.control.state_dict()
        stats = await self.control.stats()
        summary = {
            "epoch": self.epoch,
            "final": True,
            "update_index": self.update_index,
            "policy_version": self.policy_version,
            "final_sha": sha256_bytes(self.learner.weights_blob()),
            "committed_sha": self.current_sha,
            "residual_queue_ids": [
                r["rollout_id"] for r in state["coordinator"]["queue"]
            ],
            "coordinator_stats": self.control.coordinator.stats().to_dict(),
            "control_stats": stats.to_dict(),
            "stopped_at": time.time(),
        }
        atomic_write_bytes(
            self.store.root / f"summary-{self.epoch}.json",
            (json.dumps(summary, indent=2) + "\n").encode(),
        )
        for log in (self._wal, self._ledger, self._evictions):
            if log is not None:
                log.close()
        return summary


# --------------------------------------------------------------------------- #
# Controller process: framework HTTP transport + bench endpoints
# --------------------------------------------------------------------------- #


def configure_worker_auth(worker_key: str) -> None:
    """Point the framework API config at the per-run worker credential."""
    if len(worker_key) < 16:
        raise WorkloadError(f"{ENV_WORKER_KEY} must be at least 16 characters")
    os.environ["API_KEYS"] = f"{worker_key}:rollout_worker"
    os.environ["API_REQUIRE_AUTH"] = "false"
    from stateset_agents.api import config as api_config

    api_config.reload_config()


def build_controller_app(core: ControllerCore) -> Any:
    from fastapi import Depends, FastAPI, HTTPException, Request, Response

    from stateset_agents.api import attach_distributed_rollout_control_plane
    from stateset_agents.api.routers.distributed_rollouts import (
        require_rollout_worker,
        router,
    )

    app = FastAPI(title="stateset-distributed-async-controller")
    assert core.control is not None
    attach_distributed_rollout_control_plane(app, core.control)
    app.include_router(router)

    @app.middleware("http")
    async def _epoch_header(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers[EPOCH_HEADER] = core.epoch
        return response

    @app.get("/bench/v1/status")
    async def _status(_user: Any = Depends(require_rollout_worker)) -> dict[str, Any]:
        return {
            "phase": core.phase,
            "epoch": core.epoch,
            "policy_version": core.policy_version,
            "update_index": core.update_index,
        }

    @app.get("/bench/v1/artifacts/{version}")
    async def _artifact(
        version: int, _user: Any = Depends(require_rollout_worker)
    ) -> Response:
        path = core.store.artifact_path(version)
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise HTTPException(
                status_code=404, detail="artifact not retained"
            ) from exc
        return Response(content=data, media_type="application/octet-stream")

    return app


async def run_controller_async(args: argparse.Namespace) -> int:
    import uvicorn

    config = parse_config(json.loads(Path(args.config_file).read_text("utf-8")))
    torch = _torch()
    torch.set_num_threads(1)
    worker_key = os.environ.get(ENV_WORKER_KEY, "")
    configure_worker_auth(worker_key)
    device = resolve_device(config.device)
    core = ControllerCore(
        config, seed=args.seed, state_dir=Path(args.state_dir), device=device
    )
    await core.start()
    app = build_controller_app(core)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            lifespan="off",
            access_log=False,
        )
    )
    # The supervisor owns signals: SIGKILL is the injected fault, SIGTERM cleanup.
    server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
    serve_task = asyncio.create_task(server.serve())
    deadline = float(args.deadline_unix)
    try:
        while time.time() < deadline:
            if serve_task.done():
                raise WorkloadError("controller HTTP server stopped unexpectedly")
            await core.step(timeout_seconds=min(0.5, max(0.05, deadline - time.time())))
        await core.stop()
        grace_end = time.time() + float(config.shutdown_grace_seconds)
        assert core.control is not None
        while time.time() < grace_end and await core.control.health():
            await asyncio.sleep(0.2)
        core.phase = "stopped"
        await asyncio.sleep(min(2.0, float(config.shutdown_grace_seconds)))
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(serve_task, timeout=10)
    return 0


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


class TransportError(RuntimeError):
    """Connection-level failure (refused, reset, timeout)."""


@dataclass
class TransportResponse:
    status: int
    body: Any
    epoch: str | None


class HttpTransport:
    """Synchronous client for the framework's rollout transport."""

    def __init__(self, base_url: str, worker_key: str, timeout: float) -> None:
        import httpx

        self._httpx = httpx
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": worker_key},
            timeout=timeout,
        )

    def _call(self, method: str, path: str, **kwargs: Any) -> TransportResponse:
        try:
            response = self.client.request(method, path, **kwargs)
        except self._httpx.HTTPError as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc
        content_type = response.headers.get("content-type", "")
        body: Any = response.content
        if "json" in content_type:
            try:
                body = response.json()
            except ValueError:
                body = None
        return TransportResponse(
            response.status_code, body, response.headers.get(EPOCH_HEADER)
        )

    def register(self, worker_id: str) -> TransportResponse:
        return self._call("POST", f"{API_PREFIX}/{worker_id}/register")

    def heartbeat(self, worker_id: str, lease_id: str) -> TransportResponse:
        return self._call(
            "POST", f"{API_PREFIX}/{worker_id}/heartbeat", json={"lease_id": lease_id}
        )

    def submit(
        self, worker_id: str, lease_id: str, record: RolloutRecord, timeout: float
    ) -> TransportResponse:
        return self._call(
            "POST",
            f"{API_PREFIX}/{worker_id}/submit",
            json={
                "lease_id": lease_id,
                "rollout": record.to_dict(),
                "timeout_seconds": timeout,
            },
        )

    def unregister(self, worker_id: str, lease_id: str) -> TransportResponse:
        return self._call(
            "DELETE", f"{API_PREFIX}/{worker_id}", json={"lease_id": lease_id}
        )

    def status(self) -> TransportResponse:
        return self._call("GET", "/bench/v1/status")

    def artifact(self, version: int) -> TransportResponse:
        return self._call("GET", f"/bench/v1/artifacts/{version}")

    def close(self) -> None:
        self.client.close()


class RolloutWorker:
    """Actor loop: lease, weight sync, sample, submit, retry, resubmit."""

    def __init__(
        self,
        config: WorkloadConfig,
        *,
        seed: int,
        worker_id: str,
        node_id: str,
        incarnation: int,
        transport: Any,
        log_path: Path,
        cache_dir: Path,
        deadline_unix: float,
        hard_deadline_unix: float,
        device: str = "cpu",
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.worker_id = worker_id
        self.node_id = node_id
        self.incarnation = int(incarnation)
        self.transport = transport
        self.log = JsonlLog(log_path)
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.deadline = float(deadline_unix)
        self.hard_deadline = float(hard_deadline_unix)
        self.device = device
        self.clock = clock
        self.sleep = sleep
        self.model = build_policy(config, seed).to(device)
        self.task = task_matrix(config, seed)
        self.lease_id: str | None = None
        self.assigned_version: int | None = None
        self.assigned_artifact: dict[str, Any] | None = None
        self.loaded_version: int | None = None
        self.loaded_sha: str | None = None
        self.epoch: str | None = None
        self.sequence = 0
        self.pending: RolloutRecord | None = None
        self.pending_resubmit = False
        self.resubmit_queue: deque[RolloutRecord] = deque()
        self.retained: deque[RolloutRecord] = deque()
        self.last_heartbeat = -math.inf
        self.unreachable_since: float | None = None
        self.stop_reason: str | None = None
        self.counts = {"generated": 0, "accepted": 0, "rejected": 0, "errors": 0}

    # -- helpers -------------------------------------------------------------

    def _event(self, name: str, **values: Any) -> None:
        self.log.write({"e": name, "t": self.clock(), **values})

    def _observe(self, response: TransportResponse) -> None:
        self.unreachable_since = None
        epoch = response.epoch
        if epoch is None:
            return
        if self.epoch is not None and epoch != self.epoch:
            self._event("epoch", old=self.epoch, new=epoch)
            # A restarted controller lost its volatile queue: fence the old
            # lease and resubmit every admitted rollout still inside the lag
            # window. Duplicates of trained IDs are rejected by the restore.
            self.lease_id = None
            self.resubmit_queue.extend(self.retained)
        self.epoch = epoch

    def _unreachable(self, exc: Exception) -> None:
        now = self.clock()
        if self.unreachable_since is None:
            self.unreachable_since = now
        self.counts["errors"] += 1

    def _phase_is_stopping(self) -> bool:
        try:
            response = self.transport.status()
        except TransportError as exc:
            self._unreachable(exc)
            return False
        self._observe(response)
        return (
            response.status == 200
            and isinstance(response.body, Mapping)
            and response.body.get("phase") in {"stopping", "stopped"}
        )

    def _apply_lease(self, body: Mapping[str, Any]) -> None:
        self.lease_id = str(body["lease_id"])
        self.assigned_version = int(body["policy_version"])
        artifact = body.get("artifact")
        self.assigned_artifact = (
            dict(artifact) if isinstance(artifact, Mapping) else None
        )

    def _register(self) -> bool:
        try:
            response = self.transport.register(self.worker_id)
        except TransportError as exc:
            self._unreachable(exc)
            return False
        self._observe(response)
        if response.status != 200 or not isinstance(response.body, Mapping):
            self._event("reg_fail", c=response.status)
            if response.status in {401, 403}:
                raise WorkloadError("worker credential rejected")
            return False
        self._apply_lease(response.body)
        self.last_heartbeat = self.clock()
        self._event(
            "reg",
            v=self.assigned_version,
            gen=response.body.get("generation"),
        )
        return True

    def _heartbeat(self) -> None:
        assert self.lease_id is not None
        try:
            response = self.transport.heartbeat(self.worker_id, self.lease_id)
        except TransportError as exc:
            self._unreachable(exc)
            return
        self._observe(response)
        self.last_heartbeat = self.clock()
        if response.status == 200 and isinstance(response.body, Mapping):
            self._apply_lease(response.body)
        elif response.status in {409, 410}:
            self.lease_id = None

    def _sync_weights(self) -> bool:
        """Load the assigned version's verified weights; True when current."""
        if self.assigned_version is None or self.assigned_artifact is None:
            return False
        if self.loaded_version == self.assigned_version:
            return True
        started = time.perf_counter()
        version = self.assigned_version
        descriptor = PolicyArtifact.from_dict(self.assigned_artifact)
        try:
            response = self.transport.artifact(version)
        except TransportError as exc:
            self._unreachable(exc)
            return False
        self._observe(response)
        if response.status != 200 or not isinstance(response.body, bytes):
            self._event("sync_fail", v=version, why=f"http {response.status}")
            return False
        path = self.cache_dir / f"policy-v{version:09d}.bin"
        atomic_write_bytes(path, response.body)
        try:
            verify_policy_artifact(path, descriptor)
            state = deserialize_tensors(response.body)
            self.model.load_state_dict({k: v.to(self.device) for k, v in state.items()})
        except (PolicyArtifactError, WorkloadError, RuntimeError) as exc:
            self._event("sync_fail", v=version, why=str(exc)[:200])
            return False
        finally:
            for old in self.cache_dir.glob("policy-v*.bin"):
                if old != path:
                    old.unlink(missing_ok=True)
        self.loaded_version = version
        self.loaded_sha = descriptor.sha256
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._event("sync", v=version, sha=descriptor.sha256, ms=round(elapsed_ms, 3))
        self._prune_retained()
        return True

    def _prune_retained(self) -> None:
        if self.loaded_version is None:
            return
        floor = self.loaded_version - self.config.max_policy_lag
        while self.retained and self.retained[0].policy_version < floor:
            self.retained.popleft()

    def _generate(self) -> RolloutRecord:
        assert self.loaded_version is not None
        rollout_id = f"{self.worker_id}.g{self.incarnation}.{self.sequence}"
        self.sequence += 1
        record = sample_rollout(
            self.model,
            self.config,
            seed=self.seed,
            task=self.task,
            rollout_id=rollout_id,
            policy_version=self.loaded_version,
            artifact_sha256=self.loaded_sha,
            worker_id=self.worker_id,
            node_id=self.node_id,
            device=self.device,
        )
        self.counts["generated"] += 1
        # Write-ahead: every ID the controller might train is logged first.
        self._event("gen", id=rollout_id, v=self.loaded_version)
        return record

    def _submit(self, record: RolloutRecord, resubmit: bool) -> bool:
        """Return True when the record reached a terminal answer."""
        assert self.lease_id is not None
        try:
            response = self.transport.submit(
                self.worker_id,
                self.lease_id,
                record,
                min(2.0, float(self.config.request_timeout_seconds)),
            )
        except TransportError as exc:
            self._unreachable(exc)
            self._event("sub", id=record.rollout_id, v=record.policy_version, r="err")
            return False
        self._observe(response)
        if response.status == 200 and isinstance(response.body, Mapping):
            accepted = bool(response.body.get("accepted"))
            self._event(
                "sub",
                id=record.rollout_id,
                v=record.policy_version,
                r="acc" if accepted else "rej",
                rs=int(resubmit),
            )
            if accepted:
                self.counts["accepted"] += 1
                if not resubmit:
                    self.retained.append(record)
            else:
                self.counts["rejected"] += 1
            return True
        self._event(
            "sub",
            id=record.rollout_id,
            v=record.policy_version,
            r="err",
            c=response.status,
        )
        detail = (
            response.body.get("detail") if isinstance(response.body, Mapping) else ""
        )
        if "artifact" in str(detail) and "no longer available" in str(detail):
            # The version it was sampled under aged out of the retained
            # artifacts, so it is far beyond the lag bound: terminal drop.
            return True
        if response.status in {409, 410}:
            self.lease_id = None
            return False
        if response.status == 503:
            if self._phase_is_stopping():
                self.stop_reason = "controller-stopping"
            return False
        if response.status == 422:
            return True  # malformed record; logged as an error terminal
        if response.status in {401, 403}:
            raise WorkloadError("worker credential rejected")
        return False

    # -- main loop -----------------------------------------------------------

    def run(self, max_iterations: int | None = None) -> int:
        if self.log.closed:
            self.log = JsonlLog(self.log.path)
        self._event(
            "start",
            w=self.worker_id,
            n=self.node_id,
            inc=self.incarnation,
            pid=os.getpid(),
        )
        iterations = 0
        backoff = 0.1
        try:
            while self.stop_reason is None:
                iterations += 1
                if max_iterations is not None and iterations > max_iterations:
                    self.stop_reason = "iterations"
                    break
                now = self.clock()
                if now >= self.hard_deadline:
                    self.stop_reason = "hard-deadline"
                    break
                if (
                    self.unreachable_since is not None
                    and now > self.deadline
                    and now - self.unreachable_since
                    > float(self.config.shutdown_grace_seconds)
                ):
                    self.stop_reason = "controller-gone"
                    break
                if self.lease_id is None:
                    if not self._register():
                        if now > self.deadline and self._phase_is_stopping():
                            self.stop_reason = "controller-stopping"
                            break
                        self.sleep(backoff)
                        backoff = min(2.0, backoff * 2)
                        continue
                if now - self.last_heartbeat >= self.config.heartbeat_interval_seconds:
                    if self._phase_is_stopping():
                        self.stop_reason = "controller-stopping"
                        break
                    if self.lease_id is not None:
                        self._heartbeat()
                    if self.lease_id is None:
                        continue
                if not self._sync_weights():
                    self.sleep(backoff)
                    backoff = min(2.0, backoff * 2)
                    continue
                if self.pending is None and self.resubmit_queue:
                    self.pending = self.resubmit_queue.popleft()
                    self.pending_resubmit = True
                if self.pending is None:
                    cycle_started = self.clock()
                    self.pending = self._generate()
                    self.pending_resubmit = False
                else:
                    cycle_started = self.clock()
                if self.lease_id is None:
                    continue
                if self._submit(self.pending, self.pending_resubmit):
                    self.pending = None
                    backoff = 0.1
                    remaining = self.config.rollout_interval_seconds - (
                        self.clock() - cycle_started
                    )
                    if remaining > 0:
                        self.sleep(remaining)
                else:
                    self.sleep(backoff)
                    backoff = min(2.0, backoff * 2)
        finally:
            if self.lease_id is not None:
                with contextlib.suppress(TransportError):
                    self.transport.unregister(self.worker_id, self.lease_id)
            self._event("stop", why=self.stop_reason, counts=self.counts)
            self.log.close()
        return 0


def run_worker(args: argparse.Namespace) -> int:
    config = parse_config(json.loads(Path(args.config_file).read_text("utf-8")))
    _torch().set_num_threads(1)
    device = resolve_device(config.device)
    worker_key = os.environ.get(ENV_WORKER_KEY, "")
    transport = HttpTransport(
        args.controller_url, worker_key, float(config.request_timeout_seconds)
    )
    out = Path(args.out_dir)
    worker = RolloutWorker(
        config,
        seed=args.seed,
        worker_id=args.worker_id,
        node_id=args.node_id,
        incarnation=args.incarnation,
        transport=transport,
        log_path=out / "workers" / f"{args.worker_id}.jsonl",
        cache_dir=out / "cache" / args.worker_id,
        deadline_unix=args.deadline_unix,
        hard_deadline_unix=args.hard_deadline_unix,
        device=device,
    )
    try:
        return worker.run()
    finally:
        transport.close()


# --------------------------------------------------------------------------- #
# Network partition proxy (no root, no iptables)
# --------------------------------------------------------------------------- #


class PartitionProxy:
    """TCP forwarder that can sever and black-hole the inter-node link.

    While partitioned, established connections are aborted and new ones are
    accepted but never forwarded (clients time out, as with a dropped route).
    """

    def __init__(
        self, listen_host: str, listen_port: int, target_host: str, target_port: int
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.target_host = target_host
        self.target_port = int(target_port)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._partitioned = False
        self._active: set[asyncio.StreamWriter] = set()
        self._held: set[asyncio.StreamWriter] = set()
        self._server: asyncio.AbstractServer | None = None
        self._error: BaseException | None = None

    async def _pipe(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionError, OSError, asyncio.CancelledError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._partitioned:
            self._held.add(writer)
            return
        try:
            up_reader, up_writer = await asyncio.open_connection(
                self.target_host, self.target_port
            )
        except OSError:
            writer.close()
            return
        self._active.update({writer, up_writer})
        try:
            await asyncio.gather(
                self._pipe(reader, up_writer), self._pipe(up_reader, writer)
            )
        finally:
            self._active.discard(writer)
            self._active.discard(up_writer)

    async def _serve(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, self.listen_host, self.listen_port, reuse_address=True
        )
        self._ready.set()
        async with self._server:
            await self._server.serve_forever()

    def start(self) -> None:
        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            try:
                self._loop.run_until_complete(self._serve())
            except asyncio.CancelledError:
                pass
            except BaseException as exc:  # noqa: BLE001 - surfaced via start()
                self._error = exc
                self._ready.set()
            finally:
                self._loop.close()

        self._thread = threading.Thread(
            target=_run, name="partition-proxy", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(10) or self._error is not None:
            raise WorkloadError(f"partition proxy failed to start: {self._error}")

    def _call(self, func: Callable[[], None]) -> None:
        assert self._loop is not None
        done = threading.Event()

        def _wrapped() -> None:
            try:
                func()
            finally:
                done.set()

        self._loop.call_soon_threadsafe(_wrapped)
        done.wait(5)

    def partition(self) -> None:
        def _sever() -> None:
            self._partitioned = True
            for writer in list(self._active):
                writer.transport.abort()
            self._active.clear()

        self._call(_sever)

    def heal(self) -> None:
        def _restore() -> None:
            self._partitioned = False
            for writer in list(self._held):
                writer.transport.abort()
            self._held.clear()

        self._call(_restore)

    @property
    def partitioned(self) -> bool:
        return self._partitioned

    def stop(self) -> None:
        if self._loop is None or self._server is None:
            return

        def _close() -> None:
            assert self._server is not None
            self._server.close()
            for writer in list(self._active | self._held):
                writer.transport.abort()
            for task in asyncio.all_tasks(self._loop):
                task.cancel()

        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(_close)
        if self._thread is not None:
            self._thread.join(5)


# --------------------------------------------------------------------------- #
# Node supervisor and fault injection
# --------------------------------------------------------------------------- #


@dataclass
class ManagedProcess:
    """A supervised child with restart-on-unexpected-exit semantics."""

    name: str
    argv_factory: Callable[[int], list[str]]
    log_path: Path
    env: Mapping[str, str]
    restart_delay: float
    cwd: Path
    incarnation: int = 0
    process: subprocess.Popen[bytes] | None = None
    done: bool = False
    restart_at: float | None = None
    exits: list[dict[str, Any]] = field(default_factory=list)
    pids: list[int] = field(default_factory=list)
    killed_by_fault: bool = False

    def launch(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab") as log:
            self.process = subprocess.Popen(  # nosec B603 - fixed argv, no shell
                self.argv_factory(self.incarnation),
                cwd=self.cwd,
                env=dict(self.env),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self.pids.append(self.process.pid)
        self.restart_at = None


class NodeSupervisor:
    """Start roles, restart them on unexpected exit, and inject one fault."""

    def __init__(
        self,
        *,
        node_id: str,
        out_dir: Path,
        plan: ScenarioPlan,
        config: WorkloadConfig,
        processes: Sequence[ManagedProcess],
        controller_name: str | None,
        fault_targets: Mapping[str, str],
        proxy: PartitionProxy | None,
        started_at: float,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.node_id = node_id
        self.out_dir = out_dir
        self.plan = plan
        self.config = config
        self.processes = list(processes)
        self.controller_name = controller_name
        self.fault_targets = dict(fault_targets)
        self.proxy = proxy
        self.started_at = started_at
        self.deadline = started_at + plan.duration_seconds
        self.hard_deadline = (
            self.deadline + 4 * float(config.shutdown_grace_seconds) + 60
        )
        self.clock = clock
        self.sleep = sleep
        self.events = JsonlLog(out_dir / "node-events.jsonl")
        self.faults_done: set[str] = set()
        self.unplanned_exits = 0
        self.forced_stop = False

    def _event(self, name: str, **values: Any) -> None:
        self.events.write(
            {"e": name, "t": self.clock(), "node": self.node_id, **values}
        )

    def _by_name(self, name: str) -> ManagedProcess | None:
        return next((p for p in self.processes if p.name == name), None)

    def _inject(self, now: float) -> None:
        if self.plan.fault_at_seconds is None:
            return
        at = self.started_at + self.plan.fault_at_seconds
        scenario = self.plan.scenario
        if now >= at and "fault" not in self.faults_done:
            target_name = self.fault_targets.get(scenario)
            if scenario in {"worker_exit", "controller_restart"} and target_name:
                target = self._by_name(target_name)
                if target is None or target.process is None:
                    raise WorkloadError(f"fault target {target_name} is not running")
                target.killed_by_fault = True
                target.process.send_signal(signal.SIGKILL)
                self._event("fault_kill", target=target_name, pid=target.process.pid)
                self.faults_done.add("fault")
            elif scenario == "network_interruption" and self.proxy is not None:
                self.proxy.partition()
                self._event("partition_start")
                self.faults_done.add("fault")
            else:
                self.faults_done.add("fault")
        if (
            scenario == "network_interruption"
            and self.proxy is not None
            and "fault" in self.faults_done
            and "heal" not in self.faults_done
            and now >= at + self.plan.partition_seconds
        ):
            self.proxy.heal()
            self._event("partition_end")
            self.faults_done.add("heal")

    def _poll(self, item: ManagedProcess, now: float) -> None:
        if item.done:
            return
        if item.process is None:
            if item.restart_at is not None and now >= item.restart_at:
                item.incarnation += 1
                item.launch()
                self._event(
                    "restart",
                    target=item.name,
                    pid=item.process.pid if item.process else None,
                    inc=item.incarnation,
                )
            return
        code = item.process.poll()
        if code is None:
            return
        item.exits.append({"t": now, "code": code, "inc": item.incarnation})
        item.process = None
        if code == 0:
            item.done = True
            self._event("exit", target=item.name, code=code)
            return
        if item.killed_by_fault:
            item.killed_by_fault = False
            item.restart_at = now + item.restart_delay
            self._event("exit", target=item.name, code=code, planned=True)
            return
        self.unplanned_exits += 1
        self._event("exit", target=item.name, code=code, planned=False)
        if now < self.deadline:
            item.restart_at = now + item.restart_delay
        else:
            item.done = True

    def run(self) -> dict[str, Any]:
        self._event("node_start", plan=self.plan.to_dict())
        for item in self.processes:
            item.launch()
            self._event(
                "launch",
                target=item.name,
                pid=item.process.pid if item.process else None,
            )
        try:
            while not all(item.done for item in self.processes):
                now = self.clock()
                self._inject(now)
                for item in self.processes:
                    self._poll(item, now)
                if now >= self.hard_deadline:
                    self.forced_stop = True
                    self._event("forced_stop")
                    break
                self.sleep(0.05)
        finally:
            for item in self.processes:
                if item.process is not None and item.process.poll() is None:
                    item.process.send_signal(signal.SIGTERM)
            for item in self.processes:
                if item.process is not None:
                    try:
                        item.process.wait(10)
                    except subprocess.TimeoutExpired:
                        item.process.kill()
                        item.process.wait(10)
            if self.proxy is not None:
                if self.proxy.partitioned:
                    self.proxy.heal()
                self.proxy.stop()
        ended = self.clock()
        controller = (
            self._by_name(self.controller_name) if self.controller_name else None
        )
        controller_end = (
            controller.exits[-1]["t"]
            if controller is not None and controller.exits
            else ended
        )
        summary = {
            "node_id": self.node_id,
            "started_at": self.started_at,
            "ended_at": ended,
            "controller_ended_at": controller_end if controller else None,
            "duration_seconds": (controller_end if controller else ended)
            - self.started_at,
            "plan": self.plan.to_dict(),
            "forced_stop": self.forced_stop,
            "unplanned_exits": self.unplanned_exits,
            "faults": sorted(self.faults_done),
            "processes": {
                item.name: {"pids": item.pids, "exits": item.exits}
                for item in self.processes
            },
            "device": describe_device(resolve_device(self.config.device)),
            "hostname_sha256": sha256_bytes(socket.gethostname().encode())[:16],
        }
        self._event("node_end")
        self.events.close()
        atomic_write_bytes(
            self.out_dir / "node-summary.json",
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        )
        return summary


def run_node(args: argparse.Namespace) -> int:
    config_raw = parse_pinned_config(args.config_json, args.config_sha256)
    config = parse_config(config_raw)
    plan = scenario_plan(config, args.scenario, args.mode)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / "config.json"
    atomic_write_bytes(config_file, (canonical_json(config_raw) + "\n").encode())
    worker_key = os.environ.get(ENV_WORKER_KEY, "")
    if len(worker_key) < 16:
        raise WorkloadError(f"{ENV_WORKER_KEY} must be set (>= 16 characters)")
    started_at = time.time()
    deadline = started_at + plan.duration_seconds
    hard_deadline = deadline + 3 * float(config.shutdown_grace_seconds) + 30
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    script = str(Path(__file__).resolve())
    python = sys.executable
    processes: list[ManagedProcess] = []
    proxy: PartitionProxy | None = None
    controller_name: str | None = None
    fault_targets: dict[str, str] = {}
    if args.controller:
        controller_name = f"{args.node_id}-controller"
        controller_deadline = deadline

        def controller_argv(incarnation: int) -> list[str]:
            return [
                python,
                "-u",
                script,
                "controller",
                "--config-file",
                str(config_file),
                "--seed",
                str(args.seed),
                "--state-dir",
                str(out / "controller"),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.controller_port),
                "--deadline-unix",
                repr(controller_deadline),
            ]

        processes.append(
            ManagedProcess(
                name=controller_name,
                argv_factory=controller_argv,
                log_path=out / "logs" / "controller.log",
                env=env,
                restart_delay=float(config.controller_restart_delay_seconds),
                cwd=_ROOT,
            )
        )
        fault_targets["controller_restart"] = controller_name
        proxy = PartitionProxy(
            args.proxy_host, args.proxy_port, "127.0.0.1", args.controller_port
        )
        proxy.start()
    controller_url = args.controller_url or f"http://127.0.0.1:{args.controller_port}"
    for index in range(args.workers):
        worker_id = f"{args.node_id}-w{index}"

        def worker_argv(incarnation: int, worker_id: str = worker_id) -> list[str]:
            return [
                python,
                "-u",
                script,
                "worker",
                "--config-file",
                str(config_file),
                "--seed",
                str(args.seed),
                "--worker-id",
                worker_id,
                "--node-id",
                args.node_id,
                "--incarnation",
                str(incarnation),
                "--controller-url",
                controller_url,
                "--out-dir",
                str(out),
                "--deadline-unix",
                repr(deadline),
                "--hard-deadline-unix",
                repr(hard_deadline),
            ]

        processes.append(
            ManagedProcess(
                name=worker_id,
                argv_factory=worker_argv,
                log_path=out / "logs" / f"{worker_id}.log",
                env=env,
                restart_delay=float(config.worker_restart_delay_seconds),
                cwd=_ROOT,
            )
        )
    if args.worker_fault_target and any(
        item.name == args.worker_fault_target for item in processes
    ):
        fault_targets["worker_exit"] = args.worker_fault_target
    supervisor = NodeSupervisor(
        node_id=args.node_id,
        out_dir=out,
        plan=plan,
        config=config,
        processes=processes,
        controller_name=controller_name,
        fault_targets=fault_targets,
        proxy=proxy,
        started_at=started_at,
    )
    summary = supervisor.run()
    return 1 if summary["forced_stop"] else 0


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise WorkloadError("no samples")
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def audit_run(
    node_dirs: Mapping[str, Path],
    *,
    controller_node: str,
    scenario: str,
    max_policy_lag: int,
    remote_nodes: Iterable[str] = (),
) -> dict[str, Any]:
    """Account for every rollout and update across all nodes of one run."""
    controller_dir = node_dirs[controller_node] / "controller"
    ledger = read_jsonl(controller_dir / "ledger.jsonl")
    evictions = read_jsonl(controller_dir / "evictions.jsonl")
    epochs = read_jsonl(controller_dir / "epochs.jsonl")
    initial_path = controller_dir / "initial.json"
    initial = (
        json.loads(initial_path.read_text("utf-8")) if initial_path.exists() else {}
    )
    summaries = sorted(
        (
            json.loads(p.read_text("utf-8"))
            for p in controller_dir.glob("summary-*.json")
        ),
        key=lambda item: float(item.get("stopped_at", 0)),
    )
    final = summaries[-1] if summaries else None

    problems: list[str] = []
    # -- optimizer updates --------------------------------------------------
    updates: dict[int, dict[str, Any]] = {}
    duplicate_updates = 0
    for row in ledger:
        index = int(row["u"])
        if index in updates:
            duplicate_updates += 1
            if updates[index]["sha"] != row["sha"]:
                problems.append(f"update {index} committed with two digests")
            continue
        updates[index] = row
    count = max(updates) if updates else 0
    lost_updates = sum(1 for index in range(1, count + 1) if index not in updates)
    version_sha = {0: initial.get("sha")}
    chain_breaks = 0
    for index in range(1, count + 1):
        row = updates.get(index)
        if row is None:
            continue
        if int(row["to"]) != int(row["from"]) + 1 or int(row["to"]) != index:
            chain_breaks += 1
        version_sha[int(row["to"])] = row["sha"]
    lost_updates += chain_breaks

    # -- trained rollouts ---------------------------------------------------
    trained_counts: dict[str, int] = {}
    lags: list[int] = []
    for row in updates.values():
        for rollout_id in row["ids"]:
            trained_counts[rollout_id] = trained_counts.get(rollout_id, 0) + 1
        lags.extend(int(value) for value in row["lags"])
    duplicate_rollouts = sum(n - 1 for n in trained_counts.values() if n > 1)
    trained = set(trained_counts)

    # -- worker logs --------------------------------------------------------
    generated: set[str] = set()
    acknowledged: set[str] = set()
    final_result: dict[str, tuple[str, int]] = {}
    sync_ms: list[float] = []
    observed: dict[int, set[str]] = {}
    sync_failures = 0
    worker_events: dict[str, list[dict[str, Any]]] = {}
    for directory in node_dirs.values():
        for path in sorted((directory / "workers").glob("*.jsonl")):
            rows = read_jsonl(path)
            worker_events[path.stem] = rows
            for row in rows:
                kind = row.get("e")
                if kind == "gen":
                    generated.add(str(row["id"]))
                elif kind == "sub":
                    rollout_id = str(row["id"])
                    if row.get("r") == "acc":
                        acknowledged.add(rollout_id)
                    if row.get("r") in {"acc", "rej"}:
                        final_result[rollout_id] = (
                            str(row["r"]),
                            int(row.get("rs", 0)),
                        )
                elif kind == "sync":
                    sync_ms.append(float(row["ms"]))
                    observed.setdefault(int(row["v"]), set()).add(str(row["sha"]))
                elif kind == "sync_fail":
                    sync_failures += 1

    # Verification failures are retried and never loaded, so they are counted
    # separately; a mismatch is a loaded digest that disagrees with the commit.
    digest_mismatches = 0
    lost_observed_versions = 0
    for version, digests in observed.items():
        committed = version_sha.get(version)
        if committed is None:
            lost_observed_versions += 1
            continue
        digest_mismatches += sum(1 for sha in digests if sha != committed)
    lost_updates += lost_observed_versions
    digest_mismatches += sum(int(e.get("replay_digest_mismatches", 0)) for e in epochs)
    if final is not None and updates:
        if final.get("final_sha") != updates[count]["sha"]:
            digest_mismatches += 1
    phantom = len(trained - generated)

    evicted = {str(row["id"]) for row in evictions}
    residual = set(final.get("residual_queue_ids", [])) if final else set()
    stale_on_resubmit = {
        rid
        for rid, (result, resubmitted) in final_result.items()
        if result == "rej" and resubmitted
    }
    untrained_ack = acknowledged - trained
    lost_admitted = untrained_ack - evicted - residual - stale_on_resubmit

    # -- recovery -----------------------------------------------------------
    node_events: list[dict[str, Any]] = []
    for directory in node_dirs.values():
        node_events.extend(read_jsonl(directory / "node-events.jsonl"))
    recovery_seconds: float | None = None
    fault_event = None
    commit_times = sorted(float(row["t"]) for row in updates.values())
    if scenario == "steady_state_soak":
        recovery_seconds = 0.0
        unexpected = [e for e in node_events if e.get("e") == "fault_kill"]
        if unexpected:
            problems.append("soak recorded a fault injection")
    elif scenario == "controller_restart":
        fault_event = next(
            (
                e
                for e in node_events
                if e.get("e") == "fault_kill"
                and str(e.get("target", "")).endswith("controller")
            ),
            None,
        )
        if fault_event is not None:
            after = [t for t in commit_times if t > float(fault_event["t"])]
            restarts = [e for e in epochs if float(e["t"]) > float(fault_event["t"])]
            if after and restarts and restarts[0].get("recovered"):
                recovery_seconds = after[0] - float(fault_event["t"])
    elif scenario == "worker_exit":
        fault_event = next((e for e in node_events if e.get("e") == "fault_kill"), None)
        if fault_event is not None:
            target = str(fault_event["target"])
            kill_t = float(fault_event["t"])
            for row in worker_events.get(target, []):
                if (
                    row.get("e") == "sub"
                    and row.get("r") == "acc"
                    and float(row["t"]) > kill_t
                ):
                    recovery_seconds = float(row["t"]) - kill_t
                    break
    elif scenario == "network_interruption":
        fault_event = next(
            (e for e in node_events if e.get("e") == "partition_start"), None
        )
        heal = next((e for e in node_events if e.get("e") == "partition_end"), None)
        remote = set(remote_nodes)
        if fault_event is not None and heal is not None:
            heal_t = float(heal["t"])
            for row in sorted(updates.values(), key=lambda r: float(r["t"])):
                if float(row["t"]) > heal_t and remote.intersection(row["nodes"]):
                    recovery_seconds = float(row["t"]) - heal_t
                    break
            # During the partition no remote rollout may have been committed
            # from a submission that crossed the partitioned link.
    recovered = recovery_seconds is not None
    if not recovered:
        problems.append(f"{scenario}: no recovery was observed")

    rewards = [
        float(row["metrics"]["reward_mean"]) for _, row in sorted(updates.items())
    ]
    window = max(1, min(20, len(rewards) // 4 or 1))
    metrics = {
        "rollouts_attempted": len(generated),
        "rollouts_accepted": len(trained),
        "rollouts_acknowledged": len(acknowledged),
        "optimizer_updates": len(updates),
        "policy_versions": count + 1,
        "max_observed_policy_lag": max(lags) if lags else 0,
        "max_policy_lag_bound": int(max_policy_lag),
        "lost_optimizer_updates": lost_updates,
        "duplicate_optimizer_updates": duplicate_updates + duplicate_rollouts,
        "artifact_digest_mismatches": digest_mismatches,
        "duplicate_trained_rollouts": duplicate_rollouts,
        "phantom_trained_rollouts": phantom,
        "lost_admitted_rollouts": len(lost_admitted),
        "evicted_stale_rollouts": len(untrained_ack & evicted),
        "residual_queue_rollouts": len(untrained_ack & residual),
        "stale_on_resubmit_rollouts": len(
            untrained_ack & stale_on_resubmit - evicted - residual
        ),
        "worker_sync_failures": sync_failures,
        "controller_epochs": len(epochs),
        "weight_sync_samples": len(sync_ms),
        "reward_first_window": (sum(rewards[:window]) / window) if rewards else None,
        "reward_last_window": (sum(rewards[-window:]) / window) if rewards else None,
    }
    if sync_ms:
        metrics["weight_sync_latency_ms"] = {
            "p50": max(_percentile(sync_ms, 0.50), 1e-6),
            "p95": max(_percentile(sync_ms, 0.95), 1e-6),
            "max": max(max(sync_ms), 1e-6),
        }
    else:
        problems.append("no weight synchronization was measured")
    if metrics["max_observed_policy_lag"] > max_policy_lag:
        problems.append("trained rollout exceeded max_policy_lag")
    for key in (
        "lost_optimizer_updates",
        "duplicate_optimizer_updates",
        "artifact_digest_mismatches",
        "phantom_trained_rollouts",
        "lost_admitted_rollouts",
    ):
        if metrics[key]:
            problems.append(f"{key}={metrics[key]}")
    if final is None:
        problems.append("controller wrote no final summary")
    if not updates:
        problems.append("no optimizer updates were committed")
    return {
        "metrics": metrics,
        "recovery": {
            "recovered": recovered,
            "recovery_seconds": float(recovery_seconds or 0.0),
            "fault_event": fault_event,
        },
        "problems": problems,
        "complete": not problems,
    }


def adapter_from_audit(
    audit: Mapping[str, Any],
    *,
    scenario: str,
    seed: int,
    protocol: str,
    framework_version: str,
    provider: str,
    config_sha256: str,
    duration_seconds: float,
    topology: Mapping[str, Any],
    cost_total_usd: float,
    cost_source: str,
    resources_remaining: int,
    artifact_path: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The matrix-runner adapter document for one audited scenario run."""
    metrics = dict(audit["metrics"])
    accepted = int(metrics["rollouts_accepted"])
    if duration_seconds <= 0 or accepted < 1:
        raise WorkloadError("cannot build an adapter without accepted work")
    metrics["accepted_rollouts_per_second"] = accepted / float(duration_seconds)
    metrics["duration_seconds"] = float(duration_seconds)
    metrics["total_cost_usd"] = float(cost_total_usd)
    metrics["cost_per_accepted_rollout_usd"] = float(cost_total_usd) / accepted
    document: dict[str, Any] = {
        "status": "completed",
        "measured": True,
        "scenario": scenario,
        "seed": int(seed),
        "protocol": protocol,
        "framework_version": framework_version,
        "provider": provider,
        "config_sha256": config_sha256,
        "duration_seconds": float(duration_seconds),
        "topology": dict(topology),
        "metrics": metrics,
        "cost": {
            "currency": "USD",
            "source": cost_source,
            "total": float(cost_total_usd),
            "per_accepted_rollout": float(cost_total_usd) / accepted,
        },
        "recovery": {
            "recovered": bool(audit["recovery"]["recovered"]),
            "recovery_seconds": float(audit["recovery"]["recovery_seconds"]),
            "resources_remaining": int(resources_remaining),
        },
        "artifact_path": str(Path(artifact_path).resolve()),
    }
    if extra:
        document.update(dict(extra))
    return document


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="role", required=True)

    node = sub.add_parser("node", help="supervise one node's roles")
    node.add_argument("--node-id", required=True)
    node.add_argument("--scenario", choices=SCENARIOS, required=True)
    node.add_argument("--mode", choices=("measured", "preflight"), required=True)
    node.add_argument("--seed", type=int, required=True)
    node.add_argument("--config-json", required=True)
    node.add_argument("--config-sha256", required=True)
    node.add_argument("--out-dir", required=True)
    node.add_argument("--workers", type=int, required=True)
    node.add_argument("--controller", action="store_true")
    node.add_argument("--controller-port", type=int, default=29510)
    node.add_argument("--proxy-host", default="127.0.0.1")
    node.add_argument("--proxy-port", type=int, default=29500)
    node.add_argument("--controller-url", default=None)
    node.add_argument("--worker-fault-target", default=None)

    controller = sub.add_parser("controller", help="learner + control plane")
    controller.add_argument("--config-file", required=True)
    controller.add_argument("--seed", type=int, required=True)
    controller.add_argument("--state-dir", required=True)
    controller.add_argument("--host", default="127.0.0.1")
    controller.add_argument("--port", type=int, required=True)
    controller.add_argument("--deadline-unix", type=float, required=True)

    worker = sub.add_parser("worker", help="rollout actor")
    worker.add_argument("--config-file", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--node-id", required=True)
    worker.add_argument("--incarnation", type=int, default=0)
    worker.add_argument("--controller-url", required=True)
    worker.add_argument("--out-dir", required=True)
    worker.add_argument("--deadline-unix", type=float, required=True)
    worker.add_argument("--hard-deadline-unix", type=float, required=True)

    audit = sub.add_parser("audit", help="account for a finished run")
    audit.add_argument("--node", action="append", required=True, help="ID=DIR")
    audit.add_argument("--controller-node", required=True)
    audit.add_argument("--scenario", choices=SCENARIOS, required=True)
    audit.add_argument("--max-policy-lag", type=int, required=True)
    audit.add_argument("--remote-node", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.role == "node":
            return run_node(args)
        if args.role == "controller":
            return asyncio.run(run_controller_async(args))
        if args.role == "worker":
            return run_worker(args)
        nodes = {}
        for item in args.node:
            key, _, value = item.partition("=")
            nodes[key] = Path(value)
        report = audit_run(
            nodes,
            controller_node=args.controller_node,
            scenario=args.scenario,
            max_policy_lag=args.max_policy_lag,
            remote_nodes=args.remote_node,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["complete"] else 1
    except WorkloadError as exc:
        print(f"distributed async workload error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
