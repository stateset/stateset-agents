"""Exactly-once, lag, recovery, and fault-injection tests for the async workload."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.training.async_rollouts import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    RolloutRecord,
)

from .distributed_async_fakes import (
    SHA,
    CoreTransport,
    LoopThread,
    small_config,
    workload,
    write_synthetic_run,
)

pytest.importorskip("torch")


def _record(core: Any, rollout_id: str, version: int | None = None) -> RolloutRecord:
    version = core.policy_version if version is None else version
    artifact = core.store.artifact_path(version)
    model = workload.build_policy(core.config, core.seed)
    state = workload.deserialize_tensors(artifact.read_bytes())
    model.load_state_dict(state)
    return workload.sample_rollout(
        model,
        core.config,
        seed=core.seed,
        task=workload.task_matrix(core.config, core.seed),
        rollout_id=rollout_id,
        policy_version=version,
        artifact_sha256=workload.sha256_bytes(artifact.read_bytes()),
        worker_id="w",
        node_id="node-1",
    )


# --------------------------------------------------------------------------- #
# Config, policy, serialization
# --------------------------------------------------------------------------- #


def test_config_rejects_unknown_and_missing_keys() -> None:
    with pytest.raises(workload.WorkloadError, match="unknown config keys"):
        small_config(typo_key=1)
    with pytest.raises(workload.WorkloadError, match="missing"):
        workload.parse_config({"max_policy_lag": 1})
    with pytest.raises(workload.WorkloadError, match="lease_ttl"):
        small_config(lease_ttl_seconds=0.0001, heartbeat_interval_seconds=1)


def test_scenario_plan_shapes_faults_and_preflight() -> None:
    config = small_config(preflight_duration_seconds=2)
    soak = workload.scenario_plan(config, "steady_state_soak", "measured")
    assert soak.duration_seconds == 6 and soak.fault_at_seconds is None
    partition = workload.scenario_plan(config, "network_interruption", "measured")
    assert partition.fault_at_seconds == 1 and partition.partition_seconds == 0.5
    preflight = workload.scenario_plan(config, "steady_state_soak", "preflight")
    assert preflight.duration_seconds == 2


def test_pinned_config_digest_is_enforced() -> None:
    text = workload.canonical_json({"a": 1})
    with pytest.raises(workload.WorkloadError, match="config_sha256"):
        workload.parse_pinned_config(text, "0" * 64)
    import hashlib

    assert workload.parse_pinned_config(text, hashlib.sha256(text.encode()).hexdigest())


def test_policy_serialization_is_deterministic_and_pickle_free() -> None:
    config = small_config()
    first = workload.serialize_tensors(workload.build_policy(config, 3).state_dict())
    second = workload.serialize_tensors(workload.build_policy(config, 3).state_dict())
    assert first == second and first.startswith(b"SSDAPOL1")
    restored = workload.deserialize_tensors(first)
    assert workload.serialize_tensors(restored) == first
    with pytest.raises(workload.WorkloadError):
        workload.deserialize_tensors(first[:-4])


def test_learner_improves_reward_with_importance_corrected_updates() -> None:
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = small_config(batch_size=16, learning_rate=0.05)
    learner = workload.Learner(config, seed=7)
    task = workload.task_matrix(config, 7)
    rewards = []
    for update in range(120):
        records = [
            workload.sample_rollout(
                learner.model,
                config,
                seed=7,
                task=task,
                rollout_id=f"r{update}.{index}",
                policy_version=update,
                artifact_sha256=None,
                worker_id="w",
                node_id="n",
            )
            for index in range(16)
        ]
        rewards.append(learner.update(records)["reward_mean"])
    torch.set_num_threads(threads)
    assert sum(rewards[-20:]) / 20 > sum(rewards[:20]) / 20 + 0.2


def test_eviction_listener_reports_queue_evictions() -> None:
    async def scenario() -> list[tuple[str, int]]:
        coordinator = AsyncRolloutCoordinator(
            AsyncRolloutConfig(queue_capacity=4, max_batch_size=2, max_policy_lag=0)
        )
        seen: list[tuple[str, int]] = []
        coordinator.add_eviction_listener(
            lambda r, lag: seen.append((r.rollout_id, lag))
        )
        await coordinator.submit(RolloutRecord("a", 0, (-0.1,), {}))
        await coordinator.advance_policy()
        return seen

    assert asyncio.run(scenario()) == [("a", 1)]
    with pytest.raises(TypeError):
        AsyncRolloutCoordinator().add_eviction_listener(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Controller core: exactly-once, lag, crash recovery
# --------------------------------------------------------------------------- #


async def _fill_and_step(core: Any, lease: Any, prefix: str, updates: int) -> list[str]:
    trained: list[str] = []
    for update in range(updates):
        for index in range(core.config.batch_size):
            await lease_submit(core, lease, _record(core, f"{prefix}{update}.{index}"))
        entry = await core.step(timeout_seconds=1)
        assert entry is not None
        trained.extend(record["rollout_id"] for record in entry["records"])
    return trained


async def lease_submit(core: Any, lease: Any, record: RolloutRecord) -> bool:
    current = await core.control.heartbeat(lease.worker_id, lease.lease_id)
    del current
    return await core.control.submit(lease.worker_id, lease.lease_id, record)


def test_controller_crash_recovery_is_exactly_once(tmp_path: Path) -> None:
    config = small_config()
    state = tmp_path / "node-0" / "controller"

    async def scenario() -> None:
        core = workload.ControllerCore(config, seed=1, state_dir=state, epoch="first")
        await core.start()
        lease = await core.control.register("w")
        trained = await _fill_and_step(core, lease, "a", 7)
        assert core.update_index == 7 and core.store.latest_snapshot() == 6
        # Admitted after the last snapshot and never trained: volatile only.
        pending = [_record(core, f"late{i}") for i in range(2)]
        for record in pending:
            assert await lease_submit(core, lease, record)
        committed_sha = core.current_sha
        # Hard crash: the object is dropped without stop().

        restarted = workload.ControllerCore(
            config, seed=1, state_dir=state, epoch="second"
        )
        await restarted.start()
        assert restarted.recovery["recovered"] is True
        assert restarted.recovery["replayed_updates"] == 1
        assert restarted.replay_digest_mismatches == 0
        assert restarted.update_index == 7 and restarted.policy_version == 7
        assert restarted.current_sha == committed_sha
        assert workload.sha256_bytes(restarted.learner.weights_blob()) == committed_sha
        lease2 = await restarted.control.register("w")
        # Trained IDs replayed from the WAL are fenced as duplicates...
        assert not await lease_submit(
            restarted, lease2, _record(restarted, trained[-1], version=7)
        )
        assert not await lease_submit(
            restarted, lease2, _record(restarted, trained[0], version=7)
        )
        # ...while admitted-but-untrained IDs can be resubmitted.
        for record in pending:
            assert await lease_submit(restarted, lease2, record)
        for index in range(config.batch_size - 2):
            await lease_submit(restarted, lease2, _record(restarted, f"b{index}"))
        entry = await restarted.step(timeout_seconds=1)
        assert entry is not None and entry["u"] == 8
        await restarted.stop()

    asyncio.run(scenario())
    ledger = workload.read_jsonl(state / "ledger.jsonl")
    assert [row["u"] for row in ledger] == list(range(1, 9))
    ids = [rid for row in ledger for rid in row["ids"]]
    assert len(ids) == len(set(ids)) == 8 * config.batch_size
    assert max(lag for row in ledger for lag in row["lags"]) <= config.max_policy_lag


def test_torn_wal_tail_is_repaired(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"u":1}\n{"u":2')
    assert workload.repair_jsonl(path) == [{"u": 1}]
    assert path.read_bytes() == b'{"u":1}\n'


def test_stale_rollouts_are_rejected_or_evicted_and_logged(tmp_path: Path) -> None:
    config = small_config(max_policy_lag=0, batch_size=2)
    state = tmp_path / "controller"

    async def scenario() -> None:
        core = workload.ControllerCore(config, seed=2, state_dir=state)
        await core.start()
        lease = await core.control.register("w")
        for index in range(3):  # third stays queued at version 0
            await lease_submit(core, lease, _record(core, f"v0.{index}"))
        stale = _record(core, "late-v0", version=0)
        await core.step(timeout_seconds=1)
        # Queued record from version 0 is evicted when version 1 is published.
        assert core.control.coordinator.stats().dropped_stale == 1
        # Submitting a version-0 record against version 1 is rejected.
        assert not await lease_submit(core, lease, stale)
        await core.stop()

    asyncio.run(scenario())
    evicted = workload.read_jsonl(state / "evictions.jsonl")
    assert [row["id"] for row in evicted] == ["v0.2"]
    assert all(
        lag == 0
        for row in workload.read_jsonl(state / "ledger.jsonl")
        for lag in row["lags"]
    )


# --------------------------------------------------------------------------- #
# Worker against an in-process controller: partition and controller restart
# --------------------------------------------------------------------------- #


def test_worker_survives_partition_and_controller_restart(tmp_path: Path) -> None:
    config = small_config()
    node0 = tmp_path / "node-0"
    node1 = tmp_path / "node-1"
    loop = LoopThread()
    core = workload.ControllerCore(
        config, seed=5, state_dir=node0 / "controller", epoch="e1"
    )
    loop.run(core.start())
    stepping = {"on": True}

    async def learner() -> None:
        while stepping["on"]:
            await core_ref["core"].step(timeout_seconds=0.05)

    core_ref = {"core": core}
    task = asyncio.run_coroutine_threadsafe(learner(), loop.loop)
    transport = CoreTransport(loop, core)
    worker = workload.RolloutWorker(
        config,
        seed=5,
        worker_id="node-1-w0",
        node_id="node-1",
        incarnation=0,
        transport=transport,
        log_path=node1 / "workers" / "node-1-w0.jsonl",
        cache_dir=node1 / "cache",
        deadline_unix=time.time() + 600,
        hard_deadline_unix=time.time() + 600,
        sleep=lambda _s: time.sleep(0.001),
    )
    try:
        worker.run(max_iterations=150)
        assert worker.counts["accepted"] > 20
        transport.partitioned = True
        worker.stop_reason = None
        worker.run(max_iterations=20)
        transport.partitioned = False
        worker.stop_reason = None
        worker.run(max_iterations=60)
        # Controller crash: stop learning, drop the core, restart from disk.
        stepping["on"] = False
        task.result(10)
        restarted = workload.ControllerCore(
            config, seed=5, state_dir=node0 / "controller", epoch="e2"
        )
        loop.run(restarted.start())
        core_ref["core"] = restarted
        transport.core = restarted
        stepping["on"] = True
        task = asyncio.run_coroutine_threadsafe(learner(), loop.loop)
        worker.stop_reason = None
        worker.run(max_iterations=150)
        stepping["on"] = False
        task.result(10)
        loop.run(restarted.stop())
    finally:
        stepping["on"] = False
        loop.close()

    rows = workload.read_jsonl(node1 / "workers" / "node-1-w0.jsonl")
    assert any(row["e"] == "epoch" for row in rows)
    assert any(row["e"] == "sub" and row.get("rs") == 1 for row in rows)
    assert any(row["e"] == "sub" and row["r"] == "err" for row in rows)
    report = workload.audit_run(
        {"node-0": node0, "node-1": node1},
        controller_node="node-0",
        scenario="steady_state_soak",
        max_policy_lag=config.max_policy_lag,
        remote_nodes=["node-1"],
    )
    metrics = report["metrics"]
    assert metrics["duplicate_optimizer_updates"] == 0
    assert metrics["lost_optimizer_updates"] == 0
    assert metrics["artifact_digest_mismatches"] == 0
    assert metrics["phantom_trained_rollouts"] == 0
    assert metrics["lost_admitted_rollouts"] == 0, report
    assert metrics["controller_epochs"] == 2
    assert metrics["max_observed_policy_lag"] <= config.max_policy_lag


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("scenario", "recovery"),
    [("worker_exit", 1.1), ("network_interruption", 5.0), ("steady_state_soak", 0.0)],
)
def test_audit_accepts_consistent_run(
    tmp_path: Path, scenario: str, recovery: float
) -> None:
    nodes = write_synthetic_run(tmp_path, scenario=scenario)
    report = workload.audit_run(
        nodes,
        controller_node="node-0",
        scenario=scenario,
        max_policy_lag=1,
        remote_nodes=["node-1"],
    )
    assert report["complete"], report["problems"]
    assert report["recovery"]["recovery_seconds"] == pytest.approx(recovery)
    assert report["metrics"]["rollouts_accepted"] == 4
    assert report["metrics"]["policy_versions"] == 3
    assert report["metrics"]["weight_sync_latency_ms"]["max"] == 4.0


def _rewrite(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_audit_detects_duplicates_lost_updates_and_mismatches(tmp_path: Path) -> None:
    nodes = write_synthetic_run(tmp_path)
    ledger_path = nodes["node-0"] / "controller" / "ledger.jsonl"
    ledger = workload.read_jsonl(ledger_path)
    ledger[1]["ids"].append(ledger[0]["ids"][0])  # trained twice
    _rewrite(ledger_path, ledger)
    worker = nodes["node-1"] / "workers" / "node-1-w0.jsonl"
    rows = workload.read_jsonl(worker)
    rows.append(
        {"e": "sync", "v": 2, "sha": SHA[9], "ms": 1.0, "t": 21}
    )  # wrong digest
    rows.append(
        {"e": "sync", "v": 3, "sha": SHA[3], "ms": 1.0, "t": 22}
    )  # never committed
    rows.append(
        {"e": "sub", "id": "ghost", "v": 2, "r": "acc", "t": 23}
    )  # admitted, lost
    _rewrite(worker, rows)
    report = workload.audit_run(
        nodes, controller_node="node-0", scenario="worker_exit", max_policy_lag=1
    )
    metrics = report["metrics"]
    assert metrics["duplicate_optimizer_updates"] == 1
    assert metrics["artifact_digest_mismatches"] == 1
    assert metrics["lost_optimizer_updates"] == 1
    assert metrics["lost_admitted_rollouts"] == 1
    assert not report["complete"]


def test_audit_detects_gaps_phantoms_and_lag_violations(tmp_path: Path) -> None:
    nodes = write_synthetic_run(tmp_path)
    ledger_path = nodes["node-0"] / "controller" / "ledger.jsonl"
    ledger = workload.read_jsonl(ledger_path)
    ledger[1].update({"u": 3, "from": 2, "to": 3, "lags": [2, 0]})
    ledger[1]["ids"].append("never-generated")
    ledger[1]["lags"].append(0)
    _rewrite(ledger_path, ledger)
    report = workload.audit_run(
        nodes, controller_node="node-0", scenario="worker_exit", max_policy_lag=1
    )
    assert report["metrics"]["lost_optimizer_updates"] >= 1
    assert report["metrics"]["phantom_trained_rollouts"] == 1
    assert any("max_policy_lag" in problem for problem in report["problems"])


def test_audit_requires_observed_recovery(tmp_path: Path) -> None:
    nodes = write_synthetic_run(tmp_path, scenario="controller_restart")
    report = workload.audit_run(
        nodes, controller_node="node-0", scenario="controller_restart", max_policy_lag=1
    )
    assert report["recovery"]["recovered"] is False
    assert not report["complete"]


# --------------------------------------------------------------------------- #
# Fault injection: partition proxy and node supervisor
# --------------------------------------------------------------------------- #


def _echo_server() -> tuple[socket.socket, int]:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()

    def serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return

            def echo(c: socket.socket = conn) -> None:
                with c:
                    while data := c.recv(1024):
                        c.sendall(data)

            threading.Thread(target=echo, daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    return server, server.getsockname()[1]


def _roundtrip(port: int, timeout: float = 1.0) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
        conn.sendall(b"ping")
        return conn.recv(4)


def test_partition_proxy_blackholes_and_heals() -> None:
    server, port = _echo_server()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        proxy_port = probe.getsockname()[1]
    proxy = workload.PartitionProxy("127.0.0.1", proxy_port, "127.0.0.1", port)
    proxy.start()
    try:
        assert _roundtrip(proxy_port) == b"ping"
        established = socket.create_connection(("127.0.0.1", proxy_port), timeout=1)
        established.sendall(b"ping")
        assert established.recv(4) == b"ping"
        proxy.partition()
        assert proxy.partitioned
        with pytest.raises(OSError):
            established.sendall(b"ping")
            if not established.recv(4):
                raise ConnectionResetError("closed")
        with pytest.raises((socket.timeout, OSError)):
            _roundtrip(proxy_port, timeout=0.5)
        proxy.heal()
        assert _roundtrip(proxy_port) == b"ping"
    finally:
        proxy.stop()
        server.close()


def _sleeper(seconds: float) -> Any:
    return lambda _inc: [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def test_supervisor_kills_and_restarts_fault_target(tmp_path: Path) -> None:
    config = small_config(
        controller_restart_delay_seconds=0.1, shutdown_grace_seconds=1
    )
    plan = workload.ScenarioPlan("controller_restart", "measured", 3.0, 0.3, 0.0)
    processes = [
        workload.ManagedProcess(
            name="n-controller",
            argv_factory=_sleeper(1.0),
            log_path=tmp_path / "c.log",
            env={},
            restart_delay=0.1,
            cwd=tmp_path,
        ),
        workload.ManagedProcess(
            name="n-w0",
            argv_factory=_sleeper(0.5),
            log_path=tmp_path / "w.log",
            env={},
            restart_delay=0.1,
            cwd=tmp_path,
        ),
    ]
    supervisor = workload.NodeSupervisor(
        node_id="n",
        out_dir=tmp_path,
        plan=plan,
        config=config,
        processes=processes,
        controller_name="n-controller",
        fault_targets={"controller_restart": "n-controller"},
        proxy=None,
        started_at=time.time(),
    )
    summary = supervisor.run()
    events = workload.read_jsonl(tmp_path / "node-events.jsonl")
    kinds = [event["e"] for event in events]
    assert "fault_kill" in kinds and "restart" in kinds
    assert summary["forced_stop"] is False and summary["unplanned_exits"] == 0
    controller = summary["processes"]["n-controller"]
    assert len(controller["pids"]) == 2
    assert controller["exits"][0]["code"] == -9 and controller["exits"][-1]["code"] == 0


class _FakeProxy:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.partitioned = False

    def partition(self) -> None:
        self.calls.append("partition")
        self.partitioned = True

    def heal(self) -> None:
        self.calls.append("heal")
        self.partitioned = False

    def stop(self) -> None:
        self.calls.append("stop")


def test_supervisor_partitions_and_heals_on_schedule(tmp_path: Path) -> None:
    config = small_config()
    plan = workload.ScenarioPlan("network_interruption", "measured", 2.0, 0.2, 0.3)
    proxy = _FakeProxy()
    supervisor = workload.NodeSupervisor(
        node_id="n",
        out_dir=tmp_path,
        plan=plan,
        config=config,
        processes=[
            workload.ManagedProcess(
                name="n-w0",
                argv_factory=_sleeper(1.0),
                log_path=tmp_path / "w.log",
                env={},
                restart_delay=0.1,
                cwd=tmp_path,
            )
        ],
        controller_name=None,
        fault_targets={},
        proxy=proxy,  # type: ignore[arg-type]
        started_at=time.time(),
    )
    supervisor.run()
    assert proxy.calls == ["partition", "heal", "stop"]
    events = {
        event["e"]: event["t"]
        for event in workload.read_jsonl(tmp_path / "node-events.jsonl")
    }
    assert events["partition_end"] - events["partition_start"] == pytest.approx(
        0.3, abs=0.2
    )
