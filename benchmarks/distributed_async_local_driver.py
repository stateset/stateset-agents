#!/usr/bin/env python3
"""Local *diagnostic* provider driver for the distributed async matrix.

Implements the argv contract of ``run_distributed_async_matrix.py`` by running
every "node" of ``benchmarks/distributed_async_workload.py`` as its own process
group (own session, own working directory, own supervised workers) on this
host, connected over loopback TCP. The controller node exposes the
partitionable proxy that the other nodes use, exactly as a RunPod controller
pod exposes its public port.

The output is honest about what it is: ``provider="local"``,
``cost.source="local-meter"`` with a zero cost, ``interconnect`` as declared
(``loopback-tcp``), and ``topology.physical_hosts=1`` /
``topology.diagnostic=true``. It passes per-run validation (so the whole
pipeline can be exercised end to end) but the publication gate
(:func:`distributed_async_evidence.validate_matrix`) rejects it, because
process groups on one host are not physical nodes.

    python benchmarks/run_distributed_async_matrix.py \\
        benchmarks/distributed_async_local_manifest.json \\
        --output-dir /tmp/da-local --preflight
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BENCHMARKS = Path(__file__).resolve().parent
_ROOT = _BENCHMARKS.parent
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from distributed_async_workload import (  # noqa: E402
    ENV_WORKER_KEY,
    SCENARIOS,
    WorkloadError,
    adapter_from_audit,
    audit_run,
    describe_device,
    parse_config,
    parse_pinned_config,
    resolve_device,
    scenario_plan,
)

PROVIDER = "local"
COST_SOURCE = "local-meter"


class LocalDriverError(RuntimeError):
    """Raised when a local diagnostic run cannot be performed honestly."""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def split_workers(worker_count: int, node_count: int) -> list[int]:
    """Spread workers over nodes; every node gets at least one."""
    if node_count < 2 or worker_count < node_count:
        raise LocalDriverError("need >= 2 nodes and at least one worker per node")
    base, extra = divmod(worker_count, node_count)
    return [base + (1 if index < extra else 0) for index in range(node_count)]


def check_accelerator(
    declared_accelerator: str, declared_driver: str, detected: Mapping[str, str]
) -> None:
    """Refuse to label a run with hardware it did not use."""
    if declared_accelerator != detected["accelerator"]:
        raise LocalDriverError(
            f"declared accelerator {declared_accelerator!r} but this host runs "
            f"{detected['accelerator']!r}"
        )
    if declared_driver != detected["accelerator_driver"]:
        raise LocalDriverError(
            f"declared accelerator driver {declared_driver!r} but detected "
            f"{detected['accelerator_driver']!r}"
        )


def node_argv(
    args: argparse.Namespace,
    *,
    node_id: str,
    out_dir: Path,
    workers: int,
    controller: bool,
    controller_port: int,
    proxy_port: int,
    worker_fault_target: str,
) -> list[str]:
    argv = [
        sys.executable,
        "-u",
        str(_BENCHMARKS / "distributed_async_workload.py"),
        "node",
        "--node-id",
        node_id,
        "--scenario",
        args.scenario,
        "--mode",
        args.mode,
        "--seed",
        str(args.seed),
        "--config-json",
        args.config_json,
        "--config-sha256",
        args.config_sha256,
        "--out-dir",
        str(out_dir),
        "--workers",
        str(workers),
        "--controller-port",
        str(controller_port),
        "--proxy-port",
        str(proxy_port),
        "--worker-fault-target",
        worker_fault_target,
    ]
    if controller:
        argv.append("--controller")
    else:
        argv.extend(["--controller-url", f"http://127.0.0.1:{proxy_port}"])
    return argv


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_nodes(
    args: argparse.Namespace, *, timeout_seconds: float, workers_per_node: list[int]
) -> dict[str, Any]:
    """Launch one process group per node, wait, and account for leftovers."""
    artifact_dir = Path(args.artifact_dir).resolve()
    controller_port = _free_port()
    proxy_port = _free_port()
    node_ids = [f"local-node-{index}" for index in range(len(workers_per_node))]
    fault_target = f"{node_ids[-1]}-w0"
    env = dict(os.environ)
    env[ENV_WORKER_KEY] = secrets.token_urlsafe(32)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    agents: list[tuple[str, subprocess.Popen[bytes]]] = []
    codes: dict[str, int | None] = {}
    started = time.monotonic()
    try:
        for index, node_id in enumerate(node_ids):
            out_dir = artifact_dir / "nodes" / node_id
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "agent.log").open("ab") as log:
                process = subprocess.Popen(  # nosec B603 - fixed argv, no shell
                    node_argv(
                        args,
                        node_id=node_id,
                        out_dir=out_dir,
                        workers=workers_per_node[index],
                        controller=index == 0,
                        controller_port=controller_port,
                        proxy_port=proxy_port,
                        worker_fault_target=fault_target,
                    ),
                    cwd=_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            agents.append((node_id, process))
            if index == 0:
                time.sleep(0.5)
        for node_id, process in agents:
            remaining = timeout_seconds - (time.monotonic() - started)
            try:
                codes[node_id] = process.wait(max(1.0, remaining))
            except subprocess.TimeoutExpired:
                codes[node_id] = None
    finally:
        leftovers = 0
        for _, process in agents:
            if process.poll() is None:
                leftovers += 1
                process.kill()
                process.wait(10)
        time.sleep(0.2)
        for _, process in agents:
            if _group_alive(process.pid):
                leftovers += 1
                os.killpg(process.pid, signal.SIGKILL)
    return {
        "node_ids": node_ids,
        "exit_codes": codes,
        "resources_remaining": leftovers,
        "wall_seconds": time.monotonic() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("measured", "preflight"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--framework-version", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--node-count", type=int, default=2)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--accelerator", required=True)
    parser.add_argument("--accelerator-driver", required=True)
    parser.add_argument("--interconnect", default="loopback-tcp")
    return parser


def _installed_version() -> str:
    from stateset_agents import __version__

    return str(__version__)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_dir = Path(args.artifact_dir).resolve()
    try:
        if args.framework_version != _installed_version():
            raise LocalDriverError(
                f"installed stateset_agents {_installed_version()} does not match "
                f"{args.framework_version}"
            )
        if args.interconnect != "loopback-tcp":
            raise LocalDriverError("a local run can only declare loopback-tcp")
        raw = parse_pinned_config(args.config_json, args.config_sha256)
        config = parse_config(raw)
        plan = scenario_plan(config, args.scenario, args.mode)
        detected = describe_device(resolve_device(config.device))
        check_accelerator(args.accelerator, args.accelerator_driver, detected)
        workers_per_node = split_workers(args.worker_count, args.node_count)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        timeout = plan.duration_seconds + 4 * float(config.shutdown_grace_seconds) + 180
        launched = run_nodes(
            args, timeout_seconds=timeout, workers_per_node=workers_per_node
        )
        node_ids = launched["node_ids"]
        node_dirs = {node: artifact_dir / "nodes" / node for node in node_ids}
        audit = audit_run(
            node_dirs,
            controller_node=node_ids[0],
            scenario=args.scenario,
            max_policy_lag=config.max_policy_lag,
            remote_nodes=node_ids[1:],
        )
        problems = list(audit["problems"])
        for node, code in launched["exit_codes"].items():
            if code != 0:
                problems.append(f"{node} agent exited {code}")
        if launched["resources_remaining"]:
            problems.append(
                f"{launched['resources_remaining']} process groups outlived the run"
            )
        summary_path = node_dirs[node_ids[0]] / "node-summary.json"
        duration = None
        if summary_path.exists():
            duration = float(json.loads(summary_path.read_text())["duration_seconds"])
        else:
            problems.append("controller node wrote no summary")
        record = {
            "kind": "stateset-distributed-async-local-run",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "scenario": args.scenario,
            "seed": args.seed,
            "mode": args.mode,
            "plan": plan.to_dict(),
            "launched": launched,
            "detected_device": detected,
            "audit": audit,
            "problems": problems,
            "publishable": False,
        }
        (artifact_dir / "local-run.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        if problems or duration is None:
            print("local run incomplete: " + "; ".join(problems), file=sys.stderr)
            return 1
        topology = {
            "node_count": args.node_count,
            "worker_count": args.worker_count,
            "node_ids": node_ids,
            "accelerator": args.accelerator,
            "accelerator_driver": args.accelerator_driver,
            "interconnect": args.interconnect,
            "physical_hosts": 1,
            "isolation": "process-group per node, own working directory",
            "diagnostic": True,
        }
        adapter = adapter_from_audit(
            audit,
            scenario=args.scenario,
            seed=args.seed,
            protocol=args.protocol,
            framework_version=args.framework_version,
            provider=PROVIDER,
            config_sha256=args.config_sha256,
            duration_seconds=duration,
            topology=topology,
            cost_total_usd=0.0,
            cost_source=COST_SOURCE,
            resources_remaining=launched["resources_remaining"],
            artifact_path=artifact_dir,
            extra={"diagnostic": True, "publishable": False, "mode": args.mode},
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(adapter, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(adapter["metrics"], sort_keys=True))
        return 0
    except (LocalDriverError, WorkloadError, OSError) as exc:
        print(f"local distributed async driver rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
