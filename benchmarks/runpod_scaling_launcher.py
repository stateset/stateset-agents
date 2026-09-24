#!/usr/bin/env python3
"""Budget-bounded RunPod launcher for one distributed-scaling matrix row.

This argv-only adapter is called by ``run_scaling_matrix.py``. It provisions
one Secure Cloud pod per node with private global networking, launches the
exact committed workload through multi-node torchrun, retrieves rank-zero
evidence, and terminates every pod on every exit path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .runpod_backend_conformance import (
        RunPodConformanceError,
        _arm_self_destruct,
        _positive_price,
        _public_key,
        _run_checked,
        _wait_for_ssh,
        _write_lease,
        fetch_catalog,
    )
    from .scaling_comparison import load_scaling_evidence
except ImportError:  # pragma: no cover - direct script execution
    from runpod_backend_conformance import (
        RunPodConformanceError,
        _arm_self_destruct,
        _positive_price,
        _public_key,
        _run_checked,
        _wait_for_ssh,
        _write_lease,
        fetch_catalog,
    )
    from scaling_comparison import load_scaling_evidence

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.ledger import BudgetExceeded, check_budget
from stateset_agents.remote.runpod import (
    DEFAULT_RUNPOD_LEASE_DIR,
    RunPodApi,
    SshTransport,
)

REMOTE_REPOSITORY = "/workspace/stateset-agents"
REMOTE_OUTPUT = "/workspace/stateset-scaling-output"
REMOTE_EXIT = "/workspace/stateset-scaling.exit"
REMOTE_LOG = "/workspace/stateset-scaling.log"
REPOSITORY_URL = "https://github.com/stateset/stateset-agents.git"
HEX = frozenset("0123456789abcdef")


class RunPodScalingError(ValueError):
    """Raised when a RunPod scaling row cannot be safely executed."""


def _write_provider_record(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically retain one immutable lifecycle record outside evidence discovery."""
    if path.exists():
        raise RunPodScalingError(f"refusing to overwrite provider record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or value < 1:
        raise RunPodScalingError(f"{name} must be a positive integer")
    return value


def validate_request(args: argparse.Namespace) -> dict[str, Any]:
    """Validate topology, immutable inputs, and explicit cost bounds."""
    gpu_count = _positive_int(args.gpu_count, "gpu_count")
    node_count = _positive_int(args.node_count, "node_count")
    nproc = _positive_int(args.nproc_per_node, "nproc_per_node")
    if node_count * nproc != gpu_count:
        raise RunPodScalingError("node_count * nproc_per_node must equal gpu_count")
    if args.seed < 0:
        raise RunPodScalingError("seed must be non-negative")
    if (
        args.workload.name != "distributed_scaling_workload.py"
        or not args.workload.is_file()
    ):
        raise RunPodScalingError(
            "workload must be the checked-out distributed_scaling_workload.py"
        )
    _positive_int(args.container_disk_gb, "container_disk_gb")
    _positive_int(args.timeout_seconds, "timeout_seconds")
    _positive_int(args.max_lifetime_seconds, "max_lifetime_seconds")
    if not 1 <= args.master_port <= 65535:
        raise RunPodScalingError("master_port must be in [1, 65535]")
    if node_count > 1 and args.cloud_type != "SECURE":
        raise RunPodScalingError("multi-node global networking requires SECURE cloud")
    if node_count > 1 and not args.data_center_id:
        raise RunPodScalingError("multi-node execution requires --data-center-id")
    commit = args.harness_commit
    if (
        len(commit) != 40
        or any(char not in HEX for char in commit)
        or commit == "0" * 40
    ):
        raise RunPodScalingError("harness_commit must be a nonzero full commit")
    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        raise RunPodScalingError("config_json is invalid") from exc
    if not isinstance(config, Mapping) or config.get("scaling_mode") != "strong":
        raise RunPodScalingError("RunPod A+ collection requires strong scaling")
    if args.max_lifetime_seconds <= args.timeout_seconds:
        raise RunPodScalingError("max lifetime must exceed workload timeout")
    if not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0:
        raise RunPodScalingError("max_cost_usd must be finite and positive")
    if args.execute and (
        args.confirm_max_cost_usd is None
        or not math.isclose(
            args.confirm_max_cost_usd,
            args.max_cost_usd,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RunPodScalingError(
            "--execute requires --confirm-max-cost-usd exactly equal to max_cost_usd"
        )
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".pt").exists():
        raise RunPodScalingError(f"refusing to overwrite scaling output: {output}")
    return dict(config)


def build_plan(
    args: argparse.Namespace, catalog: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Price the complete multi-pod lifetime before authenticated provisioning."""
    matches = [row for row in catalog if row.get("id") == args.gpu_name]
    if len(matches) != 1:
        raise RunPodScalingError(
            f"RunPod catalog must contain exactly one GPU named {args.gpu_name!r}"
        )
    price_field = "securePrice" if args.cloud_type == "SECURE" else "communityPrice"
    availability = "secureCloud" if args.cloud_type == "SECURE" else "communityCloud"
    if matches[0].get(availability) is not True:
        raise RunPodScalingError(f"{args.gpu_name} is unavailable in {args.cloud_type}")
    unit_price = _positive_price(matches[0].get(price_field), price_field)
    try:
        worst_case = check_budget(
            unit_price,
            args.max_lifetime_seconds,
            args.max_cost_usd,
            gpu_count=args.gpu_count,
        )
    except BudgetExceeded as exc:
        raise RunPodScalingError(str(exc)) from exc
    return {
        "schema_version": 1,
        "kind": "stateset-runpod-scaling-plan",
        "provisions_hardware": False,
        "gpu": args.gpu_name,
        "gpu_count": args.gpu_count,
        "node_count": args.node_count,
        "nproc_per_node": args.nproc_per_node,
        "data_center_id": args.data_center_id,
        "global_networking": args.node_count > 1,
        "catalog_unit_gpu_cost_per_hr_usd": unit_price,
        "catalog_total_cost_per_hr_usd": unit_price * args.gpu_count,
        "max_lifetime_seconds": args.max_lifetime_seconds,
        "max_cost_usd": args.max_cost_usd,
        "worst_case_cost_usd": worst_case,
    }


def _remaining(args: argparse.Namespace, started: float, stage: str) -> int:
    remaining = args.max_lifetime_seconds - max(1, math.ceil(time.time() - started))
    if remaining < 1:
        raise RunPodScalingError(f"pod lifetime exhausted before {stage}")
    return remaining


def _remote_command(args: argparse.Namespace, node_rank: int, master: str) -> str:
    workload = f"{REMOTE_REPOSITORY}/benchmarks/{args.workload.name}"
    output = f"{REMOTE_OUTPUT}/{args.output.name}"
    values = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.node_count}",
        f"--nproc-per-node={args.nproc_per_node}",
        f"--node-rank={node_rank}",
        f"--master-addr={master}",
        f"--master-port={args.master_port}",
        workload,
        "--gpu-count",
        str(args.gpu_count),
        "--seed",
        str(args.seed),
        "--harness-commit",
        args.harness_commit,
        "--config-json",
        args.config_json,
        "--command-label",
        args.command_label,
        "--output",
        output,
    ]
    command = " ".join(shlex.quote(value) for value in values)
    env = "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
    if args.nccl_socket_ifname:
        env += f" NCCL_SOCKET_IFNAME={shlex.quote(args.nccl_socket_ifname)}"
    return f"cd {shlex.quote(REMOTE_REPOSITORY)} && {env} {command}"


def execute(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    *,
    api: Any,
    ssh_factory: Callable[[], Any],
    public_key: str,
    lease_dir: Path = DEFAULT_RUNPOD_LEASE_DIR,
    poll_seconds: float = 5.0,
) -> Path:
    """Provision all nodes, execute one row, retrieve evidence, and clean up."""
    pods: list[dict[str, Any]] = []
    leases: dict[str, Path] = {}
    transports: list[Any] = []
    prices: dict[str, float] = {}
    created_at: dict[str, float] = {}
    started = time.time()
    completed = False
    termination_errors: list[str] = []
    lease_metadata = {
        "schema_version": 1,
        "kind": "stateset-runpod-scaling-lease",
        "harness_commit": args.harness_commit,
        "seed": args.seed,
        "gpu_count": args.gpu_count,
    }
    try:
        for node_rank in range(args.node_count):
            pod = api.create_pod(
                name=f"stateset-scaling-{args.seed}-{node_rank}-{uuid.uuid4().hex[:6]}",
                image=args.image,
                gpu_type_id=args.gpu_name,
                gpu_count=args.nproc_per_node,
                ports=["22/tcp"],
                env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
                container_disk_gb=args.container_disk_gb,
                cloud_type=args.cloud_type,
                support_public_ip=True,
                global_networking=args.node_count > 1,
                data_center_id=args.data_center_id,
            )
            pod_id = str(pod.get("id") or "")
            if not pod_id:
                raise RunPodScalingError("RunPod create response omitted pod id")
            pods.append(pod)
            created_at[pod_id] = time.time()
            try:
                leases[pod_id] = _write_lease(
                    lease_dir, pod_id, lease_metadata, time.time()
                )
            except Exception:
                api.terminate_pod(pod_id)
                pods.pop()
                raise
            raw_price = pod.get("costPerHr")
            if raw_price is None:
                pod.update(api.get_pod(pod_id))
                raw_price = pod.get("costPerHr")
            prices[pod_id] = _positive_price(raw_price, "pod costPerHr")
            try:
                check_budget(
                    sum(prices.values()),
                    args.max_lifetime_seconds,
                    args.max_cost_usd,
                    gpu_count=1,
                )
            except BudgetExceeded as exc:
                raise RunPodScalingError(str(exc)) from exc

        for pod in pods:
            pod_id = str(pod["id"])
            transport = ssh_factory()
            host, port, current = _wait_for_ssh(
                api,
                pod_id,
                timeout_s=min(900, _remaining(args, started, "SSH allocation")),
            )
            pod.update(current)
            transport.wait_until_reachable(
                host, port, min(300, _remaining(args, started, "SSH readiness"))
            )
            _arm_self_destruct(
                transport,
                api,
                pod_id,
                _remaining(args, started, "watchdog arming"),
            )
            transports.append(transport)

        machine_ids = {str(pod.get("machineId") or "") for pod in pods}
        if args.node_count > 1 and (
            "" in machine_ids or len(machine_ids) != args.node_count
        ):
            raise RunPodScalingError(
                "RunPod did not attest one distinct machineId per requested node"
            )

        revision = shlex.quote(args.harness_commit)
        repo = shlex.quote(REMOTE_REPOSITORY)
        setup = (
            f"git clone --filter=blob:none {shlex.quote(REPOSITORY_URL)} {repo} && "
            f"git -C {repo} checkout --detach {revision} && "
            f"python -m pip install --quiet --no-deps {repo}"
        )
        for transport in transports:
            _run_checked(transport, setup, "scaling harness setup")

        master = f"{pods[0]['id']}.runpod.internal"
        for node_rank, transport in enumerate(transports):
            run = _remote_command(args, node_rank, master)
            _run_checked(
                transport,
                f"rm -rf {shlex.quote(REMOTE_OUTPUT)} {shlex.quote(REMOTE_EXIT)} && "
                f"mkdir -p {shlex.quote(REMOTE_OUTPUT)} && "
                f"(nohup bash -c {shlex.quote(run + f'; echo $? > {REMOTE_EXIT}')} "
                f"> {shlex.quote(REMOTE_LOG)} 2>&1 < /dev/null &)",
                f"rank {node_rank} launch",
            )

        remote_code: int | None = None
        while _remaining(args, started, "scaling completion") > 0:
            code, value = transports[0].run(
                f"test -f {shlex.quote(REMOTE_EXIT)} && cat {shlex.quote(REMOTE_EXIT)}"
            )
            if code == 0 and value.strip().isdigit():
                remote_code = int(value.strip())
                break
            time.sleep(poll_seconds)
        if remote_code != 0:
            _, tail = transports[0].run(f"tail -c 4000 {shlex.quote(REMOTE_LOG)}")
            raise RunPodScalingError(
                f"distributed scaling failed ({remote_code}); rank-zero log: {tail}"
            )

        staging = args.output.parent / f".runpod-{args.output.stem}"
        transports[0].download_dir(REMOTE_OUTPUT, staging)
        downloaded = staging / args.output.name
        artifact = staging / args.output.with_suffix(".pt").name
        if not downloaded.is_file() or not artifact.is_file():
            raise RunPodScalingError(
                "rank zero did not return evidence and policy artifact"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(args.output)
        artifact.replace(args.output.with_suffix(".pt"))
        staging.rmdir()
        load_scaling_evidence([args.output])
        completed = True
        return args.output
    finally:
        cleanup: dict[str, bool] = {}
        terminated_at: dict[str, float] = {}
        for pod in reversed(pods):
            pod_id = str(pod["id"])
            try:
                api.terminate_pod(pod_id)
            except Exception as exc:  # preserve every recovery lease
                termination_errors.append(f"{pod_id}: {exc}")
                cleanup[pod_id] = False
            else:
                cleanup[pod_id] = True
                terminated_at[pod_id] = time.time()
                lease = leases.get(pod_id)
                if lease is not None:
                    lease.unlink(missing_ok=True)
        durations = {
            pod_id: max(
                0.0,
                terminated_at.get(pod_id, time.time()) - created_at[pod_id],
            )
            for pod_id in created_at
        }
        pod_records = [
            {
                "pod_id": pod_id,
                "machine_id": str(
                    next(
                        (
                            pod.get("machineId")
                            for pod in pods
                            if pod.get("id") == pod_id
                        ),
                        "",
                    )
                    or ""
                ),
                "authoritative_cost_per_hr_usd": prices.get(pod_id),
                "lifetime_seconds": round(durations[pod_id], 3),
                "estimated_cost_usd": (
                    round(prices[pod_id] * durations[pod_id] / 3600, 6)
                    if pod_id in prices
                    else None
                ),
                "termination_confirmed": cleanup.get(pod_id, False),
                "cleanup_lease_retained": bool(
                    leases.get(pod_id) and leases[pod_id].exists()
                ),
            }
            for pod_id in created_at
        ]
        record = {
            "schema_version": 1,
            "kind": "stateset-runpod-scaling-provider-record",
            "status": (
                "completed"
                if completed and not termination_errors
                else "cleanup-pending" if termination_errors else "failed"
            ),
            "harness_commit": args.harness_commit,
            "image": args.image,
            "seed": args.seed,
            "gpu_count": args.gpu_count,
            "node_count": args.node_count,
            "nproc_per_node": args.nproc_per_node,
            "output": args.output.name,
            "config_sha256": hashlib.sha256(
                json.dumps(
                    json.loads(args.config_json), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "catalog_total_cost_per_hr_usd": plan["catalog_total_cost_per_hr_usd"],
            "pods": pod_records,
            "total_estimated_cost_usd": round(
                sum(float(pod["estimated_cost_usd"] or 0.0) for pod in pod_records),
                6,
            ),
            "all_terminations_confirmed": bool(pod_records)
            and all(bool(pod["termination_confirmed"]) for pod in pod_records),
        }
        record_path = args.output.parent.parent / "runpod-provider" / args.output.name
        try:
            _write_provider_record(record_path, record)
        except (OSError, RunPodScalingError) as exc:
            if completed:
                raise RunPodScalingError(
                    f"evidence retrieved but provider record failed: {exc}"
                ) from exc
        if completed and termination_errors:
            raise RunPodScalingError(
                "evidence retrieved but pod cleanup was not confirmed: "
                + "; ".join(termination_errors)
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--node-count", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--command-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--data-center-id")
    parser.add_argument(
        "--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE"
    )
    parser.add_argument("--container-disk-gb", type=int, default=80)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-lifetime-seconds", type=int, default=2700)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--nccl-socket-ifname")
    parser.add_argument("--ssh-public-key", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-max-cost-usd", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_request(args)
        plan = build_plan(args, fetch_catalog())
        print(json.dumps(plan, indent=2, sort_keys=True))
        if not args.execute:
            return 0
        api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
        if not api_key:
            raise RunPodScalingError("RUNPOD_API_KEY is not set")
        public_key, private_key = _public_key(args.ssh_public_key)
        execute(
            args,
            plan,
            api=RunPodApi(api_key),
            ssh_factory=lambda: SshTransport(key_path=private_key),
            public_key=public_key,
            poll_seconds=args.poll_seconds,
        )
    except (
        RunPodScalingError,
        RunPodConformanceError,
        RemoteExecutionError,
        OSError,
    ) as exc:
        print(f"RunPod scaling rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
