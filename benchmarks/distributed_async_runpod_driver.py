#!/usr/bin/env python3
"""RunPod provider driver for one (scenario, seed) of the distributed async matrix.

``run_distributed_async_matrix.py`` invokes this script once per scenario and
seed (argv list, ``shell=False``) with its nine placeholders plus fixed
topology flags. The driver provisions ``--node-count`` pods (CPU by default,
GPU with ``--gpu``), checks out the exact harness commit on each, starts one
``distributed_async_workload.py node`` supervisor per pod (the controller pod
exposes its partitionable proxy port; the other pods' workers connect to the
controller pod's public IP and mapped TCP port), polls exit markers, downloads
every node's artifacts into the runner-owned ``--artifact-dir``, terminates
every pod unconditionally, appends the cost ledger, audits the run locally,
and writes the adapter JSON **only** for a complete, verified run.

It composes the fail-closed RunPod machinery of
``runpod_backend_conformance.py``: free public-catalog planning, a recovery
lease per pod, an in-pod self-destruct armed to the remaining billable
lifetime, an authoritative post-allocation ``costPerHr`` check against the
spend ceiling, and termination in ``finally``.

Spending requires an explicit ceiling (``--confirm-max-cost-usd`` or
``STATESET_DA_CONFIRM_MAX_COST_USD``) no larger than the pinned
``config.max_cost_usd_per_scenario`` entry and no smaller than the catalog
worst case for this scenario; otherwise the driver refuses to provision.

    # free: price the full 3-seed x 4-scenario matrix from the public catalog
    python benchmarks/distributed_async_runpod_driver.py <placeholders> --plan
    # free, offline: validate inputs and print the remote commands
    python benchmarks/distributed_async_runpod_driver.py <placeholders> --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shlex
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
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
    SCENARIOS,
    WorkloadConfig,
    WorkloadError,
    adapter_from_audit,
    audit_run,
    parse_config,
    parse_pinned_config,
    scenario_plan,
)
from runpod_backend_conformance import (  # noqa: E402
    _CATALOG_URL,
    _REMOTE_REPOSITORY,
    _REPOSITORY_URL,
    RunPodConformanceError,
    _arm_self_destruct,
    _positive_price,
    _public_key,
    _run_checked,
    _write_lease,
    fetch_catalog,
)

from stateset_agents.remote.executor import RemoteExecutionError  # noqa: E402
from stateset_agents.remote.ledger import CostEntry, record_entry  # noqa: E402
from stateset_agents.remote.runpod import (  # noqa: E402
    DEFAULT_RUNPOD_LEASE_DIR,
    RunPodApi,
    SshTransport,
)

PROVIDER = "runpod"
COST_SOURCE = "runpod-api"
INTERCONNECT = "public-internet-tcp (RunPod exposed TCP port)"
ENV_CEILING = "STATESET_DA_CONFIRM_MAX_COST_USD"
DEFAULT_CPU_FLAVOR = "cpu3c"
DEFAULT_VCPUS = 4
DEFAULT_TIER = "SECURE"
DEFAULT_IMAGE = "runpod/pytorch:1.1.0-cu1290-torch271-ubuntu2204"
DEFAULT_DISK_GB = 30
DEFAULT_POLL_SECONDS = 120.0
DEFAULT_MATRIX_SEEDS = 3
#: Billable seconds beyond the scenario itself: provisioning, image pull,
#: harness install, startup, shutdown grace, artifact download, termination.
SETUP_HEADROOM_SECONDS = 1800
CONTROLLER_PORT = 29510
PROXY_PORT = 29500
_ALLOWED_TIERS = frozenset({"SECURE", "COMMUNITY"})
_REMOTE_OUTPUT = "/workspace/da-output"
_REMOTE_EXIT = "/workspace/da-node.exit"
_REMOTE_NOHUP_LOG = "/workspace/da-nohup.log"
_REMOTE_WORKER_KEY = "/workspace/.stateset-da-worker-key"
_PROVIDER_RECORD = "runpod-provider.json"


class RunPodDriverError(RuntimeError):
    """Raised when the driver cannot guarantee a safe, honest scenario run."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Inputs, catalog, and planning (no network except the public catalog)
# --------------------------------------------------------------------------- #


def load_inputs(args: argparse.Namespace, installed_version: str) -> WorkloadConfig:
    if args.framework_version != installed_version:
        raise RunPodDriverError(
            f"local stateset_agents {installed_version} does not match the pinned "
            f"framework_version {args.framework_version}"
        )
    try:
        config = parse_config(parse_pinned_config(args.config_json, args.config_sha256))
        scenario_plan(config, args.scenario, args.mode)
    except WorkloadError as exc:
        raise RunPodDriverError(str(exc)) from exc
    if config.max_cost_usd_per_scenario is None:
        raise RunPodDriverError("config.max_cost_usd_per_scenario is required")
    if args.interconnect != INTERCONNECT:
        raise RunPodDriverError(f"RunPod pods communicate over {INTERCONNECT!r}")
    if args.node_count < 2 or args.worker_count < args.node_count:
        raise RunPodDriverError("need >= 2 nodes and at least one worker per node")
    return config


def execution_settings(
    args: argparse.Namespace, config: WorkloadConfig
) -> dict[str, Any]:
    tier = str(args.provider_tier)
    if tier not in _ALLOWED_TIERS:
        raise RunPodDriverError("--provider-tier must be SECURE or COMMUNITY")
    if args.gpu:
        if config.device == "cpu":
            raise RunPodDriverError("--gpu requires config.device cuda or auto")
        if args.accelerator != args.gpu:
            raise RunPodDriverError("--accelerator must name the --gpu type exactly")
        compute = "GPU"
    else:
        if config.device == "cuda":
            raise RunPodDriverError("config.device=cuda needs --gpu")
        if args.accelerator != "CPU":
            raise RunPodDriverError("CPU pods must declare --accelerator CPU")
        if tier != "SECURE":
            raise RunPodDriverError("RunPod CPU pods are Secure Cloud only")
        compute = "CPU"
    if args.vcpus < 2:
        raise RunPodDriverError("--vcpus must be at least 2")
    return {
        "provider": PROVIDER,
        "compute_type": compute,
        "provider_tier": tier,
        "cpu_flavor": args.cpu_flavor if compute == "CPU" else None,
        "vcpus": int(args.vcpus) if compute == "CPU" else None,
        "gpu_name": args.gpu if compute == "GPU" else None,
        "node_count": int(args.node_count),
        "worker_count": int(args.worker_count),
        "container_image": args.container_image,
        "container_disk_gb": int(args.container_disk_gb),
    }


def cpu_instance_id(flavor: Mapping[str, Any], vcpus: int) -> str:
    multiplier = flavor.get("ramMultiplier")
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, int)
        or multiplier < 1
    ):
        raise RunPodDriverError("RunPod CPU flavor has no RAM multiplier")
    low, high = flavor.get("minVcpu"), flavor.get("maxVcpu")
    if isinstance(low, int) and isinstance(high, int) and not low <= vcpus <= high:
        raise RunPodDriverError(f"{flavor.get('id')} supports {low}-{high} vCPUs")
    return f"{flavor['id']}-{vcpus}-{vcpus * multiplier}"


def fetch_cpu_catalog(flavor_id: str, vcpus: int) -> dict[str, Any]:
    """Price one CPU instance from RunPod's public GraphQL catalog (no key)."""
    import requests

    def _post(query: str) -> Mapping[str, Any]:
        try:
            response = requests.post(_CATALOG_URL, json={"query": query}, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RunPodDriverError(f"RunPod catalog request failed: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("errors"):
            raise RunPodDriverError("RunPod CPU catalog returned GraphQL errors")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise RunPodDriverError("RunPod CPU catalog response is malformed")
        return data

    flavors = _post(
        "query { cpuFlavors { id displayName minVcpu maxVcpu ramMultiplier } }"
    ).get("cpuFlavors")
    matches = [
        row
        for row in (flavors or [])
        if isinstance(row, Mapping) and row.get("id") == flavor_id
    ]
    if len(matches) != 1:
        raise RunPodDriverError(f"RunPod catalog has no CPU flavor {flavor_id!r}")
    flavor = dict(matches[0])
    instance = cpu_instance_id(flavor, vcpus)
    rows = _post(
        "query { cpuFlavors { id specifics(input: {instanceId: "
        + json.dumps(instance)
        + "}) { securePrice stockStatus } } }"
    ).get("cpuFlavors")
    priced = [
        row
        for row in (rows or [])
        if isinstance(row, Mapping) and row.get("id") == flavor_id
    ]
    specifics = priced[0].get("specifics") if len(priced) == 1 else None
    if not isinstance(specifics, Mapping):
        raise RunPodDriverError(f"RunPod catalog has no price for {instance}")
    return {
        **flavor,
        "instance_id": instance,
        "securePrice": specifics.get("securePrice"),
        "stockStatus": specifics.get("stockStatus"),
    }


def unit_price(
    settings: Mapping[str, Any],
    *,
    gpu_catalog: Callable[[], Sequence[Mapping[str, Any]]],
    cpu_catalog: Callable[[str, int], Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Catalog $/h for one pod, plus what was quoted."""
    if settings["compute_type"] == "CPU":
        row = dict(cpu_catalog(str(settings["cpu_flavor"]), int(settings["vcpus"])))
        if row.get("stockStatus") in {None, "", "None"}:
            raise RunPodDriverError(f"{row.get('instance_id')} is out of stock")
        return _positive_price(row.get("securePrice"), "securePrice"), row
    tier = str(settings["provider_tier"])
    name = str(settings["gpu_name"])
    matches = [row for row in gpu_catalog() if row.get("id") == name]
    if len(matches) != 1:
        raise RunPodDriverError(f"RunPod catalog must contain exactly one GPU {name!r}")
    row = dict(matches[0])
    if row.get("secureCloud" if tier == "SECURE" else "communityCloud") is not True:
        raise RunPodDriverError(f"{name} is unavailable in {tier} cloud")
    field = "securePrice" if tier == "SECURE" else "communityPrice"
    return _positive_price(row.get(field), field), row


def lifetime_seconds(config: WorkloadConfig, scenario: str, mode: str) -> int:
    plan = scenario_plan(config, scenario, mode)
    return int(math.ceil(plan.duration_seconds + SETUP_HEADROOM_SECONDS))


def worst_case_usd(price_per_pod_hour: float, node_count: int, lifetime: int) -> float:
    return round(price_per_pod_hour * node_count * lifetime / 3600.0, 4)


def build_plan(
    settings: Mapping[str, Any],
    config: WorkloadConfig,
    *,
    price: float,
    quote: Mapping[str, Any],
    scenario: str,
    mode: str,
    seeds: int,
) -> dict[str, Any]:
    """Worst-case cost for this scenario and for the full matrix."""
    nodes = int(settings["node_count"])
    per_scenario = {}
    matrix_total = 0.0
    for name in SCENARIOS:
        lifetime = lifetime_seconds(config, name, mode)
        worst = worst_case_usd(price, nodes, lifetime)
        cap = float((config.max_cost_usd_per_scenario or {})[name])
        per_scenario[name] = {
            "lifetime_seconds": lifetime,
            "worst_case_cost_usd": worst,
            "max_cost_usd_per_scenario": cap,
            "within_cap": worst <= cap,
        }
        matrix_total += worst * seeds
    return {
        "schema_version": 1,
        "kind": "stateset-distributed-async-runpod-plan",
        "generated_at": _utc_now(),
        "settings": dict(settings),
        "mode": mode,
        "catalog_quote": dict(quote),
        "catalog_cost_per_pod_hr_usd": price,
        "catalog_cost_per_hr_usd_all_pods": round(price * nodes, 6),
        "scenario": scenario,
        "this_scenario_worst_case_cost_usd": per_scenario[scenario][
            "worst_case_cost_usd"
        ],
        "per_scenario": per_scenario,
        "matrix_seeds": seeds,
        "matrix_runs": seeds * len(SCENARIOS),
        "matrix_worst_case_cost_usd": round(matrix_total, 4),
        "setup_headroom_seconds": SETUP_HEADROOM_SECONDS,
        "catalog_source": _CATALOG_URL,
        "catalog_quote_is_authoritative": False,
        "provisions_hardware": False,
    }


def resolve_spend_ceiling(
    flag_value: float | None,
    environ: Mapping[str, str],
    *,
    cap: float,
    worst_case: float,
) -> float:
    """Explicit confirmation; refuses absent, above-cap, or below-worst-case."""
    raw: Any = flag_value
    if raw is None:
        text = environ.get(ENV_CEILING, "").strip()
        if not text:
            raise RunPodDriverError(
                "refusing to provision without an explicit spend ceiling: pass "
                f"--confirm-max-cost-usd or set {ENV_CEILING}"
            )
        try:
            raw = float(text)
        except ValueError as exc:
            raise RunPodDriverError(f"{ENV_CEILING} must be a number") from exc
    ceiling = float(raw)
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise RunPodDriverError("spend ceiling must be a positive finite number")
    if ceiling > cap + 1e-9:
        raise RunPodDriverError(
            f"spend ceiling ${ceiling:.2f} exceeds config.max_cost_usd_per_scenario "
            f"${cap:.2f}"
        )
    if worst_case > ceiling + 1e-9:
        raise RunPodDriverError(
            f"worst case ${worst_case:.2f} for this scenario exceeds the spend "
            f"ceiling ${ceiling:.2f}"
        )
    return ceiling


def resolve_harness_revision(root: Path, explicit: str | None) -> str:
    import re
    import subprocess

    if explicit is not None:
        revision = explicit
    else:
        status = subprocess.run(  # nosec B603 B607 - fixed git argv
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
        )
        if status.returncode != 0 or status.stdout.strip():
            raise RunPodDriverError(
                "harness worktree must be clean (or pass --harness-revision)"
            )
        revision = subprocess.run(  # nosec B603 B607 - fixed git argv
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or revision == "0" * 40:
        raise RunPodDriverError("harness revision must be a 40-character commit")
    return revision


# --------------------------------------------------------------------------- #
# Remote commands (pure; secrets are uploaded, never inlined)
# --------------------------------------------------------------------------- #


def node_ids(node_count: int) -> list[str]:
    return [f"node-{index}" for index in range(node_count)]


def workers_per_node(worker_count: int, node_count: int) -> list[int]:
    base, extra = divmod(worker_count, node_count)
    return [base + (1 if index < extra else 0) for index in range(node_count)]


def node_command(
    args: argparse.Namespace, *, index: int, controller_url: str | None
) -> list[str]:
    ids = node_ids(args.node_count)
    argv = [
        "python",
        "-u",
        "benchmarks/distributed_async_workload.py",
        "node",
        "--node-id",
        ids[index],
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
        f"{_REMOTE_OUTPUT}/{ids[index]}",
        "--workers",
        str(workers_per_node(args.worker_count, args.node_count)[index]),
        "--controller-port",
        str(CONTROLLER_PORT),
        "--proxy-port",
        str(PROXY_PORT),
        "--worker-fault-target",
        f"{ids[-1]}-w0",
    ]
    if index == 0:
        # The proxy is the only publicly bound listener; the controller itself
        # stays on loopback and every call is authenticated.
        argv += ["--controller", "--proxy-host", "0.0.0.0"]  # nosec B104
    else:
        if not controller_url:
            raise RunPodDriverError("worker nodes need the controller URL")
        argv += ["--controller-url", controller_url]
    return argv


def remote_commands(
    args: argparse.Namespace,
    harness_revision: str,
    *,
    index: int,
    controller_url: str | None,
) -> dict[str, str]:
    repo = shlex.quote(_REMOTE_REPOSITORY)
    setup = " && ".join(
        [
            f"git clone --filter=blob:none {shlex.quote(_REPOSITORY_URL)} {repo}",
            f"git -C {repo} checkout --detach {shlex.quote(harness_revision)}",
            f"python -m pip install --quiet -e {repo}'[api]'",
            "python -c "
            + shlex.quote(
                "import sys, torch, fastapi, uvicorn, httpx, stateset_agents as s; "
                f"sys.exit(0 if s.__version__ == {str(args.framework_version)!r} "
                "else 'installed stateset_agents ' + s.__version__)"
            ),
        ]
    )
    workload = " ".join(
        shlex.quote(part)
        for part in node_command(args, index=index, controller_url=controller_url)
    )
    node_dir = f"{_REMOTE_OUTPUT}/{node_ids(args.node_count)[index]}"
    log = shlex.quote(f"{node_dir}/node-agent.log")
    inner = (
        f'export STATESET_DA_WORKER_KEY="$(cat {shlex.quote(_REMOTE_WORKER_KEY)})"; '
        f"cd {repo} && {workload} > {log} 2>&1; "
        f"echo $? > {shlex.quote(_REMOTE_EXIT)}"
    )
    launch = (
        f"rm -f {shlex.quote(_REMOTE_EXIT)} && mkdir -p {shlex.quote(node_dir)} && "
        f"(nohup bash -c {shlex.quote(inner)} > {shlex.quote(_REMOTE_NOHUP_LOG)} "
        "2>&1 < /dev/null &)"
    )
    leftovers = "pgrep -fc distributed_async_workload.py || true"
    return {"setup": setup, "launch": launch, "leftovers": leftovers}


# --------------------------------------------------------------------------- #
# Paid execution
# --------------------------------------------------------------------------- #


def _remaining(lifetime: int, started_at: float, stage: str) -> int:
    remaining = lifetime - max(1, math.ceil(time.time() - started_at))
    if remaining < 1:
        raise RunPodDriverError(
            f"pods exhausted their billable lifetime before {stage}"
        )
    return remaining


def _wait_for_ports(
    api: Any,
    pod_id: str,
    ports: Sequence[str],
    *,
    timeout_s: int,
    poll_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while True:
        last = dict(api.get_pod(pod_id))
        mapping = last.get("portMappings") or {}
        if (
            last.get("desiredStatus") == "RUNNING"
            and last.get("publicIp")
            and all(mapping.get(port) for port in ports)
        ):
            return last
        if time.monotonic() >= deadline:
            raise RunPodDriverError(
                f"pod {pod_id} did not publish ports {list(ports)} within {timeout_s}s "
                f"(last status {last.get('desiredStatus')!r})"
            )
        sleep(poll_s)


def poll_nodes(
    transports: Sequence[Any],
    *,
    deadline_s: int,
    poll_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int | None]:
    """Wait for every node's exit marker; None for a node that never exited."""
    deadline = time.monotonic() + max(1, int(deadline_s))
    codes: list[int | None] = [None] * len(transports)
    while True:
        for index, ssh in enumerate(transports):
            if codes[index] is not None:
                continue
            code, text = ssh.run(f"cat {shlex.quote(_REMOTE_EXIT)} 2>/dev/null")
            if code == 0 and text.strip():
                try:
                    codes[index] = int(text.strip().splitlines()[-1])
                except ValueError:
                    codes[index] = 1
        print(f"{datetime.now(timezone.utc):%H:%M:%SZ} node exits: {codes}", flush=True)
        if all(code is not None for code in codes) or time.monotonic() >= deadline:
            return codes
        sleep(max(0.0, min(float(poll_s), deadline - time.monotonic())))


def _create_pod(
    api: Any,
    settings: Mapping[str, Any],
    *,
    name: str,
    ports: list[str],
    env: dict[str, str],
) -> dict[str, Any]:
    if settings["compute_type"] == "CPU":
        return dict(
            api.create_cpu_pod(
                name=name,
                image=str(settings["container_image"]),
                cpu_flavor_id=str(settings["cpu_flavor"]),
                vcpu_count=int(settings["vcpus"]),
                ports=ports,
                env=env,
                container_disk_gb=int(settings["container_disk_gb"]),
            )
        )
    return dict(
        api.create_pod(
            name=name,
            image=str(settings["container_image"]),
            gpu_type_id=str(settings["gpu_name"]),
            gpu_count=1,
            ports=ports,
            env=env,
            container_disk_gb=int(settings["container_disk_gb"]),
            cloud_type=str(settings["provider_tier"]),
        )
    )


def execute_scenario(
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    plan: Mapping[str, Any],
    harness_revision: str,
    *,
    ceiling: float,
    lifetime: int,
    api: Any,
    ssh_factory: Callable[[], Any],
    public_key: str,
    lease_dir: Path = DEFAULT_RUNPOD_LEASE_DIR,
    ledger_path: Path | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Provision every node, run the scenario, collect artifacts, terminate.

    Never raises for workload failures; raises only when the termination of a
    created pod could not be confirmed (its cleanup lease is retained).
    """
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ids = node_ids(args.node_count)
    pods: list[dict[str, Any]] = []
    error: str | None = None
    status = "failed"
    codes: list[int | None] = []
    leftovers = 0
    started_at = time.time()
    worker_key = secrets.token_urlsafe(32)
    try:
        for index, node in enumerate(ids):
            ports = ["22/tcp"] + ([f"{PROXY_PORT}/tcp"] if index == 0 else [])
            created_at = time.time()
            response = _create_pod(
                api,
                settings,
                name=f"stateset-da-{args.scenario[:12]}-s{args.seed}-{node}-{uuid.uuid4().hex[:6]}",
                ports=ports,
                env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
            )
            pod_id = str(response.get("id") or "")
            if not pod_id:
                raise RunPodDriverError("RunPod create response omitted pod id")
            record: dict[str, Any] = {
                "node": node,
                "pod_id": pod_id,
                "created_at": created_at,
                "ports": ports,
                "price": None,
                "terminated": False,
            }
            pods.append(record)
            record["lease"] = str(
                _write_lease(lease_dir, pod_id, dict(plan), created_at)
            )
            raw_price = response.get("costPerHr")
            if raw_price is None:
                raw_price = api.get_pod(pod_id).get("costPerHr")
            record["price"] = _positive_price(raw_price, "pod costPerHr")
        hourly = sum(float(pod["price"]) for pod in pods)
        worst = round(hourly * lifetime / 3600.0, 4)
        if worst > ceiling:
            raise RunPodDriverError(
                f"authoritative prices give a worst case ${worst:.2f} above the "
                f"ceiling ${ceiling:.2f}"
            )
        transports = []
        for index, pod in enumerate(pods):
            wanted = ["22"] + ([str(PROXY_PORT)] if index == 0 else [])
            info = _wait_for_ports(
                api,
                pod["pod_id"],
                wanted,
                timeout_s=min(900, _remaining(lifetime, started_at, "pod start")),
                sleep=sleep,
            )
            pod["public_ip"] = str(info["publicIp"])
            pod["port_mappings"] = dict(info.get("portMappings") or {})
            pod["machine_id"] = info.get("machineId")
            ssh = ssh_factory()
            ssh.wait_until_reachable(
                pod["public_ip"],
                int(pod["port_mappings"]["22"]),
                min(300, _remaining(lifetime, started_at, "SSH")),
            )
            _arm_self_destruct(
                ssh, api, pod["pod_id"], _remaining(lifetime, started_at, "watchdog")
            )
            ssh.upload_secret(worker_key, _REMOTE_WORKER_KEY)
            transports.append(ssh)
        controller_url = (
            f"http://{pods[0]['public_ip']}:{pods[0]['port_mappings'][str(PROXY_PORT)]}"
        )
        command_sets = [
            remote_commands(
                args,
                harness_revision,
                index=index,
                controller_url=None if index == 0 else controller_url,
            )
            for index in range(len(pods))
        ]
        for ssh, commands in zip(transports, command_sets, strict=True):
            _run_checked(ssh, commands["setup"], "harness checkout and install")
        for ssh, commands in zip(transports, command_sets, strict=True):
            _run_checked(ssh, commands["launch"], "node launch")
        codes = poll_nodes(
            transports,
            deadline_s=_remaining(lifetime, started_at, "scenario"),
            poll_s=poll_seconds,
            sleep=sleep,
        )
        for ssh, commands in zip(transports, command_sets, strict=True):
            _, text = ssh.run(commands["leftovers"])
            try:
                leftovers += int(text.strip().splitlines()[-1])
            except (ValueError, IndexError):
                leftovers += 1
        for ssh, node in zip(transports, ids, strict=True):
            try:
                ssh.download_dir(
                    f"{_REMOTE_OUTPUT}/{node}", artifact_dir / "nodes" / node
                )
            except Exception as exc:  # noqa: BLE001
                raise RunPodDriverError(
                    f"could not retrieve {node} artifacts: {exc}"
                ) from exc
        if any(code is None for code in codes):
            raise RunPodDriverError("a node did not finish within the pod lifetime")
        if any(code != 0 for code in codes):
            raise RunPodDriverError(f"node supervisors exited {codes}")
        status = "completed"
    except (
        RunPodDriverError,
        RunPodConformanceError,
        RemoteExecutionError,
        OSError,
    ) as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - still terminate and account
        error = f"unexpected {type(exc).__name__}: {exc}"
    finally:
        termination_errors: list[str] = []
        for pod in pods:
            try:
                api.terminate_pod(pod["pod_id"])
            except Exception as exc:  # noqa: BLE001
                termination_errors.append(f"{pod['pod_id']}: {exc}")
            else:
                pod["terminated"] = True
                if pod.get("lease"):
                    Path(pod["lease"]).unlink(missing_ok=True)
            pod["billed_seconds"] = float(
                max(1, math.ceil(time.time() - pod["created_at"]))
            )
            # Unrounded: a short run must never read as free.
            pod["cost_usd"] = (
                float(pod["price"]) * pod["billed_seconds"] / 3600.0
                if pod["price"] is not None
                else None
            )
            record_entry(
                CostEntry(
                    provider=PROVIDER,
                    job_id=pod["pod_id"],
                    base_model="distributed-async-mlp",
                    gpu=str(
                        settings.get("gpu_name")
                        or f"cpu:{settings.get('cpu_flavor')}-{settings.get('vcpus')}vcpu"
                    ),
                    gpu_count=1 if settings["compute_type"] == "GPU" else 0,
                    cost_per_hr=pod["price"],
                    duration_s=pod["billed_seconds"],
                    cost_usd=pod["cost_usd"],
                    status=status,
                ),
                path=ledger_path,
            )
    all_terminated = all(pod["terminated"] for pod in pods)
    if status == "completed" and not all_terminated:
        status = "cleanup-pending"
    costs = [pod.get("cost_usd") for pod in pods]
    total = (
        round(sum(float(cost) for cost in costs), 6)
        if pods and all(isinstance(cost, (int, float)) for cost in costs)
        else None
    )
    outcome = {
        "status": status,
        "error": error,
        "node_exit_codes": codes,
        "pods": [
            {key: value for key, value in pod.items() if key != "lease"} for pod in pods
        ],
        "harness_revision": harness_revision,
        "cost_usd": total,
        "termination_confirmed": all_terminated,
        "leftover_processes": leftovers,
        "resources_remaining": leftovers + sum(not pod["terminated"] for pod in pods),
    }
    provider_record = {
        **outcome,
        "schema_version": 1,
        "kind": "stateset-distributed-async-runpod-provider-record",
        "cost_source": COST_SOURCE,
        "recorded_at": _utc_now(),
        "scenario": args.scenario,
        "seed": args.seed,
        "max_cost_usd": ceiling,
        "settings": dict(settings),
        "plan_worst_case_usd": plan.get("this_scenario_worst_case_cost_usd"),
    }
    try:
        (artifact_dir / _PROVIDER_RECORD).write_text(
            json.dumps(provider_record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"provider record not written: {exc}", file=sys.stderr)
    if pods and not all_terminated:
        raise RunPodDriverError(
            "termination was not confirmed for every pod; cleanup leases retained: "
            + "; ".join(termination_errors)
        )
    return outcome


# --------------------------------------------------------------------------- #
# Adapter (only for complete, verified runs)
# --------------------------------------------------------------------------- #


def verify_and_build_adapter(
    args: argparse.Namespace,
    config: WorkloadConfig,
    outcome: Mapping[str, Any],
    ceiling: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    problems: list[str] = []
    if outcome.get("status") != "completed":
        problems.append(str(outcome.get("error") or f"status={outcome.get('status')}"))
        return None, problems
    artifact_dir = Path(args.artifact_dir)
    ids = node_ids(args.node_count)
    node_dirs = {node: artifact_dir / "nodes" / node for node in ids}
    summaries = {}
    for node, directory in node_dirs.items():
        path = directory / "node-summary.json"
        if not path.exists():
            problems.append(f"{node} wrote no node summary")
            continue
        summaries[node] = json.loads(path.read_text(encoding="utf-8"))
    if problems:
        return None, problems
    for node, summary in summaries.items():
        device = summary.get("device", {})
        if device.get("accelerator") != args.accelerator:
            problems.append(f"{node} ran on {device.get('accelerator')!r}")
        if device.get("accelerator_driver") != args.accelerator_driver:
            problems.append(f"{node} driver is {device.get('accelerator_driver')!r}")
        if summary.get("forced_stop"):
            problems.append(f"{node} supervisor was force-stopped")
    plan = scenario_plan(config, args.scenario, args.mode)
    duration = float(summaries[ids[0]]["duration_seconds"])
    if duration < plan.duration_seconds:
        problems.append("scenario ended before its planned duration")
    audit = audit_run(
        node_dirs,
        controller_node=ids[0],
        scenario=args.scenario,
        max_policy_lag=config.max_policy_lag,
        remote_nodes=ids[1:],
    )
    problems.extend(audit["problems"])
    cost = outcome.get("cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost <= 0:
        problems.append("provider cost is unknown")
    elif float(cost) > ceiling + 1e-9:
        problems.append(f"provider cost ${cost} exceeded the ceiling ${ceiling}")
    if outcome.get("resources_remaining"):
        problems.append("resources remained after the scenario")
    (artifact_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if problems:
        return None, problems
    pods = list(outcome.get("pods", []))
    machines = [pod.get("machine_id") for pod in pods]
    topology: dict[str, Any] = {
        "node_count": args.node_count,
        "worker_count": args.worker_count,
        "node_ids": [f"runpod:{pod['pod_id']}" for pod in pods],
        "accelerator": args.accelerator,
        "accelerator_driver": args.accelerator_driver,
        "interconnect": args.interconnect,
        "instance": (
            f"{args.cpu_flavor}-{args.vcpus}vcpu" if not args.gpu else args.gpu
        ),
    }
    if all(isinstance(machine, str) and machine for machine in machines):
        topology["physical_hosts"] = len(set(machines))
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
        cost_total_usd=float(cost),
        cost_source=COST_SOURCE,
        resources_remaining=int(outcome.get("resources_remaining", 0)),
        artifact_path=artifact_dir,
        extra={
            "harness_revision": outcome.get("harness_revision"),
            "pods": pods,
            "max_cost_usd": ceiling,
            "mode": args.mode,
        },
    )
    return adapter, []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


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
    parser.add_argument("--interconnect", default=INTERCONNECT)
    parser.add_argument("--plan", action="store_true", help="price only (free)")
    parser.add_argument("--dry-run", action="store_true", help="offline preview")
    parser.add_argument("--matrix-seeds", type=int, default=DEFAULT_MATRIX_SEEDS)
    parser.add_argument("--confirm-max-cost-usd", type=float)
    parser.add_argument("--cpu-flavor", default=DEFAULT_CPU_FLAVOR)
    parser.add_argument("--vcpus", type=int, default=DEFAULT_VCPUS)
    parser.add_argument("--gpu", default=None, help="use GPU pods of this type")
    parser.add_argument("--provider-tier", default=DEFAULT_TIER)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-disk-gb", type=int, default=DEFAULT_DISK_GB)
    parser.add_argument("--harness-revision", default=None)
    parser.add_argument("--ssh-public-key", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--lease-dir", type=Path, default=DEFAULT_RUNPOD_LEASE_DIR)
    parser.add_argument("--ledger", type=Path, default=None)
    return parser


def _installed_version() -> str:
    from stateset_agents import __version__

    return str(__version__)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    gpu_catalog: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    cpu_catalog: Callable[[str, int], Mapping[str, Any]] | None = None,
    api_factory: Callable[[str], Any] | None = None,
    ssh_factory: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if environ is None else environ
    make_api = api_factory or RunPodApi
    make_ssh = ssh_factory or SshTransport
    try:
        config = load_inputs(args, _installed_version())
        settings = execution_settings(args, config)
        lifetime = lifetime_seconds(config, args.scenario, args.mode)
        cap = float((config.max_cost_usd_per_scenario or {})[args.scenario])
        if args.dry_run:
            revision = args.harness_revision or "<HEAD at execution>"
            print(
                json.dumps(
                    {
                        "kind": "stateset-distributed-async-runpod-dry-run",
                        "settings": settings,
                        "lifetime_seconds": lifetime,
                        "max_cost_usd_per_scenario": cap,
                        "commands": [
                            remote_commands(
                                args,
                                revision,
                                index=index,
                                controller_url=(
                                    None
                                    if index == 0
                                    else "http://<controller-ip>:<mapped-port>"
                                ),
                            )
                            for index in range(args.node_count)
                        ],
                        "provisions_hardware": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        price, quote = unit_price(
            settings,
            gpu_catalog=gpu_catalog or fetch_catalog,
            cpu_catalog=cpu_catalog or fetch_cpu_catalog,
        )
        plan = build_plan(
            settings,
            config,
            price=price,
            quote=quote,
            scenario=args.scenario,
            mode=args.mode,
            seeds=args.matrix_seeds,
        )
        if args.plan:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        ceiling = resolve_spend_ceiling(
            args.confirm_max_cost_usd,
            env,
            cap=cap,
            worst_case=float(plan["this_scenario_worst_case_cost_usd"]),
        )
        revision = resolve_harness_revision(_ROOT, args.harness_revision)
        api_key = str(env.get("RUNPOD_API_KEY", "")).strip()
        if not api_key:
            raise RunPodDriverError("RUNPOD_API_KEY is not set")
        public_key, private_key = _public_key(args.ssh_public_key)
        outcome = execute_scenario(
            args,
            settings,
            plan,
            revision,
            ceiling=ceiling,
            lifetime=lifetime,
            api=make_api(api_key),
            ssh_factory=lambda: make_ssh(key_path=private_key),
            public_key=public_key,
            lease_dir=args.lease_dir,
            ledger_path=args.ledger,
            poll_seconds=args.poll_seconds,
            sleep=sleep,
        )
        adapter, problems = verify_and_build_adapter(args, config, outcome, ceiling)
        if adapter is None:
            print("scenario incomplete: " + "; ".join(problems), file=sys.stderr)
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(adapter, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(adapter["metrics"], sort_keys=True))
        return 0
    except (
        RunPodDriverError,
        RunPodConformanceError,
        RemoteExecutionError,
        WorkloadError,
        OSError,
    ) as exc:
        print(f"distributed async RunPod driver rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
