#!/usr/bin/env python3
"""RunPod provider driver for one seed of the StateSet flagship benchmark.

``run_flagship_matrix.py`` invokes this script once per seed (argv list,
``shell=False``) with the twelve manifest placeholders. The driver provisions
exactly one single-GPU pod, installs the exact harness commit, runs
``benchmarks/flagship_workload.py`` detached under ``nohup`` with an exit
marker, polls until the marker appears, downloads every artifact into the
runner-owned ``--artifact-dir``, terminates the pod unconditionally, appends to
the cost ledger, and writes the adapter JSON the matrix runner validates.

It composes the fail-closed RunPod machinery of
``runpod_backend_conformance.py``: free public-catalog planning, a recovery
lease, an in-pod self-destruct armed to the remaining billable lifetime, an
authoritative post-allocation ``costPerHr`` check, and termination in
``finally``.

Spending requires an explicit ceiling (``--confirm-max-cost-usd`` or
``STATESET_FLAGSHIP_CONFIRM_MAX_COST_USD``) no larger than the manifest's
``max_cost_usd_per_seed``; without it the driver refuses to provision.

    # free: price the worst case from the public catalog, print the plan
    python benchmarks/flagship_runpod_driver.py <placeholders...> --plan
    # free, offline: validate inputs and print the remote commands
    python benchmarks/flagship_runpod_driver.py <placeholders...> --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BENCHMARKS = Path(__file__).resolve().parent
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from flagship_workload import (  # noqa: E402
    WorkloadError,
    digest_json,
    parse_pinned_json,
    validate_config,
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
    _wait_for_ssh,
    _write_lease,
    fetch_catalog,
)

from stateset_agents.remote.executor import RemoteExecutionError  # noqa: E402
from stateset_agents.remote.ledger import (  # noqa: E402
    BudgetExceeded,
    CostEntry,
    check_budget,
    estimate_cost_usd,
    record_entry,
)
from stateset_agents.remote.runpod import (  # noqa: E402
    DEFAULT_RUNPOD_LEASE_DIR,
    RunPodApi,
    SshTransport,
)

COST_SOURCE = "runpod-api"
ENV_CEILING = "STATESET_FLAGSHIP_CONFIRM_MAX_COST_USD"
ENV_GPU = "STATESET_FLAGSHIP_GPU"
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"
DEFAULT_TIER = "SECURE"
DEFAULT_IMAGE = "runpod/pytorch:1.1.0-cu1290-torch271-ubuntu2204"
DEFAULT_DISK_GB = 150
DEFAULT_POLL_SECONDS = 120.0
#: Seconds of the external wall-clock budget kept for local provisioning,
#: artifact download, and termination; the pod lifetime gets the rest.
LOCAL_HEADROOM_SECONDS = 600
_ALLOWED_TIERS = frozenset({"SECURE", "COMMUNITY"})
_REMOTE_ROOT = "/workspace/flagship-output"
_REMOTE_ARTIFACTS = f"{_REMOTE_ROOT}/artifacts"
_REMOTE_RESULT = f"{_REMOTE_ROOT}/result.json"
_REMOTE_EXIT = "/workspace/flagship.exit"
_REMOTE_NOHUP_LOG = "/workspace/flagship-nohup.log"
_REMOTE_HF_TOKEN = "/workspace/.stateset-flagship-hf-token"
_RESULT_NAME = "workload_result.json"
_PROVIDER_RECORD = "runpod-provider.json"


class FlagshipDriverError(RuntimeError):
    """Raised when the driver cannot guarantee a safe, honest seed run."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Input validation and planning (no network, no spend)
# --------------------------------------------------------------------------- #


def load_pinned_inputs(
    args: argparse.Namespace, installed_version: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify version and digests locally before anything is provisioned."""
    if args.framework_version != installed_version:
        raise FlagshipDriverError(
            f"local stateset_agents {installed_version} does not match the pinned "
            f"framework_version {args.framework_version}"
        )
    try:
        config = parse_pinned_json(args.config_json, args.config_sha256, "config")
        judge = parse_pinned_json(args.judge_json, args.judge_sha256, "judge")
        validate_config(config)
    except WorkloadError as exc:
        raise FlagshipDriverError(str(exc)) from exc
    for key in ("max_wall_clock_seconds", "max_cost_usd_per_seed"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise FlagshipDriverError(f"config.{key} must be a positive number")
    for key in ("model", "revision", "rubric_revision"):
        if not isinstance(judge.get(key), str) or not judge[key]:
            raise FlagshipDriverError(f"judge.{key} must be a non-empty string")
    return config, judge


def reject_template_revisions(
    args: argparse.Namespace, judge: Mapping[str, Any]
) -> None:
    """Paid runs never execute against the all-zero placeholder revisions."""
    for label, value in (
        ("model_revision", args.model_revision),
        ("dataset_revision", args.dataset_revision),
        ("judge.revision", judge.get("revision")),
        ("judge.rubric_revision", judge.get("rubric_revision")),
    ):
        if value == "0" * 40:
            raise FlagshipDriverError(
                f"{label} is the template zero revision; publish and pin it first"
            )


def resolve_spend_ceiling(
    flag_value: float | None,
    environ: Mapping[str, str],
    per_seed_cap: float,
) -> float:
    """The explicit spend confirmation; absent, non-finite, or above cap refuses."""
    raw: Any = flag_value
    if raw is None:
        text = environ.get(ENV_CEILING, "").strip()
        if not text:
            raise FlagshipDriverError(
                "refusing to provision without an explicit spend ceiling: pass "
                f"--confirm-max-cost-usd or set {ENV_CEILING}"
            )
        try:
            raw = float(text)
        except ValueError as exc:
            raise FlagshipDriverError(f"{ENV_CEILING} must be a number") from exc
    ceiling = float(raw)
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise FlagshipDriverError("spend ceiling must be a positive finite number")
    if ceiling > float(per_seed_cap) + 1e-9:
        raise FlagshipDriverError(
            f"spend ceiling ${ceiling:.2f} exceeds the manifest's "
            f"max_cost_usd_per_seed ${float(per_seed_cap):.2f}"
        )
    return ceiling


def execution_settings(
    args: argparse.Namespace, config: Mapping[str, Any], environ: Mapping[str, str]
) -> dict[str, Any]:
    """Pod shape and lifetime derived from the pinned config and driver flags."""
    tier = str(args.provider_tier)
    if tier not in _ALLOWED_TIERS:
        raise FlagshipDriverError("--provider-tier must be SECURE or COMMUNITY")
    lifetime = int(float(config["max_wall_clock_seconds"])) - LOCAL_HEADROOM_SECONDS
    if lifetime < 1800:
        raise FlagshipDriverError(
            "config.max_wall_clock_seconds leaves under 30 minutes of pod lifetime"
        )
    return {
        "provider": "runpod",
        "provider_tier": tier,
        "gpu_name": args.gpu or environ.get(ENV_GPU, "").strip() or DEFAULT_GPU,
        "gpu_count": 1,
        "container_image": args.container_image,
        "container_disk_gb": int(args.container_disk_gb),
        "max_lifetime_seconds": lifetime,
    }


def build_plan(
    settings: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    ceiling: float,
) -> dict[str, Any]:
    """Price the full pod lifetime from the public catalog (no authentication)."""
    tier = str(settings["provider_tier"])
    gpu_name = str(settings["gpu_name"])
    matches = [row for row in catalog if row.get("id") == gpu_name]
    if len(matches) != 1:
        raise FlagshipDriverError(
            f"RunPod catalog must contain exactly one GPU named {gpu_name!r}"
        )
    row = matches[0]
    availability_field = "secureCloud" if tier == "SECURE" else "communityCloud"
    price_field = "securePrice" if tier == "SECURE" else "communityPrice"
    if row.get(availability_field) is not True:
        raise FlagshipDriverError(f"{gpu_name} is unavailable in {tier} cloud")
    unit_price = _positive_price(row.get(price_field), price_field)
    lifetime = int(settings["max_lifetime_seconds"])
    gpu_count = int(settings["gpu_count"])
    try:
        worst_case = check_budget(unit_price, lifetime, ceiling, gpu_count=gpu_count)
    except BudgetExceeded as exc:
        raise FlagshipDriverError(str(exc)) from exc
    return {
        "schema_version": 1,
        "kind": "stateset-flagship-runpod-plan",
        "generated_at": _utc_now(),
        **{key: settings[key] for key in sorted(settings)},
        "gpu_type_id": gpu_name,
        "catalog_unit_cost_per_hr_usd": unit_price,
        "catalog_total_cost_per_hr_usd": round(unit_price * gpu_count, 6),
        "worst_case_cost_usd": worst_case,
        "max_cost_usd": ceiling,
        "catalog_source": _CATALOG_URL,
        "catalog_quote_is_authoritative": False,
        "provisions_hardware": False,
    }


def resolve_harness_revision(root: Path, explicit: str | None) -> str:
    """The exact commit the pod checks out; local measured runs must be clean."""
    if explicit is not None:
        revision = explicit
    else:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0 or status.stdout.strip():
            raise FlagshipDriverError(
                "harness worktree must be clean (or pass --harness-revision)"
            )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or revision == "0" * 40:
        raise FlagshipDriverError("harness revision must be a 40-character commit")
    return revision


def workload_argv(args: argparse.Namespace) -> list[str]:
    """The on-pod workload command (argv; quoted by :func:`remote_commands`)."""
    return [
        "python",
        "-u",
        "benchmarks/flagship_workload.py",
        "--seed",
        str(args.seed),
        "--mode",
        str(args.mode),
        "--framework-version",
        str(args.framework_version),
        "--model",
        str(args.model),
        "--model-revision",
        str(args.model_revision),
        "--dataset-revision",
        str(args.dataset_revision),
        "--config-json",
        str(args.config_json),
        "--config-sha256",
        str(args.config_sha256),
        "--judge-json",
        str(args.judge_json),
        "--judge-sha256",
        str(args.judge_sha256),
        "--output",
        _REMOTE_RESULT,
        "--artifact-dir",
        _REMOTE_ARTIFACTS,
    ]


def remote_commands(
    args: argparse.Namespace,
    harness_revision: str,
    *,
    with_hf_token: bool,
) -> dict[str, str]:
    """Setup and detached-launch shell commands (pure; no secrets inline)."""
    repo = shlex.quote(_REMOTE_REPOSITORY)
    setup = " && ".join(
        [
            f"git clone --filter=blob:none {shlex.quote(_REPOSITORY_URL)} {repo}",
            f"git -C {repo} checkout --detach {shlex.quote(harness_revision)}",
            f"python -m pip install --quiet -e {repo}'[training,vllm]'",
            "python -c "
            + shlex.quote(
                "import sys, stateset_agents as s; "
                f"sys.exit(0 if s.__version__ == {str(args.framework_version)!r} "
                "else 'installed stateset_agents ' + s.__version__ + "
                f"' != ' + {str(args.framework_version)!r})"
            ),
        ]
    )
    workload = " ".join(shlex.quote(part) for part in workload_argv(args))
    log = shlex.quote(f"{_REMOTE_ARTIFACTS}/logs/workload.log")
    exports = (
        f'export HF_TOKEN="$(cat {shlex.quote(_REMOTE_HF_TOKEN)})"; '
        if with_hf_token
        else ""
    )
    inner = (
        f"{exports}cd {repo} && {workload} > {log} 2>&1; "
        f"echo $? > {shlex.quote(_REMOTE_EXIT)}"
    )
    launch = (
        f"rm -f {shlex.quote(_REMOTE_EXIT)} && "
        f"mkdir -p {shlex.quote(_REMOTE_ARTIFACTS + '/logs')} && "
        f"(nohup bash -c {shlex.quote(inner)} > {shlex.quote(_REMOTE_NOHUP_LOG)} "
        "2>&1 < /dev/null &)"
    )
    return {"setup": setup, "launch": launch}


# --------------------------------------------------------------------------- #
# Paid execution
# --------------------------------------------------------------------------- #


def _remaining(lifetime: int, started_at: float, stage: str) -> int:
    remaining = lifetime - max(1, math.ceil(time.time() - started_at))
    if remaining < 1:
        raise FlagshipDriverError(f"pod exhausted its billable lifetime before {stage}")
    return remaining


#: Provider states from which a pod never comes back on its own.
_TERMINAL_POD_STATES = frozenset({"EXITED", "TERMINATED", "DEAD"})


def poll_until_exit(
    ssh: Any,
    artifact_dir: Path,
    *,
    deadline_s: int,
    poll_s: float,
    pod_status: Callable[[], str | None] | None = None,
) -> int | None:
    """Sync logs every ``poll_s`` seconds until the exit marker appears.

    Returns the workload's exit code, or None when ``deadline_s`` elapses.
    ``pod_status`` (the provider's view of the pod) is checked every poll: a
    pod the provider stopped (e.g. "Exited by Runpod") raises immediately
    instead of polling a dead host until the lifetime runs out.
    """
    deadline = time.monotonic() + max(1, int(deadline_s))
    started = time.monotonic()
    while True:
        if pod_status is not None:
            status = pod_status()
            if status in _TERMINAL_POD_STATES:
                raise FlagshipDriverError(
                    f"provider reports the pod {status} before the workload finished"
                )
        try:
            ssh.download_dir(f"{_REMOTE_ARTIFACTS}/logs", artifact_dir / "logs")
        except Exception as exc:  # noqa: BLE001 - transient scp failure, keep polling
            print(f"log sync deferred: {exc}", file=sys.stderr, flush=True)
        _, tail = ssh.run(
            f"tail -n 1 {shlex.quote(_REMOTE_ARTIFACTS + '/logs/workload.log')} "
            "2>/dev/null"
        )
        print(
            f"{datetime.now(timezone.utc):%H:%M:%SZ} poll "
            f"elapsed={time.monotonic() - started:.0f}s | {tail.strip()[-160:]}",
            flush=True,
        )
        code, text = ssh.run(f"cat {shlex.quote(_REMOTE_EXIT)} 2>/dev/null")
        if code == 0 and text.strip():
            try:
                return int(text.strip().splitlines()[-1])
            except ValueError:
                return 1
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.0, min(float(poll_s), deadline - time.monotonic())))


def _require_credit(api: Any, plan: Mapping[str, Any]) -> None:
    """Refuse to provision when the account cannot cover the worst case.

    RunPod reports exhausted credit as an opaque HTTP 500 on pod creation; an
    unreadable balance does not block the run.
    """
    reader = getattr(api, "account_balance", None)
    balance = reader() if callable(reader) else None
    worst = float(plan.get("worst_case_cost_usd") or 0.0)
    if balance is not None and balance < worst:
        raise FlagshipDriverError(
            f"RunPod credit ${balance:.2f} is below this seed's worst case "
            f"${worst:.2f}; top up the account before provisioning (RunPod "
            "reports insufficient credit as an opaque HTTP 500)"
        )


def execute_seed(
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    plan: Mapping[str, Any],
    harness_revision: str,
    *,
    api: Any,
    ssh: Any,
    public_key: str,
    hf_token: str | None,
    lease_dir: Path = DEFAULT_RUNPOD_LEASE_DIR,
    ledger_path: Path | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> dict[str, Any]:
    """Provision one pod, run the workload, download artifacts, terminate.

    Never raises for workload failures: the returned outcome records status,
    exit code, authoritative price, billed seconds, cost and termination.
    Raises only when termination of a created pod could not be confirmed.
    """
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pod_id = ""
    lease: Path | None = None
    started_at = time.time()
    price: float | None = None
    termination_confirmed = False
    termination_error: Exception | None = None
    remote_code: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    status = "failed"
    commands = remote_commands(args, harness_revision, with_hf_token=bool(hf_token))
    _require_credit(api, plan)
    try:
        pod = api.create_pod(
            name=f"stateset-flagship-s{args.seed}-{uuid.uuid4().hex[:8]}",
            image=str(settings["container_image"]),
            gpu_type_id=str(plan["gpu_type_id"]),
            gpu_count=int(settings["gpu_count"]),
            ports=["22/tcp"],
            env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
            container_disk_gb=int(settings["container_disk_gb"]),
            cloud_type=str(settings["provider_tier"]),
        )
        pod_id = str(pod.get("id") or "")
        if not pod_id:
            raise FlagshipDriverError("RunPod create response omitted pod id")
        started_at = time.time()
        lease = _write_lease(lease_dir, pod_id, dict(plan), started_at)
        raw_price = pod.get("costPerHr")
        if raw_price is None:
            raw_price = api.get_pod(pod_id).get("costPerHr")
        price = _positive_price(raw_price, "pod costPerHr")
        lifetime = int(settings["max_lifetime_seconds"])
        try:
            check_budget(price, lifetime, float(plan["max_cost_usd"]), gpu_count=1)
        except BudgetExceeded as exc:
            raise FlagshipDriverError(str(exc)) from exc
        host, port, _ = _wait_for_ssh(
            api, pod_id, timeout_s=min(900, _remaining(lifetime, started_at, "SSH"))
        )
        ssh.wait_until_reachable(
            host, port, min(300, _remaining(lifetime, started_at, "SSH setup"))
        )
        _arm_self_destruct(
            ssh, api, pod_id, _remaining(lifetime, started_at, "watchdog arming")
        )
        if hf_token:
            ssh.upload_secret(hf_token, _REMOTE_HF_TOKEN)
        _run_checked(ssh, commands["setup"], "harness checkout and install")
        _run_checked(ssh, commands["launch"], "workload launch")

        def provider_status() -> str | None:
            try:
                return str(api.get_pod(pod_id).get("desiredStatus") or "") or None
            except Exception:  # noqa: BLE001 - an API hiccup is not a pod failure
                return None

        remote_code = poll_until_exit(
            ssh,
            artifact_dir,
            deadline_s=_remaining(lifetime, started_at, "workload"),
            poll_s=poll_seconds,
            pod_status=provider_status,
        )
        try:
            ssh.download_dir(_REMOTE_ARTIFACTS, artifact_dir)
        except Exception as exc:  # noqa: BLE001
            raise FlagshipDriverError(f"could not retrieve artifacts: {exc}") from exc
        result_path = artifact_dir / _RESULT_NAME
        if result_path.is_file():
            try:
                loaded = json.loads(result_path.read_text(encoding="utf-8"))
                result = loaded if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                result = None
        if remote_code is None:
            raise FlagshipDriverError("workload did not finish within the pod lifetime")
        if remote_code != 0:
            raise FlagshipDriverError(f"remote workload exited {remote_code}")
        if result is None:
            raise FlagshipDriverError("workload wrote no result document")
        status = "completed"
    except (
        FlagshipDriverError,
        RunPodConformanceError,
        RemoteExecutionError,
        OSError,
    ) as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - still terminate and account
        error = f"unexpected {type(exc).__name__}: {exc}"
    finally:
        if pod_id:
            try:
                api.terminate_pod(pod_id)
            except Exception as exc:  # noqa: BLE001
                termination_error = exc
            else:
                termination_confirmed = True
                if lease is not None:
                    lease.unlink(missing_ok=True)
        billed_seconds = float(max(1, math.ceil(time.time() - started_at)))
        cost = estimate_cost_usd(price, billed_seconds) if pod_id else None
        if status == "completed" and not termination_confirmed:
            status = "cleanup-pending"
        record_entry(
            CostEntry(
                provider="runpod",
                job_id=pod_id or "creation-failed",
                base_model=str(args.model),
                gpu=str(settings["gpu_name"]),
                gpu_count=int(settings["gpu_count"]),
                cost_per_hr=price,
                duration_s=billed_seconds if pod_id else 0.0,
                cost_usd=cost,
                status=status,
            ),
            path=ledger_path,
        )
    outcome = {
        "status": status,
        "error": error,
        "remote_exit_code": remote_code,
        "pod_id": pod_id,
        "harness_revision": harness_revision,
        "authoritative_cost_per_hr_usd": price,
        "billed_seconds": billed_seconds if pod_id else 0.0,
        "cost_usd": cost,
        "termination_confirmed": termination_confirmed,
        "cleanup_lease_retained": bool(lease and lease.exists()),
        "result": result,
    }
    record = {key: value for key, value in outcome.items() if key != "result"}
    record.update(
        {
            "schema_version": 1,
            "kind": "stateset-flagship-runpod-provider-record",
            "cost_source": COST_SOURCE,
            "recorded_at": _utc_now(),
            "seed": args.seed,
            "catalog_total_cost_per_hr_usd": plan.get("catalog_total_cost_per_hr_usd"),
            "max_cost_usd": plan.get("max_cost_usd"),
            "settings": dict(settings),
        }
    )
    try:
        (artifact_dir / _PROVIDER_RECORD).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"provider record not written: {exc}", file=sys.stderr)
    if pod_id and not termination_confirmed:
        raise FlagshipDriverError(
            f"termination of pod {pod_id} was not confirmed; cleanup lease "
            f"retained: {termination_error}"
        ) from termination_error
    return outcome


# --------------------------------------------------------------------------- #
# Adapter output
# --------------------------------------------------------------------------- #

_METRIC_KEYS = (
    "baseline_score",
    "final_score",
    "judge_self_disagreement",
    "train_examples",
    "eval_examples",
    "peak_vram_mb",
)


def _workload_problems(
    result: Mapping[str, Any] | None,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> list[str]:
    """Every reason a downloaded workload result is not complete evidence."""
    if not isinstance(result, Mapping):
        return ["no workload result"]
    problems: list[str] = []
    if result.get("status") != "completed" or result.get("measured") is not True:
        problems.append(
            f"workload status={result.get('status')!r} measured={result.get('measured')!r}"
        )
    if result.get("smoke"):
        problems.append("smoke results are never measured evidence")
    expected = {
        "seed": int(args.seed),
        "mode": args.mode,
        "framework_version": args.framework_version,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
        "config_sha256": args.config_sha256,
        "judge_sha256": args.judge_sha256,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            problems.append(f"workload {key} does not match the driver input")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return problems + ["workload metrics missing"]
    for key in _METRIC_KEYS:
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            problems.append(f"metrics.{key} missing or non-numeric")
    if problems:
        return problems
    for key in ("baseline_score", "final_score", "judge_self_disagreement"):
        if not 0 <= float(metrics[key]) <= 1:
            problems.append(f"metrics.{key} outside [0, 1]")
    if metrics["train_examples"] < config["num_train_examples"]:
        problems.append("fewer training examples than the config requires")
    if metrics["eval_examples"] < config["num_eval_examples"]:
        problems.append("fewer evaluation examples than the config requires")
    if float(metrics["peak_vram_mb"]) <= 0:
        problems.append("peak VRAM was not measured")
    hardware = result.get("hardware")
    if not isinstance(hardware, Mapping):
        return problems + ["hardware missing"]
    if hardware.get("gpu") != settings["gpu_name"]:
        problems.append(
            f"workload ran on {hardware.get('gpu')!r}, not {settings['gpu_name']!r}"
        )
    if hardware.get("gpu_count") != settings["gpu_count"]:
        problems.append("workload GPU count does not match the pod")
    for key in ("cuda", "driver"):
        if not isinstance(hardware.get(key), str) or not hardware[key]:
            problems.append(f"hardware.{key} missing")
    return problems


def build_adapter(
    args: argparse.Namespace,
    outcome: Mapping[str, Any],
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    ceiling: float,
) -> dict[str, Any]:
    """Adapter JSON for ``run_flagship_matrix``; completed only when whole."""
    result = outcome.get("result")
    problems: list[str] = []
    if outcome.get("status") != "completed":
        problems.append(str(outcome.get("error") or f"status={outcome.get('status')}"))
    problems.extend(_workload_problems(result, args, config, settings))
    cost = outcome.get("cost_usd")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
        problems.append("provider cost is unknown")
    elif float(cost) > ceiling:
        problems.append(f"provider cost ${cost} exceeded the ceiling ${ceiling}")
    metrics = dict(result.get("metrics", {})) if isinstance(result, Mapping) else {}
    metrics["cost_usd"] = cost
    completed = not problems
    adapter: dict[str, Any] = {
        "status": "completed" if completed else "failed",
        "measured": completed,
        "provider": "runpod",
        "cost_source": COST_SOURCE,
        "seed": int(args.seed),
        "mode": args.mode,
        "framework_version": args.framework_version,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
        "config_sha256": args.config_sha256,
        "judge_sha256": args.judge_sha256,
        "artifact_path": str(Path(args.artifact_dir).resolve()),
        "hardware": (
            dict(result.get("hardware", {})) if isinstance(result, Mapping) else {}
        ),
        "metrics": metrics,
        "diagnostics": (
            dict(result.get("diagnostics", {})) if isinstance(result, Mapping) else {}
        ),
        "pod_id": outcome.get("pod_id"),
        "harness_revision": outcome.get("harness_revision"),
        "authoritative_cost_per_hr_usd": outcome.get("authoritative_cost_per_hr_usd"),
        "billed_seconds": outcome.get("billed_seconds"),
        "termination_confirmed": outcome.get("termination_confirmed"),
        "max_cost_usd": ceiling,
        "recorded_at": _utc_now(),
    }
    if problems:
        adapter["problems"] = problems
    return adapter


def _write_adapter(path: Path, adapter: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adapter, indent=2, sort_keys=True) + "\n", "utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("measured", "preflight"), required=True)
    parser.add_argument("--framework-version", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--judge-json", required=True)
    parser.add_argument("--judge-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="price only (free)")
    parser.add_argument("--dry-run", action="store_true", help="offline preview")
    parser.add_argument("--confirm-max-cost-usd", type=float)
    parser.add_argument("--gpu", default=None, help=f"default ${ENV_GPU} or H100")
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
    catalog_fetcher: Any = None,
    api_factory: Any = None,
    ssh_factory: Any = None,
) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if environ is None else environ
    fetch = catalog_fetcher or fetch_catalog
    make_api = api_factory or RunPodApi
    make_ssh = ssh_factory or SshTransport
    root = _BENCHMARKS.parent
    try:
        config, judge = load_pinned_inputs(args, _installed_version())
        settings = execution_settings(args, config, env)
        per_seed_cap = float(config["max_cost_usd_per_seed"])
        if args.dry_run:
            revision = args.harness_revision or "<HEAD at execution>"
            print(
                json.dumps(
                    {
                        "kind": "stateset-flagship-runpod-dry-run",
                        "settings": settings,
                        "per_seed_cap_usd": per_seed_cap,
                        "config_sha256": digest_json(config),
                        "commands": remote_commands(
                            args, revision, with_hf_token=bool(env.get("HF_TOKEN"))
                        ),
                        "provisions_hardware": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.plan:
            ceiling = (
                resolve_spend_ceiling(args.confirm_max_cost_usd, env, per_seed_cap)
                if args.confirm_max_cost_usd is not None or env.get(ENV_CEILING)
                else per_seed_cap
            )
            print(json.dumps(build_plan(settings, fetch(), ceiling), indent=2))
            return 0
        ceiling = resolve_spend_ceiling(args.confirm_max_cost_usd, env, per_seed_cap)
        reject_template_revisions(args, judge)
        revision = resolve_harness_revision(root, args.harness_revision)
        plan = build_plan(settings, fetch(), ceiling)
        api_key = str(env.get("RUNPOD_API_KEY", "")).strip()
        if not api_key:
            raise FlagshipDriverError("RUNPOD_API_KEY is not set")
        public_key, private_key = _public_key(args.ssh_public_key)
        outcome = execute_seed(
            args,
            settings,
            plan,
            revision,
            api=make_api(api_key),
            ssh=make_ssh(key_path=private_key),
            public_key=public_key,
            hf_token=str(env.get("HF_TOKEN", "")).strip() or None,
            lease_dir=args.lease_dir,
            ledger_path=args.ledger,
            poll_seconds=args.poll_seconds,
        )
        adapter = build_adapter(args, outcome, config, settings, ceiling)
        _write_adapter(args.output, adapter)
        if adapter["status"] != "completed":
            print(
                "flagship seed incomplete: " + "; ".join(adapter.get("problems", [])),
                file=sys.stderr,
            )
            return 1
        print(json.dumps(adapter["metrics"], sort_keys=True))
        return 0
    except (
        FlagshipDriverError,
        RunPodConformanceError,
        RemoteExecutionError,
        OSError,
    ) as exc:
        print(f"flagship RunPod driver rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
