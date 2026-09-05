#!/usr/bin/env python3
"""Plan or execute a budget-bounded framework shootout on one RunPod GPU.

Composes the fail-closed pieces of ``runpod_backend_conformance.py`` (free
public-catalog planning, an exact spend-ceiling confirmation, a local
recovery lease, an in-pod self-destruct, an authoritative post-allocation
price check, unconditional termination, and a cost-ledger entry) around
``benchmarks/shootout.py`` so a matched StateSet-versus-TRL comparison can be
collected with one command and retained as evidence.

    # free: validate both manifests and price the worst case
    python benchmarks/runpod_shootout.py benchmarks/runpod_shootout_manifest.json

    # paid: provision, run every seed x framework, download evidence, terminate
    python benchmarks/runpod_shootout.py benchmarks/runpod_shootout_manifest.json \
        --execute --confirm-max-cost-usd <ceiling> --output-dir <dir>

The launcher manifest names the shootout manifest it drives and pins the
StateSet harness revision that the pod checks out; the shootout manifest
keeps its own protocol, model, dataset, and implementation pins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend_conformance import ConformanceError, canonical_digest, write_json_once
from runpod_backend_conformance import (
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
from shootout import load_manifest as load_shootout_manifest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.ledger import (
    BudgetExceeded,
    CostEntry,
    check_budget,
    estimate_cost_usd,
    record_entry,
)
from stateset_agents.remote.runpod import (
    DEFAULT_RUNPOD_LEASE_DIR,
    RunPodApi,
    SshTransport,
)

_REMOTE_SHOOTOUT_MANIFEST = "/workspace/shootout-manifest.json"
_REMOTE_OUTPUT = "/workspace/shootout-output"
_ALLOWED_TIERS = frozenset({"SECURE", "COMMUNITY"})
_REQUIRED_EXECUTION = (
    "provider",
    "provider_tier",
    "container_image",
    "gpu_name",
    "gpu_count",
    "container_disk_gb",
    "timeout_seconds",
    "max_lifetime_seconds",
    "max_cost_usd",
)


class RunPodShootoutError(ConformanceError):
    """Raised when a shootout launch is malformed, over budget, or fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_launcher_manifest(path: Path, root: Path) -> dict[str, Any]:
    """Validate the launcher manifest and the shootout manifest it names."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunPodShootoutError(f"{path}: invalid JSON") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise RunPodShootoutError("manifest must be an object with schema_version=1")
    revision = raw.get("harness_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise RunPodShootoutError(
            "manifest.harness_revision must be a 40-character StateSet commit"
        )
    execution = raw.get("execution")
    if not isinstance(execution, Mapping):
        raise RunPodShootoutError("manifest.execution must be an object")
    missing = [key for key in _REQUIRED_EXECUTION if key not in execution]
    if missing:
        raise RunPodShootoutError(
            "manifest.execution is missing: " + ", ".join(missing)
        )
    if execution["provider"] != "runpod":
        raise RunPodShootoutError("execution.provider must be runpod")
    tier = str(execution["provider_tier"])
    if tier not in _ALLOWED_TIERS:
        raise RunPodShootoutError("execution.provider_tier must be SECURE or COMMUNITY")
    for key in (
        "gpu_count",
        "container_disk_gb",
        "timeout_seconds",
        "max_lifetime_seconds",
    ):
        value = execution[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RunPodShootoutError(f"execution.{key} must be a positive integer")
    if int(execution["max_lifetime_seconds"]) <= int(execution["timeout_seconds"]):
        raise RunPodShootoutError(
            "execution.max_lifetime_seconds must exceed timeout_seconds "
            "(setup and evidence download need headroom)"
        )
    ceiling = execution["max_cost_usd"]
    if (
        isinstance(ceiling, bool)
        or not isinstance(ceiling, (int, float))
        or ceiling <= 0
    ):
        raise RunPodShootoutError("execution.max_cost_usd must be a positive number")
    shootout_path = raw.get("shootout_manifest")
    if not isinstance(shootout_path, str) or not shootout_path.strip():
        raise RunPodShootoutError(
            "manifest.shootout_manifest must be a repo-relative path"
        )
    local_shootout = (root / shootout_path).resolve()
    if not local_shootout.is_file():
        raise RunPodShootoutError(f"shootout manifest not found: {local_shootout}")
    shootout = load_shootout_manifest(local_shootout)
    if shootout["hardware"]["gpu"] != execution["gpu_name"] or int(
        shootout["hardware"]["gpu_count"]
    ) != int(execution["gpu_count"]):
        raise RunPodShootoutError(
            "shootout manifest hardware must match the launcher's execution GPU"
        )
    result = dict(raw)
    result["_shootout"] = shootout
    result["_shootout_path"] = str(local_shootout)
    return result


def build_plan(
    manifest: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Price the full declared pod lifetime from the public catalog (no auth)."""
    execution = manifest["execution"]
    tier = str(execution["provider_tier"])
    gpu_name = str(execution["gpu_name"])
    matches = [row for row in catalog if row.get("id") == gpu_name]
    if len(matches) != 1:
        raise RunPodShootoutError(
            f"RunPod catalog must contain exactly one GPU named {gpu_name!r}"
        )
    row = matches[0]
    availability_field = "secureCloud" if tier == "SECURE" else "communityCloud"
    price_field = "securePrice" if tier == "SECURE" else "communityPrice"
    if row.get(availability_field) is not True:
        raise RunPodShootoutError(f"{gpu_name} is unavailable in {tier} cloud")
    unit_price = _positive_price(row.get(price_field), price_field)
    gpu_count = int(execution["gpu_count"])
    lifetime = int(execution["max_lifetime_seconds"])
    ceiling = float(execution["max_cost_usd"])
    try:
        worst_case = check_budget(unit_price, lifetime, ceiling, gpu_count=gpu_count)
    except BudgetExceeded as exc:
        raise RunPodShootoutError(str(exc)) from exc
    shootout = manifest["_shootout"]
    runs = len(shootout["seeds"]) * len(shootout["implementations"])
    public = {k: v for k, v in manifest.items() if not k.startswith("_")}
    return {
        "schema_version": 1,
        "kind": "stateset-runpod-shootout-plan",
        "generated_at": _utc_now(),
        "manifest_sha256": canonical_digest(public),
        "shootout_manifest_sha256": canonical_digest(shootout),
        "protocol": shootout["protocol"],
        "frameworks": [str(i["name"]) for i in shootout["implementations"]],
        "seeds": list(shootout["seeds"]),
        "planned_runs": runs,
        "provider": "runpod",
        "provider_tier": tier,
        "gpu_name": gpu_name,
        "gpu_type_id": gpu_name,
        "gpu_count": gpu_count,
        "container_image": execution["container_image"],
        "container_disk_gb": execution["container_disk_gb"],
        "workload_timeout_seconds": execution["timeout_seconds"],
        "max_lifetime_seconds": lifetime,
        "catalog_unit_cost_per_hr_usd": unit_price,
        "catalog_total_cost_per_hr_usd": round(unit_price * gpu_count, 6),
        "worst_case_cost_usd": worst_case,
        "max_cost_usd": ceiling,
        "catalog_source": _CATALOG_URL,
        "catalog_quote_is_authoritative": False,
        "provisions_hardware": False,
    }


def _remaining(lifetime: int, started_at: float, stage: str) -> int:
    remaining = lifetime - max(1, math.ceil(time.time() - started_at))
    if remaining < 1:
        raise RunPodShootoutError(f"pod exhausted its billable lifetime before {stage}")
    return remaining


def execute(
    manifest: Mapping[str, Any],
    output_dir: Path,
    plan: Mapping[str, Any],
    *,
    api: Any,
    ssh: Any,
    public_key: str,
    lease_dir: Path = DEFAULT_RUNPOD_LEASE_DIR,
    ledger_path: Path | None = None,
) -> Path:
    """Provision exactly one pod, run the shootout, retrieve evidence, terminate."""
    if output_dir.exists():
        raise RunPodShootoutError(f"refusing to overwrite output: {output_dir}")
    execution = manifest["execution"]
    shootout = manifest["_shootout"]
    pod_id = ""
    lease: Path | None = None
    started_at = time.time()
    authoritative_price: float | None = None
    termination_confirmed = False
    status = "failed"
    remote_code: int | None = None
    try:
        pod = api.create_pod(
            name=f"stateset-shootout-{uuid.uuid4().hex[:8]}",
            image=str(execution["container_image"]),
            gpu_type_id=str(plan["gpu_type_id"]),
            gpu_count=int(execution["gpu_count"]),
            ports=["22/tcp"],
            env={"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key},
            container_disk_gb=int(execution["container_disk_gb"]),
            cloud_type=str(execution["provider_tier"]),
        )
        pod_id = str(pod.get("id") or "")
        if not pod_id:
            raise RunPodShootoutError("RunPod create response omitted pod id")
        started_at = time.time()
        try:
            lease = _write_lease(lease_dir, pod_id, manifest["_shootout"], started_at)
        except Exception:
            api.terminate_pod(pod_id)
            raise
        raw_price = pod.get("costPerHr")
        if raw_price is None:
            pod = api.get_pod(pod_id)
            raw_price = pod.get("costPerHr")
        authoritative_price = _positive_price(raw_price, "pod costPerHr")
        try:
            check_budget(
                authoritative_price,
                int(execution["max_lifetime_seconds"]),
                float(execution["max_cost_usd"]),
                gpu_count=1,  # costPerHr is the effective whole-pod rate
            )
        except BudgetExceeded as exc:
            raise RunPodShootoutError(str(exc)) from exc

        lifetime = int(execution["max_lifetime_seconds"])
        host, port, _ = _wait_for_ssh(
            api, pod_id, timeout_s=min(900, _remaining(lifetime, started_at, "SSH"))
        )
        ssh.wait_until_reachable(
            host, port, min(300, _remaining(lifetime, started_at, "SSH setup"))
        )
        _arm_self_destruct(
            ssh, api, pod_id, _remaining(lifetime, started_at, "watchdog arming")
        )
        ssh.upload_secret(
            json.dumps(shootout, sort_keys=True), _REMOTE_SHOOTOUT_MANIFEST
        )
        revision = shlex.quote(str(manifest["harness_revision"]))
        repo = shlex.quote(_REMOTE_REPOSITORY)
        steps = [
            f"git clone --filter=blob:none {shlex.quote(_REPOSITORY_URL)} {repo}",
            f"git -C {repo} checkout --detach {revision}",
            f"python -m pip install --quiet -e {repo}'[training]'",
        ]
        extra_pip = [str(p) for p in manifest.get("extra_pip", []) if str(p).strip()]
        if extra_pip:
            steps.append(
                "python -m pip install --quiet "
                + " ".join(shlex.quote(p) for p in extra_pip)
            )
        setup = " && ".join(steps)
        _run_checked(ssh, setup, "harness checkout and install")
        prewarm = manifest.get("prewarm_command")
        if isinstance(prewarm, str) and prewarm.strip():
            _run_checked(ssh, f"cd {repo} && {prewarm}", "cache prewarm")
        required = " ".join(
            f"--required-framework {shlex.quote(str(i['name']))}"
            for i in shootout["implementations"]
        )
        run = (
            f"cd {repo} && python benchmarks/shootout.py "
            f"{shlex.quote(_REMOTE_SHOOTOUT_MANIFEST)} --root {repo} "
            f"--output-dir {shlex.quote(_REMOTE_OUTPUT)} "
            f"--timeout-seconds {int(execution['timeout_seconds'])} {required}"
        )
        remote_code, output = ssh.run(run)
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            ssh.download_dir(_REMOTE_OUTPUT, output_dir)
        except Exception as exc:
            raise RunPodShootoutError(
                f"could not retrieve remote evidence: {exc}"
            ) from exc
        if remote_code != 0:
            raise RunPodShootoutError(
                f"remote shootout failed ({remote_code}); downloaded failure "
                f"evidence: {output[-2000:]}"
            )
        status = "completed"
        return output_dir
    finally:
        termination_error: Exception | None = None
        if pod_id:
            try:
                api.terminate_pod(pod_id)
            except Exception as exc:
                termination_confirmed = False
                termination_error = exc
            else:
                termination_confirmed = True
                if lease is not None:
                    lease.unlink(missing_ok=True)
        if status == "completed" and not termination_confirmed:
            status = "cleanup-pending"
        duration = max(0.0, time.time() - started_at)
        cost = estimate_cost_usd(authoritative_price, duration)
        record_entry(
            CostEntry(
                provider="runpod",
                job_id=pod_id or "creation-failed",
                base_model=str(shootout["model"]),
                gpu=str(execution["gpu_name"]),
                gpu_count=int(execution["gpu_count"]),
                cost_per_hr=authoritative_price,
                duration_s=round(duration, 3),
                cost_usd=cost,
                status=status,
            ),
            path=ledger_path,
        )
        if output_dir.exists():
            report = {
                "schema_version": 1,
                "kind": "stateset-runpod-shootout-provider-record",
                "recorded_at": _utc_now(),
                "manifest_sha256": plan["manifest_sha256"],
                "shootout_manifest_sha256": plan["shootout_manifest_sha256"],
                "harness_revision": manifest["harness_revision"],
                "container_image": execution["container_image"],
                "status": status,
                "remote_exit_code": remote_code,
                "pod_id": pod_id,
                "catalog_total_cost_per_hr_usd": plan["catalog_total_cost_per_hr_usd"],
                "authoritative_pod_cost_per_hr_usd": authoritative_price,
                "pod_lifetime_seconds": round(duration, 3),
                "estimated_cost_usd": cost,
                "termination_confirmed": termination_confirmed,
                "cleanup_lease_retained": bool(lease and lease.exists()),
            }
            try:
                write_json_once(output_dir / "runpod-provider.json", report)
            except (ConformanceError, OSError):
                pass
        if status == "cleanup-pending":
            raise RunPodShootoutError(
                f"evidence was retrieved, but termination of pod {pod_id} was not "
                f"confirmed; cleanup lease retained: {termination_error}"
            ) from termination_error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-max-cost-usd", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ssh-public-key", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = load_launcher_manifest(args.manifest, args.root)
        plan = build_plan(manifest, fetch_catalog())
        if args.plan_output:
            write_json_once(args.plan_output, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        if not args.execute:
            if args.confirm_max_cost_usd is not None or args.output_dir is not None:
                raise RunPodShootoutError(
                    "--confirm-max-cost-usd/--output-dir require --execute"
                )
            return 0
        ceiling = float(manifest["execution"]["max_cost_usd"])
        if args.confirm_max_cost_usd is None or not math.isclose(
            args.confirm_max_cost_usd, ceiling, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RunPodShootoutError(
                "--execute requires --confirm-max-cost-usd exactly equal to "
                f"the manifest ceiling ({ceiling})"
            )
        if args.output_dir is None:
            raise RunPodShootoutError("--execute requires --output-dir")
        api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
        if not api_key:
            raise RunPodShootoutError("RUNPOD_API_KEY is not set")
        public_key, private_key = _public_key(args.ssh_public_key)
        execute(
            manifest,
            args.output_dir,
            plan,
            api=RunPodApi(api_key),
            ssh=SshTransport(key_path=private_key),
            public_key=public_key,
        )
    except (
        RunPodShootoutError,
        RunPodConformanceError,
        RemoteExecutionError,
        OSError,
    ) as exc:
        print(f"RunPod shootout rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
