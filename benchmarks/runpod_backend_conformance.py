#!/usr/bin/env python3
"""Plan or execute a budget-bounded conformance run on RunPod.

Planning is the default and performs no authenticated request.  Provisioning
requires both ``--execute`` and an exact repetition of the manifest's spend
ceiling.  Every provisioned pod gets a local recovery lease, an in-pod
self-destruct, an authoritative post-allocation price check, and unconditional
termination by the launching process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from backend_conformance import (
    ConformanceError,
    canonical_digest,
    load_evidence,
    load_manifest,
    verify_harness_revision,
    write_json_once,
)

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
from stateset_agents.remote.serve_session import self_destruct_script

_CATALOG_URL = "https://api.runpod.io/graphql"
_REPOSITORY_URL = "https://github.com/stateset/stateset-agents.git"
_REMOTE_REPOSITORY = "/workspace/stateset-agents"
_REMOTE_MANIFEST = "/workspace/conformance-manifest.json"
_REMOTE_OUTPUT = "/workspace/conformance-output"
_REMOTE_KEY = "/workspace/.stateset-conformance-runpod-key"
_REMOTE_DESTRUCT = "/workspace/stateset-conformance-self-destruct.sh"
_ALLOWED_TIERS = frozenset({"SECURE", "COMMUNITY"})


class RunPodConformanceError(ConformanceError):
    """Raised when safe planning or provider execution cannot be guaranteed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_price(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunPodConformanceError(f"RunPod {field} is unavailable")
    price = float(value)
    if not math.isfinite(price) or price <= 0:
        raise RunPodConformanceError(f"RunPod {field} is unavailable")
    return price


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_catalog() -> list[dict[str, Any]]:
    """Fetch RunPod's public GPU catalog without using an API key."""
    import requests

    query = """
    query StateSetConformanceCatalog {
      gpuTypes {
        id displayName secureCloud communityCloud securePrice communityPrice
      }
    }
    """
    try:
        response = requests.post(_CATALOG_URL, json={"query": query}, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RunPodConformanceError(f"RunPod catalog request failed: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("errors"):
        raise RunPodConformanceError("RunPod catalog returned GraphQL errors")
    data = payload.get("data")
    rows = data.get("gpuTypes") if isinstance(data, Mapping) else None
    if not isinstance(rows, list):
        raise RunPodConformanceError("RunPod catalog response is malformed")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def build_plan(
    manifest: Mapping[str, Any],
    dataset: Path,
    catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate local inputs and price the full declared pod lifetime."""
    execution = manifest["execution"]
    if execution["provider"] != "runpod":
        raise RunPodConformanceError("execution.provider must be runpod")
    tier = str(execution["provider_tier"]).upper()
    if tier not in _ALLOWED_TIERS or tier != execution["provider_tier"]:
        raise RunPodConformanceError(
            "execution.provider_tier must be SECURE or COMMUNITY"
        )
    dataset = dataset.resolve()
    if not dataset.is_file():
        raise RunPodConformanceError(f"dataset is not a file: {dataset}")
    digest = _sha256_file(dataset)
    expected_digest = manifest["experiment"]["dataset_sha256"]
    if digest != expected_digest:
        raise RunPodConformanceError("local dataset SHA-256 does not match manifest")
    remote_dataset = PurePosixPath(str(manifest["experiment"]["dataset_uri"]))
    if (
        not remote_dataset.is_absolute()
        or ".." in remote_dataset.parts
        or remote_dataset == PurePosixPath("/")
    ):
        raise RunPodConformanceError(
            "experiment.dataset_uri must be a safe absolute remote path"
        )
    protected = (
        PurePosixPath(_REMOTE_REPOSITORY),
        PurePosixPath(_REMOTE_OUTPUT),
    )
    if remote_dataset in {
        PurePosixPath(_REMOTE_MANIFEST),
        PurePosixPath(_REMOTE_KEY),
        PurePosixPath(_REMOTE_DESTRUCT),
    } or any(remote_dataset.is_relative_to(path) for path in protected):
        raise RunPodConformanceError(
            "experiment.dataset_uri collides with a launcher-owned remote path"
        )
    gpu_name = str(execution["gpu_name"])
    # RunPod's ``id`` is the exact NVIDIA identity used by the pod API and
    # reported by nvidia-smi (for example ``NVIDIA A40``); ``displayName`` is
    # a shorter console label (``A40``) and must not drive hardware attestation.
    matches = [row for row in catalog if row.get("id") == gpu_name]
    if len(matches) != 1:
        raise RunPodConformanceError(
            f"RunPod catalog must contain exactly one GPU named {gpu_name!r}"
        )
    row = matches[0]
    availability_field = "secureCloud" if tier == "SECURE" else "communityCloud"
    price_field = "securePrice" if tier == "SECURE" else "communityPrice"
    if row.get(availability_field) is not True:
        raise RunPodConformanceError(f"{gpu_name} is unavailable in {tier} cloud")
    unit_price = _positive_price(row.get(price_field), price_field)
    gpu_type_id = gpu_name
    gpu_count = int(execution["gpu_count"])
    lifetime = int(execution["max_lifetime_seconds"])
    ceiling = float(execution["max_cost_usd"])
    try:
        worst_case = check_budget(unit_price, lifetime, ceiling, gpu_count=gpu_count)
    except BudgetExceeded as exc:
        raise RunPodConformanceError(str(exc)) from exc
    if worst_case is None:  # defensive: schema v3 requires a spend ceiling
        raise RunPodConformanceError("RunPod catalog cost could not be bounded")
    return {
        "schema_version": 1,
        "kind": "stateset-runpod-conformance-plan",
        "generated_at": _utc_now(),
        "manifest_sha256": canonical_digest(manifest),
        "provider": "runpod",
        "provider_tier": tier,
        "gpu_name": gpu_name,
        "gpu_type_id": gpu_type_id,
        "gpu_count": gpu_count,
        "container_image": execution["container_image"],
        "container_disk_gb": execution["container_disk_gb"],
        "workload_timeout_seconds": execution["timeout_seconds"],
        "max_lifetime_seconds": lifetime,
        "catalog_unit_cost_per_hr_usd": unit_price,
        "catalog_total_cost_per_hr_usd": round(unit_price * gpu_count, 6),
        "worst_case_cost_usd": worst_case,
        "max_cost_usd": ceiling,
        "dataset_sha256": digest,
        "catalog_source": _CATALOG_URL,
        "catalog_quote_is_authoritative": False,
        "provisions_hardware": False,
    }


def _lease_path(lease_dir: Path, pod_id: str) -> Path:
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in pod_id)
    return lease_dir / f"conformance-{safe_id}.json"


def _write_lease(
    lease_dir: Path, pod_id: str, manifest: Mapping[str, Any], created_at: float
) -> Path:
    target = _lease_path(lease_dir, pod_id)
    temporary = target.with_suffix(".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(
                {
                    "provider": "runpod",
                    "kind": "backend-conformance",
                    "pod_id": pod_id,
                    "created_at": created_at,
                    "manifest_sha256": canonical_digest(manifest),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RunPodConformanceError(f"could not record cleanup lease: {exc}") from exc
    return target


def _public_key(key_path: Path | None) -> tuple[str, Path | None]:
    candidates = (
        [key_path]
        if key_path
        else [
            Path.home() / ".ssh/id_ed25519.pub",
            Path.home() / ".ssh/id_rsa.pub",
        ]
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                private = (
                    candidate.with_suffix("") if candidate.suffix == ".pub" else None
                )
                return value, private if private and private.is_file() else None
    raise RunPodConformanceError("no usable SSH public key was found")


def _wait_for_ssh(
    api: Any, pod_id: str, *, timeout_s: int = 900, poll_s: float = 5.0
) -> tuple[str, int, dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = api.get_pod(pod_id)
        mapping = last.get("portMappings") or {}
        host, port = last.get("publicIp"), mapping.get("22")
        if last.get("desiredStatus") == "RUNNING" and host and port:
            return str(host), int(port), last
        time.sleep(poll_s)
    raise RunPodConformanceError(
        f"pod {pod_id} did not publish SSH within {timeout_s}s "
        f"(last status {last.get('desiredStatus')!r})"
    )


def _run_checked(ssh: Any, command: str, label: str) -> str:
    code, output = ssh.run(command)
    if code != 0:
        raise RunPodConformanceError(f"remote {label} failed ({code}): {output}")
    return output


def _arm_self_destruct(ssh: Any, api: Any, pod_id: str, lifetime_seconds: int) -> None:
    ssh.upload_secret(str(api.api_key), _REMOTE_KEY)
    with tempfile.TemporaryDirectory() as staging:
        script = Path(staging) / "self-destruct.sh"
        script.write_text(
            self_destruct_script(
                pod_id,
                lifetime_seconds / 3600,
                api.root,
                key_file=_REMOTE_KEY,
            ),
            encoding="utf-8",
        )
        ssh.upload(script, _REMOTE_DESTRUCT)
    command = (
        f"chmod 600 {shlex.quote(_REMOTE_KEY)} {shlex.quote(_REMOTE_DESTRUCT)} && "
        f"(nohup bash {shlex.quote(_REMOTE_DESTRUCT)} "
        "> /workspace/conformance-self-destruct.log 2>&1 < /dev/null &)"
    )
    _run_checked(ssh, command, "self-destruct arming")


def execute(
    manifest: Mapping[str, Any],
    dataset: Path,
    output_dir: Path,
    root: Path,
    plan: Mapping[str, Any],
    *,
    api: Any,
    ssh: Any,
    public_key: str,
    lease_dir: Path = DEFAULT_RUNPOD_LEASE_DIR,
    ledger_path: Path | None = None,
) -> Path:
    """Provision exactly one pod, collect evidence, and terminate it."""
    if output_dir.exists():
        raise RunPodConformanceError(f"refusing to overwrite output: {output_dir}")
    verify_harness_revision(str(manifest["harness_revision"]), root)
    execution = manifest["execution"]
    pod: dict[str, Any] = {}
    pod_id = ""
    lease: Path | None = None
    started_at = time.time()
    authoritative_price: float | None = None
    termination_confirmed = False
    status = "failed"
    try:
        pod = api.create_pod(
            name=f"stateset-conformance-{manifest['backend']}-{uuid.uuid4().hex[:8]}",
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
            raise RunPodConformanceError("RunPod create response omitted pod id")
        started_at = time.time()
        try:
            lease = _write_lease(lease_dir, pod_id, manifest, started_at)
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
                # costPerHr is the effective whole-pod rate.
                gpu_count=1,
            )
        except BudgetExceeded as exc:
            raise RunPodConformanceError(str(exc)) from exc

        lifetime = int(execution["max_lifetime_seconds"])
        remaining = lifetime - max(1, math.ceil(time.time() - started_at))
        if remaining < 1:
            raise RunPodConformanceError(
                "pod exhausted its billable lifetime before becoming reachable"
            )
        host, port, _ = _wait_for_ssh(api, pod_id, timeout_s=min(900, remaining))
        remaining = lifetime - max(1, math.ceil(time.time() - started_at))
        if remaining < 1:
            raise RunPodConformanceError(
                "pod exhausted its billable lifetime before SSH setup"
            )
        ssh.wait_until_reachable(host, port, min(300, remaining))
        remaining = lifetime - max(1, math.ceil(time.time() - started_at))
        if remaining < 1:
            raise RunPodConformanceError(
                "pod exhausted its billable lifetime before watchdog arming"
            )
        _arm_self_destruct(ssh, api, pod_id, remaining)
        remote_dataset = str(manifest["experiment"]["dataset_uri"])
        _run_checked(
            ssh,
            f"mkdir -p {shlex.quote(str(PurePosixPath(remote_dataset).parent))}",
            "dataset directory creation",
        )
        ssh.upload(dataset.resolve(), remote_dataset)
        ssh.upload_secret(json.dumps(manifest, sort_keys=True), _REMOTE_MANIFEST)
        revision = shlex.quote(str(manifest["harness_revision"]))
        setup = (
            f"git clone --filter=blob:none {shlex.quote(_REPOSITORY_URL)} "
            f"{shlex.quote(_REMOTE_REPOSITORY)} && "
            f"git -C {shlex.quote(_REMOTE_REPOSITORY)} checkout --detach {revision} "
            f"&& python -m pip install --no-deps {shlex.quote(_REMOTE_REPOSITORY)} "
            f"&& git -C {shlex.quote(_REMOTE_REPOSITORY)} clean -fdx"
        )
        _run_checked(ssh, setup, "harness checkout")
        run = (
            f"cd {shlex.quote(_REMOTE_REPOSITORY)} && "
            f"python benchmarks/backend_conformance.py {shlex.quote(_REMOTE_MANIFEST)} "
            f"--root {shlex.quote(_REMOTE_REPOSITORY)} "
            f"--output-dir {shlex.quote(_REMOTE_OUTPUT)} "
            f"--timeout-seconds {int(execution['timeout_seconds'])}"
        )
        code, output = ssh.run(run)
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            ssh.download_dir(_REMOTE_OUTPUT, output_dir)
        except Exception as exc:
            raise RunPodConformanceError(
                f"could not retrieve remote evidence: {exc}"
            ) from exc
        if code != 0:
            raise RunPodConformanceError(
                f"remote conformance failed ({code}); downloaded failure evidence: {output}"
            )
        evidence_path = output_dir / "conformance.json"
        load_evidence(evidence_path)
        status = "completed"
        return evidence_path
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
                base_model=str(manifest["experiment"]["model"]),
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
                "kind": "stateset-runpod-conformance-provider-record",
                "recorded_at": _utc_now(),
                "manifest_sha256": canonical_digest(manifest),
                "status": status,
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
            raise RunPodConformanceError(
                f"evidence was retrieved, but termination of pod {pod_id} was not "
                f"confirmed; cleanup lease retained: {termination_error}"
            ) from termination_error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-max-cost-usd", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ssh-public-key", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        plan = build_plan(manifest, args.dataset, fetch_catalog())
        if args.plan_output:
            write_json_once(args.plan_output, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        if not args.execute:
            if args.confirm_max_cost_usd is not None or args.output_dir is not None:
                raise RunPodConformanceError(
                    "--confirm-max-cost-usd/--output-dir require --execute"
                )
            return 0
        ceiling = float(manifest["execution"]["max_cost_usd"])
        if args.confirm_max_cost_usd is None or not math.isclose(
            args.confirm_max_cost_usd, ceiling, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RunPodConformanceError(
                "--execute requires --confirm-max-cost-usd exactly equal to "
                f"the manifest ceiling ({ceiling})"
            )
        if args.output_dir is None:
            raise RunPodConformanceError("--execute requires --output-dir")
        api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
        if not api_key:
            raise RunPodConformanceError("RUNPOD_API_KEY is not set")
        public_key, private_key = _public_key(args.ssh_public_key)
        execute(
            manifest,
            args.dataset,
            args.output_dir,
            args.root,
            plan,
            api=RunPodApi(api_key),
            ssh=SshTransport(key_path=private_key),
            public_key=public_key,
        )
    except (RunPodConformanceError, RemoteExecutionError, OSError) as exc:
        print(f"RunPod conformance rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
