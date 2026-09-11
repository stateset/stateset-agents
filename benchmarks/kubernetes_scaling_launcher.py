#!/usr/bin/env python3
"""Run one multi-node scaling row as a Kubernetes Indexed Job.

The adapter is provider-neutral and works with CoreWeave CKS, Nebius Managed
Kubernetes, or another conforming cluster. It creates a headless Service plus
an anti-affined Indexed Job, retrieves rank-zero evidence, and removes both
resources on every exit path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .scaling_comparison import load_scaling_evidence
except ImportError:  # pragma: no cover - direct script execution
    from scaling_comparison import load_scaling_evidence

from stateset_agents.remote.executor import RemoteExecutionError

HEX = frozenset("0123456789abcdef")
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
REMOTE_ROOT = "/workspace/stateset-scaling"


class KubernetesScalingError(ValueError):
    """Raised when Kubernetes scaling evidence cannot be safely collected."""


class KubernetesApi:
    """Small argv-only kubectl transport with an injectable process runner."""

    def __init__(
        self,
        *,
        context: str,
        namespace: str,
        runner: Any = subprocess.run,
    ) -> None:
        self.context = context
        self.namespace = namespace
        self.runner = runner

    def _command(self, args: Sequence[str]) -> list[str]:
        return [
            "kubectl",
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            *args,
        ]

    def run(self, args: Sequence[str], *, stdin: str | None = None) -> str:
        try:
            result = self.runner(
                self._command(args),
                input=stdin,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RemoteExecutionError.wrap(
                exc, "could not execute kubectl", provider="kubernetes"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise KubernetesScalingError(f"kubectl failed: {detail}")
        return result.stdout

    def apply(self, manifest: Mapping[str, Any]) -> None:
        self.run(["apply", "-f", "-"], stdin=json.dumps(manifest))

    def get_json(self, kind: str, name: str) -> dict[str, Any]:
        raw = self.run(["get", kind, name, "-o", "json"])
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KubernetesScalingError("kubectl returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise KubernetesScalingError("kubectl returned a non-object")
        return value

    def list_pods(self, selector: str) -> list[dict[str, Any]]:
        value = json.loads(self.run(["get", "pods", "-l", selector, "-o", "json"]))
        items = value.get("items") if isinstance(value, dict) else None
        if not isinstance(items, list) or any(
            not isinstance(item, dict) for item in items
        ):
            raise KubernetesScalingError("kubectl returned invalid pod inventory")
        return items

    def copy_from(self, pod: str, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self.run(["cp", f"{pod}:{remote}", str(local)])

    def logs(self, pod: str) -> str:
        return self.run(["logs", pod, "--tail=200"])

    def delete(self, kind: str, name: str) -> None:
        self.run(["delete", kind, name, "--ignore-not-found=true", "--wait=true"])

    def require_permissions(self) -> None:
        for verb, resource in (
            ("create", "jobs.batch"),
            ("create", "services"),
            ("get", "pods"),
            ("get", "pods/log"),
            ("delete", "jobs.batch"),
            ("delete", "services"),
        ):
            answer = self.run(["auth", "can-i", verb, resource]).strip().lower()
            if answer != "yes":
                raise KubernetesScalingError(
                    f"Kubernetes identity cannot {verb} {resource}"
                )


def validate_request(args: argparse.Namespace) -> dict[str, Any]:
    """Validate immutable execution inputs without contacting a cluster."""
    for field in (
        "gpu_count",
        "node_count",
        "nproc_per_node",
        "seed",
        "timeout_seconds",
        "fabric_resource_count",
    ):
        value = getattr(args, field)
        minimum = 0 if field == "seed" else 1
        if isinstance(value, bool) or value < minimum:
            raise KubernetesScalingError(f"{field} must be >= {minimum}")
    if args.node_count * args.nproc_per_node != args.gpu_count:
        raise KubernetesScalingError("node_count * nproc_per_node must equal gpu_count")
    if args.node_count > 1 and not args.fabric:
        raise KubernetesScalingError(
            "multi-node publication requires a declared high-bandwidth fabric"
        )
    if args.node_count > 1 and not args.fabric_resource:
        raise KubernetesScalingError(
            "multi-node publication requires an enforced fabric resource"
        )
    if not args.context.strip() or not args.namespace.strip():
        raise KubernetesScalingError("context and namespace must be explicit")
    if not DNS_LABEL.fullmatch(args.namespace) or len(args.namespace) > 63:
        raise KubernetesScalingError("namespace must be a DNS label")
    image_digest = args.image.rsplit("@sha256:", 1)[-1]
    if (
        "@sha256:" not in args.image
        or len(image_digest) != 64
        or any(char not in HEX for char in image_digest)
        or image_digest == "0" * 64
    ):
        raise KubernetesScalingError("image must be pinned by a sha256 digest")
    if (
        len(args.harness_commit) != 40
        or any(char not in HEX for char in args.harness_commit)
        or args.harness_commit == "0" * 40
    ):
        raise KubernetesScalingError("harness_commit must be a nonzero full commit")
    if (
        args.workload.name != "distributed_scaling_workload.py"
        or not args.workload.is_file()
    ):
        raise KubernetesScalingError("workload must be distributed_scaling_workload.py")
    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        raise KubernetesScalingError("config_json is invalid") from exc
    if not isinstance(config, dict) or config.get("scaling_mode") != "strong":
        raise KubernetesScalingError("publication collection requires strong scaling")
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        raise KubernetesScalingError(f"refusing to overwrite output: {args.output}")
    if bool(args.node_selector_key) != bool(args.node_selector_value):
        raise KubernetesScalingError(
            "node selector key and value must be supplied together"
        )
    if not 1 <= args.master_port <= 65535:
        raise KubernetesScalingError("master_port must be in [1, 65535]")
    if args.poll_seconds <= 0:
        raise KubernetesScalingError("poll_seconds must be positive")
    return config


def _container_script(args: argparse.Namespace, name: str, service: str) -> str:
    """Build the controlled in-container command for every indexed rank."""
    repository = f"{REMOTE_ROOT}/repository"
    output = f"{REMOTE_ROOT}/output/{args.output.name}"
    master = f"{name}-0.{service}"
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.node_count}",
        f"--nproc-per-node={args.nproc_per_node}",
        "--node-rank=${JOB_COMPLETION_INDEX}",
        f"--master-addr={master}",
        f"--master-port={args.master_port}",
        f"{repository}/benchmarks/distributed_scaling_workload.py",
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
    quoted = []
    for value in command:
        if value == "--node-rank=${JOB_COMPLETION_INDEX}":
            quoted.append('--node-rank="${JOB_COMPLETION_INDEX}"')
        else:
            quoted.append(shlex.quote(value))
    return " && ".join(
        (
            "set -eu",
            f"git clone --filter=blob:none {shlex.quote(args.repository_url)} {shlex.quote(repository)}",
            f"git -C {shlex.quote(repository)} checkout --detach {shlex.quote(args.harness_commit)}",
            f"python -m pip install --quiet --no-deps {shlex.quote(repository)}",
            f"mkdir -p {shlex.quote(REMOTE_ROOT + '/output')}",
            "exec " + " ".join(quoted),
        )
    )


def build_resources(
    args: argparse.Namespace, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the headless Service and anti-affined Indexed Job manifests."""
    service_name = f"{name}-headless"
    labels = {"app.kubernetes.io/name": "stateset-scaling", "stateset.ai/run": name}
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": service_name, "labels": labels},
        "spec": {
            "clusterIP": "None",
            "publishNotReadyAddresses": True,
            "selector": labels,
        },
    }
    container: dict[str, Any] = {
        "name": "trainer",
        "image": args.image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["bash", "-lc"],
        "args": [_container_script(args, name, service_name)],
        "env": [
            {
                "name": "JOB_COMPLETION_INDEX",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                    }
                },
            },
            {"name": "TORCH_NCCL_ASYNC_ERROR_HANDLING", "value": "1"},
        ],
        "resources": {
            "requests": {args.gpu_resource: args.nproc_per_node},
            "limits": {args.gpu_resource: args.nproc_per_node},
        },
    }
    if args.nccl_socket_ifname:
        container["env"].append(
            {"name": "NCCL_SOCKET_IFNAME", "value": args.nccl_socket_ifname}
        )
    if args.fabric_resource:
        container["resources"]["requests"][
            args.fabric_resource
        ] = args.fabric_resource_count
        container["resources"]["limits"][
            args.fabric_resource
        ] = args.fabric_resource_count
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "subdomain": service_name,
        "containers": [container],
        "affinity": {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {"matchLabels": labels},
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ]
            }
        },
    }
    if args.service_account:
        pod_spec["serviceAccountName"] = args.service_account
    if args.node_selector_key:
        pod_spec["nodeSelector"] = {args.node_selector_key: args.node_selector_value}
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "completions": args.node_count,
            "parallelism": args.node_count,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            "activeDeadlineSeconds": args.timeout_seconds,
            "template": {
                "metadata": {"labels": labels},
                "spec": pod_spec,
            },
        },
    }
    return service, job


def _write_record(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise KubernetesScalingError(f"refusing to overwrite provider record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def execute(args: argparse.Namespace, kube: KubernetesApi) -> Path:
    """Apply, observe, retrieve, attest, and unconditionally clean one row."""
    name = f"stateset-scale-{args.seed}-{uuid.uuid4().hex[:8]}"
    service, job = build_resources(args, name)
    service_applied = False
    job_applied = False
    cleanup_errors: list[str] = []
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    completed = False
    pods: list[dict[str, Any]] = []
    try:
        kube.require_permissions()
        kube.apply(service)
        service_applied = True
        kube.apply(job)
        job_applied = True
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            state = kube.get_json("job", name).get("status", {})
            conditions = state.get("conditions", []) if isinstance(state, dict) else []
            if any(
                c.get("type") == "Complete" and c.get("status") == "True"
                for c in conditions
            ):
                break
            if any(
                c.get("type") == "Failed" and c.get("status") == "True"
                for c in conditions
            ):
                raise KubernetesScalingError(f"Kubernetes scaling Job {name} failed")
            time.sleep(args.poll_seconds)
        else:
            raise KubernetesScalingError(f"Kubernetes scaling Job {name} timed out")

        pods = kube.list_pods(f"stateset.ai/run={name}")
        node_names = {str(pod.get("spec", {}).get("nodeName") or "") for pod in pods}
        if (
            len(pods) != args.node_count
            or "" in node_names
            or len(node_names) != args.node_count
        ):
            raise KubernetesScalingError(
                "scheduler did not attest one distinct Kubernetes node per rank"
            )
        rank_zero = [
            pod
            for pod in pods
            if pod.get("metadata", {})
            .get("annotations", {})
            .get("batch.kubernetes.io/job-completion-index")
            == "0"
        ]
        if len(rank_zero) != 1:
            raise KubernetesScalingError("could not identify exactly one rank-zero pod")
        pod_name = str(rank_zero[0]["metadata"]["name"])
        remote = f"{REMOTE_ROOT}/output"
        kube.copy_from(pod_name, f"{remote}/{args.output.name}", args.output)
        artifact = args.output.with_suffix(".pt")
        kube.copy_from(pod_name, f"{remote}/{artifact.name}", artifact)
        load_scaling_evidence([args.output])
        completed = True
        return args.output
    except Exception as exc:
        if job_applied:
            try:
                pods = kube.list_pods(f"stateset.ai/run={name}")
                tails = []
                for pod in pods:
                    pod_name = str(pod.get("metadata", {}).get("name") or "")
                    if pod_name:
                        try:
                            tails.append({"pod": pod_name, "tail": kube.logs(pod_name)})
                        except Exception:
                            pass
                if tails:
                    exc.__dict__["stateset_pod_logs"] = tails
            except Exception:
                pass
        raise
    finally:
        if job_applied:
            try:
                kube.delete("job", name)
            except Exception as exc:
                cleanup_errors.append(f"job/{name}: {exc}")
        if service_applied:
            try:
                kube.delete("service", service["metadata"]["name"])
            except Exception as exc:
                cleanup_errors.append(f"service/{service['metadata']['name']}: {exc}")
        record = {
            "schema_version": 1,
            "kind": "stateset-kubernetes-scaling-provider-record",
            "provider": args.provider,
            "status": (
                "completed"
                if completed and not cleanup_errors
                else "cleanup-pending" if cleanup_errors else "failed"
            ),
            "context": args.context,
            "namespace": args.namespace,
            "job": name,
            "harness_commit": args.harness_commit,
            "image": args.image,
            "fabric": args.fabric,
            "fabric_resource": args.fabric_resource,
            "fabric_resource_count_per_node": args.fabric_resource_count,
            "gpu_count": args.gpu_count,
            "node_count": args.node_count,
            "nproc_per_node": args.nproc_per_node,
            "seed": args.seed,
            "output": args.output.name,
            "config_sha256": hashlib.sha256(
                json.dumps(
                    json.loads(args.config_json), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "duration_seconds": round(max(0.0, time.time() - started), 3),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "node_names_sha256": sorted(
                hashlib.sha256(str(node).encode()).hexdigest()
                for node in {
                    pod.get("spec", {}).get("nodeName")
                    for pod in pods
                    if pod.get("spec", {}).get("nodeName")
                }
            ),
            "pod_uids_sha256": sorted(
                hashlib.sha256(str(uid).encode()).hexdigest()
                for uid in {
                    pod.get("metadata", {}).get("uid")
                    for pod in pods
                    if pod.get("metadata", {}).get("uid")
                }
            ),
            "resources_deleted": not cleanup_errors,
            "cleanup_errors": cleanup_errors,
        }
        record_path = (
            args.output.parent.parent / "kubernetes-provider" / args.output.name
        )
        try:
            _write_record(record_path, record)
        except Exception as exc:
            if completed:
                raise KubernetesScalingError(
                    f"evidence retrieved but provider record failed: {exc}"
                ) from exc
        if completed and cleanup_errors:
            raise KubernetesScalingError(
                "evidence retrieved but Kubernetes cleanup failed: "
                + "; ".join(cleanup_errors)
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
    parser.add_argument(
        "--provider", choices=("coreweave", "nebius", "kubernetes"), required=True
    )
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--fabric", required=True)
    parser.add_argument("--fabric-resource")
    parser.add_argument("--fabric-resource-count", type=int, default=1)
    parser.add_argument("--gpu-resource", default="nvidia.com/gpu")
    parser.add_argument("--node-selector-key")
    parser.add_argument("--node-selector-value")
    parser.add_argument("--service-account")
    parser.add_argument("--nccl-socket-ifname")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--repository-url", default="https://github.com/stateset/stateset-agents.git"
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_request(args)
        name = "stateset-scale-contract"
        service, job = build_resources(args, name)
        print(json.dumps({"service": service, "job": job}, indent=2, sort_keys=True))
        if args.execute:
            execute(
                args,
                KubernetesApi(
                    context=args.context,
                    namespace=args.namespace,
                ),
            )
    except (KubernetesScalingError, RemoteExecutionError, OSError) as exc:
        print(f"Kubernetes scaling rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
