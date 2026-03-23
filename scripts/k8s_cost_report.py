#!/usr/bin/env python3
"""
Kubernetes (GKE) cost-triage report.

This script uses `kubectl` to estimate which namespaces/pods are likely driving
cost by summing *resource requests* (CPU/memory/GPU/ephemeral-storage).

Notes:
- On GKE Autopilot, billing is largely based on requested resources, so this is
  a good proxy for cost drivers.
- On Standard GKE, billing is driven by node VMs and attached resources; pod
  requests are still useful to spot "why did we scale up", but node inventory
  matters more.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from collections.abc import Iterable

_QTY_RE = re.compile(r"^([0-9.]+)([KMGTE]i?|m)?$")
_RS_HASH_RE = re.compile(r"^(?P<name>.+)-[0-9a-f]{9,10}$")


def _run_kubectl(args: list[str]) -> str:
    try:
        return subprocess.check_output(["kubectl", *args], text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise SystemExit(f"kubectl failed: kubectl {' '.join(args)}\n{exc}") from exc


def _parse_cpu_millicores(value: str | None) -> int:
    if not value:
        return 0
    value = str(value)
    if value.endswith("m"):
        return int(float(value[:-1]))
    return int(float(value) * 1000)


def _parse_bytes(value: str | None) -> int:
    if not value:
        return 0
    value = str(value)
    m = _QTY_RE.match(value)
    if not m:
        return 0
    num = float(m.group(1))
    suffix = m.group(2)
    if not suffix:
        return int(num)
    if suffix == "m":
        # CPU millicores would use "m"; treat as 0 bytes for safety.
        return 0
    table = {
        "K": 10**3,
        "M": 10**6,
        "G": 10**9,
        "T": 10**12,
        "E": 10**15,
        "Ki": 2**10,
        "Mi": 2**20,
        "Gi": 2**30,
        "Ti": 2**40,
        "Ei": 2**50,
    }
    return int(num * table.get(suffix, 1))


def _fmt_cpu(mcpu: int) -> str:
    return f"{mcpu/1000:.2f}" if mcpu >= 1000 else f"{mcpu}m"


def _fmt_gib(bytes_value: int) -> str:
    return f"{bytes_value/2**30:.1f}Gi"


def _fmt_tib(bytes_value: int) -> str:
    return f"{bytes_value/2**40:.2f}Ti"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_str(ts: datetime | None, now: datetime) -> str:
    if ts is None:
        return "?"
    delta = now - ts
    days = delta.days
    hours = delta.seconds // 3600
    mins = (delta.seconds % 3600) // 60
    if days:
        return f"{days}d{hours}h"
    return f"{hours}h{mins}m"


@dataclass
class PodRequest:
    namespace: str
    name: str
    cpu_mcpu: int
    mem_bytes: int
    eph_bytes: int
    gpus: int
    created_at: datetime | None
    group: str


def _sum_pod_requests(pod: dict[str, Any]) -> tuple[int, int, int, int]:
    spec = pod.get("spec") or {}
    containers: Iterable[dict[str, Any]] = (spec.get("containers") or []) + (
        spec.get("initContainers") or []
    )

    cpu = mem = eph = gpu = 0
    for c in containers:
        req = ((c.get("resources") or {}).get("requests")) or {}
        cpu += _parse_cpu_millicores(req.get("cpu"))
        mem += _parse_bytes(req.get("memory"))
        eph += _parse_bytes(req.get("ephemeral-storage"))
        try:
            gpu += int(req.get("nvidia.com/gpu") or 0)
        except (TypeError, ValueError):
            gpu += 0
    return cpu, mem, eph, gpu


def _pod_group(namespace: str, pod: dict[str, Any]) -> str:
    meta = pod.get("metadata") or {}
    name = meta.get("name") or "unknown"
    owners = meta.get("ownerReferences") or []
    if owners:
        owner = owners[0] or {}
        kind = owner.get("kind") or "Unknown"
        oname = owner.get("name") or "unknown"
        if kind == "ReplicaSet":
            m = _RS_HASH_RE.match(oname)
            if m:
                return f"{namespace}/Deployment:{m.group('name')}"
        return f"{namespace}/{kind}:{oname}"

    # Fallback heuristic for naked pods (often custom controllers).
    parts = name.split("-")
    if len(parts) >= 2:
        return f"{namespace}/PodGroup:{parts[0]}-{parts[1]}"
    return f"{namespace}/Pod:{name}"


def _load_json(kind: str, args: list[str]) -> dict[str, Any]:
    raw = _run_kubectl(args)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise SystemExit(f"kubectl returned invalid JSON for {kind}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GKE/Kubernetes cost triage report")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many rows to show per section (default: 20).",
    )
    parser.add_argument(
        "--since",
        default="",
        help=(
            "Optional ISO date (YYYY-MM-DD) to highlight pods created since that date "
            "(useful for 'what changed' questions)."
        ),
    )
    parser.add_argument(
        "--exclude-namespaces",
        default="",
        help=(
            "Comma-separated namespaces to exclude from pod request summaries "
            "(e.g. kube-system,gke-gmp-system). Default: none."
        ),
    )
    args = parser.parse_args(argv)

    since_dt: datetime | None = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise SystemExit(
                f"--since must be YYYY-MM-DD (got: {args.since!r})"
            ) from exc

    now = datetime.now(timezone.utc)
    context = _run_kubectl(["config", "current-context"]).strip()

    excluded = {ns.strip() for ns in args.exclude_namespaces.split(",") if ns.strip()}

    pods = _load_json("pods", ["get", "pods", "-A", "-o", "json"])
    nodes = _load_json("nodes", ["get", "nodes", "-o", "json"])

    # Optional resources (best-effort).
    try:
        svcs = _load_json("services", ["get", "svc", "-A", "-o", "json"])
    except SystemExit:
        svcs = {"items": []}
    try:
        pvcs = _load_json("pvcs", ["get", "pvc", "-A", "-o", "json"])
    except SystemExit:
        pvcs = {"items": []}

    # Node inventory.
    node_counts: Counter[str] = Counter()
    node_counts_zone: Counter[tuple[str, str]] = Counter()
    for node in nodes.get("items", []):
        labels = (node.get("metadata") or {}).get("labels") or {}
        it = (
            labels.get("node.kubernetes.io/instance-type")
            or labels.get("beta.kubernetes.io/instance-type")
            or "?"
        )
        zone = (
            labels.get("topology.kubernetes.io/zone")
            or labels.get("failure-domain.beta.kubernetes.io/zone")
            or "?"
        )
        node_counts[it] += 1
        node_counts_zone[(zone, it)] += 1

    # Pod requests by namespace.
    ns_tot = defaultdict(lambda: {"cpu": 0, "mem": 0, "eph": 0, "gpu": 0, "pods": 0})
    group_tot = defaultdict(lambda: {"cpu": 0, "mem": 0, "eph": 0, "gpu": 0, "pods": 0})
    pod_rows: list[PodRequest] = []

    for pod in pods.get("items", []):
        meta = pod.get("metadata") or {}
        ns = meta.get("namespace") or "default"
        name = meta.get("name") or ""
        created_at = _parse_ts(meta.get("creationTimestamp"))

        if ns in excluded:
            continue

        cpu, mem, eph, gpu = _sum_pod_requests(pod)
        group = _pod_group(ns, pod)
        ns_tot[ns]["cpu"] += cpu
        ns_tot[ns]["mem"] += mem
        ns_tot[ns]["eph"] += eph
        ns_tot[ns]["gpu"] += gpu
        ns_tot[ns]["pods"] += 1
        group_tot[group]["cpu"] += cpu
        group_tot[group]["mem"] += mem
        group_tot[group]["eph"] += eph
        group_tot[group]["gpu"] += gpu
        group_tot[group]["pods"] += 1

        pod_rows.append(
            PodRequest(
                namespace=ns,
                name=name,
                cpu_mcpu=cpu,
                mem_bytes=mem,
                eph_bytes=eph,
                gpus=gpu,
                created_at=created_at,
                group=group,
            )
        )

    cpu_total = sum(v["cpu"] for v in ns_tot.values())
    mem_total = sum(v["mem"] for v in ns_tot.values())
    eph_total = sum(v["eph"] for v in ns_tot.values())
    gpu_total = sum(v["gpu"] for v in ns_tot.values())

    print(f"context: {context}")
    if excluded:
        print(f"excluded namespaces: {', '.join(sorted(excluded))}")
    print(
        "pod requests (sum): "
        f"cpu={_fmt_cpu(cpu_total)} cores={cpu_total/1000:.2f} "
        f"mem={_fmt_gib(mem_total)} eph={_fmt_tib(eph_total)} "
        f"gpu={gpu_total}"
    )
    print("")

    print("nodes (count by instance type):")
    for it, count in node_counts.most_common():
        print(f"  {it:28} {count}")
    print("")

    print("top workload groups by CPU requests:")
    for group, tot in sorted(
        group_tot.items(), key=lambda kv: kv[1]["cpu"], reverse=True
    )[: args.top]:
        print(
            f"  {group:44} cpu={_fmt_cpu(tot['cpu']):>7} "
            f"mem={_fmt_gib(tot['mem']):>9} gpu={tot['gpu']:<3} pods={tot['pods']}"
        )
    print("")

    print("top workload groups by Memory requests:")
    for group, tot in sorted(
        group_tot.items(), key=lambda kv: kv[1]["mem"], reverse=True
    )[: args.top]:
        print(
            f"  {group:44} mem={_fmt_gib(tot['mem']):>9} "
            f"cpu={_fmt_cpu(tot['cpu']):>7} gpu={tot['gpu']:<3} pods={tot['pods']}"
        )
    print("")

    print("top namespaces by CPU requests:")
    for ns, tot in sorted(ns_tot.items(), key=lambda kv: kv[1]["cpu"], reverse=True)[
        : args.top
    ]:
        print(
            f"  {ns:22} cpu={_fmt_cpu(tot['cpu']):>7} "
            f"mem={_fmt_gib(tot['mem']):>9} eph={_fmt_tib(tot['eph']):>8} "
            f"gpu={tot['gpu']:<3} pods={tot['pods']}"
        )
    print("")

    print("top namespaces by Memory requests:")
    for ns, tot in sorted(ns_tot.items(), key=lambda kv: kv[1]["mem"], reverse=True)[
        : args.top
    ]:
        print(
            f"  {ns:22} mem={_fmt_gib(tot['mem']):>9} "
            f"cpu={_fmt_cpu(tot['cpu']):>7} eph={_fmt_tib(tot['eph']):>8} "
            f"gpu={tot['gpu']:<3} pods={tot['pods']}"
        )
    print("")

    gpu_pods = [p for p in pod_rows if p.gpus]
    print("gpu-requesting pods:")
    if not gpu_pods:
        print("  (none)")
    else:
        for p in sorted(gpu_pods, key=lambda p: p.gpus, reverse=True)[: args.top * 5]:
            print(f"  {p.namespace}/{p.name}: gpu={p.gpus}")
    print("")

    print("top pods by CPU requests:")
    for p in sorted(pod_rows, key=lambda p: p.cpu_mcpu, reverse=True)[: args.top]:
        if not (p.cpu_mcpu or p.mem_bytes or p.gpus):
            break
        print(
            f"  {p.namespace:18} {p.name:55} "
            f"cpu={_fmt_cpu(p.cpu_mcpu):>7} mem={_fmt_gib(p.mem_bytes):>9} "
            f"gpu={p.gpus:<2} age={_age_str(p.created_at, now)}"
        )
    print("")

    print("top pods by Memory requests:")
    for p in sorted(pod_rows, key=lambda p: p.mem_bytes, reverse=True)[: args.top]:
        if not (p.cpu_mcpu or p.mem_bytes or p.gpus):
            break
        print(
            f"  {p.namespace:18} {p.name:55} "
            f"mem={_fmt_gib(p.mem_bytes):>9} cpu={_fmt_cpu(p.cpu_mcpu):>7} "
            f"gpu={p.gpus:<2} age={_age_str(p.created_at, now)}"
        )
    print("")

    if since_dt is not None:
        print(
            f"pods created since {since_dt.date().isoformat()} (high requests first):"
        )
        recent = [p for p in pod_rows if p.created_at and p.created_at >= since_dt]
        for p in sorted(
            recent, key=lambda p: (p.gpus, p.cpu_mcpu, p.mem_bytes), reverse=True
        )[: args.top * 2]:
            print(
                f"  {p.namespace:18} {p.name:55} "
                f"cpu={_fmt_cpu(p.cpu_mcpu):>7} mem={_fmt_gib(p.mem_bytes):>9} "
                f"gpu={p.gpus:<2} created={p.created_at.isoformat()}"
            )
        print("")

    # LoadBalancers (often low cost, but useful to spot proliferation).
    lb = []
    for svc in svcs.get("items", []):
        spec = svc.get("spec") or {}
        if (spec.get("type") or "") == "LoadBalancer":
            meta = svc.get("metadata") or {}
            ns = meta.get("namespace") or "default"
            name = meta.get("name") or ""
            lb.append(f"{ns}/{name}")
    print("services type=LoadBalancer:")
    if lb:
        for s in sorted(lb)[: args.top * 5]:
            print(f"  {s}")
    else:
        print("  (none)")
    print("")

    # PVCs (storage cost).
    pvc_tot = defaultdict(int)
    pvc_rows = []
    for pvc in pvcs.get("items", []):
        meta = pvc.get("metadata") or {}
        ns = meta.get("namespace") or "default"
        name = meta.get("name") or ""
        req = ((pvc.get("spec") or {}).get("resources") or {}).get("requests") or {}
        size = _parse_bytes(req.get("storage"))
        pvc_tot[ns] += size
        pvc_rows.append((size, ns, name))

    print("pvc requested storage (top namespaces):")
    for ns, size in sorted(pvc_tot.items(), key=lambda kv: kv[1], reverse=True)[
        : args.top
    ]:
        print(f"  {ns:22} storage={_fmt_tib(size):>8}")
    print("")
    print("largest pvcs:")
    for size, ns, name in sorted(pvc_rows, reverse=True)[: args.top]:
        if size <= 0:
            break
        print(f"  {ns:22} {name:45} storage={_fmt_tib(size):>8}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
