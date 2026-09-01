#!/usr/bin/env python3
"""Validate publication-grade multi-node asynchronous RL evidence.

Provider-specific launchers emit one JSON document per scenario and seed. This
module rejects synthetic, single-node, short, corrupt, semantically drifted,
or incomplete matrices before a summary can be published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_SCENARIOS = (
    "steady_state_soak",
    "worker_exit",
    "controller_restart",
    "network_interruption",
)
HEX_DIGITS = frozenset("0123456789abcdef")


class DistributedAsyncEvidenceError(ValueError):
    """Raised when distributed asynchronous evidence is not publishable."""


def _nonempty(data: Mapping[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DistributedAsyncEvidenceError(
            f"{source}: {key} must be a non-empty string"
        )
    return value


def _integer(
    data: Mapping[str, Any], key: str, source: Path, *, minimum: int = 0
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DistributedAsyncEvidenceError(
            f"{source}: {key} must be an integer >= {minimum}"
        )
    return value


def _number(
    data: Mapping[str, Any], key: str, source: Path, *, minimum: float = 0.0
) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistributedAsyncEvidenceError(f"{source}: {key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise DistributedAsyncEvidenceError(
            f"{source}: {key} must be finite and >= {minimum}"
        )
    return result


def _digest(data: Mapping[str, Any], key: str, source: Path) -> str:
    value = _nonempty(data, key, source)
    if len(value) != 64 or any(char not in HEX_DIGITS for char in value):
        raise DistributedAsyncEvidenceError(
            f"{source}: {key} must be 64 lowercase hexadecimal characters"
        )
    return value


def _section(data: Mapping[str, Any], key: str, source: Path) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise DistributedAsyncEvidenceError(f"{source}: {key} must be an object")
    return value


def validate_run(data: Mapping[str, Any], source: Path) -> dict[str, Any]:
    """Validate one measured multi-node scenario and its safety invariants."""
    if (
        data.get("schema_version") != 1
        or data.get("kind") != "stateset-distributed-async-evidence"
        or data.get("status") != "completed"
        or data.get("measured") is not True
    ):
        raise DistributedAsyncEvidenceError(
            f"{source}: completed measured schema_version=1 evidence is required"
        )
    for key in (
        "run_id",
        "framework_version",
        "protocol",
        "timestamp",
        "provider",
    ):
        _nonempty(data, key, source)
    commit = _nonempty(data, "harness_commit", source)
    if len(commit) != 40 or any(char not in HEX_DIGITS for char in commit):
        raise DistributedAsyncEvidenceError(
            f"{source}: harness_commit must be a full lowercase git commit"
        )
    _integer(data, "seed", source)
    duration = _number(data, "duration_seconds", source, minimum=0.001)
    scenario = _nonempty(data, "scenario", source)
    if scenario not in REQUIRED_SCENARIOS:
        raise DistributedAsyncEvidenceError(
            f"{source}: unsupported scenario {scenario!r}"
        )

    config = _section(data, "config", source)
    if not config:
        raise DistributedAsyncEvidenceError(f"{source}: config must not be empty")
    config_digest = _digest(data, "config_sha256", source)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != config_digest:
        raise DistributedAsyncEvidenceError(
            f"{source}: config_sha256 does not match canonical config"
        )
    max_policy_lag = _integer(config, "max_policy_lag", source)

    topology = _section(data, "topology", source)
    node_count = _integer(topology, "node_count", source, minimum=2)
    worker_count = _integer(topology, "worker_count", source, minimum=2)
    node_ids = topology.get("node_ids")
    if (
        not isinstance(node_ids, list)
        or len(node_ids) != node_count
        or any(not isinstance(item, str) or not item.strip() for item in node_ids)
        or len(node_ids) != len(set(node_ids))
    ):
        raise DistributedAsyncEvidenceError(
            f"{source}: topology.node_ids must contain one unique ID per node"
        )
    if worker_count < node_count:
        raise DistributedAsyncEvidenceError(
            f"{source}: topology.worker_count must cover every node"
        )
    for key in ("accelerator", "accelerator_driver", "interconnect"):
        _nonempty(topology, key, source)

    metrics = _section(data, "metrics", source)
    attempted = _integer(metrics, "rollouts_attempted", source, minimum=1)
    accepted = _integer(metrics, "rollouts_accepted", source, minimum=1)
    if accepted > attempted:
        raise DistributedAsyncEvidenceError(
            f"{source}: accepted rollouts exceed attempted rollouts"
        )
    _integer(metrics, "optimizer_updates", source, minimum=1)
    _integer(metrics, "policy_versions", source, minimum=2)
    observed_lag = _integer(metrics, "max_observed_policy_lag", source)
    if observed_lag > max_policy_lag:
        raise DistributedAsyncEvidenceError(
            f"{source}: observed policy lag exceeds the configured bound"
        )
    for key in (
        "lost_optimizer_updates",
        "duplicate_optimizer_updates",
        "artifact_digest_mismatches",
    ):
        if _integer(metrics, key, source) != 0:
            raise DistributedAsyncEvidenceError(f"{source}: {key} must be zero")
    throughput = _number(
        metrics, "accepted_rollouts_per_second", source, minimum=0.000001
    )
    if not math.isclose(throughput, accepted / duration, rel_tol=0.02):
        raise DistributedAsyncEvidenceError(
            f"{source}: rollout throughput does not match accepted work / duration"
        )

    sync = _section(metrics, "weight_sync_latency_ms", source)
    p50 = _number(sync, "p50", source, minimum=0.000001)
    p95 = _number(sync, "p95", source, minimum=0.000001)
    maximum = _number(sync, "max", source, minimum=0.000001)
    if not p50 <= p95 <= maximum:
        raise DistributedAsyncEvidenceError(
            f"{source}: weight-sync latency must satisfy p50 <= p95 <= max"
        )

    cost = _section(data, "cost", source)
    if _nonempty(cost, "currency", source) != "USD":
        raise DistributedAsyncEvidenceError(f"{source}: cost.currency must be USD")
    _nonempty(cost, "source", source)
    total_cost = _number(cost, "total", source)
    per_rollout = _number(cost, "per_accepted_rollout", source)
    if not math.isclose(per_rollout, total_cost / accepted, rel_tol=0.02, abs_tol=1e-9):
        raise DistributedAsyncEvidenceError(
            f"{source}: per-rollout cost does not match total / accepted"
        )

    recovery = _section(data, "recovery", source)
    if recovery.get("recovered") is not True:
        raise DistributedAsyncEvidenceError(f"{source}: recovery must complete")
    _number(recovery, "recovery_seconds", source)
    if _integer(recovery, "resources_remaining", source) != 0:
        raise DistributedAsyncEvidenceError(
            f"{source}: resources remain after the scenario"
        )
    _digest(data, "artifact_sha256", source)
    return dict(data)


def load_runs(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    """Load JSON files or directories and validate every discovered run."""
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise DistributedAsyncEvidenceError(f"input does not exist: {candidate}")
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not paths:
        raise DistributedAsyncEvidenceError("no distributed async evidence found")
    runs: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DistributedAsyncEvidenceError(f"{path}: invalid JSON") from exc
        if not isinstance(raw, Mapping):
            raise DistributedAsyncEvidenceError(
                f"{path}: top-level value must be an object"
            )
        runs.append(validate_run(raw, path))
    return runs


def validate_matrix(
    runs: Sequence[Mapping[str, Any]],
    *,
    min_seeds: int = 3,
    min_soak_seconds: float = 43_200.0,
    min_nodes: int = 2,
) -> None:
    """Require a matched, multi-node, three-seed fault and soak matrix."""
    if min_seeds < 1 or min_soak_seconds <= 0 or min_nodes < 2:
        raise DistributedAsyncEvidenceError("invalid matrix gate configuration")
    if not runs:
        raise DistributedAsyncEvidenceError("evidence matrix is empty")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["scenario"]), []).append(run)
    if set(grouped) != set(REQUIRED_SCENARIOS):
        raise DistributedAsyncEvidenceError(
            f"scenario matrix mismatch: required={sorted(REQUIRED_SCENARIOS)}, "
            f"actual={sorted(grouped)}"
        )

    first = runs[0]
    matched_fields = (
        "framework_version",
        "harness_commit",
        "protocol",
        "provider",
        "config_sha256",
    )
    for run in runs[1:]:
        changed = [field for field in matched_fields if run[field] != first[field]]
        for field in (
            "node_count",
            "worker_count",
            "accelerator",
            "accelerator_driver",
            "interconnect",
        ):
            if run["topology"][field] != first["topology"][field]:
                changed.append(f"topology.{field}")
        if changed:
            raise DistributedAsyncEvidenceError(
                "evidence matrix mixes " + ", ".join(changed)
            )
    if int(first["topology"]["node_count"]) < min_nodes:
        raise DistributedAsyncEvidenceError(
            f"matrix uses fewer than {min_nodes} physical nodes"
        )

    expected_seeds: set[int] | None = None
    for scenario, scenario_runs in sorted(grouped.items()):
        seeds = [int(run["seed"]) for run in scenario_runs]
        if len(seeds) != len(set(seeds)):
            raise DistributedAsyncEvidenceError(f"{scenario}: duplicate seeds")
        if len(seeds) < min_seeds:
            raise DistributedAsyncEvidenceError(
                f"{scenario}: only {len(seeds)} seeds; need {min_seeds}"
            )
        seed_set = set(seeds)
        if expected_seeds is None:
            expected_seeds = seed_set
        elif seed_set != expected_seeds:
            raise DistributedAsyncEvidenceError(f"{scenario}: seed set mismatch")
    for run in grouped["steady_state_soak"]:
        if float(run["duration_seconds"]) < min_soak_seconds:
            raise DistributedAsyncEvidenceError(
                "steady_state_soak duration is below the publication floor"
            )


def summarize(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a compact report without discarding cost or tail latency."""
    total_rollouts = sum(int(run["metrics"]["rollouts_accepted"]) for run in runs)
    total_cost = sum(float(run["cost"]["total"]) for run in runs)
    return {
        "schema_version": 1,
        "kind": "stateset-distributed-async-matrix",
        "passed": True,
        "run_count": len(runs),
        "scenarios": sorted({str(run["scenario"]) for run in runs}),
        "seeds": sorted({int(run["seed"]) for run in runs}),
        "node_count": int(runs[0]["topology"]["node_count"]),
        "accepted_rollouts": total_rollouts,
        "total_cost_usd": total_cost,
        "cost_per_accepted_rollout_usd": total_cost / total_rollouts,
        "max_weight_sync_p95_ms": max(
            float(run["metrics"]["weight_sync_latency_ms"]["p95"]) for run in runs
        ),
        "max_observed_policy_lag": max(
            int(run["metrics"]["max_observed_policy_lag"]) for run in runs
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the publication gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--min-soak-seconds", type=float, default=43_200.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        runs = load_runs(args.inputs)
        validate_matrix(
            runs,
            min_seeds=args.min_seeds,
            min_nodes=args.min_nodes,
            min_soak_seconds=args.min_soak_seconds,
        )
        report = summarize(runs)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    except DistributedAsyncEvidenceError as exc:
        print(f"distributed async evidence rejected: {exc}", file=sys.stderr)
        return 2
    print(f"validated {len(runs)} measured distributed async runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
