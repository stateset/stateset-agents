#!/usr/bin/env python3
"""Lay out, report on, and gate the A+ evidence bundle for one frozen commit.

``a_plus_gate.py`` takes thirteen evidence paths that must all describe the
same clean commit. This helper fixes one directory layout under an evidence
root, reports which gates are present, collects the gates that cost nothing
(reliability on CPU and release readiness), and runs the gate with every path
filled in.

    # which gates have evidence?
    python benchmarks/a_plus_evidence.py status --root benchmark_results/a_plus/<commit>

    # collect the free gates at the checked-out (clean) commit
    python benchmarks/a_plus_evidence.py collect-free --root ...

    # run the publication gate over the whole bundle
    python benchmarks/a_plus_evidence.py gate --root ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GateInput:
    """One ``a_plus_gate.py`` argument and where its evidence lives."""

    flag: str
    relative_path: str
    gate: str
    how: str
    required: bool = True


LAYOUT: tuple[GateInput, ...] = (
    GateInput(
        "--framework-evidence",
        "framework",
        "competitive frameworks",
        "benchmarks/runpod_shootout.py benchmarks/runpod_shootout_v6_manifest.json",
    ),
    GateInput(
        "--scaling-evidence",
        "scaling/runs",
        "scaling",
        "benchmarks/run_scaling_matrix.py with the RunPod launcher manifest",
    ),
    GateInput(
        "--scaling-provider-evidence",
        "scaling/provider",
        "scaling",
        "written by benchmarks/runpod_scaling_launcher.py",
    ),
    GateInput(
        "--scaling-image-attestation",
        "scaling/image-attestation.json",
        "scaling",
        "attest the image built from deployment/docker/Dockerfile.scaling",
    ),
    GateInput(
        "--scaling-billing-evidence",
        "scaling/billing",
        "scaling",
        "only for Kubernetes scaling providers",
        required=False,
    ),
    GateInput(
        "--reliability-evidence",
        "reliability/evidence",
        "reliability",
        "free: collect-free",
    ),
    GateInput(
        "--distributed-async-evidence",
        "distributed_async/evidence",
        "distributed async",
        "benchmarks/run_distributed_async_matrix.py with the RunPod driver",
    ),
    GateInput(
        "--provider-evidence",
        "providers",
        "providers",
        "read-only provider canaries (River, RunPod, Fireworks, CoreWeave, Nebius)",
    ),
    GateInput(
        "--agent-quality-evidence",
        "agent_quality/evidence",
        "agent quality",
        "benchmarks/run_agent_quality_matrix.py (tau3-bench + bfcl-v4)",
    ),
    GateInput(
        "--flagship-manifest",
        "flagship/manifest.json",
        "flagship",
        "copy of benchmarks/flagship_manifest.json used for the run",
    ),
    GateInput(
        "--flagship-evidence",
        "flagship/evidence",
        "flagship",
        "benchmarks/run_flagship_matrix.py benchmarks/flagship_manifest.json",
    ),
    GateInput(
        "--release-readiness",
        "release/publish-readiness-summary.json",
        "release readiness",
        "free: collect-free (scripts/publish_readiness.sh)",
    ),
)


def _present(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file()


def status(root: Path) -> list[dict[str, object]]:
    """Return one row per gate input with whether its evidence exists."""
    return [
        {
            "flag": item.flag,
            "gate": item.gate,
            "path": str(root / item.relative_path),
            "present": _present(root / item.relative_path),
            "required": item.required,
            "how": item.how,
        }
        for item in LAYOUT
    ]


def gate_argv(root: Path, expected_commit: str, output: Path) -> list[str]:
    """Build the ``a_plus_gate.py`` argument list for the bundle under ``root``."""
    missing = [
        row["flag"] for row in status(root) if row["required"] and not row["present"]
    ]
    if missing:
        raise SystemExit(
            "evidence is missing for: " + ", ".join(str(flag) for flag in missing)
        )
    argv = ["--expected-commit", expected_commit, "--output", str(output)]
    for item in LAYOUT:
        path = root / item.relative_path
        if item.required or _present(path):
            argv += [item.flag, str(path)]
    return argv


def _head_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return result.stdout.strip()


def collect_free(root: Path, python: str) -> None:
    """Collect the reliability matrix (CPU) and release readiness at HEAD."""
    commit = _head_commit()
    reliability = root / "reliability"
    subprocess.run(
        [
            python,
            "benchmarks/run_reliability_matrix.py",
            "--seeds",
            "42",
            "1337",
            "2026",
            "--device",
            "cpu",
            "--output-dir",
            str(reliability),
            "--harness-commit",
            commit,
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["bash", "scripts/publish_readiness.sh"], cwd=ROOT, check=True)
    release = root / "release"
    release.mkdir(parents=True, exist_ok=True)
    summary = ROOT / "publish-readiness-summary.json"
    (release / summary.name).write_bytes(summary.read_bytes())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "collect-free", "gate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    if args.command == "status":
        rows = status(args.root)
        print(json.dumps(rows, indent=2))
        missing = [r for r in rows if r["required"] and not r["present"]]
        print(f"{len(rows) - len(missing)}/{len(rows)} inputs present", file=sys.stderr)
        return 0 if not missing else 1
    if args.command == "collect-free":
        collect_free(args.root, args.python)
        return 0
    expected = args.expected_commit or _head_commit()
    output = args.output or args.root / "report.json"
    completed = subprocess.run(
        [
            args.python,
            "benchmarks/a_plus_gate.py",
            *gate_argv(args.root, expected, output),
        ],
        cwd=ROOT,
    )
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
