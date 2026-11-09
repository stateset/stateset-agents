"""The process that runs inside each sandbox: one run, one ``result.json``.

Imports only the standard library at module level. Whatever goes wrong while
loading the workload or its dependencies must still end in a result the
orchestrator can read, because the difference between "the workload failed" and
"the worker never ran" decides whether a retry could help.

    python -m stateset_agents.fanout.worker \\
        --spec spec.json --output-dir output --stage-dir stage
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stateset_agents.utils.atomic_io import atomic_json

__all__ = ["EVENT_MARKER", "METRICS_FILE", "RESULT_FILE", "WorkloadOutcome", "main"]

RESULT_FILE = "result.json"
METRICS_FILE = "metrics.json"
#: Prefix of a structured progress line on stdout.
EVENT_MARKER = "STATESET_FANOUT_EVENT="


@dataclass
class WorkloadOutcome:
    """What a workload produced. ``exit_code`` 0 means it succeeded."""

    exit_code: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def emit(event: dict[str, Any]) -> None:
    """Write a progress event; stdout is mirrored back by the orchestrator."""
    sys.stdout.write(EVENT_MARKER + json.dumps(event, sort_keys=True) + "\n")
    sys.stdout.flush()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_metrics(raw: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Split raw metrics into finite numbers and the names that were not."""
    kept: dict[str, float] = {}
    dropped: list[str] = []
    for name, value in raw.items():
        usable = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == value  # NaN is the only value unequal to itself.
            and value not in (float("inf"), float("-inf"))
        )
        if usable:
            kept[str(name)] = value
        else:
            dropped.append(str(name))
    return kept, dropped


def _package_version() -> str | None:
    try:
        from stateset_agents import __version__
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        return None
    return str(__version__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = _timestamp()
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": None,
        "arm_id": None,
        "seed": None,
        "params": {},
        "started_at": started_at,
        "python": platform.python_version(),
        "package_version": _package_version(),
    }
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("run spec must be a JSON object")
        for name in ("run_id", "arm_id", "seed", "params"):
            result[name] = spec.get(name)
        emit({"type": "started", "run_id": spec.get("run_id")})
        # Imported here so a broken dependency becomes a reported failure.
        from stateset_agents.fanout.workloads import run_workload

        outcome = run_workload(spec, args.output_dir, args.stage_dir)
    except Exception as exc:  # noqa: BLE001 - must report every workload failure
        traceback.print_exc()
        outcome = WorkloadOutcome(exit_code=1, error=f"{type(exc).__name__}: {exc}")

    metrics, dropped = clean_metrics(outcome.metrics)
    result.update(
        status="succeeded" if outcome.exit_code == 0 else "failed",
        exit_code=outcome.exit_code,
        metrics=metrics,
        dropped_metrics=dropped,
        error=outcome.error,
        finished_at=_timestamp(),
        duration_s=round(time.monotonic() - started, 3),
    )
    atomic_json(args.output_dir / RESULT_FILE, result)
    emit({"type": "finished", "status": result["status"], "metrics": metrics})
    sys.stdout.flush()
    return outcome.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
