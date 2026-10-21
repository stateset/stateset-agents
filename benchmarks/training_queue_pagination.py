"""Measure Python memory for one page versus loading an entire job history.

Run from the checkout with ``python -m benchmarks.training_queue_pagination``.
This uses synthetic completed SQLite jobs, not model-training quality evidence.
Timings include tracemalloc instrumentation; database caches are warmed first.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from stateset_agents.api.training_queue import TrainingQueue


def _measure(operation: Callable[[], Any], repeats: int) -> dict[str, Any]:
    samples = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        try:
            started = time.perf_counter()
            operation()
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        samples.append({"seconds": elapsed, "peak_python_bytes": peak})
    return {
        "samples": samples,
        "median_seconds": statistics.median(sample["seconds"] for sample in samples),
        "median_peak_python_bytes": statistics.median(
            sample["peak_python_bytes"] for sample in samples
        ),
    }


def main() -> None:
    """Populate disposable completed history, check identical pages, and report JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=10000)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--payload-chars", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.jobs < 1 or args.repeats < 1 or args.payload_chars < 0:
        parser.error("jobs/repeats must be positive; payload-chars must be nonnegative")
    if not 1 <= args.page_size <= 100:
        parser.error("page-size must be between 1 and 100")
    with tempfile.TemporaryDirectory(prefix="stateset-pagination-") as directory:
        queue = TrainingQueue(Path(directory) / "jobs.sqlite3")

        def rows() -> Any:
            for index in range(args.jobs):
                created = datetime(2026, 1, 1) + timedelta(seconds=index)
                job_id = f"history-{index:012d}"
                snapshot = {
                    "user_id": "owner",
                    "status": "completed",
                    "created_at": created,
                    "metrics": {"optimizer_updates": 100},
                    "config": {"context": "x" * args.payload_chars},
                }
                yield (
                    job_id,
                    "owner",
                    job_id,
                    "completed",
                    queue._encode(snapshot),
                    index,
                )

        # Seed archived history in one transaction: these are completed jobs,
        # so the pending-job admission limit does not apply.
        with queue._transaction() as connection:
            connection.executemany(
                "INSERT INTO stateset_training_jobs(id,owner,fingerprint,status,snapshot,created) VALUES(?,?,?,?,?,?)",
                rows(),
            )

        def legacy() -> tuple[list[tuple[str, dict[str, Any]]], int]:
            jobs = [
                (job_id, job)
                for job_id, job in queue.jobs().items()
                if job["user_id"] in (None, "owner") and job["status"] == "completed"
            ]
            jobs.sort(key=lambda item: item[1]["created_at"], reverse=True)
            return jobs[: args.page_size], len(jobs)

        def paginated() -> tuple[list[tuple[str, dict[str, Any]]], int]:
            return queue.page(owner="owner", status="completed", limit=args.page_size)

        assert legacy() == paginated(), "Compared paths must return identical results"
        baseline = _measure(legacy, args.repeats)
        candidate = _measure(paginated, args.repeats)
    queue_source = Path(sys.modules[TrainingQueue.__module__].__file__)
    report = {
        "schema_version": 1,
        "scope": "Synthetic SQLite job-history listing; instrumented timing and Python heap only",
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "jobs": args.jobs,
        "page_size": args.page_size,
        "payload_chars": args.payload_chars,
        "pages_equal": True,
        "queue_source_sha256": hashlib.sha256(queue_source.read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "legacy": baseline,
        "paginated": candidate,
    }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
