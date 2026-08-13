"""Append-only record of what remote runs actually cost.

Rented GPUs bill by the second, and the framework provisions them on the
user's behalf — so the money spent is the framework's responsibility to
account for, not the user's to reconstruct from a provider dashboard.

Every remote run appends one line here: what it ran, on what hardware, for
how long, and the resulting dollar amount. ``stateset-agents costs`` reads
it back. The ledger is advisory bookkeeping derived from the provider's
own ``costPerHr`` and the measured pod lifetime; it is not a bill, and it
deliberately over-reports rather than under-reports by counting the pod
from creation (when billing starts) rather than from job start.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Where the ledger lives unless overridden. Kept outside the repo: it is
#: per-user accounting, not project state.
DEFAULT_LEDGER_PATH = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "stateset-agents"
    / "cost_ledger.jsonl"
)


@dataclass
class CostEntry:
    """One remote run's cost record."""

    provider: str
    job_id: str
    base_model: str
    gpu: str
    gpu_count: int = 1
    cost_per_hr: float | None = None
    duration_s: float | None = None
    cost_usd: float | None = None
    status: str = "unknown"
    #: ISO-8601 UTC, recorded when the entry is written.
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_cost_usd(
    cost_per_hr: float | None, duration_s: float | None
) -> float | None:
    """Dollars for ``duration_s`` seconds at ``cost_per_hr``.

    Returns None when either input is missing — an unknown cost must read as
    unknown, never as zero, or a budget check would silently pass.
    """
    if cost_per_hr is None or duration_s is None:
        return None
    return round(float(cost_per_hr) * (float(duration_s) / 3600.0), 4)


def record_entry(entry: CostEntry, path: Path | None = None) -> Path:
    """Append ``entry`` to the ledger, creating it if needed.

    Never raises on IO problems: a bookkeeping failure must not turn a
    successful (already paid for) training run into a failed one.
    """
    target = Path(path) if path is not None else DEFAULT_LEDGER_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    except OSError:  # pragma: no cover - defensive
        pass
    return target


def read_entries(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every ledger entry, skipping any corrupt line."""
    target = Path(path) if path is not None else DEFAULT_LEDGER_PATH
    if not target.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            entries.append(row)
    return entries


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Total spend, run count, and per-model/per-GPU breakdowns."""
    known = [e for e in entries if isinstance(e.get("cost_usd"), (int, float))]
    by_model: dict[str, float] = {}
    by_gpu: dict[str, float] = {}
    for entry in known:
        cost = float(entry["cost_usd"])
        by_model[str(entry.get("base_model", "?"))] = round(
            by_model.get(str(entry.get("base_model", "?")), 0.0) + cost, 4
        )
        by_gpu[str(entry.get("gpu", "?"))] = round(
            by_gpu.get(str(entry.get("gpu", "?")), 0.0) + cost, 4
        )
    return {
        "runs": len(entries),
        "runs_with_known_cost": len(known),
        "total_usd": round(sum(float(e["cost_usd"]) for e in known), 4),
        "by_model": by_model,
        "by_gpu": by_gpu,
    }


class BudgetExceeded(Exception):
    """A run would cost more than the caller's ``max_cost_usd`` ceiling."""


def check_budget(
    cost_per_hr: float | None,
    timeout_s: int,
    max_cost_usd: float | None,
    *,
    gpu_count: int = 1,
) -> float | None:
    """Raise ``BudgetExceeded`` if the worst case blows the ceiling.

    The worst case is the pod running to its full timeout. An unknown
    ``cost_per_hr`` with a ceiling set is treated as a refusal-worthy
    unknown only when the provider reported nothing at all; callers that
    cannot price a pod should not silently rent it against a budget.
    """
    if max_cost_usd is None:
        return None
    if cost_per_hr is None:
        raise BudgetExceeded(
            "a --max-cost ceiling was set but the provider did not report a "
            "price for this pod, so the run cannot be checked against it"
        )
    worst_case = estimate_cost_usd(float(cost_per_hr) * max(1, gpu_count), timeout_s)
    assert worst_case is not None
    if worst_case > max_cost_usd:
        raise BudgetExceeded(
            f"this run could cost ${worst_case:.2f} "
            f"({cost_per_hr}/hr x {gpu_count} GPU(s) for up to {timeout_s}s), "
            f"above the --max-cost ceiling of ${max_cost_usd:.2f}"
        )
    return worst_case


__all__ = [
    "BudgetExceeded",
    "CostEntry",
    "DEFAULT_LEDGER_PATH",
    "check_budget",
    "estimate_cost_usd",
    "read_entries",
    "record_entry",
    "summarize",
]
