"""Shared statistical, path-confinement, and MCP authorization helpers for 3D Weight Studio.

Centralizes:
- ``bootstrap_ci_95``: Deterministic 95% confidence interval calculation.
- ``slugify_export_name``: Safe filesystem export directory slugification.
- ``resolve_confined_path``: Strict workspace path confinement preventing path
  traversal (``../../etc/passwd``) when AI agents or HTTP clients pass file
  paths to Weight Studio or MCP tools.
- ``authorize_mcp_request``: Optional bearer-token authentication and read-only
  vs. write scope enforcement for remote/team Streamable HTTP MCP deployments.
"""

from __future__ import annotations

import hmac
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


def bootstrap_ci_95(
    values: list[float],
    *,
    clamp_unit: bool = False,
) -> tuple[float, float]:
    """Compute a deterministic 95% normal-approximation confidence interval."""
    if not values:
        return (0.0, 0.0)
    mean = sum(values) / len(values)
    if len(values) == 1:
        val = max(0.0, min(1.0, mean)) if clamp_unit else mean
        return (round(val, 4), round(val, 4))
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    se = math.sqrt(var / len(values))
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    if clamp_unit:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
    return (round(lo, 4), round(hi, 4))


def slugify_export_name(name: str, default: str = "studio-export") -> str:
    """Sanitize a user/agent-provided export name into a safe single path segment."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in (name or "").strip()
    ).strip("-")
    return cleaned or default


def resolve_confined_path(
    raw_path: str | Path,
    workspace_root: Path,
    *,
    allow_external: bool | None = None,
) -> Path:
    """Resolve ``raw_path`` and verify it stays inside ``workspace_root`` unless explicitly allowed.

    Relative paths are resolved against ``workspace_root``. Absolute paths
    pointing outside ``workspace_root`` raise ``PermissionError`` unless
    ``allow_external=True`` or ``STATESET_MCP_ALLOW_EXTERNAL_PATHS=1``.
    """
    if allow_external is None:
        allow_external = os.environ.get(
            "STATESET_MCP_ALLOW_EXTERNAL_PATHS", ""
        ).strip().lower() in ("1", "true", "yes")

    ws = workspace_root.expanduser().resolve()
    p = Path(raw_path).expanduser()
    resolved = p.resolve() if p.is_absolute() else (ws / p).resolve()

    if not allow_external:
        try:
            resolved.relative_to(ws)
        except ValueError as exc:
            raise PermissionError(
                f"Path {raw_path!r} escapes workspace root {ws}. "
                "Place files inside the workspace or set STATESET_MCP_ALLOW_EXTERNAL_PATHS=1."
            ) from exc
    return resolved


def authorize_mcp_request(
    auth_header: str | None,
    *,
    required_scope: str = "read",
) -> tuple[bool, int, str, str]:
    """Validate optional bearer-token auth for Streamable HTTP MCP & SSE endpoints.

    Environment variables:
    - ``STATESET_MCP_TOKEN``: Full read+write bearer token.
    - ``STATESET_MCP_READONLY_TOKEN``: Read-only bearer token (may call read-only
      tools, read resources, and fetch prompts, but cannot invoke mutating tools).

    When neither env var is configured, local loopback mode is open (returns
    ``(True, 200, "write", "")``). Whenever either token is set, auth is
    fail-closed (missing/invalid bearer token returns HTTP 401; read-only token
    calling a write-scoped tool returns HTTP 403).
    """
    rw_token = os.environ.get("STATESET_MCP_TOKEN", "").strip()
    ro_token = os.environ.get("STATESET_MCP_READONLY_TOKEN", "").strip()
    if not rw_token and not ro_token:
        return (True, 200, "write", "")

    header = (auth_header or "").strip()
    if not header.lower().startswith("bearer "):
        return (
            False,
            401,
            "none",
            "Missing Bearer token (STATESET_MCP_TOKEN / STATESET_MCP_READONLY_TOKEN is enabled).",
        )
    supplied = header[7:].strip()

    granted_scope: str | None = None
    if rw_token and hmac.compare_digest(supplied, rw_token):
        granted_scope = "write"
    elif ro_token and hmac.compare_digest(supplied, ro_token):
        granted_scope = "read"

    if granted_scope is None:
        return (False, 401, "none", "Invalid Bearer token for Weight Studio MCP.")

    if required_scope == "write" and granted_scope != "write":
        return (
            False,
            403,
            granted_scope,
            "Read-only MCP token cannot invoke mutating surgical/export tools.",
        )

    return (True, 200, granted_scope, "")


def percentile(values: list[float], pct: float) -> float:
    """Compute a linear-interpolated percentile over a list of floats."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    idx = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = min(len(ordered) - 1, lo + 1)
    frac = idx - lo
    return round(ordered[lo] + frac * (ordered[hi] - ordered[lo]), 2)


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    """Pure-Python Pearson correlation coefficient in [-1, 1]."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 1.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den_x = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if den_x == 0.0 or den_y == 0.0:
        return 1.0 if den_x == den_y else 0.0
    return round(max(-1.0, min(1.0, num / (den_x * den_y))), 4)


def spearman_corr(xs: list[float], ys: list[float]) -> float:
    """Pure-Python Spearman rank correlation in [-1, 1] with average tie ranks."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 1.0

    def _ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda kv: kv[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    return pearson_corr(_ranks(xs[:n]), _ranks(ys[:n]))


def get_studio_ingress_config() -> dict[str, Any]:
    """Read optional HTTP ingress security limits for Weight Studio endpoints.

    Environment variables:
    - ``STATESET_STUDIO_MAX_BODY_BYTES``: Maximum request payload size (default 10 MiB).
    - ``STATESET_STUDIO_RATE_LIMIT_PER_MIN``: Per-client-IP requests/min cap (0 = disabled on localhost).
    - ``STATESET_STUDIO_CORS_ORIGINS``: Comma-separated CORS allowlist (empty = same-origin only).
    """
    try:
        max_bytes = max(
            1024,
            int(
                os.environ.get("STATESET_STUDIO_MAX_BODY_BYTES", str(10 * 1024 * 1024))
            ),
        )
    except ValueError:
        max_bytes = 10 * 1024 * 1024
    try:
        rpm = max(0, int(os.environ.get("STATESET_STUDIO_RATE_LIMIT_PER_MIN", "0")))
    except ValueError:
        rpm = 0
    raw_origins = os.environ.get("STATESET_STUDIO_CORS_ORIGINS", "").strip()
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    return {
        "max_body_bytes": max_bytes,
        "rate_limit_rpm": rpm,
        "cors_origins": origins,
    }


class SlidingWindowRateLimiter:
    """Thread-safe per-key 60-second sliding-window rate limiter."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = float(window_seconds)
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit_per_window: int) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        if limit_per_window <= 0:
            return (True, 0)
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            q = self._buckets.setdefault(key, deque())
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit_per_window:
                retry_after = max(1, int(math.ceil(q[0] + self.window_seconds - now)))
                return (False, retry_after)
            q.append(now)
            return (True, 0)


__all__ = [
    "SlidingWindowRateLimiter",
    "authorize_mcp_request",
    "bootstrap_ci_95",
    "get_studio_ingress_config",
    "pearson_corr",
    "percentile",
    "resolve_confined_path",
    "slugify_export_name",
    "spearman_corr",
]
