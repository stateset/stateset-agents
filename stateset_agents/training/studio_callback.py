"""Live RL Training -> 3D Weight Studio & SQLite Telemetry Bridge Callback.

Streams real-time RL training metrics (step, loss, reward, KL, entropy,
tokens/sec, and ``OnlineRewardHackingGuard`` alerts) into:

1. Durable local ``outputs/weight_studio_telemetry/<run_name>_metrics.jsonl``
   (automatically discoverable by 3D Weight Studio's External Telemetry Sync).
2. Optional live HTTP push to a running 3D Weight Studio server
   (``POST /api/weight-studio/traces/ingest``) for continuous OTLP/SQLite
   trace accumulation and real-time browser SSE pulses.

Uses only stdlib JSON + ``live_serving._default_http_post_json`` (zero eager
``torch`` imports, injectable ``http_caller`` for clean offline testing).
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stateset_agents.weight_studio._stats import slugify_export_name
from stateset_agents.weight_studio.live_serving import (
    HttpCaller,
    _default_http_post_json,
)

HttpGetCaller = Callable[[str, dict[str, str], float], tuple[int, dict[str, Any]]]


def _default_http_get_json(
    url: str, headers: dict[str, str], timeout: float
) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw) if raw.strip() else {}
        return int(resp.status), (
            parsed if isinstance(parsed, dict) else {"data": parsed}
        )


class StudioTelemetryCallback:
    """Non-blocking RL training callback streaming metrics & traces into 3D Weight Studio."""

    def __init__(
        self,
        *,
        run_name: str = "stateset-rl-run",
        workspace_root: Path | None = None,
        studio_url: str = "",
        studio_token: str = "",
        reward_fn: str = "customer_support",
        timeout: float = 5.0,
        http_caller: HttpCaller | None = None,
        http_get_caller: HttpGetCaller | None = None,
        on_control_signal: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.run_name = slugify_export_name(run_name, default="stateset-rl-run")
        self.workspace_root = (workspace_root or Path(".")).expanduser().resolve()
        self.studio_url = studio_url.strip().rstrip("/")
        self.studio_token = studio_token.strip()
        self.reward_fn = reward_fn
        self.timeout = float(timeout)
        self._http_post: HttpCaller = http_caller or _default_http_post_json
        self._http_get: HttpGetCaller = http_get_caller or _default_http_get_json
        self._on_control_signal = on_control_signal
        self._started_at = time.perf_counter()
        self.steps_logged = 0
        self.traces_pushed = 0
        self.push_errors = 0
        self.last_control_state: dict[str, Any] = {
            "status": "running",
            "last_action": "resume",
        }

    def _auth_headers(self) -> dict[str, str]:
        return (
            {"Authorization": f"Bearer {self.studio_token}"}
            if self.studio_token
            else {}
        )

    def poll_control(self) -> dict[str, Any]:
        """Poll ``GET /api/weight-studio/rl/control`` for Pause/Resume/Rollback directives."""
        if not self.studio_url:
            return dict(self.last_control_state)
        try:
            status, body = self._http_get(
                f"{self.studio_url}/api/weight-studio/rl/control",
                self._auth_headers(),
                self.timeout,
            )
            if 200 <= status < 300 and isinstance(body.get("control"), dict):
                self.last_control_state = dict(body["control"])
        except Exception:
            self.push_errors += 1
        return dict(self.last_control_state)

    @property
    def metrics_path(self) -> Path:
        return (
            self.workspace_root
            / "outputs"
            / "weight_studio_telemetry"
            / f"{self.run_name}_metrics.jsonl"
        )

    def on_step(
        self,
        step: int,
        metrics: dict[str, Any],
        *,
        guard_report: dict[str, Any] | None = None,
        sample_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Persist step telemetry to JSONL and optionally forward traces to Weight Studio."""
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_name": self.run_name,
            "step": int(step),
            "elapsed_sec": round(time.perf_counter() - self._started_at, 3),
            **metrics,
        }
        if guard_report is not None:
            record["guard_action"] = guard_report.get("action")
            record["guard_alerts"] = list(guard_report.get("alerts") or [])

        path = self.metrics_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        self.steps_logged += 1

        pushed = 0
        telemetry_pushed = False
        if self.studio_url:
            headers = self._auth_headers()
            # 1. Push live RL step & OnlineRewardHackingGuard telemetry to 3D RL Cockpit
            #    and capture any operator Pause/Rollback directive returned in the response.
            try:
                t_status, t_body = self._http_post(
                    f"{self.studio_url}/api/weight-studio/rl/telemetry",
                    record,
                    headers,
                    self.timeout,
                )
                telemetry_pushed = 200 <= t_status < 300
                if (
                    telemetry_pushed
                    and isinstance(t_body, dict)
                    and isinstance(t_body.get("control"), dict)
                ):
                    self.last_control_state = dict(t_body["control"])
                elif not telemetry_pushed:
                    self.push_errors += 1
            except Exception:
                self.push_errors += 1

            # 2. Optionally push sampled rollout traces to durable OTLP/SQLite store
            if sample_traces:
                url = f"{self.studio_url}/api/weight-studio/traces/ingest"
                try:
                    status, _body = self._http_post(
                        url,
                        {
                            "payload": {"traces": sample_traces},
                            "reward_fn": self.reward_fn,
                        },
                        headers,
                        self.timeout,
                    )
                    if 200 <= status < 300:
                        pushed = len(sample_traces)
                        self.traces_pushed += pushed
                    else:
                        self.push_errors += 1
                except Exception:
                    self.push_errors += 1

        ctrl_status = str(self.last_control_state.get("status") or "running")
        should_pause = ctrl_status == "paused"
        should_rollback = ctrl_status == "rollback_requested"
        if self._on_control_signal is not None and (should_pause or should_rollback):
            self._on_control_signal(
                str(self.last_control_state.get("last_action") or ctrl_status),
                dict(self.last_control_state),
            )

        return {
            "logged": True,
            "metrics_path": str(path),
            "telemetry_pushed": telemetry_pushed,
            "traces_pushed": pushed,
            "control": dict(self.last_control_state),
            "should_pause": should_pause,
            "should_rollback": should_rollback,
            "record": record,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "metrics_path": str(self.metrics_path),
            "steps_logged": self.steps_logged,
            "traces_pushed": self.traces_pushed,
            "push_errors": self.push_errors,
        }


__all__ = ["StudioTelemetryCallback"]
