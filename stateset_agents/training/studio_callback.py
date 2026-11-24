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
import urllib.parse
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
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"Studio URL must use http or https scheme, got: {url!r}")
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
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
        self.rollbacks_executed = 0
        self.pause_events_handled = 0
        self._best_reward: float | None = None
        self._best_snapshot: dict[str, Any] | None = None
        self._best_snapshot_step: int | None = None
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

    def snapshot_model(
        self,
        model: Any,
        *,
        step: int = 0,
        reward: float | None = None,
        force: bool = False,
    ) -> bool:
        """Capture a detached CPU state_dict snapshot when reward improves (or when forced)."""
        if model is None or not hasattr(model, "state_dict"):
            return False
        if (
            not force
            and reward is not None
            and self._best_reward is not None
            and float(reward) < self._best_reward
        ):
            return False
        state = model.state_dict()
        self._best_snapshot = {
            k: (v.detach().cpu().clone() if hasattr(v, "detach") else v)
            for k, v in state.items()
        }
        self._best_snapshot_step = int(step)
        if reward is not None:
            self._best_reward = float(reward)
        return True

    def restore_snapshot(self, model: Any) -> bool:
        """Restore the most recent best-reward model weight snapshot in place."""
        if (
            model is None
            or self._best_snapshot is None
            or not hasattr(model, "load_state_dict")
        ):
            return False
        # Preserve parameter devices where tensors live on non-CPU devices
        target_device = None
        if hasattr(model, "parameters"):
            try:
                first_param = next(iter(model.parameters()), None)
                if first_param is not None and hasattr(first_param, "device"):
                    target_device = first_param.device
            except Exception:
                target_device = None
        restored_state = {
            k: (
                v.to(target_device)
                if target_device is not None and hasattr(v, "to")
                else v
            )
            for k, v in self._best_snapshot.items()
        }
        model.load_state_dict(restored_state, strict=False)
        self.rollbacks_executed += 1
        return True

    def wait_while_paused(
        self,
        *,
        poll_interval: float = 0.25,
        max_wait_sec: float = 30.0,
    ) -> float:
        """Cooperatively poll Studio control until status leaves ``paused`` or timeout expires."""
        if str(self.last_control_state.get("status") or "") != "paused":
            return 0.0
        self.pause_events_handled += 1
        t0 = time.perf_counter()
        deadline = t0 + max(0.0, float(max_wait_sec))
        sleep_s = max(0.01, float(poll_interval))
        while time.perf_counter() < deadline:
            ctrl = self.poll_control()
            if str(ctrl.get("status") or "") != "paused":
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(sleep_s, remaining))
        return round(time.perf_counter() - t0, 4)

    def enforce_step(
        self,
        model: Any,
        step: int,
        metrics: dict[str, Any],
        *,
        guard_report: dict[str, Any] | None = None,
        sample_traces: list[dict[str, Any]] | None = None,
        auto_snapshot: bool = True,
        pause_poll_interval: float = 0.25,
        max_pause_wait_sec: float = 10.0,
    ) -> dict[str, Any]:
        """Log/stream RL step and automatically enforce Studio Pause/Rollback directives on ``model``."""
        reward_val = metrics.get("reward", metrics.get("average_reward"))
        reward_f = float(reward_val) if isinstance(reward_val, (int, float)) else None
        out = self.on_step(
            step,
            metrics,
            guard_report=guard_report,
            sample_traces=sample_traces,
        )
        rolled_back = False
        waited_sec = 0.0
        if out.get("should_rollback"):
            rolled_back = self.restore_snapshot(model)
        else:
            if auto_snapshot:
                self.snapshot_model(model, step=step, reward=reward_f)
            if out.get("should_pause"):
                waited_sec = self.wait_while_paused(
                    poll_interval=pause_poll_interval,
                    max_wait_sec=max_pause_wait_sec,
                )
        out["rolled_back"] = rolled_back
        out["rollback_to_step"] = self._best_snapshot_step if rolled_back else None
        out["pause_waited_sec"] = waited_sec
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "metrics_path": str(self.metrics_path),
            "steps_logged": self.steps_logged,
            "traces_pushed": self.traces_pushed,
            "push_errors": self.push_errors,
            "rollbacks_executed": self.rollbacks_executed,
            "pause_events_handled": self.pause_events_handled,
            "best_snapshot_step": self._best_snapshot_step,
        }


class StudioControlMixin:
    """Zero-eager-torch mixin giving RL trainers 1-line Studio Telemetry + Pause/Rollback enforcement."""

    def attach_studio_callback(
        self,
        callback: Any,
        *,
        auto_snapshot: bool = True,
        pause_poll_interval: float = 0.25,
        max_pause_wait_sec: float = 10.0,
    ) -> None:
        """Attach a ``StudioTelemetryCallback`` and capture an initial model weight snapshot."""
        self.studio_callback = callback
        self._studio_auto_snapshot = bool(auto_snapshot)
        self._studio_pause_poll_interval = float(pause_poll_interval)
        self._studio_max_pause_wait_sec = float(max_pause_wait_sec)
        model = getattr(self, "model", None)
        if auto_snapshot and model is not None and hasattr(callback, "snapshot_model"):
            callback.snapshot_model(model, step=0, force=True)

    def enforce_studio_step(
        self, step: int, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Stream step telemetry to 3D Weight Studio and enforce Pause/Rollback directives on ``self.model``."""
        cb = getattr(self, "studio_callback", None)
        if cb is None:
            return None
        payload: dict[str, Any] = {
            "reward": result.get("reward", result.get("average_reward")),
            "loss": result.get("loss", result.get("policy_loss")),
            "entropy": result.get("entropy"),
            "learning_rate": result.get("learning_rate"),
        }
        clean_payload = {k: v for k, v in payload.items() if v is not None}
        model = getattr(self, "model", None)
        if hasattr(cb, "enforce_step"):
            enforcement = cb.enforce_step(
                model,
                step,
                clean_payload,
                auto_snapshot=bool(getattr(self, "_studio_auto_snapshot", True)),
                pause_poll_interval=float(
                    getattr(self, "_studio_pause_poll_interval", 0.25)
                ),
                max_pause_wait_sec=float(
                    getattr(self, "_studio_max_pause_wait_sec", 10.0)
                ),
            )
            result["studio_rolled_back"] = bool(enforcement.get("rolled_back"))
            result["studio_pause_waited_sec"] = float(
                enforcement.get("pause_waited_sec", 0.0)
            )
            return dict(enforcement) if isinstance(enforcement, dict) else None
        out = cb.on_step(step, clean_payload)
        return dict(out) if isinstance(out, dict) else None


__all__ = ["StudioControlMixin", "StudioTelemetryCallback"]
