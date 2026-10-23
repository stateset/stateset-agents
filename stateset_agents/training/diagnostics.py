"""Bounded training diagnostics with explicit learning-signal failure semantics."""

from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from numbers import Integral, Real
from typing import Any

from ..core.errors import ErrorCode, TrainingError
from .callbacks import has_no_policy_signal

logger = logging.getLogger(__name__)


class TrainingSignalError(TrainingError, RuntimeError):
    """Training produced invalid metrics or repeatedly measured no policy signal.

    A :class:`~stateset_agents.core.errors.TrainingError` that is still a
    ``RuntimeError`` and prints its bare message.
    """

    message_only_str = True
    default_code = ErrorCode.TRN_NO_SIGNAL


class DiagnosticsMonitor:
    """Observe trainer events without inferring missing measurements as successes.

    Native callback dispatch propagates this monitor's errors. Consecutive zero
    advantages stop training after ``zero_signal_patience`` observations; zero
    disables that guard. Missing signal evidence resets the streak.
    """

    fail_on_error = True

    def __init__(self, config: Any) -> None:
        self.config = config
        self.start_time = time.monotonic()
        self.episode_count = 0
        self.total_rewards: list[float] = []
        self.zero_steps = 0
        self.failure: str | None = None
        self.latest: dict[str, Any] = {}
        self._last_step: Any = None
        patience = getattr(config, "zero_signal_patience", 5)
        self.patience = patience if type(patience) is int and patience >= 0 else 5

    def __call__(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Validate measured scalars and update a bounded diagnostic snapshot."""
        if self.failure is not None:
            raise TrainingSignalError(self.failure)
        data = data or {}
        if event == "training_start":
            self.start_time = time.monotonic()
            return
        if event in {"step_end", "episode_end"}:
            for name, value in data.items():
                if not isinstance(value, bool) and isinstance(value, Real):
                    try:
                        finite = math.isfinite(value)
                    except (OverflowError, ValueError):
                        finite = False
                    if not finite:
                        self._fail(
                            f"Nonfinite training metric {name!r}; inspect rewards, gradients, and token scoring"
                        )
            self.latest = {
                name: _plain_scalar(value)
                for name, value in data.items()
                if isinstance(value, (str, Real))
            }
            identity = data.get(
                "training_step", data.get("global_step", data.get("step"))
            )
            if identity is None or identity != self._last_step:
                self.zero_steps = (
                    self.zero_steps + 1 if has_no_policy_signal(data) else 0
                )
                self._last_step = _plain_scalar(identity)
                if self.patience and self.zero_steps >= self.patience:
                    self._fail(
                        f"No policy advantage signal for {self.zero_steps} consecutive observations; inspect reward variance, evaluator context, rollout diversity, and task difficulty"
                    )
            if event == "episode_end":
                self.episode_count += 1
                reward = data.get("total_reward")
                if not isinstance(reward, bool) and isinstance(reward, Real):
                    self.total_rewards.append(float(reward))
                    del self.total_rewards[:-100]
                if self.episode_count % 10 == 0:
                    average = (
                        sum(self.total_rewards) / len(self.total_rewards)
                        if self.total_rewards
                        else None
                    )
                    logger.info(
                        "Episode %s: recent reward=%s elapsed=%.1fs",
                        self.episode_count,
                        average,
                        time.monotonic() - self.start_time,
                    )
        if event == "training_end":
            logger.info(
                "Training finished after %s episodes in %.1fs",
                self.episode_count,
                time.monotonic() - self.start_time,
            )

    def _fail(self, reason: str) -> None:
        self.failure = reason
        raise TrainingSignalError(reason)

    def snapshot(self) -> dict[str, Any]:
        """Return measured health and the retained failure without claiming quality."""
        return {
            "episodes": self.episode_count,
            "zero_signal_streak": self.zero_steps,
            "elapsed_seconds": time.monotonic() - self.start_time,
            "failure": self.failure,
            "latest": dict(self.latest),
        }

    def state_dict(self) -> dict[str, Any]:
        """Retain logical history; process-local clocks restart on reload."""
        return {
            "schema_version": 1,
            "patience": self.patience,
            "episode_count": self.episode_count,
            "total_rewards": list(self.total_rewards),
            "zero_steps": self.zero_steps,
            "failure": self.failure,
            "latest": dict(self.latest),
            "last_step": self._last_step,
        }


def _plain_scalar(value: Any) -> Any:
    """Normalize measured numeric scalars for weights-only checkpoint loading."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return str(value)
    return value


def _counter(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("Invalid checkpoint diagnostic counter")
    return value


def _scalar(value: Any) -> bool:
    if type(value) in (str, bool):
        return True
    if type(value) in (int, float):
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return False


def prepare_diagnostics(trainer: Any, saved: Any) -> dict[str, Any]:
    """Stage bounded, finite diagnostic history without mutating the learner."""
    if saved is None:
        return {"observations": 0, "monitor": None}
    if not isinstance(saved, dict) or set(saved) != {"observations", "monitor"}:
        raise ValueError("Invalid checkpoint diagnostics schema")
    observations = _counter(saved["observations"])
    state = saved["monitor"]
    if state is None:
        if observations:
            raise ValueError("Checkpoint diagnostic observations require a monitor")
        return {"observations": 0, "monitor": None}
    if (
        not isinstance(state, dict)
        or set(state)
        != {
            "schema_version",
            "patience",
            "episode_count",
            "total_rewards",
            "zero_steps",
            "failure",
            "latest",
            "last_step",
        }
        or type(state["schema_version"]) is not int
        or state["schema_version"] != 1
    ):
        raise ValueError("Invalid checkpoint diagnostic monitor schema")
    monitor = DiagnosticsMonitor(trainer.config)
    if _counter(state["patience"]) != monitor.patience:
        raise ValueError("Checkpoint diagnostic patience differs from trainer")
    monitor.episode_count = _counter(state["episode_count"])
    monitor.zero_steps = _counter(state["zero_steps"])
    failure = state["failure"]
    if failure is not None and (type(failure) is not str or not failure):
        raise ValueError("Invalid checkpoint diagnostic failure")
    if monitor.patience and monitor.zero_steps >= monitor.patience and failure is None:
        raise ValueError("Checkpoint diagnostic streak requires a retained failure")
    rewards = state["total_rewards"]
    if (
        type(rewards) is not list
        or len(rewards) > min(100, monitor.episode_count)
        or any(type(v) not in (int, float) or not _scalar(v) for v in rewards)
    ):
        raise ValueError("Invalid checkpoint diagnostic reward history")
    latest = state["latest"]
    if not isinstance(latest, dict) or any(
        type(k) is not str or not _scalar(v) for k, v in latest.items()
    ):
        raise ValueError("Invalid checkpoint diagnostic metric history")
    last = state["last_step"]
    if last is not None and not _scalar(last):
        raise ValueError("Invalid checkpoint diagnostic observation identity")
    monitor.failure = failure
    monitor.total_rewards = list(rewards)
    monitor.latest = deepcopy(latest)
    monitor._last_step = last
    return {"observations": observations, "monitor": monitor}


def capture_diagnostics(trainer: Any) -> dict[str, Any]:
    """Capture validated history using weights-only compatible plain values."""
    monitor = getattr(trainer, "diagnostics", None)
    if monitor is not None and type(monitor) is not DiagnosticsMonitor:
        raise ValueError("Checkpoint diagnostics require the native DiagnosticsMonitor")
    saved = {
        "observations": getattr(trainer, "_diagnostic_observations", 0),
        "monitor": None if monitor is None else monitor.state_dict(),
    }
    if (
        not saved["observations"]
        and saved["monitor"] == DiagnosticsMonitor(trainer.config).state_dict()
    ):
        # Reload can preserve a callback's monitor object even when the saved
        # learner had not created one. Both represent the same empty history.
        saved["monitor"] = None
    prepare_diagnostics(trainer, saved)
    return saved


def diagnostic_control(trainer: Any) -> dict[str, Any]:
    """Identify guard decisions while leaving timing and reward logs local."""
    saved = capture_diagnostics(trainer)
    state = saved["monitor"]
    return {
        "protocol": "stateset-diagnostics-v1",
        "observations": saved["observations"],
        "patience": DiagnosticsMonitor(trainer.config).patience,
        "zero_steps": 0 if state is None else state["zero_steps"],
        "failure": None if state is None else state["failure"],
        "last_step": None if state is None else state["last_step"],
    }


def apply_diagnostics(trainer: Any, prepared: dict[str, Any]) -> None:
    """Restore staged history and repair existing callback monitor references."""
    current = getattr(trainer, "diagnostics", None)
    restored = prepared["monitor"]
    if type(current) is DiagnosticsMonitor:
        # A legacy checkpoint starts fresh rather than borrowing target history.
        restored = restored or DiagnosticsMonitor(trainer.config)
        current.__dict__.update(restored.__dict__)
    else:
        trainer.diagnostics = restored
    trainer._diagnostic_observations = prepared["observations"]
    trainer._runtime_rollout_metrics = {}
