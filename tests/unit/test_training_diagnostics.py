"""Measured signal failures must survive the standard callback dispatcher."""

from types import SimpleNamespace

import numpy as np
import pytest

from stateset_agents.training.callbacks import notify_episode_end, notify_step_end
from stateset_agents.training.diagnostics import DiagnosticsMonitor, TrainingSignalError


async def test_zero_advantage_streak_stops_through_real_callback_dispatch():
    monitor = DiagnosticsMonitor(SimpleNamespace(zero_signal_patience=2))
    await notify_step_end(
        [monitor], step=1, metrics={"nonzero_advantage_fraction": 0.0}
    )
    with pytest.raises(TrainingSignalError, match="reward variance"):
        await notify_step_end(
            [monitor], step=2, metrics={"nonzero_advantage_fraction": 0.0}
        )
    assert monitor.snapshot()["failure"]


async def test_missing_signal_does_not_mean_zero_and_duplicate_events_do_not_double_count():
    monitor = DiagnosticsMonitor(SimpleNamespace(zero_signal_patience=2))
    for step in range(12):
        await notify_episode_end([monitor], episode=step, metrics={})
    await notify_step_end(
        [monitor],
        step=1,
        metrics={"training_step": 1, "nonzero_advantage_fraction": 0.0},
    )
    await notify_episode_end(
        [monitor],
        episode=12,
        metrics={"training_step": 1, "nonzero_advantage_fraction": 0.0},
    )
    assert monitor.zero_steps == 1 and monitor.episode_count == 13


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10**400])
async def test_invalid_metrics_are_fatal(value):
    monitor = DiagnosticsMonitor(SimpleNamespace())
    with pytest.raises(TrainingSignalError, match="Nonfinite"):
        await notify_step_end([monitor], step=0, metrics={"grad_norm": value})


def test_numpy_metrics_roundtrip_through_weights_only_checkpoint(tmp_path):
    import torch

    from stateset_agents.training.diagnostics import (
        apply_diagnostics,
        capture_diagnostics,
        prepare_diagnostics,
    )

    trainer = SimpleNamespace(config=SimpleNamespace(), _diagnostic_observations=1)
    trainer.diagnostics = DiagnosticsMonitor(trainer.config)
    trainer.diagnostics(
        "step_end",
        {"step": np.int64(3), "reward": np.float64(0.5), "loss": np.float32(0.25)},
    )
    saved = capture_diagnostics(trainer)
    path = tmp_path / "diagnostics.pt"
    torch.save(saved, path)
    restored = torch.load(path, weights_only=True)
    apply_diagnostics(trainer, prepare_diagnostics(trainer, restored))
    assert trainer.diagnostics.latest == {"step": 3, "reward": 0.5, "loss": 0.25}
    assert type(trainer.diagnostics._last_step) is int
    assert type(trainer.diagnostics.latest["reward"]) is float


@pytest.mark.parametrize("value", [np.float32("nan"), np.float64("inf")])
def test_nonfinite_numpy_metrics_remain_fatal(value):
    monitor = DiagnosticsMonitor(SimpleNamespace())
    with pytest.raises(TrainingSignalError, match="Nonfinite"):
        monitor("step_end", {"loss": value})


async def test_runtime_guard_runs_without_a_caller_installing_callbacks():
    from stateset_agents.training.runtime_state import TrainingRuntime

    class Trainer(TrainingRuntime):
        config = SimpleNamespace(zero_signal_patience=2)

        async def train_step(self):
            return {"nonzero_advantage_fraction": 0.0}

    trainer = Trainer()
    await trainer.train_step()
    with pytest.raises(TrainingSignalError, match="No policy advantage signal"):
        await trainer.train_step()


@pytest.mark.parametrize(
    "field,value",
    [
        ("patience", True),
        ("patience", 99),
        ("zero_steps", -1),
        ("zero_steps", 2),
        ("episode_count", False),
        ("total_rewards", [float("nan")]),
        ("total_rewards", [1.0] * 101),
        ("latest", {"loss": float("inf")}),
        ("latest", {"loss": []}),
        ("last_step", []),
        ("failure", ""),
        ("schema_version", True),
    ],
)
def test_diagnostic_history_validation_is_transactional(field, value):
    from stateset_agents.training.diagnostics import (
        capture_diagnostics,
        prepare_diagnostics,
    )

    trainer = SimpleNamespace(config=SimpleNamespace(zero_signal_patience=2))
    trainer.diagnostics = DiagnosticsMonitor(trainer.config)
    trainer.diagnostics("step_end", {"step": 1, "nonzero_advantage_fraction": 0.0})
    trainer._diagnostic_observations = 1
    saved = capture_diagnostics(trainer)
    saved["monitor"][field] = value
    with pytest.raises(ValueError):
        prepare_diagnostics(trainer, saved)
    assert trainer.diagnostics.zero_steps == 1
    assert trainer.diagnostics.failure is None


def test_failed_history_stays_failed_and_callback_reference_is_repaired():
    from stateset_agents.training.diagnostics import (
        apply_diagnostics,
        capture_diagnostics,
        prepare_diagnostics,
    )

    trainer = SimpleNamespace(config=SimpleNamespace(zero_signal_patience=2))
    monitor = trainer.diagnostics = DiagnosticsMonitor(trainer.config)
    for step in (1, 2):
        try:
            monitor("step_end", {"step": step, "nonzero_advantage_fraction": 0.0})
        except TrainingSignalError:
            pass
    saved = capture_diagnostics(trainer)
    apply_diagnostics(trainer, prepare_diagnostics(trainer, saved))
    assert trainer.diagnostics is monitor
    with pytest.raises(TrainingSignalError):
        monitor("step_end", {"step": 3, "nonzero_advantage_fraction": 1.0})
    apply_diagnostics(trainer, prepare_diagnostics(trainer, None))
    assert trainer.diagnostics is monitor
    monitor("step_end", {"step": 3, "nonzero_advantage_fraction": 1.0})
    assert monitor.failure is None and monitor.zero_steps == 0
