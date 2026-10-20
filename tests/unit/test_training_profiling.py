"""Efficiency metrics distinguish recorded outcomes from positive reward guesses."""

import pytest

from stateset_agents.training.profiling import TrainingProfiler


def test_costs_stay_unknown_until_supplied_and_success_is_explicit():
    profiler = TrainingProfiler()
    profiler.record_rollout(accepted=True, success=None, tokens=4)
    profiler.record_rollout(accepted=True, success=True, tokens=2)
    profiler.record_rollout(accepted=False, success=None)
    assert profiler.snapshot()["cost_per_accepted_rollout"] is None
    result = profiler.snapshot(cost_usd=1.0)
    assert result["cost_per_accepted_rollout"] == 0.5
    assert result["cost_per_verified_success"] == 1.0
    assert result["unknown_successes"] == 1 and result["attempted_rollouts"] == 3


def test_failed_stage_time_is_retained_and_samples_are_bounded():
    profiler = TrainingProfiler()
    with pytest.raises(ValueError):
        with profiler.measure("reward"):
            raise ValueError("missing evidence")
    for _ in range(1030):
        with profiler.measure("rollout"):
            pass
    result = profiler.snapshot()
    assert result["stages"]["reward"]["failures"] == 1
    assert result["stages"]["rollout"]["recent_samples"] == 1024


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_cost_validation(value):
    with pytest.raises(ValueError):
        TrainingProfiler().snapshot(cost_usd=value)


@pytest.mark.asyncio
async def test_global_empty_batches_do_not_consume_or_reset_signal_patience():
    from types import SimpleNamespace

    from stateset_agents.training.diagnostics import TrainingSignalError
    from stateset_agents.training.runtime_state import TrainingRuntime

    class Learner(TrainingRuntime):
        def __init__(self):
            self.config = SimpleNamespace(zero_signal_patience=2)
            self.empty = True

        async def training_step(self):
            return {"global_empty_batch": self.empty, "nonzero_advantage_fraction": 0.0}

    trainer = Learner()
    metrics = await trainer.training_step()
    assert metrics["step_seconds"] >= 0
    assert not hasattr(trainer, "diagnostics")
    assert not hasattr(trainer, "_diagnostic_observations")
    trainer.empty = False
    await trainer.training_step()
    monitor = trainer.diagnostics
    assert monitor.zero_steps == trainer._diagnostic_observations == 1
    last = dict(monitor.latest)
    trainer.empty = True
    for _ in range(3):
        await trainer.training_step()
    assert monitor.latest == last
    assert monitor.zero_steps == trainer._diagnostic_observations == 1
    trainer.empty = False
    with pytest.raises(TrainingSignalError):
        await trainer.training_step()
