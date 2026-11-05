"""A failed rollout-engine weight sync is visible and, by default, bounded.

The spine absorbed every vLLM sync failure -- TypeError, KeyError and
AttributeError included, which are bugs in the sync path -- and training
went on sampling a stale policy: ``vllm_weights_stale`` was written but
never read, and ``last_rollout_sync`` never reached the step metrics.

* only engine runtime/I-O failures (``RuntimeError``, ``OSError``) are
  survivable; anything else propagates;
* the step metrics report ``rollout_weights_stale`` and
  ``rollout_sync_failures`` once a sync was attempted;
* ``rollout_sync_failure="raise"`` (default) stops training after
  ``max_consecutive_rollout_sync_failures`` (3) consecutive failures with a
  ``RolloutSyncError``; ``"warn"`` keeps training on stale rollouts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.errors import TrainingError  # noqa: E402
from stateset_agents.training.spine.rollouts import (  # noqa: E402
    RolloutSyncError,
    record_rollout_sync,
    rollout_sync_metrics,
)
from tests.unit.test_dedicated_trainer_nonfinite_grads import _dapo  # noqa: E402
from tests.unit.test_vllm_weight_sync import FakeEngine, _attach  # noqa: E402


class FailingEngine(FakeEngine):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def sync_weights(self, model: Any) -> None:
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [TypeError("bad arg"), KeyError("w"), AttributeError("x"), ValueError("v")]
)
async def test_bugs_in_the_sync_path_propagate(error: BaseException) -> None:
    trainer = _dapo(num_gradient_updates=1)
    _attach(trainer, FailingEngine(error))
    with pytest.raises(type(error)):
        await trainer.train_step(["q"])


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("cuda"), OSError("rpc down")])
async def test_engine_failures_are_reported_in_the_step_metrics(
    error: BaseException,
) -> None:
    trainer = _dapo(num_gradient_updates=1)
    _attach(trainer, FailingEngine(error))
    metrics = await trainer.train_step(["q"])
    assert metrics["optimizer_updates"] == 1.0  # the update still committed
    assert metrics["rollout_weights_stale"] is True
    assert metrics["rollout_sync_failures"] == 1
    assert trainer.vllm_weights_stale


@pytest.mark.asyncio
async def test_a_healthy_engine_reports_fresh_weights() -> None:
    trainer = _dapo(num_gradient_updates=2)
    _attach(trainer, FakeEngine())
    metrics = await trainer.train_step(["q"])
    assert metrics["rollout_weights_stale"] is False
    assert metrics["rollout_sync_failures"] == 0


@pytest.mark.asyncio
async def test_without_an_engine_no_sync_metrics_are_added() -> None:
    metrics = await _dapo(num_gradient_updates=1).train_step(["q"])
    assert "rollout_weights_stale" not in metrics


@pytest.mark.asyncio
async def test_default_policy_raises_after_three_consecutive_failures() -> None:
    trainer = _dapo(num_gradient_updates=1)
    _attach(trainer, FakeEngine(fail=True))
    for _ in range(2):
        await trainer.train_step(["q"])
    with pytest.raises(RolloutSyncError, match="3 consecutive"):
        await trainer.train_step(["q"])


@pytest.mark.asyncio
async def test_warn_policy_keeps_training_on_stale_rollouts() -> None:
    trainer = _dapo(num_gradient_updates=1, rollout_sync_failure="warn")
    _attach(trainer, FakeEngine(fail=True))
    for _ in range(5):
        metrics = await trainer.train_step(["q"])
    assert metrics["rollout_sync_failures"] == 5
    assert metrics["rollout_weights_stale"] is True


@pytest.mark.asyncio
async def test_an_engine_without_sync_weights_counts_as_a_failure() -> None:
    class NoSync:
        async def initialize(self) -> bool:
            return True

    trainer = _dapo(num_gradient_updates=1, max_consecutive_rollout_sync_failures=1)
    _attach(trainer, NoSync())
    with pytest.raises(RolloutSyncError):
        await trainer.train_step(["q"])
    assert trainer.vllm_weights_stale


def _host(**config: Any) -> Any:
    return SimpleNamespace(config=SimpleNamespace(**config), last_rollout_sync=None)


def test_a_success_resets_the_failure_streak() -> None:
    host = _host()
    record_rollout_sync(host, False, label="T")
    record_rollout_sync(host, False, label="T")
    record_rollout_sync(host, True, label="T")
    assert rollout_sync_metrics(host) == {
        "rollout_weights_stale": False,
        "rollout_sync_failures": 0,
    }
    record_rollout_sync(host, False, label="T")
    record_rollout_sync(host, False, label="T")
    with pytest.raises(RolloutSyncError) as caught:
        record_rollout_sync(host, False, label="T")
    assert isinstance(caught.value, TrainingError)
    assert isinstance(caught.value, RuntimeError)


def test_nothing_attempted_reports_nothing() -> None:
    host = _host()
    record_rollout_sync(host, None, label="T")
    assert rollout_sync_metrics(host) == {}


@pytest.mark.asyncio
async def test_grpo_trainer_applies_the_same_policy() -> None:
    from tests.unit.test_rollout_sync import _fake_grad, _model, _multi_turn_trainer

    agent = SimpleNamespace(
        model=_model(), tokenizer=None, sync_rollout_backend=lambda: False
    )
    trainer = _multi_turn_trainer(agent, max_consecutive_rollout_sync_failures=2)
    _fake_grad(agent.model)
    trainer._apply_optimizer_step(torch)
    assert rollout_sync_metrics(trainer)["rollout_weights_stale"] is True
    _fake_grad(agent.model)
    with pytest.raises(RolloutSyncError):
        trainer._apply_optimizer_step(torch)
