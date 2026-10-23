"""Dedicated trainers push committed weights into their vLLM rollout engine.

GSPO (through its trajectory generator) and DAPO sync the engine from the
spine ``OptimizerStep``'s ``on_commit``: exactly once per committed update,
never for a skipped one. Fakes stand in for vLLM.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.unit.test_dedicated_trainer_nonfinite_grads import (  # noqa: E402
    _dapo,
    _gspo,
    _gspo_classes,
    _gspo_step,
    _poison,
)


class FakeEngine:
    def __init__(self, config: Any = None, *, fail: bool = False) -> None:
        self.config = config
        self.fail = fail
        self.synced: list[Any] = []

    async def initialize(self) -> bool:
        return True

    def sync_weights(self, model: Any) -> None:
        if self.fail:
            raise RuntimeError("engine layout unknown")
        self.synced.append(model)


def _attach(host: Any, engine: FakeEngine) -> None:
    host.vllm_generator = engine
    host._vllm_initialized = True


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [0, 1], ids=["gspo", "gspo_token"])
async def test_gspo_syncs_once_per_committed_update(index: int) -> None:
    trainer = _gspo(_gspo_classes()[index], num_iterations=3)
    engine = FakeEngine()
    _attach(trainer.generator, engine)
    await _gspo_step(trainer)
    assert engine.synced == [trainer.model] * 3
    assert trainer.last_rollout_sync is True


@pytest.mark.asyncio
async def test_gspo_does_not_sync_a_skipped_update() -> None:
    trainer = _gspo(_gspo_classes()[0], nonfinite_grad_policy="skip", num_iterations=3)
    engine = FakeEngine()
    _attach(trainer.generator, engine)
    _poison(trainer.model, after_backwards=1)  # passes 2 and 3 are non-finite
    metrics = await _gspo_step(trainer)
    assert metrics["optimizer_updates"] == 1.0
    assert len(engine.synced) == 1


@pytest.mark.asyncio
async def test_dapo_syncs_once_per_committed_update_and_never_when_skipped() -> None:
    trainer = _dapo(num_gradient_updates=3, nonfinite_grad_policy="skip")
    engine = FakeEngine()
    _attach(trainer, engine)
    _poison(trainer.model, after_backwards=2)  # the third pass is non-finite
    metrics = await trainer.train_step(["q"])
    assert metrics["optimizer_updates"] == 2.0
    assert engine.synced == [trainer.model] * 2


@pytest.mark.asyncio
async def test_rollout_sync_can_be_turned_off_and_a_failed_sync_is_survivable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = _dapo(
        num_gradient_updates=1, rollout_sync=False, old_logprobs_source="sampler"
    )
    engine = FakeEngine()
    _attach(trainer, engine)
    await trainer.train_step(["q"])
    assert engine.synced == [] and trainer.last_rollout_sync is None

    trainer = _dapo(num_gradient_updates=1)
    _attach(trainer, FakeEngine(fail=True))
    with caplog.at_level(logging.WARNING):
        metrics = await trainer.train_step(["q"])
    assert metrics["optimizer_updates"] == 1.0  # the update still committed
    assert trainer.last_rollout_sync is False and trainer.vllm_weights_stale
    assert "weight sync failed" in caplog.text


@pytest.mark.asyncio
async def test_without_an_engine_nothing_is_synced() -> None:
    trainer = _dapo(num_gradient_updates=1)
    await trainer.train_step(["q"])
    assert trainer.last_rollout_sync is None


def test_dapo_with_use_vllm_builds_the_engine(monkeypatch) -> None:
    """DAPO's gate read VLLM_BACKEND_AVAILABLE, which only _setup_vllm sets,
    so use_vllm=True never built an engine."""
    from stateset_agents.training import dapo_trainer

    monkeypatch.setattr(dapo_trainer, "_load_vllm_backend", lambda: True)
    monkeypatch.setattr(dapo_trainer, "VLLMConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(dapo_trainer, "VLLMGenerator", FakeEngine)
    trainer = _dapo(use_vllm=True)
    assert isinstance(trainer.vllm_generator, FakeEngine)
    assert trainer.vllm_generator.config["model_name"] == "gpt2"
    assert trainer.using_vllm is False  # until initialize_vllm() starts it


@pytest.mark.asyncio
async def test_dapo_falls_back_without_a_vllm_backend(monkeypatch, caplog) -> None:
    from stateset_agents.training import dapo_trainer

    monkeypatch.setattr(dapo_trainer, "_load_vllm_backend", lambda: False)
    with caplog.at_level(logging.WARNING):
        trainer = _dapo(use_vllm=True)
    assert trainer.vllm_generator is None
    assert "HuggingFace fallback" in caplog.text
    assert await trainer.initialize_vllm() is False


@pytest.mark.asyncio
async def test_an_unexpected_sync_error_is_not_swallowed() -> None:
    """Only engine failures (runtime, I/O, layout and lookup errors) leave the
    engine stale; anything else is a bug and propagates."""

    class BrokenEngine(FakeEngine):
        def sync_weights(self, model: Any) -> None:
            raise ZeroDivisionError("bug in the sync path")

    trainer = _dapo(num_gradient_updates=1)
    _attach(trainer, BrokenEngine())
    with pytest.raises(ZeroDivisionError):
        await trainer.train_step(["q"])


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, OSError, KeyError, ValueError])
async def test_engine_failures_leave_the_engine_stale(error: type) -> None:
    class FailingEngine(FakeEngine):
        def sync_weights(self, model: Any) -> None:
            raise error("engine")

    trainer = _dapo(num_gradient_updates=1)
    _attach(trainer, FailingEngine())
    metrics = await trainer.train_step(["q"])
    assert metrics["optimizer_updates"] == 1.0
    assert trainer.last_rollout_sync is False and trainer.vllm_weights_stale
