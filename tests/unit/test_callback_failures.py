"""A failing callback is visible, and ``strict_callbacks`` makes it fatal.

Callback errors used to be swallowed at ``logger.debug`` (under ``pragma: no
cover``), so a broken early-stopping or checkpoint-publishing callback failed
silently for a whole run. They are now logged at WARNING with the traceback;
``TrainingConfig.strict_callbacks`` (or ``strict=True`` on the dispatchers,
or a callback's own ``fail_on_error``) raises instead.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from stateset_agents.training import callbacks as module
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.spine.loop import run_dedicated_loop


class Broken:
    """Fails in every hook and as a callable."""

    def __init__(self) -> None:
        self.calls = 0

    def _fail(self, *args: Any) -> None:
        self.calls += 1
        raise ValueError("callback bug")

    on_train_start = on_episode_end = on_step_end = on_optimizer_step = _fail
    on_evaluation_end = on_train_end = on_checkpoint_saved = _fail


EVENTS = [
    (module.notify_training_start, {"trainer": "t", "config": {}}),
    (module.notify_episode_end, {"episode": 0, "metrics": {}}),
    (module.notify_step_end, {"step": 0, "metrics": {}}),
    (module.notify_optimizer_step, {"step": 0, "metrics": {}}),
    (module.notify_evaluation_end, {"metrics": {}}),
    (module.notify_training_end, {"metrics": {}}),
    (module.notify_checkpoint_saved, {"path": "p", "step": 1, "is_best": False}),
]
IDS = [notify.__name__ for notify, _ in EVENTS]


@pytest.mark.asyncio
@pytest.mark.parametrize("notify,kwargs", EVENTS, ids=IDS)
async def test_failures_are_logged_at_warning_with_the_traceback(
    notify: Any, kwargs: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    broken, after = Broken(), Broken()
    after.on_step_end = after.on_episode_end = lambda *a: None  # type: ignore
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        await notify([broken], **kwargs)
    failures = [r for r in caplog.records if "callback bug" in r.getMessage()]
    failures += [
        r for r in caplog.records if r.exc_info and "callback bug" in str(r.exc_info[1])
    ]
    assert failures, caplog.text
    assert all(r.levelno == logging.WARNING for r in failures)
    assert any(r.exc_info for r in failures)  # the traceback is kept
    assert "strict_callbacks" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("notify,kwargs", EVENTS, ids=IDS)
async def test_strict_dispatch_raises(notify: Any, kwargs: dict[str, Any]) -> None:
    broken = Broken()
    with pytest.raises(ValueError, match="callback bug"):
        await notify([broken], strict=True, **kwargs)
    assert broken.calls == 1


@pytest.mark.asyncio
async def test_a_failing_plain_callable_is_logged_or_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def callback(event: str, data: dict[str, Any]) -> None:
        raise KeyError("missing metric")

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        await module.notify_step_end([callback], step=0, metrics={})
    assert any(r.levelno == logging.WARNING and r.exc_info for r in caplog.records)
    with pytest.raises(KeyError):
        await module.notify_step_end([callback], step=0, metrics={}, strict=True)


@pytest.mark.asyncio
async def test_dedicated_loop_honours_strict_callbacks(tmp_path: Any) -> None:
    async def step(prompts: Any) -> dict[str, Any]:
        return {"loss": 1.0}

    common: dict[str, Any] = {
        "iterations": 2,
        "batch": lambda i, rng: ["p"],
        "step": step,
        "save": lambda path: None,
        "output_dir": str(tmp_path),
        "save_steps": 100,
        "final_name": "final",
    }
    lenient = await run_dedicated_loop(**common, callbacks=[Broken()])
    assert lenient.iterations == 2
    with pytest.raises(ValueError, match="callback bug"):
        await run_dedicated_loop(**common, callbacks=[Broken()], strict_callbacks=True)


def test_strict_callbacks_is_an_opt_in_boolean() -> None:
    from stateset_agents.core.errors import ConfigurationError

    assert TrainingConfig().strict_callbacks is False
    with pytest.raises(ConfigurationError, match="strict_callbacks"):
        TrainingConfig(strict_callbacks="yes").validate_for_training()  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_grpo_trainer_reads_strict_callbacks_from_its_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, strict: bool
) -> None:
    pytest.importorskip("torch")
    from tests.unit.test_grpo_scheduler_budget import make_trainer

    trainer, _rates = make_trainer(monkeypatch, tmp_path, "multi", 1, "cosine", 0.0)
    trainer.config.strict_callbacks = strict
    broken = Broken()
    trainer.callbacks = [broken]
    if strict:
        with pytest.raises(ValueError, match="callback bug"):
            await trainer.train()
        assert broken.calls == 1
    else:
        await trainer.train()
        assert broken.calls > 1  # every event reached it; training finished
