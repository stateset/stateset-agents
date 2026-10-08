from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.training import callbacks as module
from stateset_agents.training.callbacks import (
    notify_episode_end,
    notify_training_end,
    notify_training_start,
)
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.diagnostics import DiagnosticsMonitor


@dataclass
class EpisodeRecorder:
    calls: list[tuple[int, dict[str, Any]]]

    def on_episode_end(self, episode: int, metrics: dict[str, Any]) -> None:
        self.calls.append((episode, dict(metrics)))


async def test_notify_episode_end_dispatches_callable_and_method_callbacks() -> None:
    cfg = TrainingConfig(num_episodes=1)
    diagnostics = DiagnosticsMonitor(cfg)
    recorder = EpisodeRecorder(calls=[])
    callbacks: list[Any] = [diagnostics, recorder]

    await notify_training_start(callbacks, trainer="trainer", config=cfg)
    await notify_episode_end(callbacks, episode=0, metrics={"total_reward": 1.23})
    await notify_training_end(callbacks, metrics={"final_step": 1})

    assert diagnostics.episode_count == 1
    assert diagnostics.total_rewards == [1.23]
    assert recorder.calls == [(0, {"total_reward": 1.23})]


EVENTS = [
    (
        module.notify_training_start,
        {"trainer": "trainer", "config": {}},
        "on_train_start",
    ),
    (
        module.notify_training_start,
        {"trainer": "trainer", "config": {}},
        "on_training_start",
    ),
    (module.notify_episode_end, {"episode": 0, "metrics": {}}, "on_episode_end"),
    (module.notify_step_end, {"step": 0, "metrics": {}}, "on_step_end"),
    (module.notify_evaluation_end, {"metrics": {}}, "on_evaluation_end"),
    (module.notify_evaluation_end, {"metrics": {}}, "on_eval_end"),
    (module.notify_training_end, {"metrics": {}}, "on_train_end"),
    (module.notify_training_end, {"metrics": {}}, "on_training_end"),
    (
        module.notify_checkpoint_saved,
        {"path": "checkpoint", "step": 1, "is_best": True},
        "on_checkpoint_saved",
    ),
]


@pytest.mark.parametrize("notify,kwargs,method", EVENTS)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_type_error_inside_method_never_repeats_side_effects(
    notify, kwargs, method, asynchronous
):
    calls = []

    def failed(*args):
        calls.append("side effect")
        raise TypeError("failure after side effect")

    async def failed_async(*args):
        failed(*args)

    broken = SimpleNamespace(**{method: failed_async if asynchronous else failed})
    following = SimpleNamespace(**{method: lambda *args: calls.append("following")})
    await notify([broken, following], **kwargs)
    assert calls == ["side effect", "following"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_type_error_inside_callable_never_repeats_side_effects(asynchronous):
    calls = []

    def failed(*args):
        calls.append(args)
        raise TypeError("callback body failed")

    async def failed_async(*args):
        failed(*args)

    await notify_episode_end(
        [failed_async if asynchronous else failed], episode=2, metrics={"loss": 0.5}
    )
    assert calls == [("episode_end", {"episode": 2, "loss": 0.5})]


@pytest.mark.parametrize("style", ["full", "metrics", "empty", "partial", "builtin"])
async def test_signature_variants_preserve_supported_callbacks(style):
    calls = []

    def full(episode, metrics):
        calls.append((episode, metrics))

    def metrics_only(metrics):
        calls.append(metrics)

    def empty():
        calls.append("empty")

    def tagged(tag, episode, metrics):
        calls.append((tag, episode, metrics))

    callbacks = {
        "full": full,
        "metrics": metrics_only,
        "empty": empty,
        "partial": partial(tagged, "tag"),
        "builtin": calls.append,
    }
    metrics = {"loss": 0.5}
    await notify_episode_end(
        [SimpleNamespace(on_episode_end=callbacks[style])], episode=2, metrics=metrics
    )
    assert calls == [
        {
            "full": (2, metrics),
            "metrics": metrics,
            "empty": "empty",
            "partial": ("tag", 2, metrics),
            "builtin": metrics,
        }[style]
    ]


async def test_incompatible_signature_never_enters_callback():
    def incompatible(*, required):
        pytest.fail("Incompatible callback was executed")

    with pytest.raises(TypeError):
        await module._try_call_variants(incompatible, [(1, {}), ({},), ()])


async def test_opaque_callable_gets_one_canonical_attempt():
    class Opaque:
        calls = []

        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, *args):
            self.calls.append(args)
            raise TypeError("body failed")

    callback = Opaque()
    await notify_episode_end([callback], episode=2, metrics={"loss": 0.5})
    assert callback.calls == [("episode_end", {"episode": 2, "loss": 0.5})]


@pytest.mark.parametrize("custom", [False, True])
async def test_returned_awaitable_finishes_before_next_callback(custom):
    future = asyncio.get_running_loop().create_future()
    entered = asyncio.Event()
    calls = []

    class Awaitable:
        def __await__(self):
            return future.__await__()

    def first(episode, metrics):
        calls.append("first")
        entered.set()
        return Awaitable() if custom else future

    following = SimpleNamespace(on_episode_end=lambda *args: calls.append("next"))
    task = asyncio.create_task(
        notify_episode_end(
            [SimpleNamespace(on_episode_end=first), following], episode=0, metrics={}
        )
    )
    try:
        await entered.wait()
        assert not task.done() and calls == ["first"]
        future.set_result(None)
        await task
    finally:
        if not future.done():
            future.set_result(None)
        await task
    assert calls == ["first", "next"]


async def test_future_failure_is_retrieved_without_callback_retry(caplog):
    caplog.set_level("DEBUG", logger=module.__name__)
    calls = []
    future = asyncio.get_running_loop().create_future()
    future.set_exception(TypeError("future failed"))

    def callback(*args):
        calls.append(args)
        return future

    await notify_episode_end([callback], episode=0, metrics={})
    assert len(calls) == 1
    assert "future failed" in caplog.text


async def test_future_cancellation_stops_dispatch_without_retry():
    future = asyncio.get_running_loop().create_future()
    future.cancel()
    calls = []

    def first(*args):
        calls.append("first")
        return future

    with pytest.raises(asyncio.CancelledError):
        await notify_episode_end(
            [first, lambda *args: calls.append("next")], episode=0, metrics={}
        )
    assert calls == ["first"]


@pytest.mark.parametrize("notify,kwargs,method", EVENTS)
async def test_mandatory_method_failure_stops_dispatch(notify, kwargs, method):
    calls = []

    def fail(*args):
        calls.append("failure")
        raise ValueError("mandatory callback failed")

    callback = SimpleNamespace(**{method: fail}, fail_on_error=True)
    with pytest.raises(ValueError, match="mandatory callback failed"):
        await notify([callback, lambda *args: calls.append("following")], **kwargs)
    assert calls == ["failure"]


async def test_mandatory_callable_failure_stops_dispatch():
    def fail(event, data):
        raise ValueError("mandatory callable failed")

    fail.fail_on_error = True
    with pytest.raises(ValueError, match="mandatory callable failed"):
        await notify_episode_end([fail], episode=0, metrics={})
