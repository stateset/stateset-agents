"""Checkpoint saves must leave the loop responsive and retain resource ownership."""

import asyncio
import json
import sys
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stateset_agents.training.river_refund import BENCHMARKS, campaign, prepare_run
from tests.unit.river_fakes import traced_refund_trajectory


@dataclass
class Checkpoint:
    path: str = "river://saved"
    step: int = 0
    checkpoint_type: str = "inference"


class BlockedSave:
    def __init__(self, *, fail=False):
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.closed = False
        self.calls = []
        self.thread_id = None
        self.fail = fail

    def save_weights(self, name, *, mode):
        self.thread_id = threading.get_ident()
        self.calls.append((name, mode))
        self.loop.call_soon_threadsafe(self.entered.set)
        try:
            if not self.release.wait(8):
                raise TimeoutError("Test did not release save")
            assert not self.closed, "Session was closed during checkpoint save"
            if self.fail:
                raise ConnectionError("Checkpoint outcome unavailable")
            return Checkpoint()
        finally:
            self.finished.set()


@pytest.fixture(params=["evaluate", "collect"])
def native_save(tmp_path, monkeypatch, request):
    mode = request.param
    engines = []

    class Engine:
        def __init__(self, sampler, **kwargs):
            engines.append(sampler.checkpoint)

        async def rollout(self, cases, *, group_size, **kwargs):
            for case in cases:
                trajectories = []
                for _ in range(group_size):
                    trajectory = await traced_refund_trajectory(
                        case, environment="refund-policy-v2"
                    )
                    # Explicit synthetic transport usage; the sandbox helper
                    # supplies actual action traces but does not sample tokens.
                    trajectory.generated_tokens = 10
                    trajectory.elapsed = 0.1
                    trajectories.append(trajectory)
                yield trajectories

    rl = SimpleNamespace(
        Env=object,
        Budget=lambda **kw: kw,
        Schedule=lambda **kw: kw,
        GroupCompletion=lambda **kw: kw,
        RolloutEngine=Engine,
        CheckpointSampler=lambda session, **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=rl, Checkpoint=Checkpoint)
    )
    args = SimpleNamespace(
        output=tmp_path,
        benchmark="refund-policy-v2",
        base_model="base",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=mode == "evaluate",
        collect_only=mode == "collect",
        dry_run=False,
    )
    split = "test" if mode == "evaluate" else "train"
    splits = {split: BENCHMARKS[args.benchmark][1](split, 1, 42)}
    prepare_run(args, splits)

    async def run(model):
        try:
            await campaign(
                model, object(), SimpleNamespace(tokenizer=object()), args, splits
            )
        finally:
            model.closed = True

    return mode, tmp_path, engines, run


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_rollout_consumer_failure_closes_iterator_before_provider(
    native_save, monkeypatch, cancel_cleanup
):
    import stateset_agents.training.river_refund as module

    mode, path, _, run = native_save
    model = BlockedSave()
    model.release.set()
    engine = sys.modules["river_client"].rl.RolloutEngine
    original = engine.rollout
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def tracked(self, *a, **kw):
        iterator = original(self, *a, **kw)
        try:
            async for group in iterator:
                yield group
        finally:
            entered.set()
            await release.wait()
            await iterator.aclose()
            assert not model.closed
            closed.set()

    def invalid_identity(*a, **kw):
        raise ValueError("invalid trajectory identity")

    monkeypatch.setattr(engine, "rollout", tracked)
    monkeypatch.setattr(module, "trajectory_case_identity", invalid_identity)
    task = asyncio.create_task(run(model))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not task.done() and not model.closed
        if cancel_cleanup:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and not closed.is_set() and not model.closed
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel_cleanup else ValueError):
            await asyncio.wait_for(task, 3)
    assert closed.is_set() and model.closed
    assert not (path / "test_results.json").exists()
    if mode == "collect":
        assert (
            json.loads((path / "training_candidates.json").read_text())["complete"]
            is False
        )
    else:
        assert (path / "test_attempt.json").exists()


@pytest.mark.asyncio
async def test_slow_checkpoint_save_does_not_stall_other_async_work(native_save):
    mode, path, engines, run = native_save
    model = BlockedSave()
    task = asyncio.create_task(run(model))
    try:
        await asyncio.wait_for(model.entered.wait(), 2)
        assert model.thread_id != threading.get_ident()
        assert (
            await asyncio.wait_for(asyncio.sleep(0, result="responsive"), 1)
            == "responsive"
        )
        assert not task.done() and not engines and not model.closed
    finally:
        model.release.set()
        await asyncio.wait_for(task, 5)
    assert model.finished.is_set() and model.closed
    assert model.calls == [
        (
            "refund-evaluation" if mode == "evaluate" else "refund-collection",
            "inference",
        )
    ]
    assert engines == [Checkpoint()]
    report = json.loads(
        (
            path
            / (
                "test_results.json"
                if mode == "evaluate"
                else "training_candidates.json"
            )
        ).read_text()
    )
    assert report["total"] == 1 if mode == "evaluate" else report["complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_save_drains_before_session_close_and_never_samples(
    native_save, fail
):
    _, path, engines, run = native_save
    model = BlockedSave(fail=fail)
    before = {p: p.read_bytes() for p in path.iterdir()}
    task = asyncio.create_task(run(model))
    try:
        await asyncio.wait_for(model.entered.wait(), 2)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done() and not model.finished.is_set() and not model.closed
    finally:
        model.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    assert model.finished.is_set() and model.closed
    assert len(model.calls) == 1 and not engines
    assert {p: p.read_bytes() for p in path.iterdir()} == before


@pytest.mark.asyncio
async def test_checkpoint_failure_propagates_without_sampling_or_retry(native_save):
    _, path, engines, run = native_save
    model = BlockedSave(fail=True)
    model.release.set()
    before = {p: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ConnectionError, match="Checkpoint outcome unavailable"):
        await run(model)
    assert model.finished.is_set() and model.closed
    assert len(model.calls) == 1 and not engines
    assert {p: p.read_bytes() for p in path.iterdir()} == before
