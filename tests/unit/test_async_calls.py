"""Owned synchronous calls preserve context and finish before cancellation escapes."""

import asyncio
import contextvars
import subprocess
import sys
import threading

import pytest

from stateset_agents.utils.async_calls import closing_async_generator, run_sync_owned


@pytest.mark.asyncio
async def test_owned_generator_delivers_thread_backed_items_before_producer_finishes():
    produced, closed = [], []

    async def source():
        try:
            for value in (None, 0, False, "reply"):
                item = await asyncio.to_thread(lambda value=value: value)
                produced.append(item)
                yield item
        finally:
            closed.append(True)

    async with closing_async_generator(source()) as values:
        received = []
        async for item in values:
            received.append(item)
            assert produced == received  # No prefetch or end-of-stream wait.
            assert not closed
    assert received == [None, 0, False, "reply"]
    assert closed == [True]


@pytest.mark.asyncio
async def test_empty_owned_generator_finishes_and_closes():
    closed = []

    async def source():
        try:
            return
            yield  # pragma: no cover
        finally:
            closed.append(True)

    async with closing_async_generator(source()) as values:
        assert [value async for value in values] == []
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_early", [False, True])
async def test_owned_generator_preserves_producer_context_without_prefetch(stop_early):
    context = contextvars.ContextVar("generator_context", default="caller")
    generated, finalized = [], []

    async def source():
        token = context.set("producer")
        try:
            for value in range(3):
                generated.append(value)
                yield value
        finally:
            assert context.get() == "producer"
            context.reset(token)
            finalized.append(True)

    async with closing_async_generator(source()) as values:
        assert generated == []
        async for value in values:
            assert context.get() == "caller"
            await asyncio.sleep(0)
            assert generated == list(range(value + 1))
            if stop_early:
                break
    assert finalized == [True]
    assert generated == ([0] if stop_early else [0, 1, 2])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["producer", "consumer", "cleanup"])
async def test_owned_generator_propagates_errors_and_finishes_cleanup(failure):
    error = RuntimeError(failure)
    closed = []

    async def source():
        try:
            if failure == "producer":
                raise error
            yield 1
        finally:
            closed.append(True)
            if failure == "cleanup":
                raise error

    with pytest.raises(RuntimeError) as caught:
        async with closing_async_generator(source()) as values:
            async for _ in values:
                if failure == "consumer":
                    raise error
    assert caught.value is error
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during", ["next", "close"])
async def test_owned_generator_drains_through_repeated_cancellation(cancel_during):
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    generated, consumed = [], []

    async def source():
        try:
            if cancel_during == "next":
                entered.set()
                await release.wait()
            generated.append(1)
            yield 1
            generated.append(2)
            yield 2
        finally:
            if cancel_during == "close":
                entered.set()
                await release.wait()
            closed.set()

    async def consume():
        async with closing_async_generator(source()) as values:
            async for value in values:
                consumed.append(value)
                break

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not closed.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    assert closed.is_set()
    assert generated == [1]
    assert consumed == ([] if cancel_during == "next" else [1])


@pytest.mark.asyncio
async def test_worker_call_preserves_context_arguments_and_result():
    owner = contextvars.ContextVar("operation_owner", default="unset")
    owner.set("request-42")
    caller_thread = threading.get_ident()

    def operation(value, *, suffix):
        assert threading.get_ident() != caller_thread
        assert owner.get() == "request-42"
        owner.set("worker-only")
        return value + suffix

    assert (
        await run_sync_owned(operation, "saved-", suffix="checkpoint")
        == "saved-checkpoint"
    )
    assert owner.get() == "request-42"


@pytest.mark.asyncio
async def test_worker_error_is_not_swallowed_or_retried():
    calls = []
    error = ConnectionError("unavailable")

    def operation():
        calls.append(1)
        raise error

    with pytest.raises(ConnectionError) as caught:
        await run_sync_owned(operation)
    assert caught.value is error
    assert calls == [1]


def test_async_call_utility_does_not_import_optional_sdks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from stateset_agents.utils.async_calls import run_sync_owned; assert not {'torch', 'river_client', 'wandb', 'mlflow'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
