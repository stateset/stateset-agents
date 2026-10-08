"""Own asynchronous operations until resources can be released safely."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


async def drain_owned_operation(operation: Coroutine[Any, Any, _T]) -> _T:
    """Finish owned work before propagating caller cancellation.

    Repeated cancellation cannot detach the operation. Its result or exception
    is retrieved before cancellation is raised, so the caller can then release
    its resources. This does not interrupt the operation or bound its duration.
    """
    task = asyncio.create_task(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        try:
            task.result()
        except (Exception, asyncio.CancelledError):
            pass  # Retrieve the outcome; caller cancellation takes precedence.
        raise asyncio.CancelledError
    return task.result()


async def run_sync_owned(
    function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    """Run a synchronous call off-loop and drain it before returning cancellation.

    The caller must keep resources used by the function open until this returns.
    A hung call can delay cancellation indefinitely; configure timeouts in the
    underlying SDK. Unrelated event-loop work remains able to run.
    """
    return await drain_owned_operation(asyncio.to_thread(function, *args, **kwargs))


@asynccontextmanager
async def closing_async_generator(
    generator: AsyncGenerator[_T, None],
) -> AsyncIterator[AsyncIterator[_T]]:
    """Drive and close a generator in one owned task, without prefetching.

    An error or break in an async-for body does not await generator cleanup.
    This boundary keeps generator context and cleanup together, draining the
    current next-item operation before closing on caller cancellation. Repeated
    cancellation cannot detach the producer. A hung producer can delay closing
    indefinitely; configure underlying operation timeouts.
    """
    pending_requests: asyncio.Queue[asyncio.Future[_T] | None] = asyncio.Queue()

    async def drive() -> None:
        try:
            while (response := await pending_requests.get()) is not None:
                try:
                    item = await generator.__anext__()
                except StopAsyncIteration:
                    return
                response.set_result(item)
        finally:
            await generator.aclose()

    worker = asyncio.create_task(drive())

    async def iterate() -> AsyncGenerator[_T, None]:
        while not worker.done():
            response: asyncio.Future[_T] = asyncio.get_running_loop().create_future()
            pending_requests.put_nowait(response)
            pending: set[asyncio.Future[Any]] = {response, worker}
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if response not in done:
                break
            yield response.result()
        worker.result()

    stream = iterate()
    try:
        yield stream
    finally:
        pending_requests.put_nowait(None)

        async def close() -> None:
            try:
                await worker
            finally:
                await stream.aclose()

        await drain_owned_operation(close())
