"""
Request batching for inference service.

Accumulates concurrent inference requests over a short time window and
dispatches them as a single batch to the backend, improving throughput
when multiple requests arrive simultaneously.

Usage with InferenceService::

    batcher = RequestBatcher(max_batch_size=8, max_wait_ms=50)

    # Each request calls submit() — the batcher groups them into batches
    result = await batcher.submit(payload)

The batcher is transparent: if only one request arrives within the window,
it's sent immediately with no added latency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BatcherConfig:
    """Configuration for request batching."""

    max_batch_size: int = 8  # Max requests per batch
    max_wait_ms: float = 50.0  # Max time to wait for batch to fill
    enabled: bool = True


@dataclass
class _PendingRequest:
    """A request waiting to be batched."""

    payload: dict[str, Any]
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    submitted_at: float = field(default_factory=time.monotonic)


class RequestBatcher:
    """Accumulates inference requests into batches for improved throughput.

    When multiple requests arrive within ``max_wait_ms``, they are grouped
    into a single batch and dispatched together.  This reduces per-request
    overhead (connection setup, kernel launch, etc.) and enables the backend
    to use continuous batching.

    The batcher is safe for concurrent use from multiple asyncio tasks.
    """

    def __init__(
        self,
        dispatch_fn: Any = None,
        config: BatcherConfig | None = None,
    ):
        """
        Args:
            dispatch_fn: Async callable that processes a list of payloads
                and returns a list of results (one per payload).
            config: Batching configuration.
        """
        self.dispatch_fn = dispatch_fn
        self.config = config or BatcherConfig()
        self._queue: list[_PendingRequest] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._stats = {"batches_sent": 0, "requests_batched": 0, "avg_batch_size": 0.0}

    async def submit(self, payload: dict[str, Any]) -> Any:
        """Submit a request for batched dispatch.

        Returns the result for this individual request.
        """
        if not self.config.enabled or self.dispatch_fn is None:
            # Bypass batching — send immediately
            if self.dispatch_fn is not None:
                results = await self.dispatch_fn([payload])
                return results[0] if results else None
            return None

        request = _PendingRequest(payload=payload)

        async with self._lock:
            self._queue.append(request)

            # If batch is full, flush immediately
            if len(self._queue) >= self.config.max_batch_size:
                await self._flush()
            elif self._flush_task is None or self._flush_task.done():
                # Schedule a delayed flush
                self._flush_task = asyncio.ensure_future(self._delayed_flush())

        return await request.future

    async def _delayed_flush(self) -> None:
        """Wait for the batch window, then flush."""
        await asyncio.sleep(self.config.max_wait_ms / 1000.0)
        async with self._lock:
            if self._queue:
                await self._flush()

    async def _flush(self) -> None:
        """Send the current batch to the backend."""
        if not self._queue:
            return

        batch = list(self._queue)
        self._queue.clear()

        # Update stats
        self._stats["batches_sent"] += 1
        self._stats["requests_batched"] += len(batch)
        self._stats["avg_batch_size"] = (
            self._stats["requests_batched"] / self._stats["batches_sent"]
        )

        payloads = [r.payload for r in batch]

        try:
            results = await self.dispatch_fn(payloads)

            # Distribute results to waiting futures
            for i, request in enumerate(batch):
                if i < len(results):
                    request.future.set_result(results[i])
                else:
                    request.future.set_result(None)

        except Exception as exc:
            # Propagate error to all waiting requests
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)

        logger.debug(
            "Dispatched batch of %d requests (avg batch size: %.1f)",
            len(batch),
            self._stats["avg_batch_size"],
        )

    def get_stats(self) -> dict[str, Any]:
        """Get batching statistics."""
        return dict(self._stats)

    async def drain(self) -> None:
        """Flush any remaining requests (call before shutdown)."""
        async with self._lock:
            if self._queue:
                await self._flush()


__all__ = ["BatcherConfig", "RequestBatcher"]
