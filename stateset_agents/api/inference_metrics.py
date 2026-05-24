"""
Inference-level Prometheus metrics.

Captures model-side observability that complements the request-level HTTP
metrics in :mod:`stateset_agents.api.middleware`:

* request count + status (by model and route)
* end-to-end inference latency
* time-to-first-token for streaming responses
* per-request output throughput (tokens / second)
* prompt / completion token counters
* in-flight inference gauge

The module mirrors the HTTP middleware's pattern: ``prometheus_client`` is an
optional dependency. When absent, the helpers become no-ops so importers do
not need to guard their call sites.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any
from collections.abc import Iterator

try:  # Optional: only used when Prometheus scraping is enabled.
    from prometheus_client import Counter, Gauge, Histogram

    HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover - exercised only when extras missing
    HAS_PROMETHEUS = False


# Routes carry low cardinality (a handful per service) and are useful for
# distinguishing OpenAI vs Anthropic-shaped responses and streaming variants.
ROUTE_OPENAI_RESPONSE = "openai_response"
ROUTE_OPENAI_STREAM = "openai_stream"
ROUTE_ANTHROPIC_RESPONSE = "anthropic_response"
ROUTE_ANTHROPIC_STREAM = "anthropic_stream"

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

# Latency buckets tuned for LLM inference: from very fast cache hits up to
# generation that approaches the default 120s vLLM timeout.
_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
)

# TTFT is typically sub-second on warm vLLM, but can spike on cold loads.
_TTFT_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)

# Per-request output throughput in tokens / second.
_TPS_BUCKETS = (1, 5, 10, 25, 50, 100, 200, 500, 1000)


PROM_INFERENCE_REQUESTS_TOTAL: Any | None = None
PROM_INFERENCE_DURATION_SECONDS: Any | None = None
PROM_INFERENCE_TTFT_SECONDS: Any | None = None
PROM_INFERENCE_TOKENS_PER_SECOND: Any | None = None
PROM_INFERENCE_TOKENS_TOTAL: Any | None = None
PROM_INFERENCE_INFLIGHT: Any | None = None


if HAS_PROMETHEUS:
    PROM_INFERENCE_REQUESTS_TOTAL = Counter(
        "stateset_inference_requests_total",
        "Total inference requests dispatched to the model backend.",
        ["model", "route", "status"],
    )
    PROM_INFERENCE_DURATION_SECONDS = Histogram(
        "stateset_inference_duration_seconds",
        "End-to-end inference latency in seconds.",
        ["model", "route"],
        buckets=_LATENCY_BUCKETS,
    )
    PROM_INFERENCE_TTFT_SECONDS = Histogram(
        "stateset_inference_ttft_seconds",
        "Time from request dispatch to first streamed token in seconds.",
        ["model", "route"],
        buckets=_TTFT_BUCKETS,
    )
    PROM_INFERENCE_TOKENS_PER_SECOND = Histogram(
        "stateset_inference_tokens_per_second",
        "Per-request output throughput in tokens per second.",
        ["model", "route"],
        buckets=_TPS_BUCKETS,
    )
    PROM_INFERENCE_TOKENS_TOTAL = Counter(
        "stateset_inference_tokens_total",
        "Total tokens processed by the model backend, by direction.",
        ["model", "route", "direction"],
    )
    PROM_INFERENCE_INFLIGHT = Gauge(
        "stateset_inference_inflight",
        "In-flight inference requests being processed.",
        ["model", "route"],
    )


def record_request(*, model: str, route: str, status: str = STATUS_SUCCESS) -> None:
    """Increment the request counter. No-op when Prometheus is missing."""
    if PROM_INFERENCE_REQUESTS_TOTAL is None:
        return
    PROM_INFERENCE_REQUESTS_TOTAL.labels(model=model, route=route, status=status).inc()


def record_duration(*, model: str, route: str, seconds: float) -> None:
    """Observe end-to-end inference duration in seconds."""
    if PROM_INFERENCE_DURATION_SECONDS is None or seconds < 0:
        return
    PROM_INFERENCE_DURATION_SECONDS.labels(model=model, route=route).observe(seconds)


def record_ttft(*, model: str, route: str, seconds: float) -> None:
    """Observe time-to-first-token for a streaming response."""
    if PROM_INFERENCE_TTFT_SECONDS is None or seconds < 0:
        return
    PROM_INFERENCE_TTFT_SECONDS.labels(model=model, route=route).observe(seconds)


def record_throughput(*, model: str, route: str, tokens: int, seconds: float) -> None:
    """Observe per-request output throughput in tokens/sec."""
    if PROM_INFERENCE_TOKENS_PER_SECOND is None:
        return
    if tokens <= 0 or seconds <= 0:
        return
    PROM_INFERENCE_TOKENS_PER_SECOND.labels(model=model, route=route).observe(
        tokens / seconds
    )


def record_tokens(
    *, model: str, route: str, prompt_tokens: int = 0, completion_tokens: int = 0
) -> None:
    """Increment prompt/completion token counters."""
    if PROM_INFERENCE_TOKENS_TOTAL is None:
        return
    if prompt_tokens > 0:
        PROM_INFERENCE_TOKENS_TOTAL.labels(
            model=model, route=route, direction="prompt"
        ).inc(prompt_tokens)
    if completion_tokens > 0:
        PROM_INFERENCE_TOKENS_TOTAL.labels(
            model=model, route=route, direction="completion"
        ).inc(completion_tokens)


@contextmanager
def track_inflight(*, model: str, route: str) -> Iterator[None]:
    """Increment the in-flight gauge for the duration of a request."""
    gauge = PROM_INFERENCE_INFLIGHT
    if gauge is not None:
        gauge.labels(model=model, route=route).inc()
    try:
        yield
    finally:
        if gauge is not None:
            gauge.labels(model=model, route=route).dec()


@contextmanager
def track_request(*, model: str, route: str) -> Iterator[_InferenceCall]:
    """Bundle the common request-side instrumentation.

    Usage::

        with track_request(model=model, route=ROUTE_OPENAI_RESPONSE) as call:
            response = await backend(...)
            call.tokens(prompt=..., completion=...)
    """
    call = _InferenceCall(model=model, route=route)
    started = time.monotonic()
    inflight = PROM_INFERENCE_INFLIGHT
    if inflight is not None:
        inflight.labels(model=model, route=route).inc()
    try:
        yield call
    except BaseException:
        call._status = STATUS_ERROR
        raise
    finally:
        if inflight is not None:
            inflight.labels(model=model, route=route).dec()
        duration = time.monotonic() - started
        record_duration(model=model, route=route, seconds=duration)
        record_request(model=model, route=route, status=call._status)
        if call._completion_tokens > 0:
            record_throughput(
                model=model,
                route=route,
                tokens=call._completion_tokens,
                seconds=duration,
            )


class _InferenceCall:
    """Per-request helper handed to ``track_request`` callers."""

    __slots__ = (
        "model",
        "route",
        "_status",
        "_prompt_tokens",
        "_completion_tokens",
    )

    def __init__(self, *, model: str, route: str) -> None:
        self.model = model
        self.route = route
        self._status = STATUS_SUCCESS
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def tokens(self, *, prompt: int = 0, completion: int = 0) -> None:
        """Record prompt/completion token counts (cumulative within the call)."""
        if prompt > 0:
            self._prompt_tokens += prompt
        if completion > 0:
            self._completion_tokens += completion
        record_tokens(
            model=self.model,
            route=self.route,
            prompt_tokens=max(0, prompt),
            completion_tokens=max(0, completion),
        )

    def mark_error(self) -> None:
        """Mark this call as failed for the success-rate counter."""
        self._status = STATUS_ERROR


__all__ = [
    "HAS_PROMETHEUS",
    "ROUTE_OPENAI_RESPONSE",
    "ROUTE_OPENAI_STREAM",
    "ROUTE_ANTHROPIC_RESPONSE",
    "ROUTE_ANTHROPIC_STREAM",
    "STATUS_SUCCESS",
    "STATUS_ERROR",
    "record_request",
    "record_duration",
    "record_ttft",
    "record_throughput",
    "record_tokens",
    "track_inflight",
    "track_request",
]
