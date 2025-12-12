"""
GRPO Metrics Module

Unified metrics collection for the GRPO service.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..constants import (
    PERCENTILE_P50,
    PERCENTILE_P95,
    PERCENTILE_P99,
    RATE_LIMIT_DEQUE_MAXLEN,
)


@dataclass
class GRPOMetrics:
    """
    Comprehensive metrics collector for the GRPO service.

    Consolidates previously duplicate metrics implementations.
    """

    # Request tracking
    request_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    status_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    latencies: deque = field(
        default_factory=lambda: deque(maxlen=RATE_LIMIT_DEQUE_MAXLEN)
    )
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Rate limiting
    rate_limit_hits: int = 0

    # Training metrics
    training_jobs_started: int = 0
    training_jobs_completed: int = 0
    training_jobs_failed: int = 0
    total_trajectories: int = 0
    total_computation: float = 0.0

    # Conversation metrics
    conversations_started: int = 0
    conversations_ended: int = 0
    messages_processed: int = 0

    # WebSocket metrics
    websocket_connections: int = 0
    websocket_messages: int = 0

    def record_request(
        self,
        path: str,
        method: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record a completed request."""
        key = f"{method}:{path}"
        self.request_counts[key] += 1
        self.status_counts[status_code] += 1
        self.latencies.append(latency_ms)

        if status_code >= 400:
            self.error_counts[key] += 1

    def record_rate_limit_hit(self) -> None:
        """Record a rate limit hit."""
        self.rate_limit_hits += 1

    def record_training_started(self) -> None:
        """Record training job started."""
        self.training_jobs_started += 1

    def record_training_completed(
        self,
        trajectories: int = 0,
        computation: float = 0.0,
    ) -> None:
        """Record training job completed."""
        self.training_jobs_completed += 1
        self.total_trajectories += trajectories
        self.total_computation += computation

    def record_training_failed(self) -> None:
        """Record training job failed."""
        self.training_jobs_failed += 1

    def record_conversation_started(self) -> None:
        """Record conversation started."""
        self.conversations_started += 1

    def record_conversation_ended(self) -> None:
        """Record conversation ended."""
        self.conversations_ended += 1

    def record_message(self) -> None:
        """Record message processed."""
        self.messages_processed += 1

    def record_websocket_connect(self) -> None:
        """Record WebSocket connection."""
        self.websocket_connections += 1

    def record_websocket_message(self) -> None:
        """Record WebSocket message."""
        self.websocket_messages += 1

    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles."""
        if not self.latencies:
            return {
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        latencies_list = list(self.latencies)
        avg_latency = sum(latencies_list) / len(latencies_list)
        sorted_latencies = sorted(latencies_list)

        n = len(sorted_latencies)
        p50_index = int(PERCENTILE_P50 * (n - 1))
        p95_index = int(PERCENTILE_P95 * (n - 1))
        p99_index = int(PERCENTILE_P99 * (n - 1))

        return {
            "avg_ms": round(avg_latency, 2),
            "p50_ms": round(sorted_latencies[p50_index], 2),
            "p95_ms": round(sorted_latencies[p95_index], 2),
            "p99_ms": round(sorted_latencies[p99_index], 2),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "requests": {
                "total": sum(self.request_counts.values()),
                "by_endpoint": dict(self.request_counts),
                "by_status": dict(self.status_counts),
                "errors": dict(self.error_counts),
            },
            "latency": self.get_latency_percentiles(),
            "rate_limiting": {
                "hits": self.rate_limit_hits,
            },
            "training": {
                "jobs_started": self.training_jobs_started,
                "jobs_completed": self.training_jobs_completed,
                "jobs_failed": self.training_jobs_failed,
                "total_trajectories": self.total_trajectories,
                "total_computation": round(self.total_computation, 2),
            },
            "conversations": {
                "started": self.conversations_started,
                "ended": self.conversations_ended,
                "active": self.conversations_started - self.conversations_ended,
                "messages_processed": self.messages_processed,
            },
            "websocket": {
                "total_connections": self.websocket_connections,
                "total_messages": self.websocket_messages,
            },
        }

    def snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of key metrics (legacy interface).

        Returns:
            Dictionary with essential metrics.
        """
        latency = self.get_latency_percentiles()

        return {
            "total_requests": sum(self.request_counts.values()),
            "requests_by_path": dict(self.request_counts),
            "status_codes": dict(self.status_counts),
            "average_latency_seconds": latency["avg_ms"] / 1000,
            "p95_latency_seconds": latency["p95_ms"] / 1000,
            "rate_limit_hits": self.rate_limit_hits,
        }


# Global singleton
_metrics: Optional[GRPOMetrics] = None


def get_grpo_metrics() -> GRPOMetrics:
    """Get the global GRPO metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = GRPOMetrics()
    return _metrics


def reset_metrics() -> None:
    """Reset metrics (for testing)."""
    global _metrics
    _metrics = None
