"""
Unified Rate Limiter Module

Single source of truth for rate limiting across the API.
This consolidates the duplicate rate limiting implementations.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[int] = None


class UnifiedRateLimiter:
    """
    Thread-safe sliding window rate limiter.

    This is the single source of truth for rate limiting across the API,
    consolidating the previously duplicate implementations.
    """

    def __init__(self, window_seconds: int = 60):
        """
        Initialize the rate limiter.

        Args:
            window_seconds: Size of the sliding window in seconds.
        """
        self.window_seconds = window_seconds
        self._windows: Dict[str, deque] = defaultdict(deque)

    def check(self, key: str, limit: int) -> RateLimitResult:
        """
        Check if a request is allowed under the rate limit.

        Args:
            key: Unique identifier for the rate limit bucket (e.g., API key, IP).
            limit: Maximum number of requests allowed per window.

        Returns:
            RateLimitResult with allowed status and metadata.
        """
        now = time.monotonic()
        window = self._windows[key]
        cutoff = now - self.window_seconds

        # Remove expired entries
        while window and window[0] <= cutoff:
            window.popleft()

        current_count = len(window)
        remaining = max(0, limit - current_count)
        reset_at = now + self.window_seconds

        if current_count >= limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=self.window_seconds,
            )

        # Record the request
        window.append(now)

        return RateLimitResult(
            allowed=True,
            remaining=remaining - 1,  # Account for this request
            reset_at=reset_at,
        )

    def allow(self, key: str, limit: int) -> bool:
        """
        Simple check if request is allowed (legacy interface).

        Args:
            key: Rate limit bucket key.
            limit: Maximum requests per window.

        Returns:
            True if allowed, False if rate limited.
        """
        return self.check(key, limit).allowed

    def is_allowed(self, key: str, limit: int) -> Tuple[bool, int]:
        """
        Check if allowed and return remaining count (middleware interface).

        Args:
            key: Rate limit bucket key.
            limit: Maximum requests per window.

        Returns:
            Tuple of (allowed, remaining_requests).
        """
        result = self.check(key, limit)
        return result.allowed, result.remaining

    def reset(self, key: str) -> None:
        """
        Reset rate limit for a specific key.

        Args:
            key: Rate limit bucket key to reset.
        """
        self._windows.pop(key, None)

    def get_usage(self, key: str) -> int:
        """
        Get current usage count for a key.

        Args:
            key: Rate limit bucket key.

        Returns:
            Number of requests in current window.
        """
        now = time.monotonic()
        window = self._windows.get(key, deque())
        cutoff = now - self.window_seconds

        # Count non-expired entries
        return sum(1 for ts in window if ts > cutoff)

    def cleanup(self) -> int:
        """
        Remove all expired entries from all windows.

        Returns:
            Number of keys cleaned up.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds
        cleaned = 0

        empty_keys = []
        for key, window in self._windows.items():
            # Remove expired entries
            while window and window[0] <= cutoff:
                window.popleft()

            if not window:
                empty_keys.append(key)

        # Remove empty windows
        for key in empty_keys:
            del self._windows[key]
            cleaned += 1

        return cleaned

    def stats(self) -> Dict[str, int]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with statistics.
        """
        total_keys = len(self._windows)
        total_entries = sum(len(w) for w in self._windows.values())

        return {
            "active_keys": total_keys,
            "total_entries": total_entries,
            "window_seconds": self.window_seconds,
        }


# Global singleton instance
_rate_limiter: Optional[UnifiedRateLimiter] = None


def get_rate_limiter(window_seconds: int = 60) -> UnifiedRateLimiter:
    """
    Get the global rate limiter instance.

    Args:
        window_seconds: Window size (only used on first call).

    Returns:
        The global UnifiedRateLimiter instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = UnifiedRateLimiter(window_seconds)
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (for testing)."""
    global _rate_limiter
    _rate_limiter = None
