"""
Resilience Pattern Tests

Comprehensive tests for circuit breaker, retry, timeout, bulkhead,
and health check patterns.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    RetryStrategy,
    RetryConfig,
    Bulkhead,
    BulkheadConfig,
    BulkheadFullError,
    HealthChecker,
    HealthStatus,
    HealthCheckResult,
    timeout,
    TimeoutError,
    retry,
    get_circuit_breaker,
    get_all_circuit_stats,
)


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    @pytest.mark.asyncio
    async def test_circuit_starts_closed(self):
        """Circuit should start in closed state."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_calls_keep_circuit_closed(self):
        """Successful calls should keep circuit closed."""
        breaker = CircuitBreaker("test")

        async def success():
            return "ok"

        for _ in range(10):
            result = await breaker.call(success)
            assert result == "ok"

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 0

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self):
        """Enough failures should open the circuit."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("error")

        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_requests(self):
        """Open circuit should reject requests."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=10)
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("error")

        async def success():
            return "ok"

        # Open the circuit
        with pytest.raises(ValueError):
            await breaker.call(fail)

        # Now requests should be rejected
        with pytest.raises(CircuitOpenError):
            await breaker.call(success)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self):
        """Circuit should transition to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("error")

        async def success():
            return "ok"

        # Open the circuit
        with pytest.raises(ValueError):
            await breaker.call(fail)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should transition to half-open and succeed
        result = await breaker.call(success)
        assert result == "ok"
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_closes_on_success(self):
        """Half-open circuit should close after enough successes."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1
        )
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("error")

        async def success():
            return "ok"

        # Open -> Half-open
        with pytest.raises(ValueError):
            await breaker.call(fail)

        await asyncio.sleep(0.15)
        await breaker.call(success)
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success should close
        await breaker.call(success)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_reopens_on_failure(self):
        """Half-open circuit should reopen on failure."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        breaker = CircuitBreaker("test_reopen", config)

        async def always_fail():
            raise ValueError("error")

        # First call fails, opens circuit
        with pytest.raises(ValueError):
            await breaker.call(always_fail)

        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.15)

        # Next call should transition to half-open and fail again
        with pytest.raises(ValueError):
            await breaker.call(always_fail)

        # Should be back to OPEN after failure in half-open
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_fallback_function(self):
        """Fallback should be called when circuit is open."""
        async def fallback():
            return "fallback_value"

        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=10,
            fallback=fallback
        )
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("error")

        # Open the circuit
        with pytest.raises(ValueError):
            await breaker.call(fail)

        # Fallback should be called
        result = await breaker.call(fail)
        assert result == "fallback_value"

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self):
        """Excluded exceptions should not count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            excluded_exceptions=(ValueError,)
        )
        breaker = CircuitBreaker("test", config)

        async def raise_value_error():
            raise ValueError("excluded")

        # These shouldn't count as failures
        for _ in range(5):
            with pytest.raises(ValueError):
                await breaker.call(raise_value_error)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 0

    @pytest.mark.asyncio
    async def test_decorator_usage(self):
        """Test circuit breaker as decorator."""
        breaker = CircuitBreaker("decorator_test")

        @breaker
        async def decorated_func():
            return "success"

        result = await decorated_func()
        assert result == "success"

    def test_reset(self):
        """Test manual circuit reset."""
        breaker = CircuitBreaker("test")
        breaker._stats.failure_count = 10
        breaker._stats.state = CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 0

    def test_stats_to_dict(self):
        """Test stats serialization."""
        breaker = CircuitBreaker("test")
        stats = breaker.stats.to_dict()

        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "total_requests" in stats


# ============================================================================
# Retry Strategy Tests
# ============================================================================

class TestRetryStrategy:
    """Tests for retry pattern."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self):
        """Successful call should not retry."""
        strategy = RetryStrategy(RetryConfig(max_attempts=3))
        call_count = 0

        async def success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await strategy.execute(success)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Should retry on failure."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        strategy = RetryStrategy(config)
        call_count = 0

        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("error")
            return "ok"

        result = await strategy.execute(fail_twice)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Should raise after max attempts."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        strategy = RetryStrategy(config)
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        with pytest.raises(ValueError):
            await strategy.execute(always_fail)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Non-retryable exceptions should not retry."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            non_retryable_exceptions=(TypeError,)
        )
        strategy = RetryStrategy(config)
        call_count = 0

        async def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await strategy.execute(raise_type_error)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Delays should increase exponentially."""
        config = RetryConfig(
            max_attempts=4,
            base_delay=0.1,
            max_delay=10,
            jitter=False
        )
        strategy = RetryStrategy(config)

        delays = []
        for attempt in range(1, 4):
            delay = strategy._calculate_delay(attempt)
            delays.append(delay)

        # Should be 0.1, 0.2, 0.4 (exponential)
        assert delays[0] == pytest.approx(0.1, rel=0.1)
        assert delays[1] == pytest.approx(0.2, rel=0.1)
        assert delays[2] == pytest.approx(0.4, rel=0.1)

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Delay should be capped at max_delay."""
        config = RetryConfig(
            max_attempts=10,
            base_delay=1,
            max_delay=5,
            jitter=False
        )
        strategy = RetryStrategy(config)

        delay = strategy._calculate_delay(10)
        assert delay == 5

    @pytest.mark.asyncio
    async def test_decorator_factory(self):
        """Test retry decorator factory."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("error")
            return "success"

        result = await flaky_function()
        assert result == "success"
        assert call_count == 2


# ============================================================================
# Timeout Tests
# ============================================================================

class TestTimeout:
    """Tests for timeout pattern."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Fast operations should complete normally."""
        @timeout(1.0)
        async def fast_operation():
            return "done"

        result = await fast_operation()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        """Slow operations should raise TimeoutError."""
        @timeout(0.1)
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(TimeoutError) as exc_info:
            await slow_operation()

        assert "timed out" in str(exc_info.value).lower()


# ============================================================================
# Bulkhead Tests
# ============================================================================

class TestBulkhead:
    """Tests for bulkhead pattern."""

    @pytest.mark.asyncio
    async def test_allows_concurrent_requests(self):
        """Should allow requests up to max concurrent."""
        config = BulkheadConfig(max_concurrent=5, max_waiting=10)
        bulkhead = Bulkhead("test", config)

        results = []

        async def operation():
            await asyncio.sleep(0.1)
            return "done"

        # Start 5 concurrent operations
        tasks = [bulkhead.execute(operation) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(r == "done" for r in results)

    @pytest.mark.asyncio
    async def test_queues_excess_requests(self):
        """Excess requests should wait in queue."""
        config = BulkheadConfig(max_concurrent=2, max_waiting=10, timeout_seconds=5)
        bulkhead = Bulkhead("test", config)

        execution_times = []

        async def operation():
            start = time.time()
            await asyncio.sleep(0.1)
            execution_times.append(time.time() - start)
            return "done"

        # Start 4 requests with max 2 concurrent
        tasks = [bulkhead.execute(operation) for _ in range(4)]
        results = await asyncio.gather(*tasks)

        assert all(r == "done" for r in results)
        # Some operations should have waited
        assert len(execution_times) == 4

    @pytest.mark.asyncio
    async def test_rejects_when_full(self):
        """Should reject when queue is full."""
        config = BulkheadConfig(max_concurrent=1, max_waiting=1, timeout_seconds=0.1)
        bulkhead = Bulkhead("test", config)

        async def slow_operation():
            await asyncio.sleep(1.0)
            return "done"

        # Fill up the bulkhead
        task1 = asyncio.create_task(bulkhead.execute(slow_operation))
        await asyncio.sleep(0.01)  # Let it start

        task2 = asyncio.create_task(bulkhead.execute(slow_operation))
        await asyncio.sleep(0.01)  # Let it queue

        # Third should be rejected
        with pytest.raises(BulkheadFullError):
            await bulkhead.execute(slow_operation)

        # Cleanup
        task1.cancel()
        task2.cancel()
        try:
            await task1
        except:
            pass
        try:
            await task2
        except:
            pass

    @pytest.mark.asyncio
    async def test_decorator_usage(self):
        """Test bulkhead as decorator."""
        config = BulkheadConfig(max_concurrent=5)
        bulkhead = Bulkhead("decorator_test", config)

        @bulkhead
        async def decorated_func():
            return "success"

        result = await decorated_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test bulkhead statistics."""
        config = BulkheadConfig(max_concurrent=5)
        bulkhead = Bulkhead("test", config)

        stats = bulkhead.stats
        assert stats["max_concurrent"] == 5
        assert stats["active"] == 0
        assert stats["waiting"] == 0
        assert stats["available"] == 5


# ============================================================================
# Health Checker Tests
# ============================================================================

class TestHealthChecker:
    """Tests for health check pattern."""

    @pytest.mark.asyncio
    async def test_healthy_check(self):
        """Test healthy check result."""
        checker = HealthChecker()
        checker.add_check("test", lambda: True)

        result = await checker.check("test")
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_check(self):
        """Test unhealthy check result."""
        checker = HealthChecker()
        checker.add_check("test", lambda: False)

        result = await checker.check("test")
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_exception(self):
        """Test check that raises exception."""
        checker = HealthChecker()

        def failing_check():
            raise ValueError("check failed")

        checker.add_check("test", failing_check)

        result = await checker.check("test")
        assert result.status == HealthStatus.UNHEALTHY
        assert "check failed" in result.message

    @pytest.mark.asyncio
    async def test_async_check(self):
        """Test async health check."""
        checker = HealthChecker()

        async def async_check():
            await asyncio.sleep(0.01)
            return True

        checker.add_check("async_test", async_check)

        result = await checker.check("async_test")
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_all(self):
        """Test checking all registered checks."""
        checker = HealthChecker()
        checker.add_check("db", lambda: True)
        checker.add_check("cache", lambda: True)
        checker.add_check("external", lambda: False)

        results = await checker.check_all()

        assert len(results) == 3
        assert results["db"].status == HealthStatus.HEALTHY
        assert results["cache"].status == HealthStatus.HEALTHY
        assert results["external"].status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_overall_status_healthy(self):
        """Test overall status when all healthy."""
        checker = HealthChecker()
        checker.add_check("check1", lambda: True)
        checker.add_check("check2", lambda: True)

        await checker.check_all()

        assert checker.overall_status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_overall_status_unhealthy(self):
        """Test overall status when any unhealthy."""
        checker = HealthChecker()
        checker.add_check("check1", lambda: True)
        checker.add_check("check2", lambda: False)

        await checker.check_all()

        assert checker.overall_status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that latency is measured."""
        checker = HealthChecker()

        async def slow_check():
            await asyncio.sleep(0.05)
            return True

        checker.add_check("slow", slow_check)

        result = await checker.check("slow")
        assert result.latency_ms >= 50

    def test_remove_check(self):
        """Test removing a health check."""
        checker = HealthChecker()
        checker.add_check("test", lambda: True)
        checker.remove_check("test")

        assert "test" not in checker._checks

    @pytest.mark.asyncio
    async def test_missing_check(self):
        """Test checking non-existent check."""
        checker = HealthChecker()

        result = await checker.check("nonexistent")
        assert result.status == HealthStatus.UNHEALTHY
        assert "not found" in result.message


# ============================================================================
# Global Registry Tests
# ============================================================================

class TestGlobalRegistry:
    """Tests for global circuit breaker registry."""

    def test_get_circuit_breaker(self):
        """Test getting/creating circuit breaker."""
        breaker = get_circuit_breaker("global_test")
        assert breaker is not None
        assert breaker.name == "global_test"

        # Getting same name should return same instance
        breaker2 = get_circuit_breaker("global_test")
        assert breaker is breaker2

    def test_get_all_stats(self):
        """Test getting all circuit breaker stats."""
        # Create some circuit breakers
        get_circuit_breaker("stats_test_1")
        get_circuit_breaker("stats_test_2")

        stats = get_all_circuit_stats()
        assert "stats_test_1" in stats
        assert "stats_test_2" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
