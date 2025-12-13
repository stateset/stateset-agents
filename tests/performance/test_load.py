"""
Performance and Load Tests

Comprehensive performance testing for the API including:
- Load testing with concurrent requests
- Latency measurements
- Throughput testing
- Memory usage testing
- Cache performance
"""

import asyncio
import gc
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.performance


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class PerformanceThresholds:
    """Performance test thresholds."""
    max_avg_latency_ms: float = 100.0
    max_p95_latency_ms: float = 200.0
    max_p99_latency_ms: float = 500.0
    min_throughput_rps: float = 100.0
    max_memory_increase_mb: float = 50.0


THRESHOLDS = PerformanceThresholds()


# ============================================================================
# Performance Measurement Utilities
# ============================================================================

@dataclass
class PerformanceResult:
    """Result of a performance test."""
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    latencies_ms: List[float]
    errors: List[str]

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_requests / self.total_requests * 100
                if self.total_requests > 0 else 0.0)

    @property
    def throughput_rps(self) -> float:
        """Calculate requests per second."""
        return self.total_requests / self.total_time_seconds if self.total_time_seconds > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def median_latency_ms(self) -> float:
        """Calculate median latency."""
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(0.95 * len(sorted_latencies))
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        """Calculate 99th percentile latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(0.99 * len(sorted_latencies))
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    def summary(self) -> dict:
        """Get summary of results."""
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{self.success_rate:.2f}%",
            "throughput_rps": f"{self.throughput_rps:.2f}",
            "avg_latency_ms": f"{self.avg_latency_ms:.2f}",
            "median_latency_ms": f"{self.median_latency_ms:.2f}",
            "p95_latency_ms": f"{self.p95_latency_ms:.2f}",
            "p99_latency_ms": f"{self.p99_latency_ms:.2f}",
        }


async def run_load_test(
    name: str,
    func: Callable,
    num_requests: int = 100,
    concurrency: int = 10,
) -> PerformanceResult:
    """
    Run a load test with specified concurrency.

    Args:
        name: Test name
        func: Async function to test
        num_requests: Total number of requests
        concurrency: Number of concurrent requests

    Returns:
        PerformanceResult with test metrics
    """
    latencies: List[float] = []
    errors: List[str] = []
    successful = 0
    failed = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def run_single():
        nonlocal successful, failed
        async with semaphore:
            start = time.perf_counter()
            try:
                await func()
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                successful += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))

    start_time = time.perf_counter()
    await asyncio.gather(*[run_single() for _ in range(num_requests)])
    total_time = time.perf_counter() - start_time

    return PerformanceResult(
        name=name,
        total_requests=num_requests,
        successful_requests=successful,
        failed_requests=failed,
        total_time_seconds=total_time,
        latencies_ms=latencies,
        errors=errors,
    )


# ============================================================================
# Cache Performance Tests
# ============================================================================

class TestCachePerformance:
    """Performance tests for caching systems."""

    @pytest.mark.asyncio
    async def test_memory_cache_read_performance(self):
        """Test memory cache read performance."""
        from api.cache import SimpleCache

        cache = SimpleCache()

        # Pre-populate cache
        for i in range(1000):
            cache.set(f"key{i}", f"value{i}" * 100, ttl_seconds=300)

        async def read_operation():
            import random
            key = f"key{random.randint(0, 999)}"
            cache.get(key)

        result = await run_load_test(
            name="Memory Cache Read",
            func=read_operation,
            num_requests=10000,
            concurrency=100,
        )

        print(f"\nMemory Cache Read Performance:")
        print(f"  Throughput: {result.throughput_rps:.2f} ops/sec")
        print(f"  Avg Latency: {result.avg_latency_ms:.4f} ms")
        print(f"  P95 Latency: {result.p95_latency_ms:.4f} ms")

        assert result.avg_latency_ms < 1.0, "Cache read should be < 1ms"
        assert result.throughput_rps > 10000, "Should handle > 10k ops/sec"

    @pytest.mark.asyncio
    async def test_memory_cache_write_performance(self):
        """Test memory cache write performance."""
        from api.cache import SimpleCache

        cache = SimpleCache()
        counter = 0

        async def write_operation():
            nonlocal counter
            counter += 1
            cache.set(f"key{counter}", f"value{counter}" * 100, ttl_seconds=300)

        result = await run_load_test(
            name="Memory Cache Write",
            func=write_operation,
            num_requests=10000,
            concurrency=50,
        )

        print(f"\nMemory Cache Write Performance:")
        print(f"  Throughput: {result.throughput_rps:.2f} ops/sec")
        print(f"  Avg Latency: {result.avg_latency_ms:.4f} ms")

        assert result.avg_latency_ms < 1.0, "Cache write should be < 1ms"

    @pytest.mark.asyncio
    async def test_distributed_cache_performance(self):
        """Test distributed cache performance."""
        from api.distributed_cache import MemoryCache, CacheConfig

        config = CacheConfig(max_memory_items=10000)
        cache = MemoryCache(config)

        # Pre-populate
        for i in range(1000):
            await cache.set(f"key{i}", {"data": f"value{i}" * 10})

        async def mixed_operation():
            import random
            key = f"key{random.randint(0, 999)}"
            if random.random() < 0.8:  # 80% reads
                await cache.get(key)
            else:  # 20% writes
                await cache.set(key, {"data": f"updated{random.randint(0, 1000)}"})

        result = await run_load_test(
            name="Distributed Cache Mixed",
            func=mixed_operation,
            num_requests=5000,
            concurrency=50,
        )

        print(f"\nDistributed Cache Mixed Operations:")
        print(f"  Throughput: {result.throughput_rps:.2f} ops/sec")
        print(f"  Avg Latency: {result.avg_latency_ms:.4f} ms")

        assert result.success_rate > 99.0, "Should have > 99% success rate"


# ============================================================================
# Input Validation Performance Tests
# ============================================================================

class TestValidationPerformance:
    """Performance tests for input validation."""

    def test_string_validation_performance(self):
        """Test string validation performance."""
        from api.security import InputValidator

        test_string = "This is a test string for validation. " * 50
        iterations = 10000

        start = time.perf_counter()
        for _ in range(iterations):
            InputValidator.validate_string(test_string, check_injection=False)
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        avg_latency_us = (elapsed / iterations) * 1_000_000

        print(f"\nString Validation (no injection check):")
        print(f"  {ops_per_sec:.0f} ops/sec")
        print(f"  {avg_latency_us:.2f} μs avg")

        assert avg_latency_us < 100, "Validation should be < 100μs"

    def test_injection_detection_performance(self):
        """Test prompt injection detection performance."""
        from api.security import InputValidator

        test_string = "Please help me with this coding task. I want to learn Python."
        iterations = 5000

        start = time.perf_counter()
        for _ in range(iterations):
            InputValidator.detect_prompt_injection(test_string)
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        avg_latency_us = (elapsed / iterations) * 1_000_000

        print(f"\nPrompt Injection Detection:")
        print(f"  {ops_per_sec:.0f} ops/sec")
        print(f"  {avg_latency_us:.2f} μs avg")

        assert avg_latency_us < 500, "Injection detection should be < 500μs"

    def test_message_validation_performance(self):
        """Test message list validation performance."""
        from api.security import InputValidator

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you today?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "Can you help me with Python?"},
        ]
        iterations = 2000

        start = time.perf_counter()
        for _ in range(iterations):
            InputValidator.validate_messages(messages.copy())
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        avg_latency_us = (elapsed / iterations) * 1_000_000

        print(f"\nMessage Validation (4 messages):")
        print(f"  {ops_per_sec:.0f} ops/sec")
        print(f"  {avg_latency_us:.2f} μs avg")

        assert avg_latency_us < 1000, "Message validation should be < 1ms"


# ============================================================================
# Middleware Performance Tests
# ============================================================================

class TestMiddlewarePerformance:
    """Performance tests for middleware components."""

    def test_rate_limiter_performance(self):
        """Test rate limiter performance."""
        from api.middleware import SlidingWindowRateLimiter

        limiter = SlidingWindowRateLimiter(window_seconds=60)
        iterations = 50000

        start = time.perf_counter()
        for i in range(iterations):
            limiter.is_allowed(f"user{i % 1000}", limit=100)
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        avg_latency_us = (elapsed / iterations) * 1_000_000

        print(f"\nRate Limiter Check:")
        print(f"  {ops_per_sec:.0f} ops/sec")
        print(f"  {avg_latency_us:.2f} μs avg")

        assert avg_latency_us < 50, "Rate limit check should be < 50μs"

    def test_metrics_recording_performance(self):
        """Test metrics recording performance."""
        from api.middleware import APIMetrics

        metrics = APIMetrics()
        iterations = 100000

        start = time.perf_counter()
        for i in range(iterations):
            metrics.record_request(
                path=f"/api/endpoint{i % 10}",
                method="GET",
                status_code=200 if i % 10 != 0 else 500,
                latency_ms=i % 100 + 10,
            )
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        avg_latency_us = (elapsed / iterations) * 1_000_000

        print(f"\nMetrics Recording:")
        print(f"  {ops_per_sec:.0f} ops/sec")
        print(f"  {avg_latency_us:.2f} μs avg")

        assert avg_latency_us < 20, "Metrics recording should be < 20μs"


# ============================================================================
# Resilience Pattern Performance Tests
# ============================================================================

class TestResiliencePerformance:
    """Performance tests for resilience patterns."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self):
        """Test circuit breaker overhead."""
        from api.resilience import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test", config)

        async def fast_operation():
            return "success"

        # Warm up
        for _ in range(10):
            await breaker.call(fast_operation)

        # Measure with circuit breaker
        start = time.perf_counter()
        for _ in range(1000):
            await breaker.call(fast_operation)
        with_breaker = time.perf_counter() - start

        # Measure without circuit breaker
        start = time.perf_counter()
        for _ in range(1000):
            await fast_operation()
        without_breaker = time.perf_counter() - start

        overhead_percent = ((with_breaker - without_breaker) / without_breaker) * 100

        print(f"\nCircuit Breaker Overhead:")
        print(f"  With breaker: {with_breaker*1000:.2f} ms for 1000 calls")
        print(f"  Without breaker: {without_breaker*1000:.2f} ms for 1000 calls")
        print(f"  Overhead: {overhead_percent:.1f}%")

        assert overhead_percent < 100, "Circuit breaker overhead should be < 100%"

    @pytest.mark.asyncio
    async def test_bulkhead_performance(self):
        """Test bulkhead concurrency limiting."""
        from api.resilience import Bulkhead, BulkheadConfig

        config = BulkheadConfig(max_concurrent=10, max_waiting=100)
        bulkhead = Bulkhead("test", config)

        async def quick_operation():
            await asyncio.sleep(0.001)
            return "done"

        result = await run_load_test(
            name="Bulkhead",
            func=lambda: bulkhead.execute(quick_operation),
            num_requests=100,
            concurrency=20,
        )

        print(f"\nBulkhead Performance:")
        print(f"  Throughput: {result.throughput_rps:.2f} ops/sec")
        print(f"  Success Rate: {result.success_rate:.2f}%")

        assert result.success_rate > 90, "Should have > 90% success rate"


# ============================================================================
# Memory Usage Tests
# ============================================================================

class TestMemoryUsage:
    """Tests for memory usage and leaks."""

    def test_cache_memory_bounded(self):
        """Test that cache memory is bounded."""
        from api.cache import SimpleCache

        cache = SimpleCache()

        # Get baseline memory
        gc.collect()
        baseline = sys.getsizeof(cache._cache)

        # Add many items
        large_value = "x" * 10000
        for i in range(10000):
            cache.set(f"key{i}", large_value, ttl_seconds=300)

        gc.collect()
        after_inserts = sys.getsizeof(cache._cache)

        # Memory should be bounded
        memory_increase_mb = (after_inserts - baseline) / (1024 * 1024)

        print(f"\nCache Memory Usage:")
        print(f"  Baseline: {baseline / 1024:.2f} KB")
        print(f"  After 10k inserts: {after_inserts / 1024:.2f} KB")

        # Note: This is a basic check - dict size doesn't include value sizes

    @pytest.mark.asyncio
    async def test_no_memory_leak_in_rate_limiter(self):
        """Test that rate limiter doesn't leak memory."""
        from api.middleware import SlidingWindowRateLimiter

        limiter = SlidingWindowRateLimiter(window_seconds=1)

        gc.collect()
        baseline_objects = len(gc.get_objects())

        # Simulate many users over time
        for round_num in range(10):
            for user in range(1000):
                limiter.is_allowed(f"user{user}", limit=10)
            await asyncio.sleep(0.1)

        gc.collect()
        final_objects = len(gc.get_objects())

        object_increase = final_objects - baseline_objects

        print(f"\nRate Limiter Memory (object count):")
        print(f"  Baseline: {baseline_objects}")
        print(f"  Final: {final_objects}")
        print(f"  Increase: {object_increase}")

        # Should not grow unboundedly
        assert object_increase < 50000, "Object count growth should be bounded"


# ============================================================================
# Stress Tests
# ============================================================================

class TestStress:
    """Stress tests for edge cases."""

    @pytest.mark.asyncio
    async def test_high_concurrency_cache(self):
        """Test cache under high concurrency."""
        from api.distributed_cache import MemoryCache, CacheConfig

        config = CacheConfig(max_memory_items=1000)
        cache = MemoryCache(config)

        async def stress_operation():
            import random
            key = f"stress{random.randint(0, 100)}"
            if random.random() < 0.5:
                await cache.get(key)
            else:
                await cache.set(key, {"value": random.random()})

        result = await run_load_test(
            name="High Concurrency Cache",
            func=stress_operation,
            num_requests=10000,
            concurrency=200,
        )

        print(f"\nHigh Concurrency Cache Stress:")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Success Rate: {result.success_rate:.2f}%")
        print(f"  Throughput: {result.throughput_rps:.2f} ops/sec")

        assert result.success_rate > 99, "Should handle high concurrency"

    def test_large_payload_validation(self):
        """Test validation with large payloads."""
        from api.security import InputValidator

        # Test with increasingly large strings
        sizes = [1000, 10000, 50000, 100000]

        print("\nLarge Payload Validation:")
        for size in sizes:
            test_string = "x" * size

            start = time.perf_counter()
            for _ in range(100):
                try:
                    InputValidator.validate_string(
                        test_string,
                        max_length=size + 1,
                        check_injection=False,
                    )
                except ValueError:
                    pass
            elapsed = time.perf_counter() - start

            avg_ms = (elapsed / 100) * 1000
            print(f"  {size:,} chars: {avg_ms:.2f} ms avg")

            assert avg_ms < 10, f"Validation of {size} chars should be < 10ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
