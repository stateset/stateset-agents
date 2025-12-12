"""
Performance monitoring utilities for the StateSet Agents framework.

This module provides tools for profiling, benchmarking, and monitoring
the performance of agents, training, and inference operations.
"""

import gc
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

    # Memory metrics
    memory_start: Optional[float] = None  # MB
    memory_end: Optional[float] = None  # MB
    memory_peak: Optional[float] = None  # MB
    memory_delta: Optional[float] = None  # MB

    # CPU metrics
    cpu_percent_start: Optional[float] = None
    cpu_percent_end: Optional[float] = None

    # GPU metrics (if available)
    gpu_memory_start: Optional[float] = None  # MB
    gpu_memory_end: Optional[float] = None  # MB
    gpu_memory_peak: Optional[float] = None  # MB

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor performance of operations."""

    def __init__(self):
        self.process = psutil.Process()
        self._metrics_history: List[PerformanceMetrics] = []
        self._active_operations: Dict[str, PerformanceMetrics] = {}

    def start_operation(self, name: str, **metadata) -> str:
        """Start monitoring an operation."""
        operation_id = f"{name}_{int(time.time() * 1000000)}"

        metrics = PerformanceMetrics(
            operation_name=name,
            start_time=time.time(),
            memory_start=self._get_memory_usage(),
            cpu_percent_start=self.process.cpu_percent(),
            gpu_memory_start=self._get_gpu_memory_usage(),
            metadata=metadata,
        )

        self._active_operations[operation_id] = metrics
        return operation_id

    def end_operation(self, operation_id: str) -> PerformanceMetrics:
        """End monitoring an operation."""
        if operation_id not in self._active_operations:
            raise ValueError(f"Operation {operation_id} not found")

        metrics = self._active_operations.pop(operation_id)
        metrics.end_time = time.time()
        metrics.duration = metrics.end_time - metrics.start_time

        # Collect final metrics
        metrics.memory_end = self._get_memory_usage()
        metrics.cpu_percent_end = self.process.cpu_percent()
        metrics.gpu_memory_end = self._get_gpu_memory_usage()

        if metrics.memory_start is not None and metrics.memory_end is not None:
            metrics.memory_delta = metrics.memory_end - metrics.memory_start

        # Calculate peak memory (simplified)
        metrics.memory_peak = max(metrics.memory_start or 0, metrics.memory_end or 0)
        metrics.gpu_memory_peak = max(
            metrics.gpu_memory_start or 0, metrics.gpu_memory_end or 0
        )

        self._metrics_history.append(metrics)
        return metrics

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return None

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        if not self._metrics_history:
            return {"total_operations": 0}

        operations_by_name = defaultdict(list)
        for metrics in self._metrics_history:
            operations_by_name[metrics.operation_name].append(metrics)

        summary = {
            "total_operations": len(self._metrics_history),
            "operations_by_type": {},
        }

        for op_name, metrics_list in operations_by_name.items():
            durations = [m.duration for m in metrics_list if m.duration is not None]
            memory_deltas = [
                m.memory_delta for m in metrics_list if m.memory_delta is not None
            ]

            summary["operations_by_type"][op_name] = {
                "count": len(metrics_list),
                "avg_duration": sum(durations) / len(durations) if durations else None,
                "min_duration": min(durations) if durations else None,
                "max_duration": max(durations) if durations else None,
                "avg_memory_delta": sum(memory_deltas) / len(memory_deltas)
                if memory_deltas
                else None,
            }

        return summary

    def clear_history(self):
        """Clear metrics history."""
        self._metrics_history.clear()


# Global monitor instance
_global_monitor = PerformanceMonitor()


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _global_monitor


@contextmanager
def monitor_operation(name: str, **metadata):
    """Context manager for monitoring operations."""
    operation_id = _global_monitor.start_operation(name, **metadata)
    try:
        yield
    finally:
        metrics = _global_monitor.end_operation(operation_id)
        logger.info(
            f"Operation '{name}' completed in {metrics.duration:.3f}s, "
            f"memory delta: {metrics.memory_delta:.1f}MB"
        )


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function performance."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        with monitor_operation(func_name, function_name=func.__name__):
            return func(*args, **kwargs)

    return wrapper


def profile_async_function(func: Callable) -> Callable:
    """Decorator to profile async function performance."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        with monitor_operation(func_name, function_name=func.__name__):
            return await func(*args, **kwargs)

    return wrapper


class MemoryProfiler:
    """Profile memory usage of operations."""

    def __init__(self):
        self.process = psutil.Process()

    @contextmanager
    def profile_memory(self, operation_name: str):
        """Context manager to profile memory usage."""
        start_mem = self.process.memory_info().rss / 1024 / 1024
        start_gpu_mem = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024

        try:
            yield
        finally:
            end_mem = self.process.memory_info().rss / 1024 / 1024
            end_gpu_mem = None
            peak_gpu_mem = None
            if torch.cuda.is_available():
                end_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
                peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

            logger.info(
                f"Memory profile for '{operation_name}': "
                f"RAM: {start_mem:.1f}MB -> {end_mem:.1f}MB "
                f"(Δ{end_mem - start_mem:.1f}MB)"
            )

            if torch.cuda.is_available():
                logger.info(
                    f"GPU Memory: {start_gpu_mem:.1f}MB -> {end_gpu_mem:.1f}MB "
                    f"(Peak: {peak_gpu_mem:.1f}MB)"
                )


class BenchmarkRunner:
    """Run benchmarks on functions."""

    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.monitor = PerformanceMonitor()

    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark a function."""
        func_name = f"{func.__module__}.{func.__name__}"

        # Warmup runs
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)

        # Benchmark runs
        durations = []
        memory_deltas = []

        for _ in range(self.benchmark_runs):
            operation_id = self.monitor.start_operation(f"benchmark_{func_name}")
            result = func(*args, **kwargs)
            metrics = self.monitor.end_operation(operation_id)

            if metrics.duration is not None:
                durations.append(metrics.duration)
            if metrics.memory_delta is not None:
                memory_deltas.append(metrics.memory_delta)

        # Calculate statistics
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_memory = sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        return {
            "function_name": func_name,
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "avg_memory_delta": avg_memory,
            "runs": len(durations),
            "result": result,
        }


def benchmark_async_function(
    func: Callable, *args, warmup_runs: int = 3, benchmark_runs: int = 10, **kwargs
) -> Dict[str, Any]:
    """Benchmark an async function."""
    import asyncio

    async def run_benchmark():
        func_name = f"{func.__module__}.{func.__name__}"
        monitor = PerformanceMonitor()

        # Warmup runs
        for _ in range(warmup_runs):
            await func(*args, **kwargs)

        # Benchmark runs
        durations = []
        memory_deltas = []

        for _ in range(benchmark_runs):
            operation_id = monitor.start_operation(f"benchmark_{func_name}")
            result = await func(*args, **kwargs)
            metrics = monitor.end_operation(operation_id)

            if metrics.duration is not None:
                durations.append(metrics.duration)
            if metrics.memory_delta is not None:
                memory_deltas.append(metrics.memory_delta)

        # Calculate statistics
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_memory = sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        return {
            "function_name": func_name,
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "avg_memory_delta": avg_memory,
            "runs": len(durations),
            "result": result,
        }

    return asyncio.run(run_benchmark())


# Convenience functions
def start_monitoring(operation_name: str, **metadata) -> str:
    """Start monitoring an operation."""
    return _global_monitor.start_operation(operation_name, **metadata)


def end_monitoring(operation_id: str) -> PerformanceMetrics:
    """End monitoring an operation."""
    return _global_monitor.end_operation(operation_id)


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary."""
    return _global_monitor.get_metrics_summary()
