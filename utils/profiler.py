"""
Performance profiling utilities for the StateSet Agents framework.

This module provides decorators and utilities for profiling function performance,
memory usage, and generating performance reports.
"""

import cProfile
import io
import logging
import pstats
import time
from functools import wraps
from typing import Any, Callable, Optional

import psutil
import torch

from .performance_monitor import (
    monitor_operation,
    profile_async_function,
    profile_function,
)

logger = logging.getLogger(__name__)


def profiled(func: Callable) -> Callable:
    """
    Decorator to profile function performance using cProfile.

    This decorator provides detailed profiling information including
    function call counts, execution times, and call graphs.

    Args:
        func: Function to profile

    Returns:
        Wrapped function that profiles execution
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            pr.disable()

            # Create statistics
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats("cumulative")
            ps.print_stats()

            # Log profiling results
            profile_output = s.getvalue()
            logger.info(f"Profile for {func.__name__}:\n{profile_output}")

    return wrapper


def profiled_async(func: Callable) -> Callable:
    """
    Decorator to profile async function performance.

    Args:
        func: Async function to profile

    Returns:
        Wrapped async function that profiles execution
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            pr.disable()

            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats("cumulative")
            ps.print_stats()

            profile_output = s.getvalue()
            logger.info(f"Profile for {func.__name__}:\n{profile_output}")

    return wrapper


def timed(func: Callable) -> Callable:
    """
    Simple timing decorator that logs execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function with timing
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"{func.__name__} executed in {duration:.3f} seconds")

    return wrapper


def timed_async(func: Callable) -> Callable:
    """
    Simple timing decorator for async functions.

    Args:
        func: Async function to time

    Returns:
        Wrapped async function with timing
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"{func.__name__} executed in {duration:.3f} seconds")

    return wrapper


class PerformanceReport:
    """Generate performance reports from profiling data."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = None
        self.start_gpu_memory = None

    def start(self):
        """Start performance monitoring."""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    def generate_report(self, operation_name: str) -> dict:
        """Generate a performance report."""
        if self.start_memory is None:
            raise ValueError("Performance monitoring not started. Call start() first.")

        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_delta = current_memory - self.start_memory

        report = {
            "operation": operation_name,
            "memory": {
                "start_mb": self.start_memory,
                "current_mb": current_memory,
                "delta_mb": memory_delta,
            },
        }

        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

            report["gpu_memory"] = {
                "start_mb": self.start_gpu_memory,
                "current_mb": current_gpu_memory,
                "peak_mb": peak_gpu_memory,
                "delta_mb": current_gpu_memory - (self.start_gpu_memory or 0),
            }

        return report


def create_performance_report(operation_name: str) -> PerformanceReport:
    """Create a performance report generator."""
    report = PerformanceReport()
    report.start()
    return report


# Convenience decorators that combine monitoring and profiling
def monitored_and_profiled(func: Callable) -> Callable:
    """Decorator that combines monitoring and profiling."""
    profiled_func = profiled(func)
    monitored_func = profile_function(profiled_func)
    return monitored_func


def monitored_and_profiled_async(func: Callable) -> Callable:
    """Decorator that combines monitoring and profiling for async functions."""
    profiled_func = profiled_async(func)
    monitored_func = profile_async_function(profiled_func)
    return monitored_func


# Utility functions for profiling specific operations
def profile_agent_generation(agent_class, config, messages):
    """Profile agent response generation."""

    async def _profile():
        agent = agent_class(config)
        await agent.initialize()

        with monitor_operation("agent_response_generation"):
            return await agent.generate_response(messages)

    return _profile()


def profile_training_step(trainer, agent, environment):
    """Profile a training step."""

    async def _profile():
        with monitor_operation("training_step"):
            return await trainer.train_step(agent, environment)

    return _profile()


def profile_inference_batch(agent, messages_batch):
    """Profile batch inference."""

    async def _profile():
        with monitor_operation("batch_inference"):
            responses = []
            for messages in messages_batch:
                response = await agent.generate_response(messages)
                responses.append(response)
            return responses

    return _profile()
