"""
Utilities for the StateSet Agents framework.

This package intentionally avoids importing optional/heavy dependencies at
import time. Prefer importing the specific submodule you need.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AlertManager",
    "CacheService",
    "get_cache_service",
    "get_logger",
    "MonitoringService",
    "ObservabilityManager",
    "PerformanceMonitor",
    "PerformanceReport",
    "SecurityMonitor",
    "WandBLogger",
    "init_wandb",
]


def __getattr__(name: str) -> Any:
    if name == "AlertManager":
        from .alerts import AlertManager as _AlertManager

        globals()["AlertManager"] = _AlertManager
        return _AlertManager

    if name in {"CacheService", "get_cache_service"}:
        from .cache import CacheService as _CacheService
        from .cache import get_cache_service as _get_cache_service

        globals()["CacheService"] = _CacheService
        globals()["get_cache_service"] = _get_cache_service
        return globals()[name]

    if name == "get_logger":
        from .logging import get_logger as _get_logger

        globals()["get_logger"] = _get_logger
        return _get_logger

    if name == "MonitoringService":
        from .monitoring import MonitoringService as _MonitoringService

        globals()["MonitoringService"] = _MonitoringService
        return _MonitoringService

    if name == "ObservabilityManager":
        from .observability import ObservabilityManager as _ObservabilityManager

        globals()["ObservabilityManager"] = _ObservabilityManager
        return _ObservabilityManager

    if name == "PerformanceMonitor":
        from .performance_monitor import PerformanceMonitor as _PerformanceMonitor

        globals()["PerformanceMonitor"] = _PerformanceMonitor
        return _PerformanceMonitor

    if name == "PerformanceReport":
        from .profiler import PerformanceReport as _PerformanceReport

        globals()["PerformanceReport"] = _PerformanceReport
        return _PerformanceReport

    if name == "SecurityMonitor":
        from .security import SecurityMonitor as _SecurityMonitor

        globals()["SecurityMonitor"] = _SecurityMonitor
        return _SecurityMonitor

    if name in {"WandBLogger", "init_wandb"}:
        from .wandb_integration import WandBLogger as _WandBLogger
        from .wandb_integration import init_wandb as _init_wandb

        globals()["WandBLogger"] = _WandBLogger
        globals()["init_wandb"] = _init_wandb
        return globals()[name]

    raise AttributeError(f"module 'stateset_agents.utils' has no attribute {name!r}")
