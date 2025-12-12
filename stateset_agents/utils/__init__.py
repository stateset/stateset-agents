"""
Utilities for the GRPO Agent Framework
"""

from .alerts import AlertManager
from .cache import CacheService, get_cache_service
from .logging import get_logger
from .monitoring import MonitoringService
from .observability import ObservabilityManager
from .performance_monitor import PerformanceMonitor
from .profiler import PerformanceReport
from .security import SecurityMonitor

__all__ = [
    "CacheService",
    "get_cache_service",
    "get_logger",
    "MonitoringService",
    "ObservabilityManager",
    "AlertManager",
    "SecurityMonitor",
    "PerformanceMonitor",
    "PerformanceReport",
]


def __getattr__(name):
    """Lazily import optional W&B dependencies."""
    if name in {"WandBLogger", "init_wandb"}:
        from .wandb_integration import WandBLogger, init_wandb

        globals()["WandBLogger"] = WandBLogger
        globals()["init_wandb"] = init_wandb
        return globals()[name]

    raise AttributeError(f"module 'utils' has no attribute '{name}'")
