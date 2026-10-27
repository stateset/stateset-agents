"""
Utilities for the StateSet Agents framework.

This package intentionally avoids importing optional/heavy dependencies at
import time. Prefer importing the specific submodule you need.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .._deprecation import warn_deprecated

__all__ = [
    "CacheService",
    "get_cache_service",
    "MonitoringService",
    "PerformanceMonitor",
    "SecurityMonitor",
    "WandBLogger",
    "init_wandb",
]

#: Names served from modules with no in-package users, deprecated in favour of
#: the canonical monitoring path (``utils.monitoring``, ``api.observability``,
#: ``utils.performance_monitor``, stdlib ``logging``): name -> (module,
#: replacement). Removed together with their modules.
_DEPRECATED_EXPORTS: dict[str, tuple[str, str]] = {
    "AlertManager": (".alerts", "stateset_agents.utils.monitoring.AlertManager"),
    "ObservabilityManager": (
        ".observability",
        "stateset_agents.api.observability (tracing)",
    ),
    "get_logger": (".logging", "logging.getLogger"),
    "PerformanceReport": (".profiler", "stateset_agents.utils.performance_monitor"),
}
_DEPRECATED_EXPORTS_REMOVAL = "v0.71.0"


def __getattr__(name: str) -> Any:
    deprecated = _DEPRECATED_EXPORTS.get(name)
    if deprecated is not None:
        module_name, replacement = deprecated
        warn_deprecated(
            f"stateset_agents.utils.{name}",
            removal=_DEPRECATED_EXPORTS_REMOVAL,
            replacement=replacement,
        )
        return getattr(import_module(module_name, __name__), name)

    if name in {"CacheService", "get_cache_service"}:
        from .cache import CacheService as _CacheService
        from .cache import get_cache_service as _get_cache_service

        globals()["CacheService"] = _CacheService
        globals()["get_cache_service"] = _get_cache_service
        return globals()[name]

    if name == "MonitoringService":
        from .monitoring import MonitoringService as _MonitoringService

        globals()["MonitoringService"] = _MonitoringService
        return _MonitoringService

    if name == "PerformanceMonitor":
        from .performance_monitor import PerformanceMonitor as _PerformanceMonitor

        globals()["PerformanceMonitor"] = _PerformanceMonitor
        return _PerformanceMonitor

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
