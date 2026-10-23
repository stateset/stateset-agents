"""
GRPO Agent Framework

A comprehensive framework for training multi-turn AI agents using
Group Relative Policy Optimization (GRPO).
"""

from __future__ import annotations

import warnings as _warnings
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Any

from ._deprecation import warn_deprecated

__version__ = "0.69.0"
__author__ = "StateSet Team"
__email__ = "team@stateset.ai"

OPTIONAL_IMPORT_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    ValueError,
)

_TRAINING_INSTALL_HINT = "pip install 'stateset-agents[training]'"
_AIOHTTP_INSTALL_HINT = "pip install aiohttp"


def _export_group(
    module_name: str,
    names: list[str],
    *,
    install_hint: str | None = None,
) -> dict[str, tuple[str, str, str | None]]:
    return {name: (module_name, name, install_hint) for name in names}


_LAZY_EXPORTS: dict[str, tuple[str, str, str | None]] = {}

_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.agent",
        ["Agent", "MultiTurnAgent", "ToolAgent"],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.environment",
        ["Environment", "ConversationEnvironment", "TaskEnvironment"],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.trajectory",
        ["Trajectory", "MultiTurnTrajectory", "ConversationTurn"],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.reward",
        [
            "RewardFunction",
            "CompositeReward",
            "HelpfulnessReward",
            "SafetyReward",
            "CorrectnessReward",
            "ConcisenessReward",
            "EngagementReward",
            "TaskCompletionReward",
            "CustomerServiceReward",
            "TechnicalSupportReward",
            "SalesAssistantReward",
            "DomainSpecificReward",
            "SimilarityAwareReward",
            "create_domain_reward",
            "create_adaptive_reward",
        ],
    )
)
# The framework's exception root and the typed error its public entry points
# (train(), TrainingConfig, AgentConfig) raise; ConfigurationError is also a
# ValueError.
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.errors",
        ["StateSetError", "ConfigurationError"],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.error_handling",
        [
            "GRPOException",
            "TrainingException",
            "ModelException",
            "DataException",
            "NetworkException",
            "ResourceException",
            "ValidationException",
            "ErrorCode",
            "ErrorHandler",
            "RetryConfig",
            "CircuitBreakerConfig",
            "retry_async",
            "circuit_breaker",
            "handle_error",
            "get_error_summary",
        ],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.performance_optimizer",
        [
            "PerformanceOptimizer",
            "MemoryMonitor",
            "ModelOptimizer",
            "BatchOptimizer",
            "OptimizationLevel",
            "MemoryConfig",
            "ComputeConfig",
            "DataConfig",
            "PerformanceMetrics",
        ],
        install_hint=_TRAINING_INSTALL_HINT,
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.type_system",
        [
            "TypeValidator",
            "ConfigValidator",
            "TypeSafeSerializer",
            "create_typed_config",
            "ensure_type_safety",
            "ensure_async_type_safety",
            "ModelConfig",
            "TypedTrainingConfig",
            "TrajectoryData",
            "RewardMetrics",
            "DeviceType",
            "ModelSize",
            "TrainingStage",
        ],
    )
)
_LAZY_EXPORTS["TypedConversationTurn"] = (
    "stateset_agents.core.type_system",
    "ConversationTurnDict",
    None,
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.async_pool",
        [
            "AsyncResourcePool",
            "AsyncTaskManager",
            "PooledResource",
            "get_http_pool",
            "get_task_manager",
            "managed_async_resources",
        ],
        install_hint=_AIOHTTP_INSTALL_HINT,
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.data_processing",
        [
            "DataLoader",
            "DataProcessor",
            "ConversationExample",
            "load_and_prepare_data",
        ],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.training.trainer",
        ["GRPOTrainer", "MultiTurnGRPOTrainer"],
        install_hint=_TRAINING_INSTALL_HINT,
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.training.config",
        ["TrainingConfig", "TrainingProfile", "get_config_for_task"],
        install_hint=_TRAINING_INSTALL_HINT,
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.training.train",
        ["train", "AutoTrainer"],
        install_hint=_TRAINING_INSTALL_HINT,
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.training.auto_research",
        ["AutoResearchConfig", "AutoResearchLoop", "run_auto_research"],
        install_hint=_TRAINING_INSTALL_HINT,
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.utils.wandb_integration",
        ["WandBLogger", "init_wandb"],
        install_hint=_TRAINING_INSTALL_HINT,
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.data",
        [
            "ConversationDataset",
            "ConversationDatasetConfig",
            "ConversationReplayBuffer",
            "EmbeddingCache",
        ],
    )
)
__all__ = list(_LAZY_EXPORTS)

# Stable names the framework itself never uses. Deprecated in v0.70.0; per
# docs/API_COMPATIBILITY.md they stay for one further minor release and will be
# removed in v0.72.0. Each access warns (the value is not cached) and names the
# replacement; the provisional modules keep serving them until then.
_DEPRECATED_STABLE_EXPORTS_REMOVAL = "v0.72.0"
_DEPRECATED_STABLE_EXPORTS: dict[str, str] = {
    # error_handling: the framework raises GRPOException/ValidationException/
    # ResourceException; these are unused. ``ErrorCode`` here is the legacy
    # GRPO_E### enum (``exc.error_code``); every framework error's ``code`` is
    # a ``core.errors.ErrorCode`` (deprecated in v0.70.0).
    "ErrorCode": "stateset_agents.core.errors.ErrorCode",
    "TrainingException": "stateset_agents.core.errors.TrainingError",
    "ModelException": "stateset_agents.core.errors.ModelError",
    "DataException": "stateset_agents.core.errors.DataError",
    "NetworkException": "stateset_agents.core.errors.NetworkError",
    "ErrorHandler": "stateset_agents.core.error_handling.ErrorHandler",
    "RetryConfig": "stateset_agents.core.error_handling.RetryConfig",
    "CircuitBreakerConfig": "stateset_agents.core.error_handling.CircuitBreakerConfig",
    "retry_async": "stateset_agents.core.error_handling.retry_async",
    "circuit_breaker": "stateset_agents.core.error_handling.circuit_breaker",
    "handle_error": "stateset_agents.core.error_handling.handle_error",
    "get_error_summary": "stateset_agents.core.error_handling.get_error_summary",
    # performance_optimizer: used only by experimental modules.
    **{
        name: f"stateset_agents.core.performance_optimizer.{name}"
        for name in (
            "PerformanceOptimizer",
            "MemoryMonitor",
            "ModelOptimizer",
            "BatchOptimizer",
            "OptimizationLevel",
            "MemoryConfig",
            "ComputeConfig",
            "DataConfig",
            "PerformanceMetrics",
        )
    },
    # type_system: TypedDicts and validators nothing in the framework calls.
    **{
        name: f"stateset_agents.core.type_system.{name}"
        for name in (
            "TypeValidator",
            "ConfigValidator",
            "TypeSafeSerializer",
            "create_typed_config",
            "ensure_type_safety",
            "ensure_async_type_safety",
            "ModelConfig",
            "TypedTrainingConfig",
            "TrajectoryData",
            "RewardMetrics",
            "DeviceType",
            "ModelSize",
            "TrainingStage",
        )
    },
    "TypedConversationTurn": (
        "stateset_agents.ConversationTurn (or "
        "stateset_agents.core.type_system.ConversationTurnDict)"
    ),
}


def _build_import_error(
    name: str,
    module_name: str,
    install_hint: str | None,
) -> str:
    message = (
        f"{name} could not be imported from {module_name} because optional "
        "dependencies are unavailable or failed to initialize."
    )
    if install_hint:
        message += f" Install with: {install_hint}"
    return message


def _maybe_import_submodule(name: str) -> ModuleType | None:
    """Import real subpackages like ``stateset_agents.training`` on demand."""
    module_name = f"{__name__}.{name}"
    try:
        spec = find_spec(module_name)
    except OPTIONAL_IMPORT_EXCEPTIONS:
        spec = None

    if spec is None:
        return None

    module = import_module(module_name)
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        submodule = _maybe_import_submodule(name)
        if submodule is not None:
            return submodule
        raise AttributeError(f"module 'stateset_agents' has no attribute {name!r}")

    module_name, attr_name, install_hint = export
    replacement = _DEPRECATED_STABLE_EXPORTS.get(name)
    if replacement is not None:
        warn_deprecated(
            f"stateset_agents.{name}",
            removal=_DEPRECATED_STABLE_EXPORTS_REMOVAL,
            replacement=replacement,
        )

    try:
        module = import_module(module_name)
    except OPTIONAL_IMPORT_EXCEPTIONS as exc:
        raise ImportError(_build_import_error(name, module_name, install_hint)) from exc

    try:
        if replacement is None:
            value = getattr(module, attr_name)
        else:
            # Warned above; the provisional module may warn for the same name.
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", DeprecationWarning)
                value = getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"module '{module_name}' has no attribute {attr_name!r}"
        ) from exc

    if replacement is None:
        globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
