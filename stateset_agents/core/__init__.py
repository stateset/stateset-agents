"""
Core components for the StateSet RL Agent Framework.

Note: the canonical import path is now ``stateset_agents.core``. Importing
from ``core`` continues to work for backwards compatibility but will be removed
in a future release.
"""

from __future__ import annotations

import os
import warnings
from importlib import import_module
from importlib.util import find_spec
from typing import Any

# Optional deprecation warning for legacy imports.
# Set STATESET_AGENTS_ENABLE_CORE_DEPRECATION=1 to re-enable.
if os.getenv("STATESET_AGENTS_ENABLE_CORE_DEPRECATION") == "1":
    warnings.warn(
        "Importing from the top-level 'core' package is deprecated; use "
        "'stateset_agents.core' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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
    return {
        name: (module_name, name, install_hint)
        for name in names
    }


_LAZY_EXPORTS: dict[str, tuple[str, str, str | None]] = {}

_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.agent",
        ["Agent", "MultiTurnAgent", "ToolAgent"],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.agent_config",
        ["AgentConfig", "ConfigValidationError"],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.structured_output",
        [
            "StructuredOutputConfig",
            "StructuredOutputError",
            "StructuredOutputMixin",
            "create_structured_agent_class",
            "json_schema_from_type",
            "repair_json_string",
            "extract_json_from_response",
        ],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.function_calling",
        [
            "FunctionCallingMixin",
            "FunctionDefinition",
            "FunctionParameter",
            "ToolCall",
            "ToolChoiceMode",
            "ToolResult",
            "create_function_calling_agent_class",
            "tool",
        ],
    )
)
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.input_validation",
        [
            "SecureInputValidator",
            "SecurityConfig",
            "SecurityRisk",
            "RiskSeverity",
            "SecurityThreat",
            "ValidationResult",
            "create_secure_agent_wrapper",
        ],
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
        "stateset_agents.core.long_term_planning",
        [
            "PlanningConfig",
            "PlanningManager",
            "Planner",
            "Plan",
            "PlanStep",
            "PlanStatus",
            "HeuristicPlanner",
        ],
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
_LAZY_EXPORTS.update(
    _export_group(
        "stateset_agents.core.value_function",
        ["ValueFunction", "ValueHead", "create_value_function"],
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
            "TrainingConfig",
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
    "ConversationTurn",
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

__all__ = list(_LAZY_EXPORTS)


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


def _maybe_import_submodule(name: str) -> Any | None:
    """Import real submodules like ``stateset_agents.core.enhanced_state_management``."""
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
        module = _maybe_import_submodule(name)
        if module is not None:
            return module
        raise AttributeError(f"module 'stateset_agents.core' has no attribute {name!r}")

    module_name, attr_name, install_hint = export

    try:
        module = import_module(module_name)
    except OPTIONAL_IMPORT_EXCEPTIONS as exc:
        raise ImportError(
            _build_import_error(name, module_name, install_hint)
        ) from exc

    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"module '{module_name}' has no attribute {attr_name!r}"
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
