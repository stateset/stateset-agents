"""
Compatibility bridge exposing the top-level ``core`` package through the
``stateset_agents.core`` namespace without mutating ``sys.path`` or aliasing
the entire package.

The goal is to keep backwards compatibility for ``stateset_agents.core.*``
imports while gradually consolidating implementations under the
``stateset_agents`` namespace.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Iterable, Optional

_CORE_NAMESPACE = "core"
_SUBMODULES: Iterable[str] = (
    "agent",
    "agent_backends",
    "environment",
    "trajectory",
    "reward",
    "async_pool",
    "computational_engine",
    "data_processing",
    "error_handling",
    "performance_optimizer",
    "type_system",
    "multiturn_agent",
    "advanced_monitoring",
    "enhanced_state_management",
    "intelligent_orchestrator",
    "multimodal_processing",
    "adaptive_learning_controller",
    "neural_architecture_search",
    "enhanced",
    "enhanced.enhanced_agent",
    "enhanced.advanced_evaluation",
    "enhanced.advanced_rl_algorithms",
)


def _import_core_submodule(name: str) -> Optional[ModuleType]:
    """Attempt to import a submodule from the top-level ``core`` package."""
    full_name = f"{_CORE_NAMESPACE}.{name}"
    try:
        module = importlib.import_module(full_name)
    except Exception:
        return None
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __getattr__(name: str):
    """Lazy attribute access for direct ``core`` submodules."""
    module = _import_core_submodule(name)
    if module is not None:
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Prime common submodules so dotted imports continue to work.
for _submodule in _SUBMODULES:
    _import_core_submodule(_submodule)

# Re-export frequently used symbols with compatibility guards.
try:
    from core.agent import Agent, MultiTurnAgent, ToolAgent
except Exception:  # pragma: no cover - optional heavy deps
    Agent = None  # type: ignore
    MultiTurnAgent = None  # type: ignore
    ToolAgent = None  # type: ignore

try:
    from core.environment import ConversationEnvironment, Environment, TaskEnvironment
except Exception:  # pragma: no cover
    ConversationEnvironment = None  # type: ignore
    Environment = None  # type: ignore
    TaskEnvironment = None  # type: ignore

try:
    from core.reward import CompositeReward, RewardFunction
except Exception:  # pragma: no cover
    CompositeReward = None  # type: ignore
    RewardFunction = None  # type: ignore

try:
    from core.trajectory import ConversationTurn, MultiTurnTrajectory, Trajectory
except Exception:  # pragma: no cover
    ConversationTurn = None  # type: ignore
    MultiTurnTrajectory = None  # type: ignore
    Trajectory = None  # type: ignore

try:
    from core.error_handling import ErrorHandler, RetryConfig, retry_async
except Exception:  # pragma: no cover
    ErrorHandler = RetryConfig = retry_async = None  # type: ignore

try:
    from core.performance_optimizer import OptimizationLevel, PerformanceOptimizer
except Exception:  # pragma: no cover
    OptimizationLevel = PerformanceOptimizer = None  # type: ignore

try:
    from core.type_system import TypeValidator, create_typed_config
except Exception:  # pragma: no cover
    TypeValidator = create_typed_config = None  # type: ignore

try:
    from core.async_pool import AsyncResourcePool, managed_async_resources
except Exception:  # pragma: no cover
    AsyncResourcePool = managed_async_resources = None  # type: ignore

try:
    from core.advanced_monitoring import get_monitoring_service, monitor_async_function
except Exception:  # pragma: no cover
    get_monitoring_service = monitor_async_function = None  # type: ignore

try:
    from core.enhanced_state_management import get_state_service
except Exception:  # pragma: no cover
    get_state_service = None  # type: ignore

try:
    from core.adaptive_learning_controller import (
        AdaptiveLearningController,
        CurriculumStrategy,
        ExplorationStrategy,
        create_adaptive_learning_controller,
    )
except Exception:  # pragma: no cover
    AdaptiveLearningController = None  # type: ignore
    CurriculumStrategy = None  # type: ignore
    ExplorationStrategy = None  # type: ignore
    create_adaptive_learning_controller = None  # type: ignore

try:
    from core.neural_architecture_search import (
        ArchitectureConfig,
        NeuralArchitectureSearch,
        SearchStrategy,
        create_nas_controller,
    )
except Exception:  # pragma: no cover
    ArchitectureConfig = NeuralArchitectureSearch = None  # type: ignore
    SearchStrategy = create_nas_controller = None  # type: ignore

try:
    from core.multimodal_processing import (
        FusionStrategy,
        ModalityInput,
        ModalityType,
        MultimodalProcessor,
        create_modality_input,
        create_multimodal_processor,
    )
except Exception:  # pragma: no cover
    FusionStrategy = ModalityInput = None  # type: ignore
    ModalityType = MultimodalProcessor = None  # type: ignore
    create_modality_input = create_multimodal_processor = None  # type: ignore

try:
    from core.intelligent_orchestrator import (
        IntelligentOrchestrator,
        OptimizationObjective,
        OrchestrationConfig,
        OrchestrationMode,
        create_intelligent_orchestrator,
    )
except Exception:  # pragma: no cover
    IntelligentOrchestrator = None  # type: ignore
    OptimizationObjective = None  # type: ignore
    OrchestrationConfig = None  # type: ignore
    OrchestrationMode = None  # type: ignore
    create_intelligent_orchestrator = None  # type: ignore

__all__ = [
    "Agent",
    "MultiTurnAgent",
    "ToolAgent",
    "Environment",
    "ConversationEnvironment",
    "TaskEnvironment",
    "Trajectory",
    "MultiTurnTrajectory",
    "ConversationTurn",
    "RewardFunction",
    "CompositeReward",
    "ErrorHandler",
    "RetryConfig",
    "retry_async",
    "PerformanceOptimizer",
    "OptimizationLevel",
    "TypeValidator",
    "create_typed_config",
    "AsyncResourcePool",
    "managed_async_resources",
    "get_monitoring_service",
    "monitor_async_function",
    "get_state_service",
    "AdaptiveLearningController",
    "create_adaptive_learning_controller",
    "CurriculumStrategy",
    "ExplorationStrategy",
    "NeuralArchitectureSearch",
    "create_nas_controller",
    "ArchitectureConfig",
    "SearchStrategy",
    "MultimodalProcessor",
    "create_multimodal_processor",
    "ModalityInput",
    "ModalityType",
    "create_modality_input",
    "FusionStrategy",
    "IntelligentOrchestrator",
    "create_intelligent_orchestrator",
    "OrchestrationConfig",
    "OrchestrationMode",
    "OptimizationObjective",
]
