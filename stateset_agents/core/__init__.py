"""
Proxy module for `stateset_agents.core`.

This forwards imports to the top-level `core` package so users can
import `stateset_agents.core.*` without us physically moving files.
"""

from importlib import import_module as _import_module
import sys as _sys
from pathlib import Path as _Path

# Prefer the sibling/top-level 'core' package that ships with this distribution
# by pushing its parent directory to the front of sys.path. This avoids picking
# up unrelated 'core' packages that may exist earlier on sys.path in some
# environments (e.g., monorepos or notebooks).
try:
    _root_dir = _Path(__file__).resolve().parents[2]
    _root_str = str(_root_dir)
    if _root_str not in _sys.path:
        _sys.path.insert(0, _root_str)
except Exception:
    pass

# Load underlying top-level package
_core_pkg = _import_module('core')

# Re-export common submodules for dotted imports
_submodules = (
    'agent', 'environment', 'trajectory', 'reward', 'async_pool',
    'computational_engine', 'data_processing', 'error_handling',
    'performance_optimizer', 'type_system', 'multiturn_agent',
    'advanced_monitoring', 'enhanced_state_management',
    'intelligent_orchestrator', 'multimodal_processing',
    'adaptive_learning_controller', 'neural_architecture_search'
)

for _name in _submodules:
    try:
        _mod = _import_module(f'core.{_name}')
        _sys.modules[__name__ + f'.{_name}'] = _mod
    except Exception:
        pass

# Alias this package to the underlying one for attribute access
_sys.modules[__name__] = _core_pkg

# Core components that do not hard-require torch at import time
from .trajectory import Trajectory, MultiTurnTrajectory, ConversationTurn
from .reward import RewardFunction, CompositeReward
from .environment import Environment, ConversationEnvironment, TaskEnvironment

# Delay agent import to avoid torch requirement if importing just the package
try:
    from .agent import Agent, MultiTurnAgent, ToolAgent
except Exception:  # pragma: no cover
    Agent = None
    MultiTurnAgent = None
    ToolAgent = None

# Enhanced framework components (lightweight)
from .error_handling import ErrorHandler, retry_async, RetryConfig
try:
    from .performance_optimizer import PerformanceOptimizer, OptimizationLevel
except Exception:  # pragma: no cover
    PerformanceOptimizer = None
    OptimizationLevel = None
from .type_system import TypeValidator, create_typed_config
try:
    # Optional: requires aiohttp
    from .async_pool import AsyncResourcePool, managed_async_resources
except Exception:  # pragma: no cover
    AsyncResourcePool = None  # type: ignore
    managed_async_resources = None  # type: ignore
try:
    from .advanced_monitoring import get_monitoring_service, monitor_async_function
except Exception:  # pragma: no cover
    get_monitoring_service = None  # type: ignore
    monitor_async_function = None  # type: ignore
try:
    from .enhanced_state_management import get_state_service
except Exception:  # pragma: no cover
    get_state_service = None  # type: ignore

# Advanced AI capabilities (wrap in try to avoid optional deps at import)
try:
    from .adaptive_learning_controller import (
        AdaptiveLearningController, 
        create_adaptive_learning_controller,
        CurriculumStrategy,
        ExplorationStrategy
    )
except Exception:  # pragma: no cover
    AdaptiveLearningController = None
    create_adaptive_learning_controller = None
    CurriculumStrategy = None
    ExplorationStrategy = None

try:
    from .neural_architecture_search import (
        NeuralArchitectureSearch,
        create_nas_controller,
        ArchitectureConfig,
        SearchStrategy
    )
except Exception:  # pragma: no cover
    NeuralArchitectureSearch = None
    create_nas_controller = None
    ArchitectureConfig = None
    SearchStrategy = None

try:
    from .multimodal_processing import (
        MultimodalProcessor,
        create_multimodal_processor,
        ModalityInput,
        ModalityType,
        create_modality_input,
        FusionStrategy
    )
except Exception:  # pragma: no cover
    MultimodalProcessor = None
    create_multimodal_processor = None
    ModalityInput = None
    ModalityType = None
    create_modality_input = None
    FusionStrategy = None

try:
    from .intelligent_orchestrator import (
        IntelligentOrchestrator,
        create_intelligent_orchestrator,
        OrchestrationConfig,
        OrchestrationMode,
        OptimizationObjective
    )
except Exception:  # pragma: no cover
    IntelligentOrchestrator = None
    create_intelligent_orchestrator = None
    OrchestrationConfig = None
    OrchestrationMode = None
    OptimizationObjective = None

__all__ = [
    # Core framework
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
    
    # Enhanced components
    "ErrorHandler",
    "retry_async",
    "RetryConfig",
    "PerformanceOptimizer",
    "OptimizationLevel", 
    "TypeValidator",
    "create_typed_config",
    "AsyncResourcePool",
    "managed_async_resources",
    "get_monitoring_service",
    "monitor_async_function",
    "get_state_service",
    
    # Advanced AI capabilities (may be None if optional deps missing)
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
