"""
Core abstractions for the GRPO Agent Framework
"""

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
from .async_pool import AsyncResourcePool, managed_async_resources
from .advanced_monitoring import get_monitoring_service, monitor_async_function
from .enhanced_state_management import get_state_service

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