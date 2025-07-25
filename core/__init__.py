"""
Core abstractions for the GRPO Agent Framework
"""

from .agent import Agent, MultiTurnAgent, ToolAgent
from .environment import Environment, ConversationEnvironment, TaskEnvironment
from .trajectory import Trajectory, MultiTurnTrajectory, ConversationTurn
from .reward import RewardFunction, CompositeReward

# Enhanced framework components
from .error_handling import ErrorHandler, retry_async, RetryConfig
from .performance_optimizer import PerformanceOptimizer, OptimizationLevel
from .type_system import TypeValidator, create_typed_config
from .async_pool import AsyncResourcePool, managed_async_resources
from .advanced_monitoring import get_monitoring_service, monitor_async_function
from .enhanced_state_management import get_state_service

# Advanced AI capabilities
from .adaptive_learning_controller import (
    AdaptiveLearningController, 
    create_adaptive_learning_controller,
    CurriculumStrategy,
    ExplorationStrategy
)
from .neural_architecture_search import (
    NeuralArchitectureSearch,
    create_nas_controller,
    ArchitectureConfig,
    SearchStrategy
)
from .multimodal_processing import (
    MultimodalProcessor,
    create_multimodal_processor,
    ModalityInput,
    ModalityType,
    create_modality_input,
    FusionStrategy
)
from .intelligent_orchestrator import (
    IntelligentOrchestrator,
    create_intelligent_orchestrator,
    OrchestrationConfig,
    OrchestrationMode,
    OptimizationObjective
)

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
    
    # Advanced AI capabilities
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