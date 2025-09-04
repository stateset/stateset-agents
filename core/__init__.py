"""
Core components for the StateSet RL Agent Framework.
"""

# Agent components
from .agent import Agent, MultiTurnAgent, ToolAgent
from .environment import Environment, ConversationEnvironment, TaskEnvironment
from .trajectory import Trajectory, MultiTurnTrajectory, ConversationTurn

# Reward system
from .reward import (
    RewardFunction, CompositeReward,
    HelpfulnessReward, SafetyReward, CorrectnessReward,
    ConcisenessReward, EngagementReward, TaskCompletionReward,
    CustomerServiceReward, TechnicalSupportReward, SalesAssistantReward,
    DomainSpecificReward, SimilarityAwareReward,
    create_domain_reward, create_adaptive_reward
)

# Enhanced features
from .error_handling import (
    GRPOException, TrainingException, ModelException, DataException,
    NetworkException, ResourceException, ValidationException,
    ErrorHandler, RetryConfig, CircuitBreakerConfig,
    retry_async, circuit_breaker, handle_error, get_error_summary
)

from .performance_optimizer import (
    PerformanceOptimizer, MemoryMonitor, ModelOptimizer, BatchOptimizer,
    OptimizationLevel, MemoryConfig, ComputeConfig, DataConfig, PerformanceMetrics
)

from .type_system import (
    TypeValidator, ConfigValidator, TypeSafeSerializer,
    create_typed_config, ensure_type_safety, ensure_async_type_safety,
    ModelConfig, TrainingConfig, ConversationTurn as TypedConversationTurn,
    TrajectoryData, RewardMetrics, DeviceType, ModelSize, TrainingStage
)

from .async_pool import (
    AsyncResourcePool, AsyncTaskManager, PooledResource, 
    get_http_pool, get_task_manager, managed_async_resources
)

# Data processing
from .data_processing import (
    DataLoader, DataProcessor, ConversationExample,
    load_and_prepare_data
)

# Types
from .types import *

__all__ = [
    # Core classes
    "Agent", "MultiTurnAgent", "ToolAgent",
    "Environment", "ConversationEnvironment", "TaskEnvironment",
    "Trajectory", "MultiTurnTrajectory", "ConversationTurn",
    
    # Reward system
    "RewardFunction", "CompositeReward",
    "HelpfulnessReward", "SafetyReward", "CorrectnessReward",
    "ConcisenessReward", "EngagementReward", "TaskCompletionReward",
    "CustomerServiceReward", "TechnicalSupportReward", "SalesAssistantReward",
    "DomainSpecificReward", "SimilarityAwareReward",
    "create_domain_reward", "create_adaptive_reward",
    
    # Enhanced features
    "GRPOException", "TrainingException", "ModelException", "DataException",
    "NetworkException", "ResourceException", "ValidationException",
    "ErrorHandler", "RetryConfig", "CircuitBreakerConfig",
    "retry_async", "circuit_breaker", "handle_error", "get_error_summary",
    
    # Performance optimization
    "PerformanceOptimizer", "MemoryMonitor", "ModelOptimizer", "BatchOptimizer",
    "OptimizationLevel", "MemoryConfig", "ComputeConfig", "DataConfig", "PerformanceMetrics",
    
    # Type system
    "TypeValidator", "ConfigValidator", "TypeSafeSerializer",
    "create_typed_config", "ensure_type_safety", "ensure_async_type_safety",
    "ModelConfig", "TrainingConfig", "TypedConversationTurn",
    "TrajectoryData", "RewardMetrics", "DeviceType", "ModelSize", "TrainingStage",
    
    # Async resource management
    "AsyncResourcePool", "AsyncTaskManager", "PooledResource", 
    "get_http_pool", "get_task_manager", "managed_async_resources",
    
    # Data processing
    "DataLoader", "DataProcessor", "ConversationExample",
    "load_and_prepare_data",
]
