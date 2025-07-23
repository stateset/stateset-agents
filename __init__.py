"""
GRPO Agent Framework

A comprehensive framework for training multi-turn AI agents using 
Group Relative Policy Optimization (GRPO).
"""

__version__ = "0.3.0"
__author__ = "GRPO Framework Team"
__email__ = "team@grpo-framework.ai"

# Core exports
from .core.agent import Agent, MultiTurnAgent, ToolAgent
from .core.environment import Environment, ConversationEnvironment, TaskEnvironment
from .core.trajectory import Trajectory, MultiTurnTrajectory, ConversationTurn
from .core.reward import (
    RewardFunction, CompositeReward,
    # Pre-built rewards
    HelpfulnessReward, SafetyReward, CorrectnessReward,
    ConcisenessReward, EngagementReward, TaskCompletionReward,
    # Domain-specific rewards
    CustomerServiceReward, TechnicalSupportReward, SalesAssistantReward,
    # Enhanced rewards
    DomainSpecificReward, SimilarityAwareReward,
    # Factory functions
    create_domain_reward, create_adaptive_reward
)

# New enhanced features
from .core.error_handling import (
    GRPOException, TrainingException, ModelException, DataException,
    NetworkException, ResourceException, ValidationException,
    ErrorHandler, RetryConfig, CircuitBreakerConfig,
    retry_async, circuit_breaker, handle_error, get_error_summary
)
from .core.performance_optimizer import (
    PerformanceOptimizer, MemoryMonitor, ModelOptimizer, BatchOptimizer,
    OptimizationLevel, MemoryConfig, ComputeConfig, DataConfig, PerformanceMetrics
)
from .core.type_system import (
    TypeValidator, ConfigValidator, TypeSafeSerializer,
    create_typed_config, ensure_type_safety, ensure_async_type_safety,
    ModelConfig, TrainingConfig, ConversationTurn as TypedConversationTurn,
    TrajectoryData, RewardMetrics, DeviceType, ModelSize, TrainingStage
)
from .core.async_pool import (
    AsyncResourcePool, AsyncTaskManager, PooledResource, 
    get_http_pool, get_task_manager, managed_async_resources
)

# Data processing exports
from .core.data_processing import (
    DataLoader, DataProcessor, ConversationExample,
    load_and_prepare_data
)

# Training exports
from .training.trainer import GRPOTrainer, MultiTurnGRPOTrainer
from .training.config import TrainingConfig, TrainingProfile, get_config_for_task
from .training.train import train, AutoTrainer

# Utilities
from .utils.wandb_integration import WandBLogger, init_wandb

__all__ = [
    # Core classes
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
    
    # Reward functions
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
    
    # Enhanced error handling
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
    "DataLoader",
    "DataProcessor",
    "ConversationExample",
    "load_and_prepare_data",
    
    # Training
    "GRPOTrainer",
    "MultiTurnGRPOTrainer",
    "TrainingProfile",
    "get_config_for_task",
    "train",
    "AutoTrainer",
    
    # Utilities
    "WandBLogger",
    "init_wandb",
]