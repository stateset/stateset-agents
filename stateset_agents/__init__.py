"""
GRPO Agent Framework

A comprehensive framework for training multi-turn AI agents using
Group Relative Policy Optimization (GRPO).
"""

__version__ = "0.6.0"
__author__ = "StateSet Team"
__email__ = "team@stateset.ai"

# Core exports
try:
    from .core.agent import Agent, MultiTurnAgent, ToolAgent
except Exception:  # pragma: no cover - optional heavy dependencies
    Agent = None  # type: ignore
    MultiTurnAgent = None  # type: ignore
    ToolAgent = None  # type: ignore

try:
    from .core.environment import ConversationEnvironment, Environment, TaskEnvironment
except Exception:  # pragma: no cover
    ConversationEnvironment = None  # type: ignore
    Environment = None  # type: ignore
    TaskEnvironment = None  # type: ignore

# New enhanced features
from .core.error_handling import (
    CircuitBreakerConfig,
    DataException,
    ErrorHandler,
    GRPOException,
    ModelException,
    NetworkException,
    ResourceException,
    RetryConfig,
    TrainingException,
    ValidationException,
    circuit_breaker,
    get_error_summary,
    handle_error,
    retry_async,
)
from .core.reward import (  # Pre-built rewards; Domain-specific rewards; Enhanced rewards; Factory functions
    CompositeReward,
    ConcisenessReward,
    CorrectnessReward,
    CustomerServiceReward,
    DomainSpecificReward,
    EngagementReward,
    HelpfulnessReward,
    RewardFunction,
    SafetyReward,
    SalesAssistantReward,
    SimilarityAwareReward,
    TaskCompletionReward,
    TechnicalSupportReward,
    create_adaptive_reward,
    create_domain_reward,
)
from .core.trajectory import ConversationTurn, MultiTurnTrajectory, Trajectory

try:
    from .core.performance_optimizer import (
        BatchOptimizer,
        ComputeConfig,
        DataConfig,
        MemoryConfig,
        MemoryMonitor,
        ModelOptimizer,
        OptimizationLevel,
        PerformanceMetrics,
        PerformanceOptimizer,
    )
except Exception:  # pragma: no cover - optional heavy deps (torch/transformers)
    PerformanceOptimizer = None  # type: ignore
    MemoryMonitor = None  # type: ignore
    ModelOptimizer = None  # type: ignore
    BatchOptimizer = None  # type: ignore
    OptimizationLevel = None  # type: ignore
    MemoryConfig = None  # type: ignore
    ComputeConfig = None  # type: ignore
    DataConfig = None  # type: ignore
    PerformanceMetrics = None  # type: ignore
from .core.type_system import ConfigValidator
from .core.type_system import ConversationTurn as TypedConversationTurn
from .core.type_system import (
    DeviceType,
    ModelConfig,
    ModelSize,
    RewardMetrics,
    TrainingConfig,
    TrainingStage,
    TrajectoryData,
    TypeSafeSerializer,
    TypeValidator,
    create_typed_config,
    ensure_async_type_safety,
    ensure_type_safety,
)

try:
    from .core.async_pool import (
        AsyncResourcePool,
        AsyncTaskManager,
        PooledResource,
        get_http_pool,
        get_task_manager,
        managed_async_resources,
    )
except Exception:  # pragma: no cover - optional aiohttp
    AsyncResourcePool = None  # type: ignore
    AsyncTaskManager = None  # type: ignore
    PooledResource = None  # type: ignore
    get_http_pool = None  # type: ignore
    get_task_manager = None  # type: ignore
    managed_async_resources = None  # type: ignore

# Data processing exports
try:
    from .core.data_processing import (
        ConversationExample,
        DataLoader,
        DataProcessor,
        load_and_prepare_data,
    )
except Exception:  # pragma: no cover
    ConversationExample = None  # type: ignore
    DataLoader = None  # type: ignore
    DataProcessor = None  # type: ignore
    load_and_prepare_data = None  # type: ignore

# Training exports (optional to avoid import issues)
try:
    from .training.trainer import GRPOTrainer, MultiTurnGRPOTrainer
except Exception:  # pragma: no cover - allow import without training deps
    GRPOTrainer = None
    MultiTurnGRPOTrainer = None

try:
    from .training.config import (
        TrainingConfig as TrainerTrainingConfig,
        TrainingProfile,
        get_config_for_task,
    )
    from .training.train import AutoTrainer, train
    TrainingConfig = TrainerTrainingConfig
except Exception:
    TrainingProfile = None
    get_config_for_task = None
    train = None
    AutoTrainer = None

# Utilities
try:
    from .utils.wandb_integration import WandBLogger, init_wandb
except Exception:
    WandBLogger = None
    init_wandb = None

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
    # Performance optimization
    "PerformanceOptimizer",
    "MemoryMonitor",
    "ModelOptimizer",
    "BatchOptimizer",
    "OptimizationLevel",
    "MemoryConfig",
    "ComputeConfig",
    "DataConfig",
    "PerformanceMetrics",
    # Type system
    "TypeValidator",
    "ConfigValidator",
    "TypeSafeSerializer",
    "create_typed_config",
    "ensure_type_safety",
    "ensure_async_type_safety",
    "ModelConfig",
    "TrainingConfig",
    "TypedConversationTurn",
    "TrajectoryData",
    "RewardMetrics",
    "DeviceType",
    "ModelSize",
    "TrainingStage",
    # Async resource management
    "AsyncResourcePool",
    "AsyncTaskManager",
    "PooledResource",
    "get_http_pool",
    "get_task_manager",
    "managed_async_resources",
    # Data processing
    "DataLoader",
    "DataProcessor",
    "ConversationExample",
    "load_and_prepare_data",
    # Training (optional)
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
