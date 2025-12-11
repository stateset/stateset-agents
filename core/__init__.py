"""
Core components for the StateSet RL Agent Framework.

Note: the canonical import path is now ``stateset_agents.core``. Importing
from ``core`` continues to work for backwards compatibility but will be removed
in a future release.
"""

import os
import warnings

# Optional deprecation warning for legacy imports.
# Set STATESET_AGENTS_ENABLE_CORE_DEPRECATION=1 to re-enable.
if os.getenv("STATESET_AGENTS_ENABLE_CORE_DEPRECATION") == "1":
    warnings.warn(
        "Importing from the top-level 'core' package is deprecated; use "
        "'stateset_agents.core' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

# Agent components
from .agent import Agent, AgentConfig, ConfigValidationError, MultiTurnAgent, ToolAgent

# Structured output support
from .structured_output import (
    StructuredOutputConfig,
    StructuredOutputError,
    StructuredOutputMixin,
    create_structured_agent_class,
    extract_json_from_response,
    json_schema_from_type,
    repair_json_string,
)

# OpenAI-compatible function calling
from .function_calling import (
    FunctionCallingMixin,
    FunctionDefinition,
    FunctionParameter,
    ToolCall,
    ToolChoiceMode,
    ToolResult,
    create_function_calling_agent_class,
    tool,
)

# Input validation and security
from .input_validation import (
    RiskSeverity,
    SecureInputValidator,
    SecurityConfig,
    SecurityRisk,
    SecurityThreat,
    ValidationResult,
    create_secure_agent_wrapper,
)
from .async_pool import (
    AsyncResourcePool,
    AsyncTaskManager,
    PooledResource,
    get_http_pool,
    get_task_manager,
    managed_async_resources,
)

# Data processing
from .data_processing import (
    ConversationExample,
    DataLoader,
    DataProcessor,
    load_and_prepare_data,
)
from .environment import ConversationEnvironment, Environment, TaskEnvironment

# Enhanced features
from .error_handling import (
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
from .performance_optimizer import (
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

# Reward system
from .reward import (
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

# Value function for advantage estimation
from .value_function import (
    ValueFunction,
    ValueHead,
    create_value_function,
)
from .trajectory import ConversationTurn, MultiTurnTrajectory, Trajectory
from .type_system import ConfigValidator
from .type_system import ConversationTurn as TypedConversationTurn
from .type_system import (
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

# Types
from .types import *

__all__ = [
    # Core classes
    "Agent",
    "AgentConfig",
    "ConfigValidationError",
    "MultiTurnAgent",
    "ToolAgent",
    # Structured output
    "StructuredOutputConfig",
    "StructuredOutputError",
    "StructuredOutputMixin",
    "create_structured_agent_class",
    "json_schema_from_type",
    "repair_json_string",
    "extract_json_from_response",
    # Function calling
    "FunctionCallingMixin",
    "FunctionDefinition",
    "FunctionParameter",
    "ToolCall",
    "ToolChoiceMode",
    "ToolResult",
    "create_function_calling_agent_class",
    "tool",
    # Input validation
    "SecureInputValidator",
    "SecurityConfig",
    "SecurityRisk",
    "RiskSeverity",
    "SecurityThreat",
    "ValidationResult",
    "create_secure_agent_wrapper",
    "Environment",
    "ConversationEnvironment",
    "TaskEnvironment",
    "Trajectory",
    "MultiTurnTrajectory",
    "ConversationTurn",
    # Reward system
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
    # Value function
    "ValueFunction",
    "ValueHead",
    "create_value_function",
    # Enhanced features
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
]
