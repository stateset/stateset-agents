"""
GRPO Agent Framework

A comprehensive framework for training multi-turn AI agents using 
Group Relative Policy Optimization (GRPO).
"""

__version__ = "0.2.0"
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
    
    # Data processing
    "DataLoader",
    "DataProcessor",
    "ConversationExample",
    "load_and_prepare_data",
    
    # Training
    "GRPOTrainer",
    "MultiTurnGRPOTrainer",
    "TrainingConfig",
    "TrainingProfile",
    "get_config_for_task",
    "train",
    "AutoTrainer",
    
    # Utilities
    "WandBLogger",
    "init_wandb",
]