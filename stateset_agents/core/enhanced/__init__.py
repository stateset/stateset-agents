"""
Enhanced StateSet Agents Framework

This module contains advanced enhancements to the core framework including:

- Enhanced Agent Architecture with memory, reasoning, and dynamic personas
- Advanced RL Algorithms (PPO, DPO, A2C) with automatic algorithm selection
- Comprehensive Evaluation Framework with automated testing
- Production-ready features and monitoring capabilities

All components are designed to work seamlessly with the existing framework
while providing significant improvements in capabilities and performance.
"""

from .advanced_evaluation import (
    AdvancedEvaluator,
    AutomatedTestSuite,
    ComparativeAnalyzer,
    ContinuousMonitor,
    EvaluationMetrics,
    EvaluationResult,
    create_evaluation_config,
    quick_agent_comparison,
)
from .advanced_rl_algorithms import (
    A2CTrainer,
    AdvancedRLOrchestrator,
    DPOTrainer,
    PPOTrainer,
    create_a2c_trainer,
    create_advanced_rl_orchestrator,
    create_dpo_trainer,
    create_ppo_trainer,
)
from .enhanced_agent import (
    EnhancedMultiTurnAgent,
    PersonaProfile,
    ReasoningEngine,
    VectorMemory,
    create_domain_specific_agent,
    create_enhanced_agent,
)

__all__ = [
    # Enhanced Agents
    "EnhancedMultiTurnAgent",
    "VectorMemory",
    "ReasoningEngine",
    "PersonaProfile",
    "create_enhanced_agent",
    "create_domain_specific_agent",
    # Advanced RL
    "PPOTrainer",
    "DPOTrainer",
    "A2CTrainer",
    "AdvancedRLOrchestrator",
    "create_ppo_trainer",
    "create_dpo_trainer",
    "create_a2c_trainer",
    "create_advanced_rl_orchestrator",
    # Advanced Evaluation
    "AdvancedEvaluator",
    "AutomatedTestSuite",
    "ComparativeAnalyzer",
    "ContinuousMonitor",
    "EvaluationMetrics",
    "EvaluationResult",
    "create_evaluation_config",
    "quick_agent_comparison",
]

__version__ = "0.5.0"
