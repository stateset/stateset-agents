"""
Reward functions for the GRPO Agent Framework

This module provides various reward computation strategies:
- Rule-based rewards (RulerRewardFunction)
- Multi-objective rewards (MultiObjectiveRewardFunction)
- LLM-as-Judge rewards (LLMJudge) for RLAIF
- Domain-specific reward templates
"""

from .kimi_k25_reward import (
    CodeExecutionReward,
    LongContextReward,
    MultimodalConsistencyReward,
    ThinkingModeReward,
    create_kimi_k25_conversational_reward,
    create_kimi_k25_customer_service_reward,
)
from .llm_reward import RewardResult
from .multi_objective_reward import (
    MultiObjectiveRewardFunction,
    create_customer_service_reward,
    create_domain_reward,
    create_sales_reward,
    create_technical_support_reward,
)
from .ruler_reward import RulerRewardFunction
from .ruler_reward import RulerRewardFunction as LLMRewardFunction
from .ruler_reward import create_customer_service_ruler, create_general_ruler
from .symbolic_physics_reward import SymbolicPhysicsRewardFunction, SymbolicRewardConfig

# LLM-as-Judge for RLAIF
try:
    from .llm_judge import (
        EvaluationCriteria,
        JudgeConfig,
        JudgeProvider,
        LLMJudge,
        create_llm_judge_reward,
    )

    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLM_JUDGE_AVAILABLE = False

# LLM Judge training adapters (always available — gracefully degrade)
from .llm_judge_adapter import (
    LLMJudgeReward,
    LLMJudgeRewardComponent,
    LLMJudgeRewardWithFallback,
    create_rlaif_reward,
    create_rlaif_component,
)

__all__ = [
    "LLMRewardFunction",
    "RewardResult",
    "MultiObjectiveRewardFunction",
    "create_customer_service_reward",
    "create_domain_reward",
    "create_sales_reward",
    "create_technical_support_reward",
    "RulerRewardFunction",
    "create_customer_service_ruler",
    "create_general_ruler",
    "SymbolicPhysicsRewardFunction",
    "SymbolicRewardConfig",
    "ThinkingModeReward",
    "MultimodalConsistencyReward",
    "CodeExecutionReward",
    "LongContextReward",
    "create_kimi_k25_customer_service_reward",
    "create_kimi_k25_conversational_reward",
]

# Add LLM Judge exports if available
if LLM_JUDGE_AVAILABLE:
    __all__.extend(
        [
            "LLMJudge",
            "JudgeConfig",
            "JudgeProvider",
            "EvaluationCriteria",
            "create_llm_judge_reward",
        ]
    )

__all__.extend(
    [
        "LLMJudgeReward",
        "LLMJudgeRewardComponent",
        "LLMJudgeRewardWithFallback",
        "create_rlaif_reward",
        "create_rlaif_component",
    ]
)
