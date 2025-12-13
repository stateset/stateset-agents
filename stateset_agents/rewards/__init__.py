"""
Reward functions for the GRPO Agent Framework

This module provides various reward computation strategies:
- Rule-based rewards (RulerRewardFunction)
- Multi-objective rewards (MultiObjectiveRewardFunction)
- LLM-as-Judge rewards (LLMJudge) for RLAIF
- Domain-specific reward templates
"""

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

# LLM-as-Judge for RLAIF
try:
    from .llm_judge import (
        LLMJudge,
        JudgeConfig,
        JudgeProvider,
        EvaluationCriteria,
        create_llm_judge_reward,
    )
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLM_JUDGE_AVAILABLE = False

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
]

# Add LLM Judge exports if available
if LLM_JUDGE_AVAILABLE:
    __all__.extend([
        "LLMJudge",
        "JudgeConfig",
        "JudgeProvider",
        "EvaluationCriteria",
        "create_llm_judge_reward",
    ])
