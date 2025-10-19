"""
RULER (Reinforcement Learning with User Response Evaluation) Reward Functions
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

from stateset_agents.core import reward as core_reward

RewardFunction = core_reward.RewardFunction
from .llm_reward import RewardResult

logger = logging.getLogger(__name__)


@dataclass
class RulerConfig:
    """Configuration for RULER reward function"""

    model: str = "openai/gpt-4"
    temperature: float = 0.3
    max_tokens: int = 500
    weight: float = 1.0
    fallback_enabled: bool = True
    fallback_score: float = 0.5


class RulerRewardFunction(RewardFunction):
    """
    RULER reward function using LLM as a judge for conversational quality
    """

    def __init__(self, config: RulerConfig):
        super().__init__(weight=config.weight, name="RulerReward")
        self.config = config
        self.fallback_enabled = config.fallback_enabled
        self.fallback_score = config.fallback_score

    async def compute_reward(
        self, turns: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """
        Compute RULER reward using LLM evaluation
        """
        try:
            # Build conversation context
            conversation_text = self._format_conversation(turns)

            # Create evaluation prompt
            eval_prompt = self._create_evaluation_prompt(conversation_text, context)

            # Get LLM evaluation
            evaluation = await self._get_llm_evaluation(eval_prompt)

            # Parse evaluation result
            score, reasoning = self._parse_evaluation(evaluation)

            return RewardResult(
                score=score * self.weight,
                breakdown={
                    "ruler_score": score,
                    "reasoning": reasoning,
                    "model": self.config.model,
                    "evaluation": evaluation,
                },
            )

        except Exception as e:
            logger.error(f"RULER evaluation failed: {e}")
            if self.fallback_enabled:
                logger.info("Using fallback score for RULER evaluation")
                return RewardResult(
                    score=self.fallback_score * self.weight,
                    breakdown={
                        "error": str(e),
                        "fallback_used": True,
                        "fallback_score": self.fallback_score,
                    },
                )
            else:
                raise

    def _format_conversation(self, turns: List[Dict[str, str]]) -> str:
        """Format conversation turns for evaluation"""
        formatted = []
        for turn in turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            formatted.append(f"{role.title()}: {content}")
        return "\n".join(formatted)

    def _create_evaluation_prompt(
        self, conversation: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create evaluation prompt for LLM judge"""

        domain = context.get("topic", "general") if context else "general"

        prompt = f"""
You are an expert evaluator of conversational AI quality. Evaluate the following conversation on a scale from 0.0 to 1.0, where:

- 1.0 = Perfect response: Highly relevant, helpful, natural, and contextually appropriate
- 0.8 = Very good response: Relevant and helpful with minor issues
- 0.6 = Good response: Generally appropriate but could be improved
- 0.4 = Fair response: Somewhat relevant but has notable issues
- 0.2 = Poor response: Minimally relevant or helpful
- 0.0 = Very poor response: Irrelevant, unhelpful, or inappropriate

Domain: {domain}

Conversation:
{conversation}

Please provide your evaluation in the following format:
SCORE: [0.0-1.0]
REASONING: [Brief explanation of your evaluation]
"""
        return prompt.strip()

    async def _get_llm_evaluation(self, prompt: str) -> str:
        """Get evaluation from LLM"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available for RULER evaluation")

        # This is a placeholder - actual OpenAI API call would go here
        # For now, return a mock response
        return "SCORE: 0.8\nREASONING: The response is relevant and helpful, showing good understanding of the user's query."

    def _parse_evaluation(self, evaluation: str) -> tuple:
        """Parse LLM evaluation response"""
        lines = evaluation.strip().split("\n")
        score = 0.5  # default
        reasoning = "No reasoning provided"

        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))  # clamp to [0,1]
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse score from: {line}")
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return score, reasoning


def create_customer_service_ruler(
    model: str = "openai/gpt-4", weight: float = 0.3, fallback_enabled: bool = True
) -> RulerRewardFunction:
    """
    Create a RULER reward function optimized for customer service conversations
    """
    config = RulerConfig(
        model=model,
        weight=weight,
        fallback_enabled=fallback_enabled,
        temperature=0.2,  # Lower temperature for more consistent evaluations
    )

    return RulerRewardFunction(config)


def create_general_ruler(
    model: str = "openai/gpt-4", weight: float = 0.25, fallback_enabled: bool = True
) -> RulerRewardFunction:
    """
    Create a general-purpose RULER reward function
    """
    config = RulerConfig(model=model, weight=weight, fallback_enabled=fallback_enabled)

    return RulerRewardFunction(config)
