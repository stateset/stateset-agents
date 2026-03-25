"""
Adapters to wire LLM-as-Judge into training reward pipelines.

This module bridges the LLMJudge evaluation system with the framework's
reward interfaces (RewardFunction, MultiObjectiveRewardFunction), enabling
RLAIF training across all 5 algorithms.

Three adapter classes:
- LLMJudgeReward: RewardFunction adapter for GRPO single/multi-turn
- LLMJudgeRewardComponent: BaseRewardComponent for GSPO MultiObjective
- LLMJudgeRewardWithFallback: Composite that falls back to heuristics
  when no API key is available

Factory functions:
- create_rlaif_reward(): Build a composite LLM judge + heuristic reward
- create_rlaif_component(): Build a MultiObjective component
"""

from __future__ import annotations

import logging
from typing import Any

from stateset_agents.core.reward_base import (
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_query_and_response(
    turns: list[ConversationTurn],
    context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Extract the user query and assistant response from conversation turns."""
    query = ""
    response = ""

    # Try context first
    if context:
        query = str(context.get("user_query", context.get("prompt", "")))

    # Extract from turns
    for turn in reversed(turns):
        text = turn.content if isinstance(turn.content, str) else ""
        if turn.role == "assistant" and not response:
            response = text
        elif turn.role == "user" and not query:
            query = text

    return query or "Hello", response or ""


# ---------------------------------------------------------------------------
# RewardFunction adapter (GRPO single-turn / multi-turn)
# ---------------------------------------------------------------------------


class LLMJudgeReward(RewardFunction):
    """Wraps an LLMJudge as a RewardFunction for use with GRPO trainers.

    Example::

        from stateset_agents.rewards.llm_judge import LLMJudge
        reward = LLMJudgeReward(LLMJudge(provider="openai"))
        trainer = SingleTurnGRPOTrainer(agent, env, reward_fn=reward, ...)
    """

    def __init__(
        self,
        judge: Any,
        weight: float = 1.0,
        name: str = "LLMJudgeReward",
    ):
        super().__init__(weight=weight, reward_type=RewardType.IMMEDIATE, name=name)
        self.judge = judge

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        query, response = _extract_query_and_response(turns, context)

        if not response:
            return RewardResult(score=0.0, metadata={"error": "no response"})

        try:
            score = await self.judge.evaluate(query=query, response=response)
        except Exception as exc:
            logger.warning("LLMJudge evaluation failed: %s", exc)
            score = 0.5  # Neutral fallback

        return RewardResult(
            score=float(score),
            components={"llm_judge": float(score)},
            metadata={"judge_model": getattr(self.judge.config, "model_name", "unknown")},
        )


# ---------------------------------------------------------------------------
# BaseRewardComponent adapter (GSPO MultiObjectiveRewardFunction)
# ---------------------------------------------------------------------------


class LLMJudgeRewardComponent:
    """Wraps an LLMJudge as a BaseRewardComponent for MultiObjectiveRewardFunction.

    Example::

        from stateset_agents.rewards.multi_objective_reward import MultiObjectiveRewardFunction
        comp = LLMJudgeRewardComponent(judge, weight=0.5)
        reward = MultiObjectiveRewardFunction(components=[comp, ...])
    """

    def __init__(
        self,
        judge: Any,
        name: str = "llm_judge",
        weight: float = 0.5,
    ):
        self.judge = judge
        self.name = name
        self.weight = weight
        self.config: dict[str, Any] = {}

    async def compute_score(
        self,
        turns: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> float:
        """Compute score for MultiObjectiveRewardFunction."""
        query = ""
        response = ""

        for turn in reversed(turns):
            role = turn.get("role", "")
            content = str(turn.get("content", ""))
            if role == "assistant" and not response:
                response = content
            elif role == "user" and not query:
                query = content

        if context:
            query = query or str(context.get("user_query", context.get("prompt", "")))

        if not response:
            return 0.0

        try:
            return float(await self.judge.evaluate(query=query, response=response))
        except Exception as exc:
            logger.warning("LLMJudge component failed: %s", exc)
            return 0.5

    def get_info(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight, "config": self.config}


# ---------------------------------------------------------------------------
# Fallback composite: LLM judge + heuristic
# ---------------------------------------------------------------------------


class LLMJudgeRewardWithFallback(RewardFunction):
    """Composite reward that uses LLM judge when available, heuristic otherwise.

    This is the recommended reward for production RLAIF training:
    - When an API key is configured, uses LLM judge for semantic evaluation
    - When unavailable, gracefully falls back to HelpfulnessReward + SafetyReward
    - Always applies safety check via heuristic (fast, no API call)

    Example::

        reward = LLMJudgeRewardWithFallback()  # auto-detects API key
        trainer = SingleTurnGRPOTrainer(agent, env, reward_fn=reward, ...)
    """

    def __init__(
        self,
        judge: Any | None = None,
        judge_weight: float = 0.7,
        heuristic_weight: float = 0.3,
        name: str = "LLMJudgeRewardWithFallback",
    ):
        super().__init__(weight=1.0, reward_type=RewardType.IMMEDIATE, name=name)
        self.judge = judge
        self.judge_weight = judge_weight
        self.heuristic_weight = heuristic_weight
        self._judge_available = judge is not None

        # Lazy-init heuristic fallback
        self._heuristic: RewardFunction | None = None

    def _get_heuristic(self) -> RewardFunction:
        if self._heuristic is None:
            from stateset_agents.core.reward import (
                CompositeReward,
                HelpfulnessReward,
                SafetyReward,
            )

            self._heuristic = CompositeReward(
                reward_functions=[
                    HelpfulnessReward(weight=0.6),
                    SafetyReward(weight=0.4),
                ],
            )
        return self._heuristic

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        # Always run heuristic (fast, no API cost)
        heuristic_result = await self._get_heuristic().compute_reward(turns, context)
        heuristic_score = heuristic_result.score

        if self._judge_available and self.judge is not None:
            query, response = _extract_query_and_response(turns, context)
            try:
                judge_score = await self.judge.evaluate(query=query, response=response)
                combined = (
                    self.judge_weight * judge_score
                    + self.heuristic_weight * heuristic_score
                )
                return RewardResult(
                    score=combined,
                    components={
                        "llm_judge": judge_score,
                        "heuristic": heuristic_score,
                    },
                    metadata={"mode": "llm_judge"},
                )
            except Exception as exc:
                logger.warning("LLM judge failed, using heuristic: %s", exc)

        # Fallback to heuristic only
        return RewardResult(
            score=heuristic_score,
            components={"heuristic": heuristic_score},
            metadata={"mode": "heuristic_fallback"},
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_rlaif_reward(
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    criteria: list[str] | None = None,
    judge_weight: float = 0.7,
    heuristic_weight: float = 0.3,
    **kwargs: Any,
) -> LLMJudgeRewardWithFallback:
    """Create an RLAIF reward function with LLM judge and heuristic fallback.

    If no API key is available (from params or env), returns a heuristic-only
    reward that still works with all training algorithms.

    Args:
        provider: API provider (openai, anthropic, auto). Defaults to auto.
        model_name: Judge model name.
        api_key: API key. Falls back to env vars.
        criteria: Evaluation criteria names.
        judge_weight: Weight for LLM judge scores (0-1).
        heuristic_weight: Weight for heuristic scores (0-1).

    Returns:
        LLMJudgeRewardWithFallback compatible with all trainers.
    """
    try:
        from stateset_agents.rewards.llm_judge import (
            JudgeConfig,
            LLMJudge,
            EvaluationCriteria,
        )

        eval_criteria = None
        if criteria:
            eval_criteria = [
                EvaluationCriteria(c) if isinstance(c, str) else c
                for c in criteria
            ]

        config = JudgeConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            criteria=eval_criteria
            or [EvaluationCriteria.HELPFULNESS, EvaluationCriteria.CORRECTNESS],
            **kwargs,
        )

        # Check if API key was resolved
        if config.api_key:
            judge = LLMJudge(config)
            logger.info(
                "RLAIF reward: LLM judge active (provider=%s, model=%s)",
                config.provider,
                config.model_name,
            )
            return LLMJudgeRewardWithFallback(
                judge=judge,
                judge_weight=judge_weight,
                heuristic_weight=heuristic_weight,
            )
        else:
            logger.info("RLAIF reward: no API key found, using heuristic fallback")
            return LLMJudgeRewardWithFallback(judge=None)

    except Exception as exc:
        logger.warning("Failed to initialize LLM judge: %s — using heuristic", exc)
        return LLMJudgeRewardWithFallback(judge=None)


def create_rlaif_component(
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    weight: float = 0.5,
    **kwargs: Any,
) -> LLMJudgeRewardComponent | None:
    """Create an LLM judge component for MultiObjectiveRewardFunction.

    Returns None if no API key is available (caller should handle gracefully).
    """
    try:
        from stateset_agents.rewards.llm_judge import JudgeConfig, LLMJudge

        config = JudgeConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            **kwargs,
        )

        if not config.api_key:
            return None

        judge = LLMJudge(config)
        return LLMJudgeRewardComponent(judge=judge, weight=weight)

    except Exception as exc:
        logger.warning("Failed to create LLM judge component: %s", exc)
        return None


__all__ = [
    "LLMJudgeReward",
    "LLMJudgeRewardComponent",
    "LLMJudgeRewardWithFallback",
    "create_rlaif_reward",
    "create_rlaif_component",
]
