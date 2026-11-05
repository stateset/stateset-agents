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

import inspect
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from stateset_agents.core.reward_base import (
    RewardFunction,
    RewardResult,
    RewardType,
    RewardUnavailableError,
)
from stateset_agents.core.reward_context import REWARD_FALLBACK_KEY, reward_fallback
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.llm_judge import JudgeConversation, judge_conversation

logger = logging.getLogger(__name__)

#: What building a judge can raise when it cannot be configured: a missing
#: client package, an unknown provider or criterion, a bad keyword. The
#: factories fall back to the heuristic on these only; anything else is a
#: bug and propagates.
_JUDGE_SETUP_ERRORS = (ImportError, ValueError, TypeError, OSError)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_query_and_response(
    turns: list[ConversationTurn],
    context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """The user query and assistant response a judge sees.

    The response is the last assistant turn; the query is the last user turn
    before it, and ``context["user_query"]`` (else ``context["prompt"]``) only
    when no user turn precedes the response. See ``judge_conversation``,
    which also returns the earlier turns.
    """
    conversation = judge_conversation(turns, context)
    return conversation.query, conversation.response


#: Context keys that hold a scenario's reference answer, in order of
#: preference (each also looked up in a nested ``scenario``).
REFERENCE_KEYS = (
    "reference",
    "reference_answer",
    "reference_response",
    "expected_response",
    "ideal_response",
    "gold_response",
    "ground_truth",
)


def scenario_reference(context: Mapping[str, Any] | None) -> str | None:
    """The scenario's reference answer, if the context carries one.

    Read from ``REFERENCE_KEYS`` at the top level or in ``context["scenario"]``;
    a ``ground_truth`` mapping contributes its ``"answer"`` (or
    ``"reference"``). Returns ``None`` when there is none.
    """
    if not isinstance(context, Mapping):
        return None
    sources = [context]
    if isinstance(context.get("scenario"), Mapping):
        sources.append(context["scenario"])
    for key in REFERENCE_KEYS:
        for source in sources:
            value = source.get(key)
            if isinstance(value, Mapping):
                value = value.get("answer", value.get("reference"))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _accepts(judge: Any, name: str) -> bool:
    """Whether ``judge.evaluate`` takes a ``name`` keyword."""
    try:
        parameters = inspect.signature(judge.evaluate).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    return name in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def _accepts_reference(judge: Any) -> bool:
    return _accepts(judge, "reference")


#: Judge types already warned that they cannot be shown earlier turns.
_WARNED_NO_HISTORY: set[type] = set()


async def judge_score(
    judge: Any,
    query: str,
    response: str,
    context: Mapping[str, Any] | None,
    history: Sequence[Mapping[str, str]] | None = None,
) -> float:
    """Ask ``judge`` for a verdict, showing it the scenario's reference
    answer when the context carries one, and the earlier turns of the
    conversation (``history``), when the judge accepts them."""
    kwargs: dict[str, Any] = {}
    reference = scenario_reference(context)
    if reference is not None and _accepts_reference(judge):
        kwargs["reference"] = reference
    if history:
        if _accepts(judge, "history"):
            kwargs["history"] = list(history)
        elif type(judge) not in _WARNED_NO_HISTORY:
            _WARNED_NO_HISTORY.add(type(judge))
            logger.warning(
                "%s.evaluate takes no 'history'; it judges multi-turn replies "
                "without the earlier turns",
                type(judge).__name__,
            )
    verdict = await judge.evaluate(query=query, response=response, **kwargs)
    return _checked_judge_score(verdict)


async def judge_conversation_score(
    judge: Any, conversation: JudgeConversation, context: Mapping[str, Any] | None
) -> float:
    """:func:`judge_score` for a :class:`JudgeConversation`."""
    return await judge_score(
        judge,
        conversation.query,
        conversation.response,
        context,
        conversation.history,
    )


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
        conversation = judge_conversation(turns, context)

        if not conversation.response:
            return RewardResult(score=0.0, metadata={"error": "no response"})

        # A judge failure propagates: composers raise (or, opted in,
        # redistribute the component's weight), environments flag the episode
        # and trainers drop it. A neutral stand-in score would be trained on
        # as if it were a verdict.
        score = await judge_conversation_score(self.judge, conversation, context)

        return RewardResult(
            score=float(score),
            components={"llm_judge": float(score)},
            metadata={
                "judge_model": getattr(self.judge.config, "model_name", "unknown")
            },
        )


def _checked_judge_score(score: Any) -> float:
    """A judge verdict as a float in [0, 1]; anything else is a judge failure."""
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise RewardUnavailableError(f"LLM judge returned no score: {score!r}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RewardUnavailableError(
            f"LLM judge score must be a finite value in [0, 1], got {score}"
        )
    return value


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

    #: A judge: ``MultiObjectiveRewardFunction`` passes it the whole
    #: conversation, earlier replies included, even when scoring one step.
    sees_whole_conversation = True

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
        conversation = judge_conversation(turns, context)
        if not conversation.response:
            return 0.0

        # Failures propagate, so MultiObjectiveRewardFunction raises (or,
        # opted in, leaves this component out) instead of scoring it 0.5.
        return await judge_conversation_score(self.judge, conversation, context)

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
        heuristic: RewardFunction | None = None,
    ):
        super().__init__(weight=1.0, reward_type=RewardType.IMMEDIATE, name=name)
        self.judge = judge
        self.judge_weight = judge_weight
        self.heuristic_weight = heuristic_weight
        self._judge_available = judge is not None

        # Injected heuristic, or lazy-init default fallback.
        self._heuristic: RewardFunction | None = heuristic

    def _get_heuristic(self) -> RewardFunction:
        if self._heuristic is None:
            from stateset_agents.core.grounding import ground_shaping_rewards
            from stateset_agents.core.reward import (
                CompositeReward,
                HelpfulnessReward,
                SafetyReward,
            )

            # HelpfulnessReward is a style heuristic: gated on grounding, as
            # in every shipped factory, so a canned template cannot carry the
            # fallback.
            self._heuristic = CompositeReward(
                reward_functions=ground_shaping_rewards(
                    [HelpfulnessReward(weight=0.6), SafetyReward(weight=0.4)]
                ),
            )
        return self._heuristic

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        """Blend the judge's verdict with the heuristic, or score with the
        heuristic alone when no judge is configured.

        When a configured judge is unavailable the heuristic replaces it for
        this episode: the result is flagged ``metadata["reward_fallback"]``
        (``REWARD_FALLBACK_KEY``) and records ``judge_error``, so
        environments and ``score_episode`` copy the flag to the trajectory and
        trainers can drop the episode rather than mix heuristic and judge
        scores in one group. A heuristic result's own fallback flag is kept.
        """
        # Always run heuristic (fast, no API cost)
        heuristic_result = await self._get_heuristic().compute_reward(turns, context)
        heuristic_score = heuristic_result.score
        metadata: dict[str, Any] = {}
        inherited = reward_fallback(heuristic_result)
        if inherited is not None:
            metadata[REWARD_FALLBACK_KEY] = f"heuristic: {inherited}"

        if self._judge_available and self.judge is not None:
            conversation = judge_conversation(turns, context)
            try:
                verdict = await judge_conversation_score(
                    self.judge, conversation, context
                )
            except RewardUnavailableError as exc:
                # The documented fallback of this class covers judge outages
                # only, and is flagged; bugs still propagate.
                logger.warning("LLM judge failed, using heuristic: %s", exc)
                reason = f"judge unavailable ({type(exc).__name__}); heuristic score"
                if inherited is not None:
                    reason = f"{reason}; heuristic: {inherited}"
                return RewardResult(
                    score=heuristic_score,
                    components={"heuristic": heuristic_score},
                    metadata={
                        **metadata,
                        "mode": "heuristic_fallback",
                        "judge_error": f"{type(exc).__name__}: {exc}",
                        REWARD_FALLBACK_KEY: reason,
                    },
                )
            combined = (
                self.judge_weight * verdict + self.heuristic_weight * heuristic_score
            )
            return RewardResult(
                score=combined,
                components={"llm_judge": verdict, "heuristic": heuristic_score},
                metadata={**metadata, "mode": "llm_judge"},
            )

        # No judge configured: the heuristic is this reward, not a stand-in.
        return RewardResult(
            score=heuristic_score,
            components={"heuristic": heuristic_score},
            metadata={**metadata, "mode": "heuristic_fallback"},
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
            EvaluationCriteria,
            JudgeConfig,
            LLMJudge,
        )

        eval_criteria = None
        if criteria:
            eval_criteria = [
                EvaluationCriteria(c) if isinstance(c, str) else c for c in criteria
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

    except _JUDGE_SETUP_ERRORS as exc:
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

    except _JUDGE_SETUP_ERRORS as exc:
        logger.warning("Failed to create LLM judge component: %s", exc)
        return None


__all__ = [
    "REFERENCE_KEYS",
    "judge_conversation_score",
    "judge_score",
    "scenario_reference",
    "LLMJudgeReward",
    "LLMJudgeRewardComponent",
    "LLMJudgeRewardWithFallback",
    "create_rlaif_reward",
    "create_rlaif_component",
]
