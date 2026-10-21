"""Explicitly budgeted whole-conversation judging, with no neutral fallback."""

from __future__ import annotations

import hashlib
import inspect
import math
from typing import Any

from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.support_judge import ALL_CHECKS, JudgeItem, JudgeResult
from stateset_agents.rewards.support_conversation_reward import (
    SupportConversationRewardV3,
)


def checked_judgement(result: JudgeResult, max_disagreement: float) -> float:
    """Reject missing, malformed, unstable or partially parsed judge output."""
    if len(result.pass_scores) != 2 or len(result.checks) != 2:
        raise ValueError("Judge must return exactly two passes")
    values = (result.score, result.disagreement, *result.pass_scores)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ):
        raise ValueError("Judge scores and disagreement must be finite in [0, 1]")
    if (
        type(result.parse_failures) is not int
        or result.parse_failures != 0
        or any(
            set(checks) != set(ALL_CHECKS)
            or any(type(value) is not bool for value in checks.values())
            for checks in result.checks
        )
    ):
        raise ValueError("Judge evaluation must parse both complete passes")
    if not math.isclose(
        result.score, sum(result.pass_scores) / 2, abs_tol=1e-12
    ) or not math.isclose(
        result.disagreement,
        abs(result.pass_scores[0] - result.pass_scores[1]),
        abs_tol=1e-12,
    ):
        raise ValueError("Judge result disagrees with its pass scores")
    if result.disagreement > max_disagreement:
        raise ValueError("Judge self-disagreement exceeds the configured limit")
    return float(result.score)


class SupportJudgeReward(RewardFunction):
    """Score entire conversations, or blend a deterministic subset with v3.

    ``judge_weight=judge_fraction=1`` gives an independent evaluation reward.
    For training, a fraction below one chooses transcripts by content hash,
    without consuming training RNG state. The episode budget counts attempted
    judgements, including failures; each may require two passes plus the judge's
    parse retries. Reaching the budget aborts instead of changing the objective.
    This is a call bound, not a provider dollar cap. Inject a suitably bounded
    judge transport. Synchronous judges execute synchronously.
    """

    fail_on_error = True
    name = "support_judge_blend_v1"

    def __init__(
        self,
        judge: Any,
        *,
        max_judged_episodes: int,
        judge_weight: float = 1.0,
        judge_fraction: float = 1.0,
        max_disagreement: float = 0.05,
    ) -> None:
        super().__init__(weight=1.0, reward_type=RewardType.SPARSE, name=self.name)
        for name, value in (
            ("judge_weight", judge_weight),
            ("judge_fraction", judge_fraction),
            ("max_disagreement", max_disagreement),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite in [0, 1]")
        if type(max_judged_episodes) is not int or max_judged_episodes < 1:
            raise ValueError("max_judged_episodes must be a positive integer")
        self.judge = judge
        self.judge_weight = float(judge_weight)
        self.judge_fraction = float(judge_fraction)
        self.max_disagreement = float(max_disagreement)
        self.max_judged_episodes = max_judged_episodes
        self.judged_episodes = 0
        self.rules = SupportConversationRewardV3()

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Keep hard safety gates and propagate every required judge failure."""
        ctx = dict(context or {})
        transcript = tuple(
            (str(turn.role), turn.content or "")
            for turn in turns
            if turn.role in {"user", "assistant"}
        )
        rule = await self.rules.compute_reward(turns, ctx)
        breakdown = dict(rule.breakdown or {})
        # Neither a model judge nor a partial sampling policy can override vetoes.
        if (
            not any(role == "assistant" and text.strip() for role, text in transcript)
            or breakdown.get("must_avoid_hits", 0) > 0
            or breakdown.get("base_safety_score", 1) < 1
        ):
            return RewardResult(
                score=0.0, breakdown={**breakdown, "judge_sampled": 0.0}
            )
        identity = content_hash({"transcript": transcript, "scenario": ctx})
        bucket = int(hashlib.sha256(identity.encode()).hexdigest(), 16)
        sampled = self.judge_weight > 0 and bucket < self.judge_fraction * (1 << 256)
        if not sampled:
            return RewardResult(
                score=rule.score, breakdown={**breakdown, "judge_sampled": 0.0}
            )
        if self.judged_episodes >= self.max_judged_episodes:
            raise RuntimeError("Whole-conversation judge episode budget exhausted")
        self.judged_episodes += 1
        item = JudgeItem(
            id=str(ctx.get("id", identity)), scenario=ctx, transcript=transcript
        )
        results = self.judge.score([item])
        if inspect.isawaitable(results):
            results = await results
        if len(results) != 1 or results[0].id != item.id:
            raise ValueError("Judge did not return exactly the requested conversation")
        score = checked_judgement(results[0], self.max_disagreement)
        return RewardResult(
            score=(1 - self.judge_weight) * rule.score + self.judge_weight * score,
            breakdown={
                **breakdown,
                "judge_sampled": 1.0,
                "judge_score": score,
                "judge_disagreement": results[0].disagreement,
                "rule_score": rule.score,
            },
        )
