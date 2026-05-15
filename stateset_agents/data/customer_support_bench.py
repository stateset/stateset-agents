"""
Multi-turn customer-support benchmark — the framework's differentiator over TRL.

GSM8K is the obvious "show me numbers" benchmark, but it's single-turn and
verifiable by exact match. The framework's real value proposition is
**multi-turn agents trained against a simulated user**, where the reward is
a composition of intent resolution + tone + safety.

This module provides:

* :class:`SupportScenario` — a parsed customer scenario with goal, intent, and
  expected resolution criteria.
* :func:`load_support_scenarios` — bundled, reproducible mini-corpus of 24
  scenarios across 4 intent categories (refund, technical, billing, general).
* :class:`SupportRewardComposite` — composite reward: intent classification
  (rule-based), brand voice (length + tone heuristic), and a safety filter.

The bundled corpus is deliberately small and rule-based so that the benchmark
is reproducible without external API calls. For production use, swap the
heuristic rewards for ``LLMJudgeReward`` (already in the framework).

Usage::

    from stateset_agents.data.customer_support_bench import (
        load_support_scenarios,
        SupportRewardComposite,
    )
    from stateset_agents.core import ConversationEnvironment

    scenarios = load_support_scenarios()
    env = ConversationEnvironment(
        scenarios=[s.to_scenario() for s in scenarios],
        reward_fn=SupportRewardComposite(),
        max_turns=4,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.reward_base import RewardFunction, RewardResult, RewardType
from ..core.trajectory import ConversationTurn


# Bundled corpus — deliberately small and explicit so the benchmark is
# reproducible without external data. Real production use would swap in a
# larger dataset (e.g., Bitext customer-support) via the same scenario schema.
_BUNDLED_SCENARIOS: list[dict[str, Any]] = [
    # Refund intent (6)
    {"intent": "refund", "user_query": "I want my money back for order #4521",
     "must_acknowledge": ["refund", "order"], "must_avoid": ["impossible", "can't help"]},
    {"intent": "refund", "user_query": "The shoes don't fit, please refund",
     "must_acknowledge": ["refund", "return"], "must_avoid": ["impossible"]},
    {"intent": "refund", "user_query": "I was charged twice — I need a refund for the duplicate",
     "must_acknowledge": ["refund", "duplicate"], "must_avoid": ["impossible"]},
    {"intent": "refund", "user_query": "Cancel my subscription and refund last month",
     "must_acknowledge": ["refund", "cancel"], "must_avoid": ["impossible"]},
    {"intent": "refund", "user_query": "My order arrived damaged — I want a refund",
     "must_acknowledge": ["refund", "damaged"], "must_avoid": ["impossible"]},
    {"intent": "refund", "user_query": "I never received the package — refund please",
     "must_acknowledge": ["refund", "package"], "must_avoid": ["impossible"]},

    # Technical (6)
    {"intent": "technical", "user_query": "The app crashes every time I open it",
     "must_acknowledge": ["app", "crash"], "must_avoid": ["working as designed", "your fault"]},
    {"intent": "technical", "user_query": "I can't log in — keeps saying invalid password",
     "must_acknowledge": ["password", "login"], "must_avoid": ["your fault"]},
    {"intent": "technical", "user_query": "The website won't load on my browser",
     "must_acknowledge": ["website", "browser"], "must_avoid": ["your fault"]},
    {"intent": "technical", "user_query": "Email notifications stopped working last week",
     "must_acknowledge": ["email", "notification"], "must_avoid": ["your fault"]},
    {"intent": "technical", "user_query": "The mobile app won't sync with my account",
     "must_acknowledge": ["sync", "app"], "must_avoid": ["your fault"]},
    {"intent": "technical", "user_query": "I'm getting an error code 502 when checking out",
     "must_acknowledge": ["error", "checkout"], "must_avoid": ["your fault"]},

    # Billing (6)
    {"intent": "billing", "user_query": "Why is my bill higher this month?",
     "must_acknowledge": ["bill", "month"], "must_avoid": ["impossible"]},
    {"intent": "billing", "user_query": "I don't recognize this charge on my statement",
     "must_acknowledge": ["charge", "statement"], "must_avoid": ["impossible"]},
    {"intent": "billing", "user_query": "I need a copy of last month's invoice",
     "must_acknowledge": ["invoice"], "must_avoid": ["impossible"]},
    {"intent": "billing", "user_query": "Can I switch to annual billing for a discount?",
     "must_acknowledge": ["annual", "billing"], "must_avoid": ["impossible"]},
    {"intent": "billing", "user_query": "My promo code wasn't applied at checkout",
     "must_acknowledge": ["promo", "code"], "must_avoid": ["impossible"]},
    {"intent": "billing", "user_query": "Update my credit card on file",
     "must_acknowledge": ["credit card"], "must_avoid": ["impossible"]},

    # General (6)
    {"intent": "general", "user_query": "What are your business hours?",
     "must_acknowledge": ["hours"], "must_avoid": []},
    {"intent": "general", "user_query": "How do I contact a human agent?",
     "must_acknowledge": ["agent", "human"], "must_avoid": []},
    {"intent": "general", "user_query": "Do you ship internationally?",
     "must_acknowledge": ["ship", "international"], "must_avoid": []},
    {"intent": "general", "user_query": "What's your return policy?",
     "must_acknowledge": ["return", "policy"], "must_avoid": []},
    {"intent": "general", "user_query": "Where is your office located?",
     "must_acknowledge": ["office", "located"], "must_avoid": []},
    {"intent": "general", "user_query": "Are you hiring?",
     "must_acknowledge": ["hiring", "careers"], "must_avoid": []},
]


@dataclass
class SupportScenario:
    """A single customer-support scenario with a graded reward rubric."""

    intent: str  # one of: refund, technical, billing, general
    user_query: str
    must_acknowledge: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)

    def to_scenario(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "intent": self.intent,
            "must_acknowledge": list(self.must_acknowledge),
            "must_avoid": list(self.must_avoid),
        }


def load_support_scenarios(
    intent_filter: str | None = None, limit: int | None = None
) -> list[SupportScenario]:
    """Load the bundled support-scenario corpus.

    Args:
        intent_filter: If set, return only scenarios matching this intent.
        limit: If set, return at most this many scenarios.
    """
    scenarios = [
        SupportScenario(
            intent=s["intent"],
            user_query=s["user_query"],
            must_acknowledge=list(s["must_acknowledge"]),
            must_avoid=list(s["must_avoid"]),
        )
        for s in _BUNDLED_SCENARIOS
    ]
    if intent_filter is not None:
        scenarios = [s for s in scenarios if s.intent == intent_filter]
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def _contains_any(text: str, terms: list[str]) -> int:
    """Count how many terms (case-insensitive) appear anywhere in text."""
    lower = text.lower()
    return sum(1 for term in terms if term.lower() in lower)


def _safety_check(text: str) -> tuple[float, str | None]:
    """Heuristic safety filter — returns (score in [0,1], failure reason)."""
    if not text:
        return 0.0, "empty_response"
    lower = text.lower()
    red_flags = [
        ("hate", "hate-speech-like-term"),
        ("kill yourself", "self-harm-suggestion"),
        ("you're stupid", "personal-attack"),
        ("password is", "credential-leak"),
        ("ssn is", "pii-leak"),
    ]
    for needle, reason in red_flags:
        if needle in lower:
            return 0.0, reason
    return 1.0, None


def _brand_voice_score(text: str) -> tuple[float, dict[str, float]]:
    """Heuristic brand voice — length window + politeness keywords."""
    breakdown: dict[str, float] = {}
    n_words = len(text.split())

    # Length window: 10–120 words is the brand-voice target band.
    if n_words == 0:
        length_score = 0.0
    elif n_words < 10:
        length_score = max(0.0, n_words / 10.0)
    elif n_words <= 120:
        length_score = 1.0
    else:
        # Soft penalty past 120 words.
        length_score = max(0.0, 1.0 - (n_words - 120) / 100.0)
    breakdown["length_score"] = length_score
    breakdown["n_words"] = float(n_words)

    polite_terms = ["thank you", "happy to help", "of course", "i understand",
                    "i'd be glad", "please", "sorry to hear"]
    politeness = min(1.0, _contains_any(text, polite_terms) / 2.0)
    breakdown["politeness"] = politeness

    return 0.5 * length_score + 0.5 * politeness, breakdown


class SupportRewardComposite(RewardFunction):
    """Composite reward for customer-support agents.

    Combines three rule-based signals:

    * **Intent acknowledgement** — does the response mention the things the
      scenario requires (e.g., "refund", "order")?
    * **Brand voice** — length-window check + politeness keyword count.
    * **Safety** — multiplicative gate; any safety failure zeroes the score.

    The weighting matches what production customer-support deployments tend
    to use: ~60% on intent (the task), ~30% on brand voice (the experience),
    ~10% as a safety multiplier.

    Like ``GSM8KReward``, this is deliberately rule-based so the benchmark
    is reproducible without external API calls. For production use, swap in
    ``LLMJudgeReward`` (also shipped with the framework).
    """

    name = "support_composite"

    def __init__(
        self,
        weight: float = 1.0,
        intent_weight: float = 0.6,
        brand_voice_weight: float = 0.3,
        require_safety: bool = True,
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=self.name)
        self.intent_weight = intent_weight
        self.brand_voice_weight = brand_voice_weight
        self.require_safety = require_safety

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        if not turns:
            return RewardResult(score=0.0, breakdown={"no_response": 1.0})

        ctx = context or {}
        must_acknowledge: list[str] = ctx.get("must_acknowledge", [])
        must_avoid: list[str] = ctx.get("must_avoid", [])
        intent = ctx.get("intent", "unknown")

        # Aggregate assistant text across the trajectory (multi-turn).
        assistant_text = "\n".join(
            t.content for t in turns if t.role == "assistant" and t.content
        )
        if not assistant_text:
            return RewardResult(score=0.0, breakdown={"no_assistant_turn": 1.0})

        # 1) Intent acknowledgement: fraction of required terms present.
        if must_acknowledge:
            ack_count = _contains_any(assistant_text, must_acknowledge)
            intent_score = ack_count / len(must_acknowledge)
        else:
            intent_score = 1.0

        # Penalize avoided terms.
        if must_avoid:
            avoid_count = _contains_any(assistant_text, must_avoid)
            avoid_penalty = min(1.0, avoid_count / len(must_avoid))
            intent_score = max(0.0, intent_score - 0.5 * avoid_penalty)
        else:
            avoid_penalty = 0.0

        # 2) Brand voice.
        voice_score, voice_breakdown = _brand_voice_score(assistant_text)

        # 3) Safety (multiplicative).
        safety_score, safety_reason = _safety_check(assistant_text)

        composite = (
            self.intent_weight * intent_score
            + self.brand_voice_weight * voice_score
        ) * (safety_score if self.require_safety else 1.0)

        breakdown = {
            "intent_score": intent_score,
            "intent_avoid_penalty": avoid_penalty,
            "brand_voice_score": voice_score,
            "safety_score": safety_score,
        }
        breakdown.update({f"voice_{k}": v for k, v in voice_breakdown.items()})

        explanation = (
            f"intent={intent} ack_score={intent_score:.2f} voice={voice_score:.2f}"
            f" safety={safety_score:.2f}"
        )
        if safety_reason:
            explanation += f" safety_fail={safety_reason}"

        return RewardResult(
            score=float(composite),
            breakdown=breakdown,
            explanation=explanation,
        )


def make_support_scenarios(scenarios: list[SupportScenario]) -> list[dict[str, Any]]:
    """Convert ``SupportScenario`` objects into ``ConversationEnvironment`` scenarios."""
    return [s.to_scenario() for s in scenarios]


__all__ = [
    "SupportRewardComposite",
    "SupportScenario",
    "load_support_scenarios",
    "make_support_scenarios",
]
