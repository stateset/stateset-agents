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
    {
        "intent": "refund",
        "user_query": "I want my money back for order #4521",
        "must_acknowledge": ["refund", "order"],
        "must_avoid": ["impossible", "can't help"],
    },
    {
        "intent": "refund",
        "user_query": "The shoes don't fit, please refund",
        "must_acknowledge": ["refund", "return"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "refund",
        "user_query": "I was charged twice — I need a refund for the duplicate",
        "must_acknowledge": ["refund", "duplicate"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "refund",
        "user_query": "Cancel my subscription and refund last month",
        "must_acknowledge": ["refund", "cancel"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "refund",
        "user_query": "My order arrived damaged — I want a refund",
        "must_acknowledge": ["refund", "damaged"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "refund",
        "user_query": "I never received the package — refund please",
        "must_acknowledge": ["refund", "package"],
        "must_avoid": ["impossible"],
    },
    # Technical (6)
    {
        "intent": "technical",
        "user_query": "The app crashes every time I open it",
        "must_acknowledge": ["app", "crash"],
        "must_avoid": ["working as designed", "your fault"],
    },
    {
        "intent": "technical",
        "user_query": "I can't log in — keeps saying invalid password",
        "must_acknowledge": ["password", "login"],
        "must_avoid": ["your fault"],
    },
    {
        "intent": "technical",
        "user_query": "The website won't load on my browser",
        "must_acknowledge": ["website", "browser"],
        "must_avoid": ["your fault"],
    },
    {
        "intent": "technical",
        "user_query": "Email notifications stopped working last week",
        "must_acknowledge": ["email", "notification"],
        "must_avoid": ["your fault"],
    },
    {
        "intent": "technical",
        "user_query": "The mobile app won't sync with my account",
        "must_acknowledge": ["sync", "app"],
        "must_avoid": ["your fault"],
    },
    {
        "intent": "technical",
        "user_query": "I'm getting an error code 502 when checking out",
        "must_acknowledge": ["error", "checkout"],
        "must_avoid": ["your fault"],
    },
    # Billing (6)
    {
        "intent": "billing",
        "user_query": "Why is my bill higher this month?",
        "must_acknowledge": ["bill", "month"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "billing",
        "user_query": "I don't recognize this charge on my statement",
        "must_acknowledge": ["charge", "statement"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "billing",
        "user_query": "I need a copy of last month's invoice",
        "must_acknowledge": ["invoice"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "billing",
        "user_query": "Can I switch to annual billing for a discount?",
        "must_acknowledge": ["annual", "billing"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "billing",
        "user_query": "My promo code wasn't applied at checkout",
        "must_acknowledge": ["promo", "code"],
        "must_avoid": ["impossible"],
    },
    {
        "intent": "billing",
        "user_query": "Update my credit card on file",
        "must_acknowledge": ["credit card"],
        "must_avoid": ["impossible"],
    },
    # General (6)
    {
        "intent": "general",
        "user_query": "What are your business hours?",
        "must_acknowledge": ["hours"],
        "must_avoid": [],
    },
    {
        "intent": "general",
        "user_query": "How do I contact a human agent?",
        "must_acknowledge": ["agent", "human"],
        "must_avoid": [],
    },
    {
        "intent": "general",
        "user_query": "Do you ship internationally?",
        "must_acknowledge": ["ship", "international"],
        "must_avoid": [],
    },
    {
        "intent": "general",
        "user_query": "What's your return policy?",
        "must_acknowledge": ["return", "policy"],
        "must_avoid": [],
    },
    {
        "intent": "general",
        "user_query": "Where is your office located?",
        "must_acknowledge": ["office", "located"],
        "must_avoid": [],
    },
    {
        "intent": "general",
        "user_query": "Are you hiring?",
        "must_acknowledge": ["hiring", "careers"],
        "must_avoid": [],
    },
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

    polite_terms = [
        "thank you",
        "happy to help",
        "of course",
        "i understand",
        "i'd be glad",
        "please",
        "sorry to hear",
    ]
    politeness = min(1.0, _contains_any(text, polite_terms) / 2.0)
    breakdown["politeness"] = politeness

    return 0.5 * length_score + 0.5 * politeness, breakdown


# Markers of a reply that *does something*: a commitment, an action taken, or a
# concrete information request that moves the case forward.
_ACTION_MARKERS = [
    "i'll",
    "i will",
    "i've",
    "i have",
    "i'm going to",
    "let me",
    "i can ",
    "we'll",
    "we will",
    "here's",
    "here is",
    "next step",
    "walk you through",
    "process",
    "issued",
    "initiated",
    "escalate",
    "right away",
    "could you share",
    "could you provide",
    "can you share",
    "can you provide",
    "may i have",
    "please share",
    "please provide",
]

# Markers of a concrete timeframe.
_TIMEFRAME_MARKERS = [
    "one moment",
    "a moment",
    "right now",
    "right away",
    "today",
    "shortly",
    "immediately",
    "business day",
    "within",
    "24 hours",
    "48 hours",
]

# Markers of deflection: acknowledging the customer while sending them away
# without doing anything.
_DEFLECTION_MARKERS = [
    "check the website",
    "check our website",
    "read the faq",
    "check the faq",
    "look it up",
    "not something this channel",
    "not something we handle",
    "we don't handle",
    "we do not handle",
    "can't help with that",
    "cannot help with that",
    "yourself for more information",
    "on your own",
    "somewhere else",
    "someone else",
    "figure it out",
    "google it",
]

# A concrete reference: an order/ticket id, or an explicit ask for one.
_REFERENCE_RE = re.compile(
    r"#\d+|\b(?:order|ticket|case|reference|account|invoice)\s*(?:number|id|#)",
    re.IGNORECASE,
)


def _resolution_score(text: str) -> tuple[float, dict[str, float]]:
    """Heuristic resolution/concreteness — does the reply move the case forward?

    Polite-but-useless "deflection" replies (acknowledge the customer, then
    send them away with no action, timeframe, or concrete reference) used to
    score 0.75 under the intent+voice composite and slip past the 0.7 curation
    threshold. This component makes concreteness a measured signal:

    * **action** (60%) — commitments, actions taken, or concrete information
      requests (``I'll``, ``let me``, ``could you share`` ...).
    * **timeframe** (20%) — an explicit "when" (``one moment``, ``within`` ...).
    * **reference** (20%) — an order/ticket id (``#4521``) or an explicit ask
      for one (``your order number``).
    * **deflection penalty** — each deflection marker ("check the website",
      "we don't handle" ...) subtracts 0.5.
    """
    breakdown: dict[str, float] = {}
    lower = text.lower()

    action_hits = _contains_any(text, _ACTION_MARKERS)
    timeframe_hits = _contains_any(text, _TIMEFRAME_MARKERS)
    reference = 1.0 if _REFERENCE_RE.search(text) else 0.0
    deflection_hits = sum(1 for marker in _DEFLECTION_MARKERS if marker in lower)

    raw = (
        0.6 * min(1.0, action_hits / 2.0)
        + 0.2 * min(1.0, float(timeframe_hits))
        + 0.2 * reference
        - 0.5 * deflection_hits
    )
    score = max(0.0, min(1.0, raw))

    breakdown["action_hits"] = float(action_hits)
    breakdown["timeframe_hits"] = float(timeframe_hits)
    breakdown["reference"] = reference
    breakdown["deflection_hits"] = float(deflection_hits)
    return score, breakdown


def _persona_score(
    turns: list[ConversationTurn], persona: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """Optional persona fidelity — expected opener/signoff substrings.

    ``persona`` is ``{"opener": [str, ...], "signoff": [str, ...]}``. The
    opener must appear (case-insensitive) in the *first* assistant turn, the
    signoff in the *last* assistant turn. Each present list contributes an
    equal share of the score; an empty persona scores 1.0 (nothing required).
    """
    assistant_turns = [t for t in turns if t.role == "assistant" and t.content]
    first = assistant_turns[0].content.lower() if assistant_turns else ""
    last = assistant_turns[-1].content.lower() if assistant_turns else ""

    checks: list[float] = []
    breakdown: dict[str, float] = {}
    openers = [str(s).lower() for s in persona.get("opener", []) if str(s).strip()]
    signoffs = [str(s).lower() for s in persona.get("signoff", []) if str(s).strip()]
    if openers:
        opener_ok = 1.0 if any(s in first for s in openers) else 0.0
        breakdown["persona_opener"] = opener_ok
        checks.append(opener_ok)
    if signoffs:
        signoff_ok = 1.0 if any(s in last for s in signoffs) else 0.0
        breakdown["persona_signoff"] = signoff_ok
        checks.append(signoff_ok)

    score = sum(checks) / len(checks) if checks else 1.0
    return score, breakdown


class SupportRewardComposite(RewardFunction):
    """Composite reward for customer-support agents.

    Combines four rule-based signals:

    * **Intent acknowledgement** — does the response mention the things the
      scenario requires (e.g., "refund", "order")?
    * **Brand voice** — length-window check + politeness keyword count.
    * **Resolution** — concreteness heuristic: commitments, timeframes, and
      order/ticket references score; deflection phrasing ("check the website
      yourself") is penalized. Closes the polite-but-useless grader gap where
      deflection replies scored 0.75 and slipped past a 0.7 curation
      threshold.
    * **Safety** — multiplicative gate; any safety failure zeroes the score.

    Weights: ~45% intent (the task), ~25% brand voice (the experience),
    ~30% resolution (did the reply move the case forward), with safety as a
    multiplier. An optional ``persona`` config ({"opener": [...],
    "signoff": [...]}) adds a persona-fidelity component that takes
    ``persona_weight`` of the total (the other weights are scaled down
    proportionally).

    Like ``GSM8KReward``, this is deliberately rule-based so the benchmark
    is reproducible without external API calls. For production use, swap in
    ``LLMJudgeReward`` (also shipped with the framework).
    """

    name = "support_composite"

    def __init__(
        self,
        weight: float = 1.0,
        intent_weight: float = 0.45,
        brand_voice_weight: float = 0.25,
        resolution_weight: float = 0.3,
        require_safety: bool = True,
        persona: dict[str, Any] | None = None,
        persona_weight: float = 0.15,
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=self.name)
        self.intent_weight = intent_weight
        self.brand_voice_weight = brand_voice_weight
        self.resolution_weight = resolution_weight
        self.require_safety = require_safety
        self.persona = persona or None
        self.persona_weight = persona_weight

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

        # 3) Resolution / concreteness.
        resolution_score, resolution_breakdown = _resolution_score(assistant_text)

        # 4) Safety (multiplicative).
        safety_score, safety_reason = _safety_check(assistant_text)

        weighted = (
            self.intent_weight * intent_score
            + self.brand_voice_weight * voice_score
            + self.resolution_weight * resolution_score
        )

        # 5) Optional persona fidelity: takes persona_weight of the total,
        # scaling the rule-based portion down proportionally.
        if self.persona:
            persona_score, persona_breakdown = _persona_score(turns, self.persona)
            weighted = (
                1.0 - self.persona_weight
            ) * weighted + self.persona_weight * persona_score
        else:
            persona_score, persona_breakdown = 1.0, {}

        composite = weighted * (safety_score if self.require_safety else 1.0)

        breakdown = {
            "intent_score": intent_score,
            "intent_avoid_penalty": avoid_penalty,
            "brand_voice_score": voice_score,
            "resolution_score": resolution_score,
            "safety_score": safety_score,
        }
        breakdown.update({f"voice_{k}": v for k, v in voice_breakdown.items()})
        breakdown.update(
            {f"resolution_{k}": v for k, v in resolution_breakdown.items()}
        )
        if self.persona:
            breakdown["persona_score"] = persona_score
            breakdown.update(persona_breakdown)

        explanation = (
            f"intent={intent} ack_score={intent_score:.2f} voice={voice_score:.2f}"
            f" resolution={resolution_score:.2f} safety={safety_score:.2f}"
        )
        if self.persona:
            explanation += f" persona={persona_score:.2f}"
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
