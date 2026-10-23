"""
Domain-specific reward functions for specialized agent training.

This module provides reward functions tailored to specific domains:
- CustomerServiceReward
- TechnicalSupportReward
- SalesAssistantReward
- DomainSpecificReward (base class)
"""

import re
from abc import abstractmethod
from typing import Any

from .reward_base import RewardFunction, RewardResult, RewardType
from .reward_context import assistant_turns_in_scope
from .trajectory import ConversationTurn


def _mentions(text: str, keyword: str) -> bool:
    """Whole-word keyword match ("help" matches "helping", not "whelp")."""
    return re.search(r"\b" + re.escape(keyword) + r"\w*", text) is not None


def _turn_text(turn: ConversationTurn) -> str:
    """Normalize nullable turn content to plain text."""
    return turn.content if isinstance(turn.content, str) else ""


class DomainSpecificReward(RewardFunction):
    """
    Base class for domain-specific reward functions
    """

    #: Keyword-based style heuristics: shipped factories gate them on grounding.
    shaping = True

    def __init__(
        self,
        weight: float = 1.0,
        domain_keywords: dict[str, list[str]] | None = None,
        expected_responses: dict[str, str] | None = None,
    ):
        super().__init__(weight, RewardType.IMMEDIATE, self.__class__.__name__)
        self.domain_keywords = domain_keywords or {}
        self.expected_responses = expected_responses or {}

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Compute domain-specific reward"""

        all_assistant_turns = [t for t in turns if t.role == "assistant"]
        if not all_assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        # One step's reward scores only the newest reply (see reward_context).
        scored = len(assistant_turns_in_scope(turns, context))
        first = len(all_assistant_turns) - scored
        assistant_turns = all_assistant_turns[first:]

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns, start=first):
            # Get expected response if available
            query = self._get_query_for_turn(turns, i)
            expected = self.expected_responses.get(query) if query else None

            # Compute turn score
            turn_score = await self._compute_turn_score(turn, expected, context)
            total_score += turn_score
            breakdown[f"turn_{i}_score"] = turn_score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={
                "num_assistant_turns": len(assistant_turns),
                "domain": self.__class__.__name__,
            },
        )

    def _get_query_for_turn(
        self, turns: list[ConversationTurn], assistant_idx: int
    ) -> str | None:
        """Get the user query preceding an assistant turn"""
        # Find the user turn before this assistant turn
        assistant_count = 0
        for i, turn in enumerate(turns):
            if turn.role == "assistant":
                if assistant_count == assistant_idx:
                    # Look backwards for user turn
                    for j in range(i - 1, -1, -1):
                        if turns[j].role == "user":
                            content = turns[j].content
                            return (
                                content
                                if isinstance(content, str) and content
                                else None
                            )
                    break
                assistant_count += 1
        return None

    @abstractmethod
    async def _compute_turn_score(
        self,
        turn: ConversationTurn,
        expected: str | None,
        context: dict[str, Any] | None,
    ) -> float:
        """Compute score for a single turn"""
        pass


class CustomerServiceReward(DomainSpecificReward):
    """
    Reward function specifically for customer service interactions
    """

    def __init__(
        self, weight: float = 1.0, expected_responses: dict[str, str] | None = None
    ):
        domain_keywords = {
            "empathy": [
                "sorry",
                "understand",
                "help",
                "happy",
                "glad",
                "assist",
                "apologize",
            ],
            "action": [
                "visit",
                "email",
                "click",
                "check",
                "provide",
                "contact",
                "follow",
            ],
            "resolution": ["resolved", "fixed", "completed", "processed", "updated"],
            "professionalism": ["please", "thank you", "appreciate", "kindly"],
        }
        super().__init__(weight, domain_keywords, expected_responses)

    async def _compute_turn_score(
        self,
        turn: ConversationTurn,
        expected: str | None,
        context: dict[str, Any] | None,
    ) -> float:
        """Compute customer service quality score"""
        response = _turn_text(turn)
        response_lower = response.lower()
        score = 0.0

        # Base similarity reward if expected response provided
        if expected:
            import difflib

            similarity = difflib.SequenceMatcher(
                None, response_lower, expected.lower()
            ).ratio()
            score += similarity * 0.4

        # Empathy score — redistribute weight when no expected response
        empathy_weight = 0.3 if not expected else 0.2
        empathy_matches = sum(
            1
            for word in self.domain_keywords["empathy"]
            if _mentions(response_lower, word)
        )
        empathy_score = min(empathy_matches / 3, 1.0)
        score += empathy_score * empathy_weight

        # Action-oriented score
        action_weight = 0.3 if not expected else 0.2
        action_matches = sum(
            1
            for word in self.domain_keywords["action"]
            if _mentions(response_lower, word)
        )
        action_score = min(action_matches / 2, 1.0)
        score += action_score * action_weight

        # Professionalism score
        prof_weight = 0.2 if not expected else 0.1
        prof_matches = sum(
            1
            for word in self.domain_keywords["professionalism"]
            if _mentions(response_lower, word)
        )
        prof_score = min(prof_matches / 2, 1.0)
        score += prof_score * prof_weight

        # Resolution keywords bonus
        resolution_matches = sum(
            1
            for word in self.domain_keywords["resolution"]
            if _mentions(response_lower, word)
        )
        if resolution_matches > 0:
            score += min(resolution_matches * 0.05, 0.1)

        # Length appropriateness
        word_count = len(response.split())
        if 20 <= word_count <= 80:
            score += 0.1
        elif word_count < 10:
            score -= 0.1
        elif word_count > 150:
            score -= 0.05

        return max(0.0, min(1.0, score))


class TechnicalSupportReward(DomainSpecificReward):
    """
    Reward function for technical support interactions
    """

    def __init__(self, weight: float = 1.0):
        domain_keywords = {
            "diagnostic": ["check", "verify", "test", "diagnose", "inspect"],
            "solution": ["try", "restart", "update", "reinstall", "configure"],
            "explanation": ["because", "due to", "caused by", "reason", "why"],
            "steps": ["first", "second", "then", "next", "finally"],
        }
        super().__init__(weight, domain_keywords)

    async def _compute_turn_score(
        self,
        turn: ConversationTurn,
        expected: str | None,
        context: dict[str, Any] | None,
    ) -> float:
        """Compute technical support quality score"""
        response_lower = _turn_text(turn).lower()
        score = 0.0

        # Diagnostic approach
        diagnostic_score = sum(
            1
            for word in self.domain_keywords["diagnostic"]
            if _mentions(response_lower, word)
        )
        score += min(diagnostic_score * 0.1, 0.2)

        # Solution-oriented
        solution_score = sum(
            1
            for word in self.domain_keywords["solution"]
            if _mentions(response_lower, word)
        )
        score += min(solution_score * 0.15, 0.3)

        # Clear explanation
        explanation_score = sum(
            1
            for word in self.domain_keywords["explanation"]
            if _mentions(response_lower, word)
        )
        score += min(explanation_score * 0.1, 0.2)

        # Step-by-step guidance
        steps_score = sum(
            1
            for word in self.domain_keywords["steps"]
            if _mentions(response_lower, word)
        )
        if steps_score >= 2:  # Multiple step indicators
            score += 0.2

        # Technical accuracy (simplified check)
        if any(
            _mentions(response_lower, tech_term)
            for tech_term in ["settings", "configuration", "system", "software"]
        ):
            score += 0.1

        return min(1.0, score)


class SalesAssistantReward(DomainSpecificReward):
    """
    Reward function for sales and product recommendation
    """

    def __init__(self, weight: float = 1.0):
        domain_keywords = {
            "benefits": ["benefit", "advantage", "feature", "quality", "value"],
            "personalization": [
                "your needs",
                "for you",
                "recommend",
                "suggest",
                "perfect for",
            ],
            "urgency": ["limited", "offer", "today", "now", "special"],
            "trust": ["guarantee", "warranty", "return", "satisfaction", "trusted"],
        }
        super().__init__(weight, domain_keywords)

    async def _compute_turn_score(
        self,
        turn: ConversationTurn,
        expected: str | None,
        context: dict[str, Any] | None,
    ) -> float:
        """Compute sales effectiveness score"""
        response = _turn_text(turn)
        response_lower = response.lower()
        score = 0.0

        # Benefits highlighting
        benefits_score = sum(
            1
            for word in self.domain_keywords["benefits"]
            if _mentions(response_lower, word)
        )
        score += min(benefits_score * 0.1, 0.25)

        # Personalization
        personal_phrases = sum(
            1
            for phrase in self.domain_keywords["personalization"]
            if _mentions(response_lower, phrase)
        )
        score += min(personal_phrases * 0.15, 0.25)

        # Trust building
        trust_score = sum(
            1
            for word in self.domain_keywords["trust"]
            if _mentions(response_lower, word)
        )
        score += min(trust_score * 0.1, 0.2)

        # Not too pushy (penalize excessive urgency)
        urgency_score = sum(
            1
            for word in self.domain_keywords["urgency"]
            if _mentions(response_lower, word)
        )
        if urgency_score <= 1:
            score += 0.1
        else:
            score -= 0.1  # Too pushy

        # Question asking (engagement)
        if "?" in response:
            score += 0.1

        return max(0.0, min(1.0, score))
