"""
Domain-specific reward functions for specialized agent training.

This module provides reward functions tailored to specific domains:
- CustomerServiceReward
- TechnicalSupportReward
- SalesAssistantReward
- DomainSpecificReward (base class)
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .reward_base import RewardFunction, RewardResult, RewardType
from .trajectory import ConversationTurn


class DomainSpecificReward(RewardFunction):
    """
    Base class for domain-specific reward functions
    """

    def __init__(
        self,
        weight: float = 1.0,
        domain_keywords: Optional[Dict[str, List[str]]] = None,
        expected_responses: Optional[Dict[str, str]] = None,
    ):
        super().__init__(weight, RewardType.IMMEDIATE, self.__class__.__name__)
        self.domain_keywords = domain_keywords or {}
        self.expected_responses = expected_responses or {}

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute domain-specific reward"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
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
        self, turns: List[ConversationTurn], assistant_idx: int
    ) -> Optional[str]:
        """Get the user query preceding an assistant turn"""
        # Find the user turn before this assistant turn
        assistant_count = 0
        for i, turn in enumerate(turns):
            if turn.role == "assistant":
                if assistant_count == assistant_idx:
                    # Look backwards for user turn
                    for j in range(i - 1, -1, -1):
                        if turns[j].role == "user":
                            return turns[j].content
                    break
                assistant_count += 1
        return None

    @abstractmethod
    async def _compute_turn_score(
        self,
        turn: ConversationTurn,
        expected: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Compute score for a single turn"""
        pass


class CustomerServiceReward(DomainSpecificReward):
    """
    Reward function specifically for customer service interactions
    """

    def __init__(
        self, weight: float = 1.0, expected_responses: Optional[Dict[str, str]] = None
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
        expected: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Compute customer service quality score"""
        response_lower = turn.content.lower()
        score = 0.0

        # Base similarity reward if expected response provided
        if expected:
            import difflib

            similarity = difflib.SequenceMatcher(
                None, response_lower, expected.lower()
            ).ratio()
            score += similarity * 0.4

        # Empathy score
        empathy_matches = sum(
            1 for word in self.domain_keywords["empathy"] if word in response_lower
        )
        empathy_score = min(empathy_matches / 3, 1.0)  # Cap at 3 empathy words
        score += empathy_score * 0.2

        # Action-oriented score
        action_matches = sum(
            1 for word in self.domain_keywords["action"] if word in response_lower
        )
        action_score = min(action_matches / 2, 1.0)  # Cap at 2 action words
        score += action_score * 0.2

        # Professionalism score
        prof_matches = sum(
            1
            for word in self.domain_keywords["professionalism"]
            if word in response_lower
        )
        prof_score = min(prof_matches / 2, 1.0)
        score += prof_score * 0.1

        # Length appropriateness
        word_count = len(turn.content.split())
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
        expected: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Compute technical support quality score"""
        response_lower = turn.content.lower()
        score = 0.0

        # Diagnostic approach
        diagnostic_score = sum(
            1 for word in self.domain_keywords["diagnostic"] if word in response_lower
        )
        score += min(diagnostic_score * 0.1, 0.2)

        # Solution-oriented
        solution_score = sum(
            1 for word in self.domain_keywords["solution"] if word in response_lower
        )
        score += min(solution_score * 0.15, 0.3)

        # Clear explanation
        explanation_score = sum(
            1 for word in self.domain_keywords["explanation"] if word in response_lower
        )
        score += min(explanation_score * 0.1, 0.2)

        # Step-by-step guidance
        steps_score = sum(
            1 for word in self.domain_keywords["steps"] if word in response_lower
        )
        if steps_score >= 2:  # Multiple step indicators
            score += 0.2

        # Technical accuracy (simplified check)
        if any(
            tech_term in response_lower
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
        expected: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Compute sales effectiveness score"""
        response_lower = turn.content.lower()
        score = 0.0

        # Benefits highlighting
        benefits_score = sum(
            1 for word in self.domain_keywords["benefits"] if word in response_lower
        )
        score += min(benefits_score * 0.1, 0.25)

        # Personalization
        personal_phrases = sum(
            1
            for phrase in self.domain_keywords["personalization"]
            if phrase in response_lower
        )
        score += min(personal_phrases * 0.15, 0.25)

        # Trust building
        trust_score = sum(
            1 for word in self.domain_keywords["trust"] if word in response_lower
        )
        score += min(trust_score * 0.1, 0.2)

        # Not too pushy (penalize excessive urgency)
        urgency_score = sum(
            1 for word in self.domain_keywords["urgency"] if word in response_lower
        )
        if urgency_score <= 1:
            score += 0.1
        else:
            score -= 0.1  # Too pushy

        # Question asking (engagement)
        if "?" in turn.content:
            score += 0.1

        return max(0.0, min(1.0, score))
