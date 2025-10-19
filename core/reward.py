"""
Reward modeling framework for multi-turn agent training

This module provides flexible reward functions for evaluating agent performance
in multi-turn conversations and task-oriented interactions.
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of rewards"""

    IMMEDIATE = "immediate"  # Reward for single turn
    CUMULATIVE = "cumulative"  # Reward for entire conversation
    SPARSE = "sparse"  # Reward only at episode end
    DENSE = "dense"  # Reward at every step


@dataclass
class RewardResult:
    """Result of reward computation"""

    score: float
    breakdown: Dict[str, float]
    metadata: Dict[str, Any]
    explanation: Optional[str] = None


class RewardFunction(ABC):
    """
    Abstract base class for reward functions
    """

    def __init__(
        self,
        weight: float = 1.0,
        reward_type: RewardType = RewardType.IMMEDIATE,
        name: Optional[str] = None,
    ):
        self.weight = weight
        self.reward_type = reward_type
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward for conversation turns"""
        pass

    async def compute_turn_reward(
        self,
        turn: ConversationTurn,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> RewardResult:
        """Compute reward for a single turn"""
        turns = conversation_history or []
        turns = turns + [turn]
        return await self.compute_reward(turns, context)

    def __call__(self, *args, **kwargs):
        """Make reward function callable"""
        return asyncio.run(self.compute_reward(*args, **kwargs))


class CompositeReward(RewardFunction):
    """
    Combines multiple reward functions with weights
    """

    def __init__(
        self,
        reward_functions: List[RewardFunction],
        combination_method: str = "weighted_sum",
    ):
        super().__init__(name="CompositeReward")
        self.reward_functions = reward_functions
        self.combination_method = combination_method

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute composite reward from all functions"""

        results = []
        for reward_fn in self.reward_functions:
            result = await reward_fn.compute_reward(turns, context)
            results.append(result)

        # Combine rewards
        if self.combination_method == "weighted_sum":
            total_score = sum(
                r.score * rf.weight for r, rf in zip(results, self.reward_functions)
            )
        elif self.combination_method == "average":
            total_score = np.mean([r.score for r in results])
        elif self.combination_method == "min":
            total_score = min(r.score for r in results)
        elif self.combination_method == "max":
            total_score = max(r.score for r in results)
        else:
            total_score = sum(
                r.score * rf.weight for r, rf in zip(results, self.reward_functions)
            )

        # Combine breakdowns
        combined_breakdown = {}
        for result, reward_fn in zip(results, self.reward_functions):
            for key, value in result.breakdown.items():
                combined_key = f"{reward_fn.name}_{key}"
                combined_breakdown[combined_key] = value

        return RewardResult(
            score=total_score,
            breakdown=combined_breakdown,
            metadata={
                "component_scores": [r.score for r in results],
                "component_names": [rf.name for rf in self.reward_functions],
                "combination_method": self.combination_method,
            },
        )


# Pre-built reward functions


class HelpfulnessReward(RewardFunction):
    """
    Rewards helpful, informative responses
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "HelpfulnessReward")

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Evaluate helpfulness of assistant responses"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_helpfulness(turn.content)
            total_score += score
            breakdown[f"turn_{i}_helpfulness"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_helpfulness(self, content: str) -> float:
        """Evaluate helpfulness of a single response"""
        score = 0.0

        # Length bonus (not too short, not too long)
        length = len(content)
        if 20 <= length <= 500:
            score += 0.3
        elif length > 500:
            score += 0.2  # Slight penalty for very long responses

        # Information density
        sentences = content.split(".")
        if len(sentences) >= 2:
            score += 0.2

        # Helpful phrases
        helpful_phrases = [
            "let me help",
            "i can assist",
            "here's how",
            "to solve this",
            "the answer is",
            "you can try",
            "here are some options",
        ]
        content_lower = content.lower()
        phrase_matches = sum(1 for phrase in helpful_phrases if phrase in content_lower)
        score += min(phrase_matches * 0.1, 0.3)

        # Question answering
        if "?" in content and any(
            word in content_lower for word in ["because", "since", "due to"]
        ):
            score += 0.2

        return min(score, 1.0)


class SafetyReward(RewardFunction):
    """
    Rewards safe, appropriate responses
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "SafetyReward")
        self.unsafe_patterns = [
            r"\b(hack|hacking|illegal|violence|harmful)\b",
            r"\b(steal|fraud|scam|cheat)\b",
            r"\b(drug|weapon|bomb)\b",
        ]

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Evaluate safety of assistant responses"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=1.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_safety(turn.content)
            total_score += score
            breakdown[f"turn_{i}_safety"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_safety(self, content: str) -> float:
        """Evaluate safety of a single response"""
        content_lower = content.lower()

        # Check for unsafe patterns
        for pattern in self.unsafe_patterns:
            if re.search(pattern, content_lower):
                return 0.0  # Immediate penalty for unsafe content

        # Positive safety indicators
        score = 1.0  # Start with max safety score

        # Bonus for explicitly safe language
        safe_phrases = [
            "i can't help with",
            "that's not appropriate",
            "i don't recommend",
        ]
        if any(phrase in content_lower for phrase in safe_phrases):
            score = 1.0

        return score


class CorrectnessReward(RewardFunction):
    """
    Rewards factually correct responses
    """

    def __init__(
        self, weight: float = 1.0, ground_truth: Optional[Dict[str, Any]] = None
    ):
        super().__init__(weight, RewardType.IMMEDIATE, "CorrectnessReward")
        self.ground_truth = ground_truth or {}

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Evaluate correctness of assistant responses"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        # Use ground truth from context if available
        ground_truth = (
            context.get("ground_truth", self.ground_truth)
            if context
            else self.ground_truth
        )

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_correctness(turn.content, ground_truth)
            total_score += score
            breakdown[f"turn_{i}_correctness"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_correctness(
        self, content: str, ground_truth: Dict[str, Any]
    ) -> float:
        """Evaluate correctness against ground truth"""
        if not ground_truth:
            return 0.5  # Neutral score if no ground truth

        score = 0.0
        content_lower = content.lower()

        # Check for correct facts
        correct_facts = ground_truth.get("correct_facts", [])
        for fact in correct_facts:
            if isinstance(fact, str) and fact.lower() in content_lower:
                score += 0.3

        # Check for correct answer
        if "answer" in ground_truth:
            expected_answer = str(ground_truth["answer"]).lower()
            if expected_answer in content_lower:
                score += 0.5

        # Penalty for incorrect information
        incorrect_facts = ground_truth.get("incorrect_facts", [])
        for fact in incorrect_facts:
            if isinstance(fact, str) and fact.lower() in content_lower:
                score -= 0.3

        return max(0.0, min(1.0, score))


class ConcisenessReward(RewardFunction):
    """
    Rewards concise, to-the-point responses
    """

    def __init__(self, weight: float = 1.0, optimal_length: int = 150):
        super().__init__(weight, RewardType.IMMEDIATE, "ConcisenessReward")
        self.optimal_length = optimal_length

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Evaluate conciseness of assistant responses"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_conciseness(turn.content)
            total_score += score
            breakdown[f"turn_{i}_conciseness"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_conciseness(self, content: str) -> float:
        """Evaluate conciseness of a single response"""
        length = len(content)

        if length == 0:
            return 0.0

        # Optimal length gets full score
        if length <= self.optimal_length:
            return 1.0

        # Gradual penalty for longer responses
        excess_ratio = (length - self.optimal_length) / self.optimal_length
        penalty = min(excess_ratio * 0.5, 0.8)  # Max 80% penalty

        return max(0.2, 1.0 - penalty)


class EngagementReward(RewardFunction):
    """
    Rewards engaging, interesting responses
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "EngagementReward")

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Evaluate engagement level of assistant responses"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_engagement(turn.content)
            total_score += score
            breakdown[f"turn_{i}_engagement"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_engagement(self, content: str) -> float:
        """Evaluate engagement of a single response"""
        score = 0.0
        content_lower = content.lower()

        # Questions to user (shows interest)
        question_count = content.count("?")
        score += min(question_count * 0.15, 0.3)

        # Emotional language
        emotional_words = [
            "exciting",
            "interesting",
            "amazing",
            "wonderful",
            "great",
            "fantastic",
            "awesome",
            "incredible",
            "fascinating",
        ]
        emotion_matches = sum(1 for word in emotional_words if word in content_lower)
        score += min(emotion_matches * 0.1, 0.2)

        # Personal touch
        personal_phrases = ["i think", "in my opinion", "i believe", "personally"]
        if any(phrase in content_lower for phrase in personal_phrases):
            score += 0.2

        # Varied sentence structure
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            if len(set(lengths)) > 1:  # Varied sentence lengths
                score += 0.1

        # Avoid repetitive language
        words = content_lower.split()
        unique_words = len(set(words))
        if len(words) > 0:
            uniqueness_ratio = unique_words / len(words)
            if uniqueness_ratio > 0.7:
                score += 0.2

        return min(score, 1.0)


class TaskCompletionReward(RewardFunction):
    """
    Rewards successful task completion
    """

    def __init__(
        self, weight: float = 1.0, task_criteria: Optional[Dict[str, Any]] = None
    ):
        super().__init__(weight, RewardType.CUMULATIVE, "TaskCompletionReward")
        self.task_criteria = task_criteria or {}

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Evaluate task completion"""

        task_context = context.get("task", {}) if context else {}
        criteria = task_context.get("criteria", self.task_criteria)

        if not criteria:
            return RewardResult(score=0.5, breakdown={}, metadata={"no_criteria": True})

        # Check completion criteria
        completion_score = 0.0
        breakdown = {}

        required_actions = criteria.get("required_actions", [])
        completed_actions = context.get("completed_actions", []) if context else []

        if required_actions:
            completion_ratio = len(completed_actions) / len(required_actions)
            completion_score = completion_ratio
            breakdown["action_completion"] = completion_ratio

        # Check for goal achievement
        if criteria.get("goal_keywords"):
            all_text = " ".join(
                turn.content for turn in turns if turn.role == "assistant"
            )
            goal_matches = sum(
                1
                for keyword in criteria["goal_keywords"]
                if keyword.lower() in all_text.lower()
            )
            goal_score = min(goal_matches / len(criteria["goal_keywords"]), 1.0)
            completion_score = max(completion_score, goal_score)
            breakdown["goal_achievement"] = goal_score

        return RewardResult(
            score=completion_score,
            breakdown=breakdown,
            metadata={
                "required_actions": len(required_actions),
                "completed_actions": len(completed_actions),
                "task_criteria": criteria,
            },
        )


# Utility functions for creating common reward combinations


def create_helpful_agent_reward() -> CompositeReward:
    """Create reward function for helpful assistant"""
    return CompositeReward(
        [
            HelpfulnessReward(weight=0.4),
            SafetyReward(weight=0.3),
            CorrectnessReward(weight=0.2),
            EngagementReward(weight=0.1),
        ]
    )


def create_customer_service_reward() -> CompositeReward:
    """Create reward function for customer service agent"""
    return CompositeReward(
        [
            HelpfulnessReward(weight=0.35),
            SafetyReward(weight=0.25),
            ConcisenessReward(weight=0.2),
            EngagementReward(weight=0.2),
        ]
    )


def create_task_agent_reward(task_criteria: Dict[str, Any]) -> CompositeReward:
    """Create reward function for task-oriented agent"""
    return CompositeReward(
        [
            TaskCompletionReward(weight=0.5, task_criteria=task_criteria),
            CorrectnessReward(weight=0.3),
            SafetyReward(weight=0.2),
        ]
    )


def create_tutoring_reward() -> CompositeReward:
    """Create reward function for tutoring agent"""
    return CompositeReward(
        [
            HelpfulnessReward(weight=0.3),
            CorrectnessReward(weight=0.3),
            EngagementReward(weight=0.25),
            SafetyReward(weight=0.15),
        ]
    )


# Custom reward function decorator
def reward_function(
    weight: float = 1.0, reward_type: RewardType = RewardType.IMMEDIATE
):
    """Decorator for creating custom reward functions"""

    def decorator(func):
        class CustomReward(RewardFunction):
            def __init__(self):
                super().__init__(weight, reward_type, func.__name__)
                self.func = func

            async def compute_reward(self, turns, context=None):
                score = await self.func(turns, context)
                return RewardResult(
                    score=score,
                    breakdown={self.name: score},
                    metadata={"custom_function": True},
                )

        return CustomReward()

    return decorator


# Example usage of custom reward
@reward_function(weight=0.5)
async def politeness_reward(
    turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
) -> float:
    """Custom reward for politeness"""
    polite_phrases = [
        "please",
        "thank you",
        "you're welcome",
        "excuse me",
        "i apologize",
    ]

    assistant_turns = [t for t in turns if t.role == "assistant"]
    if not assistant_turns:
        return 0.0

    total_score = 0.0
    for turn in assistant_turns:
        content_lower = turn.content.lower()
        phrase_count = sum(1 for phrase in polite_phrases if phrase in content_lower)
        turn_score = min(phrase_count * 0.2, 1.0)
        total_score += turn_score

    return total_score / len(assistant_turns)


# Domain-Specific Reward Functions


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


# Enhanced composite reward with similarity matching


class SimilarityAwareReward(RewardFunction):
    """
    Reward function that considers similarity to expected responses
    """

    def __init__(
        self,
        weight: float = 1.0,
        expected_responses_map: Optional[Dict[str, str]] = None,
        base_reward_fn: Optional[RewardFunction] = None,
    ):
        super().__init__(weight, RewardType.IMMEDIATE, "SimilarityAwareReward")
        self.expected_responses_map = expected_responses_map or {}
        self.base_reward_fn = base_reward_fn

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward with similarity consideration"""

        # Get base reward if provided
        base_score = 0.0
        base_breakdown = {}
        if self.base_reward_fn:
            base_result = await self.base_reward_fn.compute_reward(turns, context)
            base_score = base_result.score * 0.5  # Weight base reward
            base_breakdown = base_result.breakdown

        # Compute similarity scores
        similarity_scores = []
        similarity_breakdown = {}

        for i, turn in enumerate(turns):
            if turn.role == "assistant":
                # Get the prompt/query for this turn
                prompt = context.get("prompt") if context else None
                expected = self.expected_responses_map.get(prompt) if prompt else None

                if expected:
                    import difflib

                    similarity = difflib.SequenceMatcher(
                        None, turn.content.lower(), expected.lower()
                    ).ratio()
                    similarity_scores.append(similarity)
                    similarity_breakdown[f"turn_{i}_similarity"] = similarity

        # Combine scores
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            total_score = base_score + (avg_similarity * 0.5)
        else:
            total_score = base_score

        # Merge breakdowns
        breakdown = {**base_breakdown, **similarity_breakdown}

        return RewardResult(
            score=total_score,
            breakdown=breakdown,
            metadata={
                "has_expected_responses": bool(similarity_scores),
                "avg_similarity": sum(similarity_scores) / len(similarity_scores)
                if similarity_scores
                else 0.0,
            },
        )


# Factory functions for creating domain-specific rewards


def create_domain_reward(
    domain: str,
    weight: float = 1.0,
    expected_responses: Optional[Dict[str, str]] = None,
    **kwargs,
) -> RewardFunction:
    """Create a domain-specific reward function"""

    domain_lower = domain.lower()

    if domain_lower in ["customer_service", "support", "cs"]:
        return CustomerServiceReward(weight, expected_responses)
    elif domain_lower in ["technical", "tech_support", "it"]:
        return TechnicalSupportReward(weight)
    elif domain_lower in ["sales", "marketing", "product"]:
        return SalesAssistantReward(weight)
    else:
        # Default to a general helpful agent reward
        return create_helpful_agent_reward()


def create_adaptive_reward(
    base_rewards: List[RewardFunction],
    expected_responses: Optional[Dict[str, str]] = None,
    similarity_weight: float = 0.3,
) -> CompositeReward:
    """Create an adaptive reward that combines base rewards with similarity matching"""

    reward_functions = base_rewards.copy()

    # Add similarity reward if expected responses provided
    if expected_responses:
        similarity_reward = SimilarityAwareReward(
            weight=similarity_weight,
            expected_responses_map=expected_responses,
            base_reward_fn=None,  # Don't double-count base rewards
        )
        reward_functions.append(similarity_reward)

    # Normalize weights
    total_weight = sum(rf.weight for rf in reward_functions)
    for rf in reward_functions:
        rf.weight = rf.weight / total_weight

    return CompositeReward(reward_functions)
