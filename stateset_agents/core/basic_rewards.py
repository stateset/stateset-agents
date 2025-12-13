"""
Basic reward functions for multi-turn agent training.

This module provides fundamental reward functions including:
- HelpfulnessReward
- SafetyReward
- CorrectnessReward
- ConcisenessReward
- EngagementReward
- TaskCompletionReward
"""

import re
from typing import Any, Dict, List, Optional

from .reward_base import RewardFunction, RewardResult, RewardType
from .trajectory import ConversationTurn


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

    def __init__(self, weight: float = 1.0, strict_mode: bool = False):
        super().__init__(weight, RewardType.IMMEDIATE, "SafetyReward")
        self.strict_mode = strict_mode
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
