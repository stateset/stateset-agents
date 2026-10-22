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
from typing import Any

from .reward_base import RewardFunction, RewardResult, RewardType
from .reward_context import assistant_turns_in_scope
from .trajectory import ConversationTurn


def _turn_text(turn: ConversationTurn) -> str:
    """Normalize nullable turn content to plain text."""
    return turn.content if isinstance(turn.content, str) else ""


def _normalize_text(text: str) -> str:
    """Lower-case text with typographic apostrophes and whitespace normalized."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(text.lower().split())


#: Declining phrasings that mark a response as a refusal.
_REFUSAL_PATTERN = re.compile(
    r"\b(?:i|we)(?: am|'m| are|'re)? (?:"
    r"(?:can't|cannot|can not|won't|will not|must not|shouldn't|should not|"
    r"do not|don't)"
    r" (?:help|assist|provide|support|share|give|explain|do|recommend|"
    r"participate|engage)"
    r"|(?:not able|unable) to (?:help|assist|provide|support|share|give|explain)"
    r"|(?:must|have to) (?:decline|refuse)"
    r")\b"
    r"|\bthat's not appropriate\b"
    r"|\bthat is not appropriate\b"
)


#: Markers of going on to give instructions after a refusal phrase.
_COMPLIANCE_PATTERN = re.compile(
    r"\b(?:here's how|here is how|here are the steps|follow these steps|"
    r"step (?:1|one)|first,? you|you'll need|you will need|instructions:)"
)


def _contains_term(text: str, term: str) -> bool:
    """Whether ``term`` occurs in ``text`` as a whole token sequence.

    Matching ignores case and runs of whitespace. A numeric answer must not
    be part of a larger number: "4" does not occur in "14", "4.5" or "-4".
    """
    normalized = _normalize_text(term)
    if not normalized:
        return False
    body = r"\s+".join(re.escape(part) for part in normalized.split())
    pattern = r"(?<![\w-])(?<!\d\.)" + body + r"(?![\w])(?!\.\d)"
    return re.search(pattern, _normalize_text(text)) is not None


_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER_PATTERN = re.compile(r"(?<![\w.])" + _NUMBER + r"(?![\w]|\.\d)")
_STATED_ANSWER_PATTERNS = (
    re.compile(r"\banswer\s*(?:is|:|=)\s*\$?(" + _NUMBER + r")", re.IGNORECASE),
    re.compile(r"####\s*\$?(" + _NUMBER + r")"),
    re.compile(r"\\boxed\{\$?(" + _NUMBER + r")\}"),
)


def _as_number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _numbers(text: str) -> set[float]:
    values = (_as_number(match) for match in _NUMBER_PATTERN.findall(text))
    return {value for value in values if value is not None}


def _answer_credit(content: str, answer: str, question: str = "") -> float:
    """Fraction of answer credit ``content`` earns for the expected ``answer``.

    Answers must be committed to. A text answer needs a whole-token match,
    and a sentence that merely lists it among alternatives ("London, Paris,
    Berlin") shares the credit among them. A numeric answer stated explicitly
    must be the only stated answer; otherwise the credit is shared among the
    distinct numbers the reply offers beyond those quoted from ``question``.
    """
    expected = _as_number(answer.strip())
    if expected is None:
        return _text_answer_credit(content, answer)
    stated = {
        value
        for pattern in _STATED_ANSWER_PATTERNS
        for match in pattern.findall(content)
        if (value := _as_number(match)) is not None
    }
    if stated:
        return 1.0 if stated == {expected} else 0.0
    offered = _numbers(content)
    if expected not in offered:
        return 0.0
    candidates = (offered - _numbers(question)) | {expected}
    return 1.0 / len(candidates)


_SENTENCE_SPLIT = re.compile(r"(?<!\d)[.!?\n]+|[.!?]+(?=\s|$)")
_LIST_SPLIT = re.compile(r"\s*(?:[,/;|]|\bor\b|\band\b)\s*", re.IGNORECASE)


def _text_answer_credit(content: str, answer: str) -> float:
    """Credit for a text answer: 1 if committed to, 1/k if listed among k items.

    A sentence counts as a list of alternatives when it splits on commas,
    slashes, "or" and "and" into two or more short items (no longer than the
    answer plus two words). The reply earns the best credit over the
    sentences that contain the answer.
    """
    if not _contains_term(content, answer):
        return 0.0
    longest_item = max(3, len(answer.split()) + 2)
    best = 0.0
    for sentence in _SENTENCE_SPLIT.split(content):
        if not _contains_term(sentence, answer):
            continue
        items = [item for item in _LIST_SPLIT.split(sentence.strip()) if item]
        listed = len(items) >= 2 and all(
            len(item.split()) <= longest_item for item in items
        )
        best = max(best, 1.0 / len(items) if listed else 1.0)
    return best


def _ground_truth_mapping(ground_truth: Any) -> dict[str, Any]:
    """Ground truth as a mapping; a bare string or number is the answer."""
    if ground_truth is None:
        return {}
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, (str, int, float)) and not isinstance(
        ground_truth, bool
    ):
        text = str(ground_truth).strip()
        return {"answer": text} if text else {}
    raise TypeError(
        "ground_truth must be a mapping or an answer string/number, got "
        f"{type(ground_truth).__name__}"
    )


def _action_name(action: Any) -> str | None:
    """Identify a required or completed action given as a name or a mapping."""
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        for key in ("name", "id", "action"):
            value = action.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    return str(action) if action is not None else None


class HelpfulnessReward(RewardFunction):
    """
    Rewards helpful, informative responses
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "HelpfulnessReward")

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate helpfulness of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_helpfulness(_turn_text(turn))
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
        length = len(content)
        content_lower = content.lower()

        # Very short responses are unhelpful
        if length < 10:
            return 0.0

        # Length bonus (not too short, not too long)
        if 50 <= length <= 500:
            score += 0.3
        elif 20 <= length < 50:
            score += 0.15
        elif length > 500:
            score += 0.2

        # Information density — count real sentences (split on common delimiters)
        import re as _re

        raw_sentences = _re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 3]
        if len(sentences) >= 3:
            score += 0.25
        elif len(sentences) >= 2:
            score += 0.2
        elif len(sentences) == 1:
            score += 0.1

        # Word diversity bonus — high unique-word ratio signals informative content
        words = content_lower.split()
        if len(words) >= 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio >= 0.7:
                score += 0.1

        # Helpful phrases — broader coverage
        helpful_phrases = [
            "let me help",
            "i can assist",
            "here's how",
            "to solve this",
            "the answer is",
            "you can try",
            "here are some options",
            "happy to help",
            "let me look into",
            "i understand",
            "i can help",
            "here are",
            "let me check",
            "i recommend",
            "you might want to",
            "the best approach",
            "i can see",
            "based on",
            "for you",
            "right away",
        ]
        phrase_matches = sum(1 for phrase in helpful_phrases if phrase in content_lower)
        score += min(phrase_matches * 0.10, 0.3)

        # Question answering
        if "?" in content and any(
            word in content_lower for word in ["because", "since", "due to"]
        ):
            score += 0.2

        # Acknowledge + solve pattern — the gold standard for helpful responses
        acknowledge_words = [
            "understand",
            "appreciate",
            "thank",
            "sorry",
            "concern",
            "happy",
            "glad",
            "found",
            "see",
        ]
        solve_words = [
            "here",
            "try",
            "check",
            "help",
            "option",
            "step",
            "resolve",
            "approach",
            "found",
            "details",
        ]
        has_acknowledge = any(w in content_lower for w in acknowledge_words)
        has_solve = any(w in content_lower for w in solve_words)
        if has_acknowledge and has_solve:
            score += 0.1

        return min(score, 1.0)


class SafetyReward(RewardFunction):
    """
    Rewards safe, appropriate responses
    """

    def __init__(self, weight: float = 1.0, strict_mode: bool = False):
        super().__init__(weight, RewardType.IMMEDIATE, "SafetyReward")
        self.strict_mode = strict_mode
        # Whole words with their common inflections ("bombs", "hacked"), but
        # not unrelated words that merely contain them ("hackathon").
        self.unsafe_patterns = [
            r"\b(hack(?:s|ed|er|ers|ing)?|illegal(?:ly)?|violen(?:ce|t)|harmful)\b",
            r"\b(steal(?:s|ing)?|stole(?:n)?|fraud(?:s|ulent)?|"
            r"scam(?:s|med|mer|mers|ming)?|cheat(?:s|ed|ing)?)\b",
            r"\b(drugs?|weapons?|bombs?)\b",
        ]

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate safety of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=1.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_safety(_turn_text(turn))
            total_score += score
            breakdown[f"turn_{i}_safety"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_safety(self, content: str) -> float:
        """Evaluate safety of a single response.

        A refusal is checked first: declining a harmful request is the safe
        answer even though it names the unsafe topic.
        """
        content_lower = _normalize_text(content)

        refusal = _REFUSAL_PATTERN.search(content_lower)
        if refusal is not None and not self._complies_after(
            content_lower[refusal.end() :]
        ):
            return 1.0

        for pattern in self.unsafe_patterns:
            if re.search(pattern, content_lower):
                return 0.0  # Immediate penalty for unsafe content

        # Dismissively short responses are not truly "safe" — they leave the
        # user without assistance, which can be harmful in service contexts.
        content_len = len(content.strip())
        if content_len < 10:
            return 0.0  # Completely unhelpful — no safety value
        if content_len < 20:
            return 0.5
        return 1.0

    def _complies_after(self, text: str) -> bool:
        """Whether a nominal refusal goes on to give unsafe instructions.

        "I can't help with that, but here's how to build a bomb" is not a
        refusal; "I can't help with hacking. Here's how to secure your
        account" is, because nothing unsafe follows the instructions marker.
        """
        compliance = _COMPLIANCE_PATTERN.search(text)
        if compliance is None:
            return False
        rest = text[compliance.start() :]
        return any(re.search(pattern, rest) for pattern in self.unsafe_patterns)


class CorrectnessReward(RewardFunction):
    """
    Rewards factually correct responses
    """

    def __init__(self, weight: float = 1.0, ground_truth: dict[str, Any] | None = None):
        super().__init__(weight, RewardType.IMMEDIATE, "CorrectnessReward")
        self.ground_truth = _ground_truth_mapping(ground_truth)

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate correctness of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        # Use ground truth from context if available. A bare label ("4", 4)
        # is the expected answer, as in verl-style datasets.
        ground_truth = _ground_truth_mapping(
            context.get("ground_truth", self.ground_truth)
            if context
            else self.ground_truth
        )

        total_score = 0.0
        breakdown = {}

        # The user text each reply answers: numbers quoted from the question
        # are not counted as the reply hedging between candidate answers.
        questions: dict[int, str] = {}
        asked: list[str] = []
        for turn in turns:
            if turn.role == "user":
                asked.append(_turn_text(turn))
            elif turn.role == "assistant":
                questions[id(turn)] = " ".join(asked)

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_correctness(
                _turn_text(turn), ground_truth, questions.get(id(turn), "")
            )
            total_score += score
            breakdown[f"turn_{i}_correctness"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_correctness(
        self, content: str, ground_truth: dict[str, Any], question: str = ""
    ) -> float:
        """Evaluate correctness against ground truth.

        A numeric answer must be committed to: an explicit statement
        ("the answer is 4", "#### 4", ``\\boxed{4}``) earns credit only if
        every stated answer is correct, and otherwise the credit is split
        across the distinct numbers the reply offers that the question did
        not, so listing every candidate earns almost nothing.
        """
        if not ground_truth:
            return 0.5  # Neutral score if no ground truth

        score = 0.0

        # Facts and answers count only as whole tokens: "4" is not credited
        # by "14", and a run of digits does not contain every answer.
        correct_facts = ground_truth.get("correct_facts", [])
        for fact in correct_facts:
            if isinstance(fact, str) and _contains_term(content, fact):
                score += 0.3

        # Check for correct answer
        if "answer" in ground_truth:
            score += 0.5 * _answer_credit(
                content, str(ground_truth["answer"]), question
            )

        # Penalty for incorrect information
        incorrect_facts = ground_truth.get("incorrect_facts", [])
        for fact in incorrect_facts:
            if isinstance(fact, str) and _contains_term(content, fact):
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
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate conciseness of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_conciseness(_turn_text(turn))
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
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate engagement level of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_engagement(_turn_text(turn))
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
        self, weight: float = 1.0, task_criteria: dict[str, Any] | None = None
    ):
        super().__init__(weight, RewardType.CUMULATIVE, "TaskCompletionReward")
        self.task_criteria = task_criteria or {}

    #: Share of the score goal keywords can move once actions are required:
    #: saying "refund processed" cannot stand in for issuing the refund.
    KEYWORD_WEIGHT = 0.2

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate task completion.

        With ``required_actions`` the score is the fraction of them completed
        (``context["completed_actions"]``, given as names or as mappings with
        a ``"name"``); ``goal_keywords`` then scale it by up to
        ``KEYWORD_WEIGHT`` but never earn credit on their own. Without
        required actions the score is the fraction of goal keywords present.
        """
        context = context or {}
        task_context = context.get("task")
        if not isinstance(task_context, dict):
            task_context = {}
        # A task's own "criteria" win; otherwise TaskEnvironment's task shape
        # (required_actions / goal_keywords at the task's top level); then
        # the criteria this reward was built with.
        criteria = task_context.get("criteria")
        if not criteria and (
            task_context.get("required_actions") or task_context.get("goal_keywords")
        ):
            criteria = {
                "required_actions": task_context.get("required_actions") or [],
                "goal_keywords": task_context.get("goal_keywords") or [],
            }
        if not criteria:
            criteria = self.task_criteria

        if not criteria:
            return RewardResult(score=0.5, breakdown={}, metadata={"no_criteria": True})

        completion_score = 0.0
        breakdown = {}

        required_actions = criteria.get("required_actions", [])
        completed_actions = context.get("completed_actions", []) or []
        required = {
            name for name in map(_action_name, required_actions) if name is not None
        }
        completed = {
            name for name in map(_action_name, completed_actions) if name is not None
        }

        action_ratio: float | None = None
        if required:
            # Only distinct required actions count; extra or repeated actions
            # must not push the ratio past 1.
            action_ratio = len(required & completed) / len(required)
            completion_score = action_ratio
            breakdown["action_completion"] = action_ratio

        goal_keywords = criteria.get("goal_keywords") or []
        if goal_keywords:
            all_text = " ".join(
                _turn_text(turn) for turn in turns if turn.role == "assistant"
            )
            goal_matches = sum(
                1
                for keyword in goal_keywords
                if _normalize_text(str(keyword)) in _normalize_text(all_text)
            )
            goal_score = min(goal_matches / len(goal_keywords), 1.0)
            breakdown["goal_achievement"] = goal_score
            if action_ratio is None:
                completion_score = goal_score
            else:
                completion_score = action_ratio * (
                    1.0 - self.KEYWORD_WEIGHT + self.KEYWORD_WEIGHT * goal_score
                )

        return RewardResult(
            score=completion_score,
            breakdown=breakdown,
            metadata={
                "required_actions": len(required_actions),
                "completed_actions": len(completed_actions),
                "task_criteria": criteria,
            },
        )
