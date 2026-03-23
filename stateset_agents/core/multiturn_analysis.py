"""Analysis and transcript helpers for multi-turn agents."""

from __future__ import annotations

import re
from typing import Any

from .multiturn_context import ConversationContext


def analyze_user_input(context: ConversationContext, user_message: str) -> None:
    """Analyze user input for intent and entities."""
    lowered_message = user_message.lower()
    if any(word in lowered_message for word in ["help", "problem", "issue", "error"]):
        context.intent = "support"
    elif any(word in lowered_message for word in ["buy", "purchase", "price", "cost"]):
        context.intent = "sales"
    elif any(
        word in lowered_message for word in ["learn", "how", "what", "explain"]
    ):
        context.intent = "education"

    entities: dict[str, Any] = {}

    numbers = re.findall(r"\d+", user_message)
    if numbers:
        entities["numbers"] = numbers

    emails = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        user_message,
    )
    if emails:
        entities["emails"] = emails

    context.entities.update(entities)


def extract_technical_terms(text: str) -> list[str]:
    """Extract technical terms from text."""
    technical_keywords = [
        "api",
        "database",
        "server",
        "client",
        "http",
        "ssl",
        "tcp",
        "ip",
        "json",
        "xml",
        "rest",
        "soap",
        "auth",
        "token",
        "session",
        "cache",
        "queue",
        "load",
        "performance",
        "latency",
        "throughput",
    ]

    text_lower = text.lower()
    return [term for term in technical_keywords if term in text_lower]


def extract_document_reference(text: str) -> str | None:
    """Extract document reference from text."""
    pattern = r"(?:doc|document)[-_]?(\w+)"
    matches = re.findall(pattern, text.lower())
    return matches[0] if matches else None


def create_conversation_summary(turns: list[dict[str, Any]]) -> str:
    """Create a summary of conversation turns."""
    user_messages = [turn["content"] for turn in turns if turn.get("role") == "user"]
    assistant_messages = [
        turn["content"] for turn in turns if turn.get("role") == "assistant"
    ]

    summary_parts = []
    if user_messages:
        summary_parts.append(f"User discussed: {', '.join(user_messages[:3])}...")
    if assistant_messages:
        summary_parts.append(
            f"Assistant provided: {', '.join(assistant_messages[:3])}..."
        )

    return " ".join(summary_parts)


def compress_conversation_context(
    context: ConversationContext,
    max_conversation_turns: int,
) -> None:
    """Compress conversation context to manage memory."""
    if len(context.history) <= max_conversation_turns:
        return

    keep_start = 2
    keep_end = max_conversation_turns - keep_start - 1
    middle_turns = context.history[keep_start:-keep_end]

    if middle_turns:
        summary = create_conversation_summary(middle_turns)
        context.history = (
            context.history[:keep_start]
            + [
                {
                    "role": "system",
                    "content": f"[Summary of previous turns: {summary}]",
                }
            ]
            + context.history[-keep_end:]
        )


def evaluate_context_coherence(context: ConversationContext) -> float:
    """Evaluate how coherent the conversation context is."""
    if len(context.history) < 2:
        return 1.0

    topics = []
    for turn in context.history:
        if turn.get("role") == "user":
            content = turn.get("content", "").lower()
            topics.extend(content.split())

    if not topics:
        return 0.5

    unique_topics = set(topics)
    topic_consistency = len(unique_topics) / len(topics)
    return max(0.0, 1.0 - topic_consistency)


def evaluate_goal_achievement(
    context: ConversationContext,
    ground_truth: dict[str, Any] | None,
) -> float:
    """Evaluate if conversation achieved its goal."""
    if not ground_truth:
        return 0.5

    expected_outcome = ground_truth.get("expected_outcome", "")
    if not expected_outcome:
        return 0.5

    assistant_responses = [
        turn["content"] for turn in context.history if turn.get("role") == "assistant"
    ]
    if not assistant_responses:
        return 0.0

    last_response = assistant_responses[-1].lower()
    expected_words = set(expected_outcome.lower().split())
    response_words = set(last_response.split())
    overlap = expected_words & response_words

    if expected_words:
        return len(overlap) / len(expected_words)
    return 0.0


__all__ = [
    "analyze_user_input",
    "compress_conversation_context",
    "create_conversation_summary",
    "evaluate_context_coherence",
    "evaluate_goal_achievement",
    "extract_document_reference",
    "extract_technical_terms",
]
