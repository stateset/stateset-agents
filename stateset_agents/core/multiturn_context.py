"""Conversation types and context helpers for multi-turn agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ConversationContext:
    """Context for multi-turn conversations."""

    conversation_id: str
    user_id: str | None = None
    topic: str | None = None
    intent: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)

    def add_turn(self, turn: dict[str, Any]):
        """Add a turn to the conversation history."""
        turn_with_timestamp = {**turn, "timestamp": datetime.now().isoformat()}
        self.history.append(turn_with_timestamp)

    def get_recent_history(self, max_turns: int = 10) -> list[dict[str, Any]]:
        """Get recent conversation history."""
        return self.history[-max_turns:]

    def get_context_summary(self) -> dict[str, Any]:
        """Get a summary of the conversation context."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "topic": self.topic,
            "intent": self.intent,
            "entities": self.entities,
            "turn_count": len(self.history),
            "started_at": self.started_at.isoformat(),
        }


class DialogueDatabase:
    """Simple searchable database of dialogue examples."""

    def __init__(self, dialogues: list[dict[str, Any]]):
        self.dialogues = dialogues
        self.index = self._build_index()

    def _build_index(self) -> dict[str, list[int]]:
        """Build a simple keyword index."""
        index: dict[str, list[int]] = {}
        for i, dialogue in enumerate(self.dialogues):
            content = dialogue.get("content", "").lower()
            words = content.split()

            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(i)

        return index

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search for relevant dialogues."""
        query_words = query.lower().split()

        scores: dict[int, int] = {}
        for word in query_words:
            if word in self.index:
                for dialogue_idx in self.index[word]:
                    scores[dialogue_idx] = scores.get(dialogue_idx, 0) + 1

        sorted_dialogues = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for dialogue_idx, score in sorted_dialogues[:top_k]:
            dialogue = self.dialogues[dialogue_idx].copy()
            dialogue["relevance_score"] = score
            results.append(dialogue)

        return results

    def get_dialogue_by_id(self, dialogue_id: str) -> dict[str, Any] | None:
        """Get a specific dialogue by ID."""
        for dialogue in self.dialogues:
            if dialogue.get("id") == dialogue_id:
                return dialogue
        return None


def apply_context_update(
    context: ConversationContext,
    update: dict[str, Any] | None,
) -> None:
    """Merge context updates into the active conversation context."""
    if not update:
        return

    update_dict = dict(update)
    if "user_id" in update_dict and update_dict["user_id"]:
        context.user_id = update_dict["user_id"]
    if "topic" in update_dict and update_dict["topic"]:
        context.topic = update_dict["topic"]
    if "intent" in update_dict and update_dict["intent"]:
        context.intent = update_dict["intent"]

    entities = update_dict.get("entities")
    if isinstance(entities, dict):
        context.entities.update(entities)

    history = update_dict.get("history")
    if isinstance(history, list):
        context.history.extend(history)

    metadata = update_dict.get("metadata")
    if isinstance(metadata, dict):
        context.metadata.update(metadata)

    skip_keys = {
        "conversation_id",
        "started_at",
        "user_id",
        "topic",
        "intent",
        "entities",
        "history",
        "metadata",
    }
    for key, value in update_dict.items():
        if key in skip_keys:
            continue
        context.metadata[key] = value


__all__ = [
    "ConversationContext",
    "DialogueDatabase",
    "apply_context_update",
]
