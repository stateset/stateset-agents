"""Conversation types and context helpers for multi-turn agents."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any

from .conversation import ConversationResetError


class ConversationConflict(RuntimeError):
    """A named conversation has competing work or changed during an operation."""


class ConversationClosedError(ConversationResetError):
    """The named conversation was ended or replaced while work was in flight."""


@dataclass
class _ContextLifecycle:
    state: str = "active"
    revision: int = 0
    operation: dict[str, Any] | None = None
    last_operation: dict[str, Any] | None = None
    snapshot_operation: dict[str, Any] | None = None
    last_snapshot: dict[str, Any] | None = None
    lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        with self.lock:
            if self.operation is not None or self.snapshot_operation is not None:
                raise RuntimeError(
                    "Cannot copy a conversation while an operation is active"
                )
            return {key: value for key, value in vars(self).items() if key != "lock"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = RLock()


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

    def _owner(self) -> _ContextLifecycle:
        if not hasattr(self, "_lifecycle"):
            self._lifecycle = _ContextLifecycle()
        return self._lifecycle

    def get_status(self) -> dict[str, Any]:
        """Return detached turn and snapshot receipts without invoking a backend."""
        owner = self._owner()
        with owner.lock:
            return copy.deepcopy(
                {
                    "state": owner.state,
                    "revision": owner.revision,
                    "active": owner.operation,
                    "last_finished": owner.last_operation,
                    "snapshot_active": owner.snapshot_operation,
                    "last_snapshot": owner.last_snapshot,
                }
            )

    def add_turn(self, turn: dict[str, Any]) -> None:
        """Add a turn to the conversation history."""
        turn_with_timestamp = {
            **copy.deepcopy(turn),
            "timestamp": datetime.now().isoformat(),
        }
        owner = self._owner()
        with owner.lock:
            if owner.state != "active":
                raise ConversationClosedError("Conversation has ended")
            self.history.append(turn_with_timestamp)
            owner.revision += 1

    def get_recent_history(self, max_turns: int = 10) -> list[dict[str, Any]]:
        """Get recent conversation history."""
        with self._owner().lock:
            return copy.deepcopy(self.history[-max_turns:])

    def get_context_summary(self) -> dict[str, Any]:
        """Get a summary of the conversation context."""
        with self._owner().lock:
            return copy.deepcopy(
                {
                    "conversation_id": self.conversation_id,
                    "user_id": self.user_id,
                    "topic": self.topic,
                    "intent": self.intent,
                    "entities": self.entities,
                    "turn_count": len(self.history),
                    "started_at": self.started_at.isoformat(),
                }
            )


def context_data(context: ConversationContext) -> dict[str, Any]:
    """Copy canonical context data without copying its operation ownership."""
    return copy.deepcopy(
        {
            name: getattr(context, name)
            for name in ConversationContext.__dataclass_fields__
        }
    )


def publish_context(context: ConversationContext, data: dict[str, Any]) -> None:
    """Publish prepared canonical data; the caller must hold the context lock."""
    for name in ConversationContext.__dataclass_fields__:
        setattr(context, name, data[name])
    context._owner().revision += 1


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

        sorted_dialogues = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )

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

    update_dict = copy.deepcopy(update)
    owner = context._owner()
    with owner.lock:
        if owner.state != "active":
            raise ConversationClosedError("Conversation has ended")
        _apply_context_update(context, update_dict)
        owner.revision += 1


def _apply_context_update(
    context: ConversationContext, update_dict: dict[str, Any]
) -> None:
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
    "ConversationConflict",
    "ConversationClosedError",
    "ConversationContext",
    "DialogueDatabase",
    "apply_context_update",
]
