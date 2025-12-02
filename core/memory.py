"""
Conversation Memory and Context Management for StateSet Agents

This module provides sophisticated memory management for multi-turn conversations,
including:
- Short-term memory (recent context window)
- Long-term memory (persistent storage with retrieval)
- Episodic memory (conversation summaries)
- Semantic memory (key facts and entities)

Example:
    >>> from core.memory import ConversationMemory, MemoryConfig
    >>>
    >>> memory = ConversationMemory(MemoryConfig(
    ...     max_short_term_turns=10,
    ...     enable_summarization=True,
    ...     enable_entity_extraction=True,
    ... ))
    >>>
    >>> memory.add_turn({"role": "user", "content": "My name is Alice"})
    >>> memory.add_turn({"role": "assistant", "content": "Hello Alice!"})
    >>>
    >>> context = memory.get_context_for_generation()
    >>> print(context.entities)  # {"person": ["Alice"]}
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory storage."""

    SHORT_TERM = "short_term"  # Recent conversation turns
    LONG_TERM = "long_term"  # Persistent facts and summaries
    EPISODIC = "episodic"  # Conversation episode summaries
    SEMANTIC = "semantic"  # Extracted entities and facts
    WORKING = "working"  # Current task context


@dataclass
class MemoryConfig:
    """Configuration for conversation memory.

    Attributes:
        max_short_term_turns: Maximum turns in short-term memory
        max_short_term_tokens: Maximum tokens in short-term memory
        enable_summarization: Enable automatic summarization of old turns
        enable_entity_extraction: Extract and track entities
        enable_fact_extraction: Extract key facts from conversation
        summary_threshold: Number of turns before triggering summarization
        importance_decay: How quickly importance decays over time
        retrieval_top_k: Number of memories to retrieve for context
        persistence_backend: Backend for long-term storage ("memory", "redis", "sqlite")
    """

    max_short_term_turns: int = 20
    max_short_term_tokens: int = 4000
    enable_summarization: bool = True
    enable_entity_extraction: bool = True
    enable_fact_extraction: bool = True
    summary_threshold: int = 10
    importance_decay: float = 0.95
    retrieval_top_k: int = 5
    persistence_backend: str = "memory"


@dataclass
class MemoryEntry:
    """A single memory entry.

    Attributes:
        id: Unique identifier
        content: The memory content
        memory_type: Type of memory
        timestamp: When the memory was created
        importance: Importance score (0-1)
        metadata: Additional metadata
        embedding: Optional vector embedding for retrieval
    """

    id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def decay_importance(self, factor: float = 0.95) -> None:
        """Decay importance over time."""
        self.importance *= factor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata,
        }


@dataclass
class Entity:
    """Extracted entity from conversation.

    Attributes:
        name: Entity name/value
        entity_type: Type of entity (person, location, etc.)
        mentions: Number of times mentioned
        first_seen: When first mentioned
        last_seen: When last mentioned
        context: Context where entity appears
    """

    name: str
    entity_type: str
    mentions: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    context: List[str] = field(default_factory=list)

    def update(self, context: str) -> None:
        """Update entity with new mention."""
        self.mentions += 1
        self.last_seen = datetime.now()
        if context not in self.context:
            self.context.append(context)
            # Keep only recent context
            self.context = self.context[-5:]


@dataclass
class ConversationSummary:
    """Summary of a conversation segment.

    Attributes:
        id: Unique identifier
        summary: Text summary
        turn_range: Range of turns covered
        key_points: Key points from the segment
        entities_mentioned: Entities mentioned
        timestamp: When summary was created
    """

    id: str
    summary: str
    turn_range: Tuple[int, int]
    key_points: List[str] = field(default_factory=list)
    entities_mentioned: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContextWindow:
    """Context window for generation.

    Attributes:
        messages: Recent messages within context window
        summary: Summary of earlier conversation
        entities: Extracted entities
        facts: Key facts
        working_memory: Current task context
        total_tokens: Estimated total tokens
    """

    messages: List[Dict[str, str]]
    summary: Optional[str] = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    facts: List[str] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0

    def to_messages(self, include_summary: bool = True) -> List[Dict[str, str]]:
        """Convert to message format for LLM."""
        result = []

        # Add summary as system context if available
        if include_summary and self.summary:
            context_parts = [f"Previous conversation summary: {self.summary}"]

            if self.entities:
                entity_str = "; ".join(
                    f"{k}: {', '.join(v)}" for k, v in self.entities.items()
                )
                context_parts.append(f"Key entities: {entity_str}")

            if self.facts:
                context_parts.append(f"Key facts: {'; '.join(self.facts[:5])}")

            result.append({
                "role": "system",
                "content": "\n".join(context_parts),
            })

        result.extend(self.messages)
        return result


class EntityExtractor:
    """Extract entities from text using pattern matching.

    For production use, consider using spaCy or a dedicated NER model.
    """

    # Simple patterns for common entity types
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "url": r"https?://[^\s]+",
        "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "money": r"\$[\d,]+(?:\.\d{2})?",
        "number": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
    }

    # Name patterns (capitalized words)
    NAME_PATTERN = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"

    def __init__(self):
        self._compiled_patterns = {
            k: re.compile(v) for k, v in self.PATTERNS.items()
        }
        self._name_pattern = re.compile(self.NAME_PATTERN)

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping entity types to lists of extracted values
        """
        entities: Dict[str, List[str]] = defaultdict(list)

        # Extract using patterns
        for entity_type, pattern in self._compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type].extend(set(matches))

        # Extract potential names (simple heuristic)
        # Filter out common words that might be capitalized
        common_words = {
            "I", "The", "This", "That", "What", "When", "Where", "How",
            "Why", "Hello", "Hi", "Yes", "No", "Please", "Thank", "Thanks",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday", "January", "February", "March", "April",
            "May", "June", "July", "August", "September", "October",
            "November", "December",
        }

        names = self._name_pattern.findall(text)
        filtered_names = [
            n for n in names
            if n not in common_words and len(n) > 1
        ]
        if filtered_names:
            entities["person"].extend(set(filtered_names))

        return dict(entities)


class FactExtractor:
    """Extract key facts from conversation turns."""

    # Patterns that indicate factual statements
    FACT_INDICATORS = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"i work at ([^.]+)",
        r"i live in ([^.]+)",
        r"my (?:email|phone|address) is ([^.]+)",
        r"i need help with ([^.]+)",
        r"the (?:issue|problem) is ([^.]+)",
        r"i want to ([^.]+)",
        r"i'm looking for ([^.]+)",
    ]

    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.FACT_INDICATORS]

    def extract(self, text: str, role: str = "user") -> List[str]:
        """Extract facts from text.

        Args:
            text: Input text
            role: Role of the speaker

        Returns:
            List of extracted facts
        """
        facts = []

        for pattern in self._patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Format as a fact
                fact = f"{role.capitalize()} stated: {match.strip()}"
                facts.append(fact)

        return facts


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    async def save(self, key: str, entry: MemoryEntry) -> None:
        """Save a memory entry."""
        pass

    @abstractmethod
    async def load(self, key: str) -> Optional[MemoryEntry]:
        """Load a memory entry."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """Search for relevant memories."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a memory entry."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        pass


class InMemoryStore(MemoryStore):
    """In-memory storage backend."""

    def __init__(self):
        self._store: Dict[str, MemoryEntry] = {}

    async def save(self, key: str, entry: MemoryEntry) -> None:
        self._store[key] = entry

    async def load(self, key: str) -> Optional[MemoryEntry]:
        return self._store.get(key)

    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """Simple keyword-based search."""
        query_lower = query.lower()
        results = []

        for entry in self._store.values():
            if memory_type and entry.memory_type != memory_type:
                continue

            # Simple relevance scoring
            content_lower = entry.content.lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            score *= entry.importance

            if score > 0:
                results.append((score, entry))

        # Sort by score and return top_k
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:top_k]]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()


class ConversationMemory:
    """Main conversation memory manager.

    Manages short-term, long-term, episodic, and semantic memory
    for multi-turn conversations.

    Example:
        >>> memory = ConversationMemory()
        >>> memory.add_turn({"role": "user", "content": "Hello"})
        >>> memory.add_turn({"role": "assistant", "content": "Hi there!"})
        >>> context = memory.get_context_for_generation()
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        conversation_id: Optional[str] = None,
    ):
        """Initialize conversation memory.

        Args:
            config: Memory configuration
            conversation_id: Unique conversation identifier
        """
        self.config = config or MemoryConfig()
        self.conversation_id = conversation_id or self._generate_id()

        # Memory stores
        self._short_term: List[Dict[str, Any]] = []
        self._entities: Dict[str, Entity] = {}
        self._facts: List[str] = []
        self._summaries: List[ConversationSummary] = []
        self._working_memory: Dict[str, Any] = {}

        # Long-term store
        self._store = InMemoryStore()

        # Extractors
        self._entity_extractor = EntityExtractor()
        self._fact_extractor = FactExtractor()

        # Metrics
        self._turn_count = 0
        self._last_summarization = 0

    def _generate_id(self) -> str:
        """Generate unique conversation ID."""
        return hashlib.md5(
            f"{time.time()}-{id(self)}".encode()
        ).hexdigest()[:12]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(text) // 4

    def add_turn(
        self,
        message: Dict[str, str],
        importance: float = 0.5,
        extract_info: bool = True,
    ) -> None:
        """Add a conversation turn to memory.

        Args:
            message: Message dict with 'role' and 'content'
            importance: Importance score for this turn
            extract_info: Whether to extract entities and facts
        """
        self._turn_count += 1

        # Add to short-term memory
        turn_data = {
            **message,
            "turn_number": self._turn_count,
            "timestamp": datetime.now().isoformat(),
            "importance": importance,
        }
        self._short_term.append(turn_data)

        content = message.get("content", "")
        role = message.get("role", "unknown")

        # Extract entities
        if extract_info and self.config.enable_entity_extraction:
            entities = self._entity_extractor.extract(content)
            for entity_type, values in entities.items():
                for value in values:
                    key = f"{entity_type}:{value.lower()}"
                    if key in self._entities:
                        self._entities[key].update(content[:100])
                    else:
                        self._entities[key] = Entity(
                            name=value,
                            entity_type=entity_type,
                            context=[content[:100]],
                        )

        # Extract facts
        if extract_info and self.config.enable_fact_extraction:
            facts = self._fact_extractor.extract(content, role)
            self._facts.extend(facts)
            # Keep facts unique and limited
            self._facts = list(dict.fromkeys(self._facts))[-50:]

        # Check if we need to summarize
        if self.config.enable_summarization:
            turns_since_summary = self._turn_count - self._last_summarization
            if turns_since_summary >= self.config.summary_threshold:
                asyncio.create_task(self._maybe_summarize())

        # Decay importance of older memories
        for turn in self._short_term[:-1]:
            turn["importance"] *= self.config.importance_decay

        # Trim short-term memory if needed
        self._trim_short_term()

    def _trim_short_term(self) -> None:
        """Trim short-term memory to fit constraints."""
        # By turn count
        while len(self._short_term) > self.config.max_short_term_turns:
            removed = self._short_term.pop(0)
            # Move to long-term if important
            if removed.get("importance", 0) > 0.3:
                asyncio.create_task(self._save_to_long_term(removed))

        # By token count
        total_tokens = sum(
            self._estimate_tokens(t.get("content", ""))
            for t in self._short_term
        )
        while total_tokens > self.config.max_short_term_tokens and len(self._short_term) > 1:
            removed = self._short_term.pop(0)
            total_tokens -= self._estimate_tokens(removed.get("content", ""))
            if removed.get("importance", 0) > 0.3:
                asyncio.create_task(self._save_to_long_term(removed))

    async def _save_to_long_term(self, turn: Dict[str, Any]) -> None:
        """Save a turn to long-term memory."""
        entry = MemoryEntry(
            id=f"turn_{turn.get('turn_number', 0)}",
            content=turn.get("content", ""),
            memory_type=MemoryType.LONG_TERM,
            importance=turn.get("importance", 0.5),
            metadata={
                "role": turn.get("role"),
                "turn_number": turn.get("turn_number"),
            },
        )
        await self._store.save(entry.id, entry)

    async def _maybe_summarize(self) -> None:
        """Summarize older turns if needed."""
        # Get turns to summarize
        turns_to_summarize = self._short_term[:-5]  # Keep last 5 turns

        if len(turns_to_summarize) < 3:
            return

        # Create simple summary (for production, use LLM)
        summary_parts = []
        for turn in turns_to_summarize:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")[:100]
            summary_parts.append(f"{role}: {content}")

        summary_text = " | ".join(summary_parts)

        # Get entities mentioned
        entities_in_range = [
            e.name for e in self._entities.values()
            if any(
                turn.get("turn_number", 0) >= self._last_summarization
                for turn in turns_to_summarize
            )
        ]

        summary = ConversationSummary(
            id=f"summary_{len(self._summaries)}",
            summary=summary_text[:500],
            turn_range=(self._last_summarization + 1, self._turn_count - 5),
            entities_mentioned=entities_in_range[:10],
        )
        self._summaries.append(summary)

        self._last_summarization = self._turn_count - 5

    def get_context_for_generation(
        self,
        max_tokens: Optional[int] = None,
        include_summary: bool = True,
        include_entities: bool = True,
        include_facts: bool = True,
    ) -> ContextWindow:
        """Get context window for response generation.

        Args:
            max_tokens: Maximum tokens for context
            include_summary: Include conversation summary
            include_entities: Include extracted entities
            include_facts: Include extracted facts

        Returns:
            ContextWindow with formatted context
        """
        max_tokens = max_tokens or self.config.max_short_term_tokens

        # Get recent messages
        messages = []
        total_tokens = 0

        for turn in reversed(self._short_term):
            turn_tokens = self._estimate_tokens(turn.get("content", ""))
            if total_tokens + turn_tokens > max_tokens:
                break
            messages.insert(0, {
                "role": turn.get("role", "user"),
                "content": turn.get("content", ""),
            })
            total_tokens += turn_tokens

        # Build summary from older summaries
        summary = None
        if include_summary and self._summaries:
            summary = self._summaries[-1].summary

        # Format entities
        entities: Dict[str, List[str]] = defaultdict(list)
        if include_entities:
            for key, entity in self._entities.items():
                entities[entity.entity_type].append(entity.name)

        # Get recent facts
        facts = self._facts[-10:] if include_facts else []

        return ContextWindow(
            messages=messages,
            summary=summary,
            entities=dict(entities),
            facts=facts,
            working_memory=self._working_memory.copy(),
            total_tokens=total_tokens,
        )

    def set_working_memory(self, key: str, value: Any) -> None:
        """Set a value in working memory.

        Working memory is for temporary task-specific context.

        Args:
            key: Memory key
            value: Value to store
        """
        self._working_memory[key] = value

    def get_working_memory(self, key: str, default: Any = None) -> Any:
        """Get a value from working memory.

        Args:
            key: Memory key
            default: Default value if not found

        Returns:
            Stored value or default
        """
        return self._working_memory.get(key, default)

    def clear_working_memory(self) -> None:
        """Clear all working memory."""
        self._working_memory.clear()

    def get_entities(self, entity_type: Optional[str] = None) -> List[Entity]:
        """Get extracted entities.

        Args:
            entity_type: Filter by entity type

        Returns:
            List of entities
        """
        entities = list(self._entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return sorted(entities, key=lambda e: e.mentions, reverse=True)

    def get_facts(self) -> List[str]:
        """Get extracted facts."""
        return self._facts.copy()

    async def search_memory(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Search long-term memory.

        Args:
            query: Search query
            memory_type: Filter by memory type
            top_k: Number of results

        Returns:
            List of relevant memory entries
        """
        top_k = top_k or self.config.retrieval_top_k
        return await self._store.search(query, memory_type, top_k)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary of memory statistics
        """
        return {
            "conversation_id": self.conversation_id,
            "turn_count": self._turn_count,
            "short_term_turns": len(self._short_term),
            "entity_count": len(self._entities),
            "fact_count": len(self._facts),
            "summary_count": len(self._summaries),
            "working_memory_keys": list(self._working_memory.keys()),
            "estimated_tokens": sum(
                self._estimate_tokens(t.get("content", ""))
                for t in self._short_term
            ),
        }

    async def clear(self) -> None:
        """Clear all memory."""
        self._short_term.clear()
        self._entities.clear()
        self._facts.clear()
        self._summaries.clear()
        self._working_memory.clear()
        await self._store.clear()
        self._turn_count = 0
        self._last_summarization = 0

    def export(self) -> Dict[str, Any]:
        """Export memory state for persistence.

        Returns:
            Serializable dictionary of memory state
        """
        return {
            "conversation_id": self.conversation_id,
            "config": {
                "max_short_term_turns": self.config.max_short_term_turns,
                "max_short_term_tokens": self.config.max_short_term_tokens,
                "enable_summarization": self.config.enable_summarization,
            },
            "short_term": self._short_term,
            "entities": {k: v.to_dict() if hasattr(v, 'to_dict') else str(v)
                       for k, v in self._entities.items()},
            "facts": self._facts,
            "summaries": [
                {
                    "id": s.id,
                    "summary": s.summary,
                    "turn_range": s.turn_range,
                    "key_points": s.key_points,
                }
                for s in self._summaries
            ],
            "working_memory": self._working_memory,
            "turn_count": self._turn_count,
        }

    @classmethod
    def from_export(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Create memory from exported state.

        Args:
            data: Exported memory data

        Returns:
            Restored ConversationMemory instance
        """
        config = MemoryConfig(**data.get("config", {}))
        memory = cls(config=config, conversation_id=data.get("conversation_id"))

        memory._short_term = data.get("short_term", [])
        memory._facts = data.get("facts", [])
        memory._turn_count = data.get("turn_count", 0)
        memory._working_memory = data.get("working_memory", {})

        return memory


# Convenience function to create memory with agent
def create_memory_enhanced_agent(agent_class, config, memory_config: Optional[MemoryConfig] = None):
    """Create an agent with enhanced memory capabilities.

    Args:
        agent_class: Agent class to enhance
        config: Agent configuration
        memory_config: Memory configuration

    Returns:
        Agent instance with memory
    """
    class MemoryEnhancedAgent(agent_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memory = ConversationMemory(memory_config)

        async def generate_response(self, messages, context=None):
            # Add messages to memory
            for msg in messages if isinstance(messages, list) else [{"role": "user", "content": messages}]:
                if msg not in [m for m in self.memory._short_term]:
                    self.memory.add_turn(msg)

            # Get context with memory
            memory_context = self.memory.get_context_for_generation()

            # Generate with enhanced context
            enhanced_messages = memory_context.to_messages()
            response = await super().generate_response(enhanced_messages, context)

            # Add response to memory
            self.memory.add_turn({"role": "assistant", "content": response})

            return response

    return MemoryEnhancedAgent(config)


__all__ = [
    "MemoryType",
    "MemoryConfig",
    "MemoryEntry",
    "Entity",
    "ConversationSummary",
    "ContextWindow",
    "EntityExtractor",
    "FactExtractor",
    "MemoryStore",
    "InMemoryStore",
    "ConversationMemory",
    "create_memory_enhanced_agent",
]
