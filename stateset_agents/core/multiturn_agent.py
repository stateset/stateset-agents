"""
Multi-turn Conversational Agent for GRPO Agent Framework

This module provides advanced multi-turn conversation capabilities with
context management, tool usage, and sophisticated dialogue strategies.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import utils.cache

from .agent import Agent
from .environment import Environment
from .reward import RewardFunction, RewardResult
from .trajectory import Trajectory

CacheService = utils.cache.CacheService

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Context for multi-turn conversations"""

    conversation_id: str
    user_id: Optional[str] = None
    topic: Optional[str] = None
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)

    def add_turn(self, turn: Dict[str, Any]):
        """Add a turn to the conversation history"""
        turn_with_timestamp = {**turn, "timestamp": datetime.now().isoformat()}
        self.history.append(turn_with_timestamp)

    def get_recent_history(self, max_turns: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.history[-max_turns:]

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation context"""
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
    """Simple searchable database of dialogue examples"""

    def __init__(self, dialogues: List[Dict[str, Any]]):
        self.dialogues = dialogues
        self.index = self._build_index()

    def _build_index(self) -> Dict[str, List[int]]:
        """Build a simple keyword index"""
        index = {}
        for i, dialogue in enumerate(self.dialogues):
            content = dialogue.get("content", "").lower()
            words = content.split()

            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(i)

        return index

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant dialogues"""
        query_words = query.lower().split()

        # Score dialogues based on word matches
        scores = {}
        for word in query_words:
            if word in self.index:
                for dialogue_idx in self.index[word]:
                    scores[dialogue_idx] = scores.get(dialogue_idx, 0) + 1

        # Sort by score and return top results
        sorted_dialogues = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for dialogue_idx, score in sorted_dialogues[:top_k]:
            dialogue = self.dialogues[dialogue_idx].copy()
            dialogue["relevance_score"] = score
            results.append(dialogue)

        return results

    def get_dialogue_by_id(self, dialogue_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific dialogue by ID"""
        for dialogue in self.dialogues:
            if dialogue.get("id") == dialogue_id:
                return dialogue
        return None


class MultiTurnAgent(Agent):
    """
    Advanced multi-turn conversational agent with context management
    and sophisticated dialogue strategies.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        max_context_length: int = 2048,
        max_conversation_turns: int = 20,
        context_compression_threshold: float = 0.8,
        dialogue_database: Optional[DialogueDatabase] = None,
        cache_service: Optional[CacheService] = None,
        **kwargs,
    ):
        super().__init__(model_config, **kwargs)

        self.max_context_length = max_context_length
        self.max_conversation_turns = max_conversation_turns
        self.context_compression_threshold = context_compression_threshold
        self.dialogue_database = dialogue_database
        self.cache = cache_service

        # Active conversations
        self.active_conversations: Dict[str, ConversationContext] = {}

        # Conversation strategies
        self.strategies = {
            "default": self._default_strategy,
            "customer_service": self._customer_service_strategy,
            "technical_support": self._technical_support_strategy,
            "educational": self._educational_strategy,
            "sales": self._sales_strategy,
        }

        # Tool registry
        self.tools = {}

        # Response generation backend controls (pluggable)
        self._use_hf_backend: bool = bool(kwargs.get("use_hf_backend", False))
        self._hf_backend_config: Dict[str, Any] = dict(kwargs.get("hf_backend_config", {}) or {})
        self._hf_agent = None  # Lazy-initialized HF-backed agent
        self.response_backend: Optional[
            Callable[[Union[str, List[Dict[str, Any]]]], Union[str, Awaitable[str]]]
        ] = kwargs.get("response_backend")

    def register_tool(self, name: str, tool_func: callable):
        """Register a tool for the agent to use"""
        self.tools[name] = tool_func

    async def generate_response(
        self, messages_or_prompt: Union[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate a response using pluggable backends.

        Order of precedence:
        1) `response_backend` callable if provided (sync or async)
        2) HF backend (lazy-initialized) if `use_hf_backend=True`
        3) Lightweight heuristic response (fast default for tests)
        """
        # 1) Custom response backend
        if self.response_backend is not None:
            try:
                maybe = self.response_backend(messages_or_prompt)
                if asyncio.iscoroutine(maybe):
                    return await maybe  # type: ignore[return-value]
                return str(maybe)
            except Exception as e:
                logger.warning("Custom backend failed, falling back: %s", e)

        # 2) HF backend
        if self._use_hf_backend:
            try:
                hf_resp = await self._generate_with_hf_backend(messages_or_prompt)
                if isinstance(hf_resp, str) and hf_resp:
                    return hf_resp
            except Exception as e:  # pragma: no cover - safety fallback
                logger.warning("HF backend failed, falling back to heuristic: %s", e)

        # 3) Heuristic fallback
        return self._generate_heuristic(messages_or_prompt)

    def register_response_backend(
        self,
        backend: Callable[[Union[str, List[Dict[str, Any]]]], Union[str, Awaitable[str]]],
    ) -> None:
        """Register a custom response backend callable."""
        self.response_backend = backend

    def _generate_heuristic(
        self, messages_or_prompt: Union[str, List[Dict[str, Any]]]
    ) -> str:
        """Fast deterministic response useful for tests and simple flows."""
        if isinstance(messages_or_prompt, str):
            last_user = messages_or_prompt
        else:
            last_user = ""
            for m in reversed(messages_or_prompt):
                if m.get("role") == "user":
                    last_user = str(m.get("content", ""))
                    break

        base = "I understand. Here's a helpful response to your message."
        lower = last_user.lower()

        # Small keyword hints for friendlier replies in tests
        if any(k in lower for k in ("order", "billing", "charge")):
            base = "I’m here to help with your account and order questions."
        elif any(k in lower for k in ("error", "bug", "issue", "technical")):
            base = "I can assist with technical issues. Let’s troubleshoot together."
        elif any(k in lower for k in ("learn", "explain", "teach")):
            base = "Here’s a concise explanation to help you learn this quickly."

        return base

    async def _generate_with_hf_backend(
        self, messages_or_prompt: Union[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate using the HF-backed agent from `core.agent` (lazy)."""
        if self._hf_agent is None:
            # Lazy import to avoid heavy deps when unused
            from .agent import AgentConfig as HFConfig
            from .agent import MultiTurnAgent as HFMultiTurnAgent

            model_name = (
                self._hf_backend_config.get("model_name")
                or self.model_config.get("model_name")
                or "gpt2"
            )
            cfg_kwargs = {
                k: v for k, v in self._hf_backend_config.items() if k != "model_name"
            }
            hf_cfg = HFConfig(model_name=model_name, **cfg_kwargs)
            hf_agent = HFMultiTurnAgent(hf_cfg)
            await hf_agent.initialize()
            self._hf_agent = hf_agent

        if isinstance(messages_or_prompt, str):
            messages = [{"role": "user", "content": messages_or_prompt}]
        else:
            messages = messages_or_prompt

        return await self._hf_agent.generate_response(messages)  # type: ignore[attr-defined]

    async def start_conversation(
        self,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> ConversationContext:
        """Start a new conversation"""
        conversation_id = str(uuid.uuid4())

        context = ConversationContext(
            conversation_id=conversation_id, user_id=user_id, **(initial_context or {})
        )

        self.active_conversations[conversation_id] = context

        logger.info(f"Started conversation {conversation_id} for user {user_id}")

        return context

    async def continue_conversation(
        self,
        conversation_id: str,
        user_message: str,
        strategy: str = "default",
        max_turns: int = 5,
    ) -> List[Dict[str, Any]]:
        """Continue an existing conversation"""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        context = self.active_conversations[conversation_id]

        # Add user message to context
        context.add_turn({"role": "user", "content": user_message})

        # Generate response using selected strategy
        strategy_func = self.strategies.get(strategy, self._default_strategy)
        response = await strategy_func(context)

        # Add assistant response to context
        context.add_turn({"role": "assistant", "content": response})

        # Return recent conversation history
        return context.get_recent_history(max_turns)

    async def generate_multiturn_response(
        self,
        conversation_id: str,
        user_message: str,
        strategy: str = "default",
        use_tools: bool = True,
    ) -> str:
        """Generate a single response in a multi-turn conversation"""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        context = self.active_conversations[conversation_id]

        # Check if context needs compression
        if len(context.history) > self.max_conversation_turns:
            await self._compress_context(context)

        # Update context with user message
        context.add_turn({"role": "user", "content": user_message})

        # Analyze user intent and entities
        await self._analyze_user_input(context, user_message)

        # Generate response using strategy
        strategy_func = self.strategies.get(strategy, self._default_strategy)
        response = await strategy_func(context)

        # Apply tools if requested
        if use_tools:
            response = await self._apply_tools(context, response)

        return response

    async def _default_strategy(self, context: ConversationContext) -> str:
        """Default conversation strategy"""
        # Get recent history
        history = context.get_recent_history(5)

        # Build prompt
        prompt = self._build_prompt(history, context)

        # Generate response
        response = await self.generate_response(prompt)

        return response

    async def _customer_service_strategy(self, context: ConversationContext) -> str:
        """Customer service conversation strategy"""
        # Search for relevant dialogue examples
        recent_user_messages = [
            turn["content"] for turn in context.history if turn.get("role") == "user"
        ]

        if recent_user_messages and self.dialogue_database:
            query = recent_user_messages[-1]
            examples = self.dialogue_database.search(query, top_k=2)

            # Use examples to inform response
            if examples:
                context.metadata["relevant_examples"] = examples

        # Build customer service prompt
        prompt = self._build_customer_service_prompt(context)

        # Generate response
        response = await self.generate_response(prompt)

        return response

    async def _technical_support_strategy(self, context: ConversationContext) -> str:
        """Technical support conversation strategy"""
        # Extract technical entities
        last_user_message = (
            context.history[-1].get("content", "") if context.history else ""
        )

        # Look for technical terms
        technical_terms = self._extract_technical_terms(last_user_message)
        if technical_terms:
            context.entities.update({"technical_terms": technical_terms})

        # Build technical support prompt
        prompt = self._build_technical_support_prompt(context)

        # Generate response
        response = await self.generate_response(prompt)

        return response

    async def _educational_strategy(self, context: ConversationContext) -> str:
        """Educational conversation strategy"""
        # Adapt to learner level
        user_level = context.metadata.get("user_level", "beginner")

        # Build educational prompt
        prompt = self._build_educational_prompt(context, user_level)

        # Generate response
        response = await self.generate_response(prompt)

        return response

    async def _sales_strategy(self, context: ConversationContext) -> str:
        """Sales conversation strategy"""
        # Track sales funnel stage
        stage = context.metadata.get("sales_stage", "awareness")

        # Build sales prompt
        prompt = self._build_sales_prompt(context, stage)

        # Generate response
        response = await self.generate_response(prompt)

        return response

    def _build_prompt(
        self, history: List[Dict[str, Any]], context: ConversationContext
    ) -> str:
        """Build prompt from conversation history"""
        prompt_parts = []

        # Add system context
        if context.topic:
            prompt_parts.append(f"Topic: {context.topic}")

        if context.intent:
            prompt_parts.append(f"Intent: {context.intent}")

        # Add conversation history
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")

        # Add assistant prompt
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _build_customer_service_prompt(self, context: ConversationContext) -> str:
        """Build customer service specific prompt"""
        prompt_parts = [
            "You are a helpful customer service representative.",
            "Be empathetic, professional, and solution-focused.",
            "",
        ]

        # Add relevant examples if available
        examples = context.metadata.get("relevant_examples", [])
        if examples:
            prompt_parts.append("Similar situations:")
            for example in examples:
                prompt_parts.append(f"- {example.get('content', '')[:100]}...")
            prompt_parts.append("")

        # Add conversation history
        history = context.get_recent_history(5)
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")

        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _build_technical_support_prompt(self, context: ConversationContext) -> str:
        """Build technical support specific prompt"""
        prompt_parts = [
            "You are a technical support specialist.",
            "Provide accurate, step-by-step solutions.",
            "Be clear and precise in your explanations.",
            "",
        ]

        # Add technical context
        technical_terms = context.entities.get("technical_terms", [])
        if technical_terms:
            prompt_parts.append(
                f"Technical terms mentioned: {', '.join(technical_terms)}"
            )
            prompt_parts.append("")

        # Add conversation history
        history = context.get_recent_history(5)
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")

        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _build_educational_prompt(
        self, context: ConversationContext, user_level: str
    ) -> str:
        """Build educational specific prompt"""
        prompt_parts = [
            "You are a patient and knowledgeable educator.",
            f"Adapt your explanations to a {user_level} level.",
            "Use examples and check for understanding.",
            "",
        ]

        # Add conversation history
        history = context.get_recent_history(5)
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")

        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _build_sales_prompt(self, context: ConversationContext, stage: str) -> str:
        """Build sales specific prompt"""
        prompt_parts = [
            "You are a professional sales consultant.",
            f"The customer is in the {stage} stage of the sales funnel.",
            "Focus on understanding needs and providing value.",
            "",
        ]

        # Add conversation history
        history = context.get_recent_history(5)
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")

        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    async def _analyze_user_input(
        self, context: ConversationContext, user_message: str
    ):
        """Analyze user input for intent and entities"""
        # Simple intent detection
        if any(
            word in user_message.lower()
            for word in ["help", "problem", "issue", "error"]
        ):
            context.intent = "support"
        elif any(
            word in user_message.lower()
            for word in ["buy", "purchase", "price", "cost"]
        ):
            context.intent = "sales"
        elif any(
            word in user_message.lower() for word in ["learn", "how", "what", "explain"]
        ):
            context.intent = "education"

        # Simple entity extraction
        entities = {}

        # Extract numbers
        import re

        numbers = re.findall(r"\d+", user_message)
        if numbers:
            entities["numbers"] = numbers

        # Extract email addresses
        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", user_message
        )
        if emails:
            entities["emails"] = emails

        # Update context
        context.entities.update(entities)

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text"""
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

        terms = []
        text_lower = text.lower()

        for term in technical_keywords:
            if term in text_lower:
                terms.append(term)

        return terms

    async def _apply_tools(self, context: ConversationContext, response: str) -> str:
        """Apply tools based on response content"""
        if not self.tools:
            return response

        # Check if response suggests using tools
        if "search" in response.lower() and "search" in self.tools:
            # Extract search query
            query = context.history[-1].get("content", "") if context.history else ""

            try:
                search_results = await self.tools["search"](query)
                response += f"\n\nSearch results: {search_results}"
            except Exception as e:
                logger.error(f"Search tool failed: {e}")

        if "read" in response.lower() and "read" in self.tools:
            # Extract document ID or reference
            doc_id = self._extract_document_reference(response)
            if doc_id:
                try:
                    content = await self.tools["read"](doc_id)
                    response += f"\n\nDocument content: {content}"
                except Exception as e:
                    logger.error(f"Read tool failed: {e}")

        return response

    def _extract_document_reference(self, text: str) -> Optional[str]:
        """Extract document reference from text"""
        # Simple pattern matching for document IDs
        import re

        # Look for patterns like "doc_123", "document-456", etc.
        pattern = r"(?:doc|document)[-_]?(\w+)"
        matches = re.findall(pattern, text.lower())

        return matches[0] if matches else None

    async def _compress_context(self, context: ConversationContext):
        """Compress conversation context to manage memory"""
        if len(context.history) <= self.max_conversation_turns:
            return

        # Keep first few turns (conversation start)
        keep_start = 2

        # Keep last few turns (recent context)
        keep_end = self.max_conversation_turns - keep_start - 1

        # Create summary of middle turns
        middle_turns = context.history[keep_start:-keep_end]

        if middle_turns:
            # Create a summary
            summary = self._create_conversation_summary(middle_turns)

            # Replace middle turns with summary
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

    def _create_conversation_summary(self, turns: List[Dict[str, Any]]) -> str:
        """Create a summary of conversation turns"""
        user_messages = [
            turn["content"] for turn in turns if turn.get("role") == "user"
        ]
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

    async def compute_multiturn_rewards(
        self,
        conversation_id: str,
        reward_functions: List[RewardFunction],
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute rewards for multi-turn conversation"""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        context = self.active_conversations[conversation_id]
        turns = context.history

        # Compute rewards from each function
        total_reward = 0.0
        reward_breakdown = {}

        for reward_func in reward_functions:
            try:
                result = await reward_func.compute_reward(
                    turns, context.get_context_summary()
                )
                total_reward += result.score * reward_func.weight
                reward_breakdown[reward_func.__class__.__name__] = result.score
            except Exception as e:
                logger.error(
                    f"Reward function {reward_func.__class__.__name__} failed: {e}"
                )
                reward_breakdown[reward_func.__class__.__name__] = 0.0

        # Add conversation-specific rewards
        reward_breakdown.update(
            {
                "conversation_length": min(1.0, len(turns) / 10),
                "context_coherence": self._evaluate_context_coherence(context),
                "goal_achievement": self._evaluate_goal_achievement(
                    context, ground_truth
                ),
            }
        )

        return total_reward, reward_breakdown

    def _evaluate_context_coherence(self, context: ConversationContext) -> float:
        """Evaluate how coherent the conversation context is"""
        if len(context.history) < 2:
            return 1.0

        # Simple coherence check based on topic consistency
        topics = []
        for turn in context.history:
            if turn.get("role") == "user":
                content = turn.get("content", "").lower()
                # Extract potential topics (simplified)
                words = content.split()
                topics.extend(words)

        if not topics:
            return 0.5

        # Check topic consistency
        unique_topics = set(topics)
        topic_consistency = len(unique_topics) / len(topics)

        return max(0.0, 1.0 - topic_consistency)

    def _evaluate_goal_achievement(
        self, context: ConversationContext, ground_truth: Optional[Dict[str, Any]]
    ) -> float:
        """Evaluate if conversation achieved its goal"""
        if not ground_truth:
            return 0.5  # Neutral if no ground truth

        expected_outcome = ground_truth.get("expected_outcome", "")
        if not expected_outcome:
            return 0.5

        # Check if last assistant response matches expected outcome
        assistant_responses = [
            turn["content"]
            for turn in context.history
            if turn.get("role") == "assistant"
        ]

        if not assistant_responses:
            return 0.0

        last_response = assistant_responses[-1].lower()
        expected_lower = expected_outcome.lower()

        # Simple keyword matching
        expected_words = set(expected_lower.split())
        response_words = set(last_response.split())

        overlap = expected_words & response_words
        if expected_words:
            return len(overlap) / len(expected_words)

        return 0.0

    def end_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """End a conversation and return its context"""
        if conversation_id in self.active_conversations:
            context = self.active_conversations.pop(conversation_id)
            logger.info(f"Ended conversation {conversation_id}")
            return context
        return None

    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs"""
        return list(self.active_conversations.keys())

    def get_conversation_summary(
        self, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get summary of a conversation"""
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id].get_context_summary()
        return None
