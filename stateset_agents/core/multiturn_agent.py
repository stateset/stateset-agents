"""
Multi-turn Conversational Agent for GRPO Agent Framework

This module provides advanced multi-turn conversation capabilities with
context management, tool usage, and sophisticated dialogue strategies.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from stateset_agents.utils.cache import CacheService

from .agent import Agent
from .long_term_planning import PlanningConfig, PlanningManager
from .multiturn_analysis import (
    analyze_user_input,
    compress_conversation_context,
    create_conversation_summary,
    evaluate_context_coherence,
    evaluate_goal_achievement,
    extract_document_reference,
    extract_technical_terms,
)
from .multiturn_context import (
    ConversationContext,
    DialogueDatabase,
    apply_context_update,
)
from .reward import RewardFunction

logger = logging.getLogger(__name__)

MULTITURN_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
    asyncio.TimeoutError,
)
BACKEND_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    OSError,
    asyncio.TimeoutError,
)
TOOL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    OSError,
    asyncio.TimeoutError,
)

class MultiTurnAgent(Agent):
    """
    Advanced multi-turn conversational agent with context management
    and sophisticated dialogue strategies.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        max_context_length: int = 2048,
        max_conversation_turns: int = 20,
        context_compression_threshold: float = 0.8,
        dialogue_database: DialogueDatabase | None = None,
        cache_service: CacheService | None = None,
        planning_manager: PlanningManager | None = None,
        **kwargs,
    ):
        super().__init__(model_config, **kwargs)

        self.max_context_length = max_context_length
        self.max_conversation_turns = max_conversation_turns
        self.context_compression_threshold = context_compression_threshold
        self.dialogue_database = dialogue_database
        self.cache = cache_service
        if planning_manager is None:
            enable_planning = bool(model_config.get("enable_planning", False))
            if enable_planning:
                planning_kwargs = model_config.get("planning_config", {}) or {}
                try:
                    if isinstance(planning_kwargs, dict):
                        planning_kwargs = dict(planning_kwargs)
                        planning_kwargs.pop("enabled", None)
                    planning_cfg = PlanningConfig(enabled=True, **planning_kwargs)
                    planning_manager = PlanningManager(planning_cfg)
                except MULTITURN_EXCEPTIONS as exc:
                    logger.warning("Failed to init PlanningManager: %s", exc)
        self.planning_manager = planning_manager

        # Active conversations
        self.active_conversations: dict[str, ConversationContext] = {}

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
        self._hf_backend_config: dict[str, Any] = dict(
            kwargs.get("hf_backend_config", {}) or {}
        )
        self._hf_agent = None  # Lazy-initialized HF-backed agent
        self.response_backend: Callable[[str | list[dict[str, Any]]], str | Awaitable[str]] | None = kwargs.get("response_backend")

    def register_tool(self, name: str, tool_func: callable):
        """Register a tool for the agent to use"""
        self.tools[name] = tool_func

    async def generate_response(
        self,
        messages_or_prompt: str | list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a response using pluggable backends.

        Order of precedence:
        1) `response_backend` callable if provided (sync or async)
        2) HF backend (lazy-initialized) if `use_hf_backend=True`
        3) Lightweight heuristic response (fast default for tests)
        """
        messages_for_backend = messages_or_prompt
        if self.planning_manager is not None and isinstance(messages_or_prompt, list):
            plan_message = self.planning_manager.build_plan_message(
                messages_or_prompt, context=context or {}
            )
            if plan_message is not None:
                messages_for_backend = list(messages_or_prompt)
                insert_at = 0
                while (
                    insert_at < len(messages_for_backend)
                    and messages_for_backend[insert_at].get("role") == "system"
                ):
                    insert_at += 1
                messages_for_backend.insert(insert_at, plan_message)

        # 1) Custom response backend
        if self.response_backend is not None:
            try:
                maybe = self.response_backend(messages_for_backend)
                if asyncio.iscoroutine(maybe):
                    return await maybe  # type: ignore[return-value]
                return str(maybe)
            except BACKEND_EXCEPTIONS as e:
                logger.warning("Custom backend failed, falling back: %s", e)

        # 2) HF backend
        if self._use_hf_backend:
            try:
                hf_resp = await self._generate_with_hf_backend(messages_for_backend)
                if isinstance(hf_resp, str) and hf_resp:
                    return hf_resp
            except BACKEND_EXCEPTIONS as e:  # pragma: no cover - safety fallback
                logger.warning("HF backend failed, falling back to heuristic: %s", e)

        # 3) Heuristic fallback
        return self._generate_heuristic(messages_for_backend)

    def register_response_backend(
        self,
        backend: Callable[
            [str | list[dict[str, Any]]], str | Awaitable[str]
        ],
    ) -> None:
        """Register a custom response backend callable."""
        self.response_backend = backend

    def _generate_heuristic(
        self, messages_or_prompt: str | list[dict[str, Any]]
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
        self, messages_or_prompt: str | list[dict[str, Any]]
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
        user_id: str | None = None,
        initial_context: dict[str, Any] | None = None,
    ) -> ConversationContext:
        """Start a new conversation"""
        conversation_id = str(uuid.uuid4())

        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        self._apply_context_update(context, initial_context)

        self.active_conversations[conversation_id] = context

        logger.info(f"Started conversation {conversation_id} for user {user_id}")

        return context

    def _apply_context_update(
        self,
        context: ConversationContext,
        update: dict[str, Any] | None,
    ) -> None:
        """Merge context updates into the active conversation context."""
        apply_context_update(context, update)

    async def continue_conversation(
        self,
        conversation_id: str,
        user_message: str,
        strategy: str = "default",
        max_turns: int = 5,
        context_update: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Continue an existing conversation, applying any context updates."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        context = self.active_conversations[conversation_id]
        self._apply_context_update(context, context_update)

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
        context_update: dict[str, Any] | None = None,
    ) -> str:
        """Generate a single response in a multi-turn conversation."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        context = self.active_conversations[conversation_id]
        self._apply_context_update(context, context_update)

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

        context.add_turn({"role": "assistant", "content": response})
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

    def _build_planning_context(self, context: ConversationContext) -> dict[str, Any]:
        plan_context = dict(context.metadata) if context.metadata else {}
        plan_update = plan_context.pop("plan_update", None)
        if plan_update is not None:
            context.metadata.pop("plan_update", None)
            plan_context["plan_update"] = plan_update
        plan_goal = plan_context.pop("plan_goal", None)
        if plan_goal is not None:
            context.metadata.pop("plan_goal", None)
            plan_context["plan_goal"] = plan_goal

        plan_context["conversation_id"] = context.conversation_id
        if context.user_id:
            plan_context["user_id"] = context.user_id
        if context.topic:
            plan_context["topic"] = context.topic
        if context.intent:
            plan_context["intent"] = context.intent
        if context.entities:
            plan_context["entities"] = context.entities

        return plan_context

    def _get_plan_summary(
        self,
        history: list[dict[str, Any]],
        context: ConversationContext,
    ) -> str | None:
        if self.planning_manager is None:
            return None

        plan_message = self.planning_manager.build_plan_message(
            history, context=self._build_planning_context(context)
        )
        if plan_message and plan_message.get("content"):
            return str(plan_message["content"])
        return None

    def _build_prompt(
        self, history: list[dict[str, Any]], context: ConversationContext
    ) -> str:
        """Build prompt from conversation history"""
        prompt_parts = []

        # Add system context
        if context.topic:
            prompt_parts.append(f"Topic: {context.topic}")

        if context.intent:
            prompt_parts.append(f"Intent: {context.intent}")

        plan_summary = self._get_plan_summary(history, context)
        if plan_summary:
            prompt_parts.append(plan_summary)
            prompt_parts.append("")

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

        history = context.get_recent_history(5)
        plan_summary = self._get_plan_summary(history, context)
        if plan_summary:
            prompt_parts.append(plan_summary)
            prompt_parts.append("")

        # Add conversation history
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

        history = context.get_recent_history(5)
        plan_summary = self._get_plan_summary(history, context)
        if plan_summary:
            prompt_parts.append(plan_summary)
            prompt_parts.append("")

        # Add conversation history
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

        history = context.get_recent_history(5)
        plan_summary = self._get_plan_summary(history, context)
        if plan_summary:
            prompt_parts.append(plan_summary)
            prompt_parts.append("")

        # Add conversation history
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

        history = context.get_recent_history(5)
        plan_summary = self._get_plan_summary(history, context)
        if plan_summary:
            prompt_parts.append(plan_summary)
            prompt_parts.append("")

        # Add conversation history
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
        analyze_user_input(context, user_message)

    def _extract_technical_terms(self, text: str) -> list[str]:
        """Extract technical terms from text"""
        return extract_technical_terms(text)

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
            except TOOL_EXCEPTIONS as e:
                logger.error(f"Search tool failed: {e}")

        if "read" in response.lower() and "read" in self.tools:
            # Extract document ID or reference
            doc_id = self._extract_document_reference(response)
            if doc_id:
                try:
                    content = await self.tools["read"](doc_id)
                    response += f"\n\nDocument content: {content}"
                except TOOL_EXCEPTIONS as e:
                    logger.error(f"Read tool failed: {e}")

        return response

    def _extract_document_reference(self, text: str) -> str | None:
        """Extract document reference from text"""
        return extract_document_reference(text)

    async def _compress_context(self, context: ConversationContext):
        """Compress conversation context to manage memory"""
        compress_conversation_context(context, self.max_conversation_turns)

    def _create_conversation_summary(self, turns: list[dict[str, Any]]) -> str:
        """Create a summary of conversation turns"""
        return create_conversation_summary(turns)

    async def compute_multiturn_rewards(
        self,
        conversation_id: str,
        reward_functions: list[RewardFunction],
        ground_truth: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, float]]:
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
            except MULTITURN_EXCEPTIONS as e:
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
        return evaluate_context_coherence(context)

    def _evaluate_goal_achievement(
        self, context: ConversationContext, ground_truth: dict[str, Any] | None
    ) -> float:
        """Evaluate if conversation achieved its goal"""
        return evaluate_goal_achievement(context, ground_truth)

    def end_conversation(self, conversation_id: str) -> ConversationContext | None:
        """End a conversation and return its context"""
        if conversation_id in self.active_conversations:
            context = self.active_conversations.pop(conversation_id)
            if self.planning_manager is not None:
                self.planning_manager.clear_conversation(conversation_id)
            logger.info(f"Ended conversation {conversation_id}")
            return context
        return None

    def get_active_conversations(self) -> list[str]:
        """Get list of active conversation IDs"""
        return list(self.active_conversations.keys())

    def get_conversation_summary(
        self, conversation_id: str
    ) -> dict[str, Any] | None:
        """Get summary of a conversation"""
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            summary = context.get_context_summary()
            if self.planning_manager is not None:
                plan = self.planning_manager.get_plan(conversation_id)
                if plan is not None:
                    summary["plan"] = {
                        "goal": plan.goal,
                        "progress": plan.progress(),
                        "summary": plan.summarize(
                            max_steps=self.planning_manager.config.max_steps,
                            keep_completed=self.planning_manager.config.keep_completed,
                        ),
                    }
            return summary
        return None


__all__ = [
    "ConversationContext",
    "DialogueDatabase",
    "MultiTurnAgent",
]
