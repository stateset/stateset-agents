"""
Multi-turn Conversational Agent for GRPO Agent Framework

This module provides advanced multi-turn conversation capabilities with
context management, tool usage, and sophisticated dialogue strategies.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from stateset_agents.utils.cache import CacheService

from .agent import Agent
from .agent_config import AgentConfig
from .conversation import ConversationResetError, conversation_call
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
    ConversationClosedError,
    ConversationConflict,
    ConversationContext,
    DialogueDatabase,
    apply_context_update,
)
from .multiturn_lifecycle import (
    check_turn,
    consume_planning_commands,
    planning_commit,
    run_turn,
)
from .reward import RewardFunction
from .trajectory import ConversationTurn

if TYPE_CHECKING:
    from ..experimental.long_term_planning import PlanningManager

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

    _conversation_snapshots: dict[str, Any]

    def __init__(
        self,
        model_config: dict[str, Any],
        max_context_length: int = 2048,
        max_conversation_turns: int = 20,
        context_compression_threshold: float = 0.8,
        dialogue_database: DialogueDatabase | None = None,
        cache_service: CacheService | None = None,
        planning_manager: PlanningManager | None = None,
        **kwargs: Any,
    ) -> None:
        agent_config_kwargs = {
            field_name: value
            for field_name, value in model_config.items()
            if field_name in AgentConfig.__dataclass_fields__
        }
        if "model_name" not in agent_config_kwargs:
            agent_config_kwargs["model_name"] = "stub://test"
        use_hf_backend = bool(kwargs.pop("use_hf_backend", False))
        hf_backend_config = dict(kwargs.pop("hf_backend_config", {}) or {})
        response_backend = kwargs.pop("response_backend", None)
        super().__init__(AgentConfig(**agent_config_kwargs), **kwargs)
        self.model_config = dict(model_config)

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
                    from ..experimental.long_term_planning import (
                        PlanningConfig,
                        PlanningManager,
                    )

                    if isinstance(planning_kwargs, dict):
                        planning_kwargs = dict(planning_kwargs)
                        planning_kwargs.pop("enabled", None)
                    planning_cfg = PlanningConfig(enabled=True, **planning_kwargs)
                    planning_manager = PlanningManager(planning_cfg)
                except MULTITURN_EXCEPTIONS as exc:
                    logger.warning("Failed to init PlanningManager: %s", exc)
        self.planning_manager = planning_manager

        # Arbitrary caller-attached metadata (e.g. the API layer records the
        # checkpoint path/base model a registered agent was loaded from here
        # so /v1/agents/{id} can surface it). Not used by the agent itself.
        self.metadata: dict[str, Any] = {}

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
        self.tools: dict[str, Callable[..., Any]] = {}

        # Response generation backend controls (pluggable)
        self._use_hf_backend = use_hf_backend
        self._hf_backend_config = hf_backend_config
        self._hf_agent: Agent | None = None  # Lazy-initialized HF-backed agent
        self.response_backend: (
            Callable[[str | list[dict[str, Any]]], str | Awaitable[str]] | None
        ) = response_backend

    def register_tool(self, name: str, tool_func: Callable[..., Any]) -> None:
        """Register a tool for the agent to use"""
        self.tools[name] = tool_func

    @conversation_call
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
        check_turn(self)
        messages_for_backend = messages_or_prompt
        if self.planning_manager is not None and isinstance(messages_or_prompt, list):
            from ..experimental.long_term_planning import _planning_scope

            with _planning_scope(lambda: planning_commit(self)):
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
                check_turn(self)
                maybe = self.response_backend(messages_for_backend)
                if inspect.isawaitable(maybe):
                    maybe = await maybe
                check_turn(self)
                return str(maybe)
            except ConversationResetError:
                raise
            except BACKEND_EXCEPTIONS as e:
                logger.warning("Custom backend failed, falling back: %s", e)

        # 2) HF backend
        if self._use_hf_backend:
            try:
                check_turn(self)
                hf_resp = await self._generate_with_hf_backend(messages_for_backend)
                check_turn(self)
                if isinstance(hf_resp, str) and hf_resp:
                    return hf_resp
            except ConversationResetError:
                raise
            except BACKEND_EXCEPTIONS as e:  # pragma: no cover - safety fallback
                logger.warning("HF backend failed, falling back to heuristic: %s", e)

        # 3) Heuristic fallback
        check_turn(self)
        return self._generate_heuristic(messages_for_backend)

    def register_response_backend(
        self,
        backend: Callable[[str | list[dict[str, Any]]], str | Awaitable[str]],
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
            backend_agent = HFMultiTurnAgent(hf_cfg)
            await backend_agent.initialize()
            self._hf_agent = backend_agent

        if isinstance(messages_or_prompt, str):
            messages = [{"role": "user", "content": messages_or_prompt}]
        else:
            messages = messages_or_prompt

        hf_agent = self._hf_agent
        if hf_agent is None:
            raise RuntimeError("HF backend agent was not initialized")
        return str(await hf_agent.generate_response(messages))

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

        with self._conversation_owner().lock:
            context._owner()
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

    @conversation_call
    async def continue_conversation(
        self,
        conversation_id: str,
        user_message: str,
        strategy: str = "default",
        max_turns: int = 5,
        context_update: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Continue an existing conversation, applying any context updates."""
        context = await run_turn(
            self,
            conversation_id,
            user_message,
            strategy,
            context_update,
            analyze=False,
            use_tools=False,
        )
        return context.get_recent_history(max_turns)

    def get_conversation_status(
        self, conversation_id: str | None = None
    ) -> dict[str, Any]:
        """Inspect a named context, or the agent's reset scope when no ID is given."""
        if conversation_id is None:
            return super().get_conversation_status()
        with self._conversation_owner().lock:
            context = self.active_conversations.get(conversation_id)
            if context is None:
                raise ValueError(f"Conversation {conversation_id} not found")
            return context.get_status()

    @conversation_call
    async def save_conversation(
        self,
        conversation_id: str,
        path: str | os.PathLike[str],
        *,
        expected_revision: str | None = None,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Save a settled named context as portable JSON and return its receipt.

        The parent directory must exist. ``None`` is create-only; replacing a
        file requires its current revision. Later turns may proceed after capture.
        Built-in planning state is included; model/tool implementations are not.
        Inspect the receipt after interruption before deciding whether to retry.
        """
        from .conversation_snapshot import save_conversation

        return await save_conversation(
            self, conversation_id, path, expected_revision, max_bytes
        )

    @conversation_call
    async def restore_conversation(
        self,
        path: str | os.PathLike[str],
        *,
        expected_revision: str | None = None,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> ConversationContext:
        """Attach a validated snapshot without replacing an existing conversation.

        Uses this agent's backend and tool registry. Failed turns are not replayed.
        Planning snapshots require a built-in manager on the destination agent;
        its planner callbacks and configuration are preserved, never deserialized.
        This does not restore a gateway's separate user/access-control registry.
        """
        from .conversation_snapshot import restore_conversation

        return await restore_conversation(self, path, expected_revision, max_bytes)

    @staticmethod
    async def inspect_conversation_snapshot(
        path: str | os.PathLike[str], *, max_bytes: int = 64 * 1024 * 1024
    ) -> dict[str, Any]:
        """Validate snapshot metadata without attaching context or invoking tools."""
        from .conversation_snapshot import inspect_snapshot

        return await inspect_snapshot(path, max_bytes)

    def get_conversation_snapshot_status(self) -> dict[str, Any]:
        """Return detached active/latest snapshot receipts, including file paths."""
        from .conversation_snapshot import get_snapshot_status

        return get_snapshot_status(self)

    @conversation_call
    async def generate_multiturn_response(
        self,
        conversation_id: str,
        user_message: str,
        strategy: str = "default",
        use_tools: bool = True,
        context_update: dict[str, Any] | None = None,
    ) -> str:
        """Generate a single response in a multi-turn conversation."""
        context = await run_turn(
            self,
            conversation_id,
            user_message,
            strategy,
            context_update,
            analyze=True,
            use_tools=use_tools,
        )
        return str(context.history[-1]["content"])

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
        consume_planning_commands(self, context)
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

        from ..experimental.long_term_planning import _planning_scope

        with _planning_scope(lambda: planning_commit(self)):
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
    ) -> None:
        """Analyze user input for intent and entities"""
        analyze_user_input(context, user_message)

    def _extract_technical_terms(self, text: str) -> list[str]:
        """Extract technical terms from text"""
        return list(extract_technical_terms(text))

    async def _apply_tools(self, context: ConversationContext, response: str) -> str:
        """Apply tools based on response content"""
        check_turn(self)
        if not self.tools:
            return response

        # Check if response suggests using tools
        if "search" in response.lower() and "search" in self.tools:
            # Extract search query
            query = context.history[-1].get("content", "") if context.history else ""

            try:
                check_turn(self)
                search_results = await self.tools["search"](query)
                check_turn(self)
                response += f"\n\nSearch results: {search_results}"
            except ConversationResetError:
                raise
            except TOOL_EXCEPTIONS as e:
                logger.error(f"Search tool failed: {e}")

        if "read" in response.lower() and "read" in self.tools:
            # Extract document ID or reference
            doc_id = self._extract_document_reference(response)
            if doc_id:
                try:
                    check_turn(self)
                    content = await self.tools["read"](doc_id)
                    check_turn(self)
                    response += f"\n\nDocument content: {content}"
                except ConversationResetError:
                    raise
                except TOOL_EXCEPTIONS as e:
                    logger.error(f"Read tool failed: {e}")

        return response

    def _extract_document_reference(self, text: str) -> str | None:
        """Extract document reference from text"""
        reference = extract_document_reference(text)
        return None if reference is None else str(reference)

    async def _compress_context(self, context: ConversationContext) -> None:
        """Compress conversation context to manage memory"""
        compress_conversation_context(context, self.max_conversation_turns)

    def _create_conversation_summary(self, turns: list[dict[str, Any]]) -> str:
        """Create a summary of conversation turns"""
        return str(create_conversation_summary(turns))

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
        reward_turns = [
            ConversationTurn(
                role=str(turn.get("role", "assistant")),
                content=str(turn.get("content", "")),
                metadata={
                    key: value
                    for key, value in turn.items()
                    if key not in {"role", "content"}
                },
            )
            for turn in turns
        ]

        # Compute rewards from each function
        total_reward = 0.0
        reward_breakdown = {}

        for reward_func in reward_functions:
            try:
                result = await reward_func.compute_reward(
                    reward_turns, context.get_context_summary()
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
        return float(evaluate_context_coherence(context))

    def _evaluate_goal_achievement(
        self, context: ConversationContext, ground_truth: dict[str, Any] | None
    ) -> float:
        """Evaluate if conversation achieved its goal"""
        return float(evaluate_goal_achievement(context, ground_truth))

    def end_conversation(self, conversation_id: str) -> ConversationContext | None:
        """End a conversation and return its context"""
        with self._conversation_owner().lock:
            context = self.active_conversations.pop(conversation_id, None)
            if context is not None:
                with context._owner().lock:
                    context._owner().state = "ended"
                    context._owner().revision += 1
        if context is not None:
            if self.planning_manager is not None:
                self.planning_manager.clear_conversation(conversation_id)
            logger.info(f"Ended conversation {conversation_id}")
            return context
        return None

    def get_active_conversations(self) -> list[str]:
        """Get list of active conversation IDs"""
        with self._conversation_owner().lock:
            return list(self.active_conversations.keys())

    def get_conversation_summary(self, conversation_id: str) -> dict[str, Any] | None:
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
            return dict(summary)
        return None


__all__ = [
    "ConversationClosedError",
    "ConversationConflict",
    "ConversationContext",
    "DialogueDatabase",
    "MultiTurnAgent",
]
