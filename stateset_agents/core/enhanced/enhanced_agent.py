"""
Enhanced Agent Architecture for StateSet Agents

This module provides advanced agent capabilities including:
- Memory augmentation with vector retrieval
- Assessed preparation with optional reasoning providers
- Dynamic persona adaptation
- Multi-modal processing
- Self-improvement mechanisms
"""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from stateset_agents.utils.async_calls import closing_async_generator

from ..agent import AgentConfig, MultiTurnAgent
from ..agent_initialization import BackendInitializationState
from ..conversation import conversation_call, conversation_stream
from ..trajectory import ConversationTurn
from . import generation as enhanced_generation
from .reasoning import ReasoningEngine as ReasoningEngine
from .reasoning import ReasoningStep as ReasoningStep
from .vector_memory import MemoryEntry as MemoryEntry
from .vector_memory import VectorMemory

logger = logging.getLogger(__name__)


@dataclass
class PersonaProfile:
    """Dynamic persona profile"""

    name: str
    traits: dict[str, float] = field(default_factory=dict)
    expertise_areas: list[str] = field(default_factory=list)
    communication_style: dict[str, Any] = field(default_factory=dict)
    adaptation_rate: float = 0.1

    def adapt(self, feedback: dict[str, Any]):
        """Adapt persona based on feedback"""
        for trait, adjustment in feedback.get("trait_adjustments", {}).items():
            if trait in self.traits:
                self.traits[trait] = max(
                    0.0,
                    min(1.0, self.traits[trait] + adjustment * self.adaptation_rate),
                )


@dataclass
class _MemorySetup:
    system: VectorMemory
    complete: bool = False
    error: BaseException | None = None


class EnhancedMultiTurnAgent(MultiTurnAgent):
    """
    Enhanced multi-turn agent with advanced capabilities
    """

    def __init__(
        self,
        config: AgentConfig,
        memory_system: VectorMemory | None = None,
        reasoning_engine: ReasoningEngine | None = None,
        persona: PersonaProfile | None = None,
        *,
        memory_enabled: bool = True,
        reasoning_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        # Advanced capabilities
        self.memory_enabled = memory_enabled
        self.reasoning_enabled = reasoning_enabled
        self.memory_system = (
            memory_system
            if memory_system is not None
            else (VectorMemory() if memory_enabled else None)
        )
        self.reasoning_engine = (
            reasoning_engine
            if reasoning_engine is not None
            else (ReasoningEngine() if reasoning_enabled else None)
        )
        self.persona = persona if persona is not None else PersonaProfile("Assistant")
        self._memory_setup: _MemorySetup | None = None
        self._generations = enhanced_generation._GenerationLedger()

        # Enhanced state
        self.reasoning_history: list[list[ReasoningStep]] = []
        self.performance_metrics: dict[str, Any] = {}
        self.self_improvement_data: list[dict[str, Any]] = []

        # Multi-modal support
        self.multimodal_processors: dict[str, Any] = {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Keep previously serialized enhanced agents readable."""
        self.__dict__.update(state)
        self.memory_enabled = state.get("memory_enabled", True)
        self.reasoning_enabled = state.get("reasoning_enabled", True)
        self._memory_setup = state.get("_memory_setup")
        self._generations = state.get(
            "_generations", enhanced_generation._GenerationLedger()
        )

    def _generation_ledger(self) -> enhanced_generation._GenerationLedger:
        return self._generations

    def get_generation_status(self, generation_id: str | None = None) -> dict[str, Any]:
        """Inspect active/latest generation and retained batch memory receipts."""
        return self._generation_ledger().inspect(generation_id)

    @conversation_call
    async def remember_candidate(self, turn: ConversationTurn) -> None:
        """Remember one unchanged batch candidate, at most once per retained batch.

        The latest 32 idle batch receipts are retained, plus in-flight writes.
        A failed write is unconfirmed and cannot be replayed automatically.
        Candidate selection rights are excluded from deep copies and pickles.
        """
        await enhanced_generation.remember_candidate(self, turn)

    async def _initialize_extensions(self) -> None:
        """Retain memory setup effects inside the agent initialization attempt."""
        if self.reasoning_enabled and self.reasoning_engine is None:
            raise ValueError("Enabled reasoning requires a reasoning engine")
        if not self.memory_enabled:
            return
        memory = self.memory_system
        if memory is None:
            raise ValueError("Enabled memory requires a memory system")
        previous = self._memory_setup
        if previous is not None and previous.system is memory:
            if previous.error is not None and type(memory) is not VectorMemory:
                raise RuntimeError(
                    "Previous custom memory initialization failed after it began; "
                    "provide a fresh memory system or agent before retrying"
                ) from previous.error
            if previous.complete and type(memory) is not VectorMemory:
                return
        # Built-in VectorMemory prepares a detached candidate and can safely retry.
        # Custom systems may mutate borrowed resources before raising.
        setup = _MemorySetup(memory)
        self._memory_setup = setup
        try:
            await memory.initialize()
            if self.memory_system is not memory:
                raise ValueError("Memory system changed during initialization")
        except BaseException as error:
            setup.error = error
            raise
        else:
            setup.complete = True

    def _ensure_backend_usable(self) -> None:
        super()._ensure_backend_usable()
        if (
            self.memory_enabled
            and self._initialization_owner().state is BackendInitializationState.READY
            and (
                self._memory_setup is None
                or self.memory_system is not self._memory_setup.system
                or not self._memory_setup.complete
            )
        ):
            raise RuntimeError(
                "Enhanced memory system changed; initialize the agent before use"
            )
        if self.memory_enabled and isinstance(self.memory_system, VectorMemory):
            self.memory_system._ensure_usable()
        if self.reasoning_enabled and self.reasoning_engine is None:
            raise RuntimeError("Enabled reasoning requires a reasoning engine")

    @conversation_call
    async def generate_response(
        self,
        messages_or_prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Prepare enhancements once and remember one completed response."""
        async with self._generation_ledger().attempt("response") as receipt:
            request = await enhanced_generation.prepare(
                self, receipt, messages_or_prompt, context
            )
            response = await enhanced_generation.sample(
                super().generate_response(request.messages, request.context),
                receipt,
                lambda response: [response],
            )
            await enhanced_generation.complete(self, request)
            return response

    @conversation_call
    async def generate_turn(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Apply enhancements while preserving native sampler metadata."""
        async with self._generation_ledger().attempt("turn") as receipt:
            request = await enhanced_generation.prepare(
                self, receipt, messages, context
            )
            turn = await enhanced_generation.sample(
                super().generate_turn(request.messages, request.context),
                receipt,
                lambda turn: [enhanced_generation.output_text(turn)],
            )
            await enhanced_generation.complete(self, request)
            enhanced_generation.annotate(turn, receipt, 0)
            return turn

    @conversation_call
    async def generate_turns(
        self,
        messages: str | list[dict[str, str]],
        n: int,
        context: dict[str, Any] | None = None,
    ) -> list[ConversationTurn]:
        """Sample from one prepared context; remember only an explicitly selected candidate."""
        n = int(n)
        if n <= 0:
            return []
        async with self._generation_ledger().attempt("batch") as receipt:
            request = await enhanced_generation.prepare(
                self, receipt, messages, context
            )
            turns = await enhanced_generation.sample(
                super().generate_turns(request.messages, n, request.context),
                receipt,
                lambda turns: [enhanced_generation.output_text(turn) for turn in turns],
            )
            await enhanced_generation.complete(self, request, candidates=True)
            for index, turn in enumerate(turns):
                enhanced_generation.annotate(turn, receipt, index)
            return turns

    @conversation_stream
    async def generate_response_stream(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream one prepared request, committing memory only on normal exhaustion."""
        async with self._generation_ledger().attempt("stream") as receipt:
            request = await enhanced_generation.prepare(
                self, receipt, messages, context
            )
            source = super().generate_response_stream(request.messages, request.context)
            async with closing_async_generator(source) as stream:
                async for chunk in stream:
                    receipt.partial_text += chunk
                    yield chunk
            response = self._clean_response(receipt.partial_text, native_protocol=False)
            receipt.outputs = [response]
            await enhanced_generation.complete(self, request)

    def _should_reason(self, query: str) -> bool:
        """Determine if complex reasoning is needed"""
        complex_indicators = [
            "why",
            "how",
            "explain",
            "analyze",
            "compare",
            "solve",
            "design",
            "create",
            "optimize",
            "troubleshoot",
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in complex_indicators)

    def _adapt_persona(self, context: dict[str, Any]) -> None:
        """Adapt persona based on context."""
        persona = self.persona
        preview = copy.deepcopy(persona)
        self._adapt_persona_profile(preview, context)
        with self._conversation_owner().commit():
            if self.persona is not persona:
                raise RuntimeError("Persona changed during adaptation")
            vars(persona).update(vars(preview))

    @staticmethod
    def _adapt_persona_profile(
        persona: PersonaProfile, context: dict[str, Any]
    ) -> None:
        """Apply context traits to either a detached preview or the live profile."""
        # Simple adaptation logic
        if context.get("urgency") == "high":
            persona.traits["responsiveness"] = min(
                1.0, persona.traits.get("responsiveness", 0.5) + 0.1
            )

        if context.get("technical_level") == "expert":
            persona.traits["technical_depth"] = min(
                1.0, persona.traits.get("technical_depth", 0.5) + 0.1
            )

    def _calculate_importance(self, query: str, response: str) -> float:
        """Calculate importance score for memory storage"""
        importance = 1.0

        # Increase importance for complex queries
        if len(query.split()) > 20:
            importance += 0.5

        # Increase for detailed responses
        if len(response.split()) > 50:
            importance += 0.3

        # Increase for questions (learning opportunities)
        if "?" in query:
            importance += 0.2

        return min(importance, 2.0)

    def _update_performance_metrics(self, response: str, context: dict[str, Any]):
        """Update performance tracking"""
        with self._conversation_owner().commit():
            self.performance_metrics["total_responses"] = (
                self.performance_metrics.get("total_responses", 0) + 1
            )
            self.performance_metrics["avg_response_length"] = (
                self.performance_metrics.get("avg_response_length", 0)
                * (self.performance_metrics["total_responses"] - 1)
                + len(response.split())
            ) / self.performance_metrics["total_responses"]

    @conversation_call
    async def self_improve(self, feedback: dict[str, Any]):
        """Self-improvement mechanism"""
        self._ensure_backend_usable()
        with self._conversation_owner().commit():
            self.self_improvement_data.append(
                {
                    "timestamp": datetime.now(),
                    "feedback": feedback,
                    "current_metrics": self.performance_metrics.copy(),
                }
            )

        # Adapt persona based on feedback
        if "persona_feedback" in feedback:
            persona = self.persona
            preview = copy.deepcopy(persona)
            preview.adapt(feedback["persona_feedback"])
            with self._conversation_owner().commit():
                if self.persona is not persona:
                    raise RuntimeError("Persona changed during adaptation")
                vars(persona).update(vars(preview))

        # Store improvement insights in memory
        improvement_insight = (
            f"Self-improvement: {feedback.get('insight', 'General improvement')}"
        )
        if self.memory_enabled and self.memory_system is not None:
            self._conversation_owner().check()
            await self.memory_system.add_entry(
                content=improvement_insight,
                context={"type": "self_improvement", "feedback": feedback},
                importance=1.5,
            )

    def get_agent_status(self) -> dict[str, Any]:
        """Get comprehensive agent status"""
        return copy.deepcopy(
            {
                "base_config": self.config.__dict__,
                "initialization": self.get_initialization_status(),
                "conversation": self.get_conversation_status(),
                "memory_backend": (
                    self.memory_system.get_initialization_status()
                    if self.memory_system is not None
                    and isinstance(self.memory_system, VectorMemory)
                    else None
                ),
                "memory_snapshot": (
                    self.memory_system.get_snapshot_status()
                    if self.memory_system is not None
                    and isinstance(self.memory_system, VectorMemory)
                    else None
                ),
                "memory_entries": (
                    len(self.memory_system.entries)
                    if self.memory_system is not None
                    else 0
                ),
                "reasoning_sessions": len(self.reasoning_history),
                "reasoning_backend": (
                    (
                        self.reasoning_engine.get_status()
                        if isinstance(self.reasoning_engine, ReasoningEngine)
                        else {"mode": "custom", "status": "unavailable"}
                    )
                    if self.reasoning_enabled and self.reasoning_engine is not None
                    else None
                ),
                "performance_metrics": self.performance_metrics,
                "persona": {
                    "name": self.persona.name,
                    "traits": self.persona.traits,
                    "expertise_areas": self.persona.expertise_areas,
                },
                "capabilities": [
                    *(["vector_memory"] if self.memory_enabled else []),
                    *(["chain_of_thought"] if self.reasoning_enabled else []),
                    *(["reasoning_preparation"] if self.reasoning_enabled else []),
                    "dynamic_persona",
                    "self_improvement",
                ],
            }
        )


# Factory functions for creating enhanced agents


def create_enhanced_agent(
    model_name: str = "gpt2",
    memory_enabled: bool = True,
    reasoning_enabled: bool = True,
    persona_name: str = "Assistant",
    *,
    memory_system: VectorMemory | None = None,
    reasoning_engine: ReasoningEngine | None = None,
    **kwargs,
) -> EnhancedMultiTurnAgent:
    """Create an enhanced agent with advanced capabilities"""

    config = AgentConfig(model_name=model_name, **kwargs)

    # Create components
    if memory_system is None and memory_enabled:
        memory_system = VectorMemory()
    if reasoning_engine is None and reasoning_enabled:
        reasoning_engine = ReasoningEngine()

    # Create persona
    persona = PersonaProfile(
        name=persona_name,
        traits={
            "helpfulness": 0.8,
            "technical_depth": 0.6,
            "creativity": 0.7,
            "responsiveness": 0.9,
        },
        expertise_areas=["general_assistance", "problem_solving"],
    )

    return EnhancedMultiTurnAgent(
        config=config,
        memory_system=memory_system,
        reasoning_engine=reasoning_engine,
        persona=persona,
        memory_enabled=memory_enabled,
        reasoning_enabled=reasoning_enabled,
    )


def create_domain_specific_agent(
    domain: str, model_name: str = "gpt2", **kwargs
) -> EnhancedMultiTurnAgent:
    """Create an agent specialized for a specific domain"""

    domain_configs = {
        "customer_service": {
            "persona_name": "Customer Service Assistant",
            "traits": {
                "helpfulness": 0.9,
                "empathy": 0.8,
                "responsiveness": 0.9,
                "technical_depth": 0.6,
            },
            "expertise_areas": [
                "customer_support",
                "problem_resolution",
                "communication",
            ],
        },
        "technical_support": {
            "persona_name": "Technical Support Engineer",
            "traits": {
                "helpfulness": 0.8,
                "technical_depth": 0.9,
                "analytical": 0.8,
                "patience": 0.9,
            },
            "expertise_areas": [
                "troubleshooting",
                "technical_analysis",
                "system_diagnostics",
            ],
        },
        "sales": {
            "persona_name": "Sales Consultant",
            "traits": {
                "helpfulness": 0.8,
                "persuasiveness": 0.8,
                "creativity": 0.7,
                "relationship_building": 0.9,
            },
            "expertise_areas": [
                "product_knowledge",
                "negotiation",
                "relationship_management",
            ],
        },
    }

    config = domain_configs.get(domain, domain_configs["customer_service"])

    return create_enhanced_agent(
        model_name=model_name, persona_name=str(config["persona_name"]), **kwargs
    )
