"""
Enhanced Agent Architecture for StateSet Agents

This module provides advanced agent capabilities including:
- Memory augmentation with vector retrieval
- Chain-of-thought reasoning
- Dynamic persona adaptation
- Multi-modal processing
- Self-improvement mechanisms
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

from ..agent import Agent, AgentConfig, MultiTurnAgent
from ..reward import RewardFunction, RewardResult
from ..trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Enhanced memory entry with vector embeddings"""

    content: str
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def update_access(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class ReasoningStep:
    """Step in chain-of-thought reasoning"""

    thought: str
    action: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PersonaProfile:
    """Dynamic persona profile"""

    name: str
    traits: Dict[str, float] = field(default_factory=dict)
    expertise_areas: List[str] = field(default_factory=list)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    adaptation_rate: float = 0.1

    def adapt(self, feedback: Dict[str, Any]):
        """Adapt persona based on feedback"""
        for trait, adjustment in feedback.get("trait_adjustments", {}).items():
            if trait in self.traits:
                self.traits[trait] = max(
                    0.0,
                    min(1.0, self.traits[trait] + adjustment * self.adaptation_rate),
                )


class VectorMemory:
    """Vector-based memory system for semantic retrieval"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_entries: int = 1000,
    ):
        self.embedding_model = embedding_model
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.tokenizer = None
        self.model = None
        if torch is not None:
            cuda_available = False
            try:  # pragma: no cover - device probing
                cuda_available = torch.cuda.is_available()
            except Exception:
                cuda_available = False
            self.device = torch.device("cuda" if cuda_available else "cpu")
        else:
            self.device = "cpu"

    async def initialize(self):
        """Initialize the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.embedding_model)
            self.model.to(self.device)
            logger.info(f"Vector memory initialized with {self.embedding_model}")
        except ImportError:
            logger.warning(
                "sentence-transformers not available, using simple embeddings"
            )
            self.model = None

    async def add_entry(
        self, content: str, context: Dict[str, Any] = None, importance: float = 1.0
    ):
        """Add a new memory entry"""
        entry = MemoryEntry(
            content=content, context=context or {}, importance=importance
        )

        # Generate embedding
        if self.model:
            embedding = await self._generate_embedding(content)
            entry.embedding = embedding

        self.entries.append(entry)

        # Maintain size limit (remove least important entries)
        if len(self.entries) > self.max_entries:
            self.entries.sort(
                key=lambda x: x.importance * (1 + x.access_count * 0.1), reverse=True
            )
            self.entries = self.entries[: self.max_entries]

    async def retrieve_relevant(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories based on semantic similarity"""
        if not self.entries or not self.model:
            return []

        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # Calculate similarities
        similarities = []
        for entry in self.entries:
            if entry.embedding is not None:
                similarity = np.dot(query_embedding, entry.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                )
                similarities.append((similarity, entry))

        # Sort by similarity and recency
        similarities.sort(
            key=lambda x: x[0] * (1 + x[1].access_count * 0.1), reverse=True
        )

        # Update access metadata
        relevant_entries = []
        for _, entry in similarities[:top_k]:
            entry.update_access()
            relevant_entries.append(entry)

        return relevant_entries

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.model:
            if torch is not None and hasattr(torch, "no_grad"):
                with torch.no_grad():
                    embedding = self.model.encode(text, convert_to_numpy=True)
            else:
                embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        else:
            # Simple fallback embedding
            return np.random.rand(384)  # Match MiniLM dimension


class ReasoningEngine:
    """Chain-of-thought reasoning engine"""

    def __init__(self, max_steps: int = 5, confidence_threshold: float = 0.7):
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.reasoning_templates = {
            "analytical": "Let me analyze this step by step...",
            "creative": "Let me think creatively about this...",
            "problem_solving": "Let me break this problem down...",
            "decision_making": "Let me weigh the options...",
        }

    async def reason(
        self, query: str, context: Dict[str, Any] = None
    ) -> List[ReasoningStep]:
        """Perform chain-of-thought reasoning"""
        steps = []
        current_context = context or {}

        # Determine reasoning type
        reasoning_type = self._classify_reasoning_type(query)

        # Generate reasoning steps
        for step_num in range(self.max_steps):
            step = await self._generate_reasoning_step(
                query, reasoning_type, step_num, steps, current_context
            )

            steps.append(step)

            # Stop if confidence is high enough or we have a clear answer
            if step.confidence >= self.confidence_threshold:
                break

            # Update context with new information
            current_context.update({"step": step_num + 1, "last_thought": step.thought})

        return steps

    def _classify_reasoning_type(self, query: str) -> str:
        """Classify the type of reasoning needed"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["analyze", "compare", "evaluate"]):
            return "analytical"
        elif any(word in query_lower for word in ["create", "design", "generate"]):
            return "creative"
        elif any(word in query_lower for word in ["solve", "fix", "troubleshoot"]):
            return "problem_solving"
        else:
            return "decision_making"

    async def _generate_reasoning_step(
        self,
        query: str,
        reasoning_type: str,
        step_num: int,
        previous_steps: List[ReasoningStep],
        context: Dict[str, Any],
    ) -> ReasoningStep:
        """Generate a single reasoning step"""

        # Template-based reasoning for now
        # In practice, this would use an LLM for more sophisticated reasoning

        thought_templates = {
            "analytical": [
                "First, I need to understand the key components of this query.",
                "Now let me examine the relationships between different elements.",
                "Based on my analysis, I can identify the main patterns.",
                "Let me verify my understanding by checking against known facts.",
                "Finally, I can synthesize my findings into a coherent response.",
            ],
            "creative": [
                "Let me start by brainstorming different approaches.",
                "I can combine these ideas in interesting ways.",
                "This combination could lead to innovative solutions.",
                "Let me refine this concept further.",
                "The final creative solution emerges from this process.",
            ],
            "problem_solving": [
                "The core problem seems to be...",
                "Let me identify the root cause by examining symptoms.",
                "I can apply systematic problem-solving techniques.",
                "Testing potential solutions against requirements.",
                "The most effective solution appears to be...",
            ],
            "decision_making": [
                "I need to consider all available options.",
                "Let me evaluate the pros and cons of each choice.",
                "Based on the criteria, some options stand out.",
                "Considering risks and uncertainties.",
                "The optimal decision balances all factors.",
            ],
        }

        template = thought_templates.get(
            reasoning_type, thought_templates["analytical"]
        )

        thought = template[min(step_num, len(template) - 1)]
        action = f"Continue {reasoning_type} reasoning"
        confidence = min(0.5 + step_num * 0.1, 0.9)  # Increasing confidence

        return ReasoningStep(
            thought=thought,
            action=action,
            confidence=confidence,
            evidence=[f"Step {i+1}" for i in range(step_num)],
        )


class EnhancedMultiTurnAgent(MultiTurnAgent):
    """
    Enhanced multi-turn agent with advanced capabilities
    """

    def __init__(
        self,
        config: AgentConfig,
        memory_system: Optional[VectorMemory] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        persona: Optional[PersonaProfile] = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        # Advanced capabilities
        self.memory_system = memory_system or VectorMemory()
        self.reasoning_engine = reasoning_engine or ReasoningEngine()
        self.persona = persona or PersonaProfile("Assistant")

        # Enhanced state
        self.reasoning_history: List[List[ReasoningStep]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.self_improvement_data: List[Dict[str, Any]] = []

        # Multi-modal support
        self.multimodal_processors: Dict[str, Any] = {}

    async def initialize(self):
        """Initialize the enhanced agent"""
        await super().initialize()

        # Initialize memory system
        await self.memory_system.initialize()

        logger.info("Enhanced MultiTurnAgent initialized with advanced capabilities")

    async def generate_response(
        self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate enhanced response with reasoning and memory"""

        # Extract current query
        current_query = messages[-1]["content"] if messages else ""

        # Retrieve relevant memories
        relevant_memories = await self.memory_system.retrieve_relevant(
            current_query, top_k=3
        )

        # Perform reasoning if needed
        reasoning_steps = []
        if self._should_reason(current_query):
            reasoning_steps = await self.reasoning_engine.reason(current_query, context)
            self.reasoning_history.append(reasoning_steps)

        # Adapt persona based on context
        if context:
            self._adapt_persona(context)

        # Enhance context with memories and reasoning
        enhanced_context = {
            **(context or {}),
            "relevant_memories": [mem.content for mem in relevant_memories],
            "reasoning_steps": [step.thought for step in reasoning_steps],
            "persona_traits": self.persona.traits,
        }

        # Generate response using base agent
        response = await super().generate_response(messages, enhanced_context)

        # Store interaction in memory
        memory_content = f"Query: {current_query}\nResponse: {response}"
        await self.memory_system.add_entry(
            content=memory_content,
            context={"query": current_query, "response": response},
            importance=self._calculate_importance(current_query, response),
        )

        # Update performance metrics
        self._update_performance_metrics(response, context)

        return response

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

    def _adapt_persona(self, context: Dict[str, Any]):
        """Adapt persona based on context"""
        # Simple adaptation logic
        if context.get("urgency") == "high":
            self.persona.traits["responsiveness"] = min(
                1.0, self.persona.traits.get("responsiveness", 0.5) + 0.1
            )

        if context.get("technical_level") == "expert":
            self.persona.traits["technical_depth"] = min(
                1.0, self.persona.traits.get("technical_depth", 0.5) + 0.1
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

    def _update_performance_metrics(self, response: str, context: Dict[str, Any]):
        """Update performance tracking"""
        self.performance_metrics["total_responses"] = (
            self.performance_metrics.get("total_responses", 0) + 1
        )
        self.performance_metrics["avg_response_length"] = (
            self.performance_metrics.get("avg_response_length", 0)
            * (self.performance_metrics["total_responses"] - 1)
            + len(response.split())
        ) / self.performance_metrics["total_responses"]

    async def self_improve(self, feedback: Dict[str, Any]):
        """Self-improvement mechanism"""
        self.self_improvement_data.append(
            {
                "timestamp": datetime.now(),
                "feedback": feedback,
                "current_metrics": self.performance_metrics.copy(),
            }
        )

        # Adapt persona based on feedback
        if "persona_feedback" in feedback:
            self.persona.adapt(feedback["persona_feedback"])

        # Store improvement insights in memory
        improvement_insight = (
            f"Self-improvement: {feedback.get('insight', 'General improvement')}"
        )
        await self.memory_system.add_entry(
            content=improvement_insight,
            context={"type": "self_improvement", "feedback": feedback},
            importance=1.5,
        )

    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "base_config": self.config.__dict__,
            "memory_entries": len(self.memory_system.entries),
            "reasoning_sessions": len(self.reasoning_history),
            "performance_metrics": self.performance_metrics,
            "persona": {
                "name": self.persona.name,
                "traits": self.persona.traits,
                "expertise_areas": self.persona.expertise_areas,
            },
            "capabilities": [
                "vector_memory",
                "chain_of_thought",
                "dynamic_persona",
                "self_improvement",
            ],
        }


# Factory functions for creating enhanced agents


def create_enhanced_agent(
    model_name: str = "gpt2",
    memory_enabled: bool = True,
    reasoning_enabled: bool = True,
    persona_name: str = "Assistant",
    **kwargs,
) -> EnhancedMultiTurnAgent:
    """Create an enhanced agent with advanced capabilities"""

    config = AgentConfig(model_name=model_name, **kwargs)

    # Create components
    memory_system = VectorMemory() if memory_enabled else None
    reasoning_engine = ReasoningEngine() if reasoning_enabled else None

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
        model_name=model_name, persona_name=config["persona_name"], **kwargs
    )
