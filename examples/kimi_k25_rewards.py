"""
Kimi-K2.5 Specific Reward Functions

This module provides reward functions optimized for Kimi-K2.5's unique features:
- Thinking mode with reasoning traces
- Multimodal capabilities
- Agent swarm coordination
- Long context handling
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RewardComponent:
    """Base class for reward components."""

    name: str
    weight: float
    threshold: float

    def compute(self, **kwargs) -> float:
        raise NotImplementedError


@dataclass
class ThinkingQualityReward(RewardComponent):
    """
    Rewards quality of reasoning in thinking mode.
    Kimi-K2.5 generates detailed reasoning traces that we can evaluate.
    """

    name: str = "thinking_quality"
    weight: float = 0.3
    threshold: float = 0.7

    def compute(
        self,
        reasoning_content: Optional[str] = None,
        response_content: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Evaluate the quality of reasoning:
        - Logical coherence
        - Depth of analysis
        - Relevance to query
        - Consistency with final answer
        """
        if not reasoning_content:
            return 0.0

        score = 0.0

        # Check for logical structure markers
        logical_markers = [
            "because",
            "since",
            "therefore",
            "thus",
            "however",
            "on the other hand",
            "first",
            "second",
            "third",
            "furthermore",
            "finally",
            "given that",
            "given the",
            "implies",
            "suggests",
        ]
        logical_count = sum(
            1
            for marker in logical_markers
            if marker.lower() in reasoning_content.lower()
        )
        score += min(logical_count / 5.0, 0.3)

        # Check for step-by-step reasoning
        if any(
            marker in reasoning_content.lower()
            for marker in ["step", "let me", "consider", "analyze", "evaluate"]
        ):
            score += 0.2

        # Check depth (length and complexity)
        reasoning_length = len(reasoning_content.split())
        if reasoning_length > 100:
            score += 0.2
        elif reasoning_length > 50:
            score += 0.1

        # Check consistency with final answer
        if response_content and query:
            # Simple check: reasoning should contain key terms from query
            query_terms = set(query.lower().split())
            reasoning_terms = set(reasoning_content.lower().split())
            overlap = len(query_terms & reasoning_terms) / max(len(query_terms), 1)
            score += min(overlap * 0.3, 0.3)

        return min(score, 1.0)


@dataclass
class MultimodalIntegrationReward(RewardComponent):
    """
    Rewards proper utilization of multimodal capabilities.
    Kimi-K2.5 can process images, videos, and generate code from visual specs.
    """

    name: str = "multimodal_integration"
    weight: float = 0.2
    threshold: float = 0.5

    def compute(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        response_content: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Evaluate if the agent properly uses visual inputs:
        - References visual content when provided
        - Generates code for UI specifications
        - Processes video content appropriately
        """
        if not messages or not response_content:
            return 0.0

        score = 0.0

        # Check if there are visual inputs
        has_visual = any(
            any(
                content.get("type") in ["image_url", "video_url"]
                for content in msg.get("content", [])
            )
            if isinstance(msg.get("content"), list)
            else False
            for msg in messages
        )

        if not has_visual:
            return 0.0

        # Check if response acknowledges visual content
        visual_references = [
            "image",
            "picture",
            "screenshot",
            "diagram",
            "chart",
            "video",
            "shown",
            "displayed",
            "depicted",
            "illustrated",
            "in the",
            "from the image",
            "from the video",
        ]
        ref_count = sum(
            1 for ref in visual_references if ref in response_content.lower()
        )
        score += min(ref_count / 2.0, 0.4)

        # Check for code generation if it looks like a UI spec
        ui_indicators = ["ui", "interface", "design", "layout", "component"]
        has_ui_query = any(
            indicator in str(messages).lower() for indicator in ui_indicators
        )
        if has_ui_query and "```" in response_content:
            score += 0.3

        # Check for detailed description of visual content
        if len(response_content) > 200:
            score += 0.3

        return min(score, 1.0)


@dataclass
class AgentCoordinationReward(RewardComponent):
    """
    Rewards effective agent swarm coordination.
    Kimi-K2.5 can decompose tasks into parallel sub-tasks.
    """

    name: str = "agent_coordination"
    weight: float = 0.2
    threshold: float = 0.6

    def compute(
        self,
        response_content: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Evaluate agent coordination quality:
        - Task decomposition
        - Parallel execution planning
        - Sub-task results integration
        """
        if not response_content:
            return 0.0

        score = 0.0

        # Check for task decomposition
        decomposition_markers = [
            "first",
            "second",
            "third",
            "meanwhile",
            "in parallel",
            "sub-task",
            "subtask",
            "parallel",
            "break down",
            "decompose",
        ]
        deco_count = sum(
            1
            for marker in decomposition_markers
            if marker.lower() in response_content.lower()
        )
        score += min(deco_count / 3.0, 0.3)

        # Check for integration of results
        integration_markers = [
            "combining",
            "together",
            "集成",
            "整合",
            "综合",
            "overall",
            "finally",
            "summary",
            "conclusion",
        ]
        integ_count = sum(
            1
            for marker in integration_markers
            if marker.lower() in response_content.lower()
        )
        score += min(integ_count / 2.0, 0.3)

        # Check for structured approach
        if any(marker in response_content for marker in ["1)", "2)", "3)", "- ", "* "]):
            score += 0.2

        # Check if response directly addresses the complex query
        if query and len(query) > 100:
            query_terms = set(query.lower().split()[:10])
            response_terms = set(response_content.lower().split())
            overlap = len(query_terms & response_terms) / max(len(query_terms), 1)
            score += min(overlap * 0.2, 0.2)

        return min(score, 1.0)


@dataclass
class LongContextUtilizationReward(RewardComponent):
    """
    Rewards effective use of long context (256K tokens).
    Kimi-K2.5 should maintain coherence across long conversations.
    """

    name: str = "long_context_utilization"
    weight: float = 0.15
    threshold: float = 0.5

    def compute(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        response_content: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Evaluate long context handling:
        - References earlier parts of conversation
        - Maintains consistency
        - Uses context clues appropriately
        """
        score = 0.0

        if not conversation_history or not response_content:
            return 0.0

        # Check conversation length
        if len(conversation_history) < 3:
            return 0.0

        # Extract key terms from earlier in conversation
        key_terms = set()
        for msg in conversation_history[:-2]:
            content = msg.get("content", "")
            if isinstance(content, str):
                key_terms.update(content.lower().split()[:5])

        # Check if response references earlier context
        for term in key_terms:
            if term in response_content.lower():
                score += 0.1

        score = min(score, 0.5)

        # Check for coherence markers
        coherence_markers = [
            "as mentioned earlier",
            "previously",
            "as discussed",
            "building on",
            "following up",
            "continuing",
            "如前所述",
            "之前",
            "基于",
            "接着",
        ]
        if any(marker in response_content.lower() for marker in coherence_markers):
            score += 0.3

        # Check for consistency in tone/style
        if len(response_content) > 50:
            score += 0.2

        return min(score, 1.0)


@dataclass
class CodeGenerationReward(RewardComponent):
    """
    Rewards high-quality code generation.
    Kimi-K2.5 excels at coding, especially from visual specs.
    """

    name: str = "code_generation"
    weight: float = 0.15
    threshold: float = 0.6

    def compute(
        self,
        response_content: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Evaluate code quality:
        - Syntax correctness (basic check)
        - Completeness
        - Documentation/comments
        - Best practices
        """
        score = 0.0

        if not query or not response_content:
            return 0.0

        # Check if code is requested
        code_indicators = ["code", "function", "implement", "write", "代码", "实现"]
        needs_code = any(indicator in query.lower() for indicator in code_indicators)

        if not needs_code:
            return 0.0

        # Check for code blocks
        code_blocks = response_content.count("```")
        if code_blocks >= 2:
            score += 0.3

        # Check for common code elements
        code_elements = [
            "def ",
            "class ",
            "import ",
            "function",
            "return",
            "var ",
            "const ",
        ]
        element_count = sum(1 for elem in code_elements if elem in response_content)
        score += min(element_count / 3.0, 0.3)

        # Check for comments/docstrings
        comment_markers = ["#", "//", "/**", "'''", '"""', "/*"]
        if any(marker in response_content for marker in comment_markers):
            score += 0.2

        # Check for error handling or edge cases
        if any(
            keyword in response_content.lower()
            for keyword in ["try", "catch", "error", "except", "throw"]
        ):
            score += 0.2

        return min(score, 1.0)


class KimiMultiObjectiveReward:
    """
    Multi-objective reward function combining all Kimi-K2.5 specific components.
    """

    def __init__(
        self, components: Optional[List[RewardComponent]] = None, threshold: float = 0.6
    ):
        if components is None:
            components = [
                ThinkingQualityReward(),
                MultimodalIntegrationReward(),
                AgentCoordinationReward(),
                LongContextUtilizationReward(),
                CodeGenerationReward(),
            ]

        self.components = components
        self.threshold = threshold

    async def compute_reward(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        response: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        query: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> float:
        """
        Compute weighted reward from all components.
        """
        if not response:
            return 0.0

        total_reward = 0.0
        total_weight = 0.0

        for component in self.components:
            component_reward = component.compute(
                messages=messages,
                response_content=response,
                reasoning_content=reasoning_content,
                query=query,
                conversation_history=conversation_history,
                **kwargs,
            )

            total_reward += component_reward * component.weight
            total_weight += component.weight

            logger.debug(
                f"{component.name}: {component_reward:.3f} (weight: {component.weight})"
            )

        final_reward = total_reward / total_weight if total_weight > 0 else 0.0

        # Apply threshold
        if final_reward < self.threshold:
            final_reward = final_reward * 0.5

        logger.debug(f"Final reward: {final_reward:.3f}")

        return final_reward


def create_kimi_conversational_reward(
    thinking_weight: float = 0.3,
    multimodal_weight: float = 0.2,
    coordination_weight: float = 0.2,
    context_weight: float = 0.15,
    code_weight: float = 0.15,
    threshold: float = 0.6,
) -> KimiMultiObjectiveReward:
    """
    Create a reward function optimized for Kimi-K2.5 conversational tasks.
    """
    components = [
        ThinkingQualityReward(weight=thinking_weight),
        MultimodalIntegrationReward(weight=multimodal_weight),
        AgentCoordinationReward(weight=coordination_weight),
        LongContextUtilizationReward(weight=context_weight),
        CodeGenerationReward(weight=code_weight),
    ]

    return KimiMultiObjectiveReward(components=components, threshold=threshold)


def create_kimi_coding_reward(
    thinking_weight: float = 0.2,
    context_weight: float = 0.1,
    code_weight: float = 0.7,
    threshold: float = 0.7,
) -> KimiMultiObjectiveReward:
    """
    Create a reward function optimized for coding tasks.
    """
    components = [
        ThinkingQualityReward(weight=thinking_weight),
        LongContextUtilizationReward(weight=context_weight),
        CodeGenerationReward(weight=code_weight),
    ]

    return KimiMultiObjectiveReward(components=components, threshold=threshold)
