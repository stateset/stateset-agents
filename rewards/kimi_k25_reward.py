"""
Kimi-K2.5 Reward System

This module implements specialized reward functions for training Kimi-K2.5 models,
including multimodal reasoning rewards and thinking-mode specific rewards.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction as MultiObjectiveReward,
    RewardComponent,
)


logger = logging.getLogger(__name__)


class ThinkingModeReward(RewardComponent):
    """
    Reward component for evaluating the quality of reasoning in thinking mode.
    
    Kimi-K2.5 reasoning_content provides detailed step-by-step reasoning.
    This component evaluates:
    - Logical coherence of reasoning
    - Correctness of intermediate steps
    - Final answer consistency with reasoning
    """

    def __init__(
        self,
        weight: float = 0.3,
        min_reasoning_length: int = 100,
        max_reasoning_length: int = 4000,
    ):
        """
        Initialize ThinkingModeReward.
        
        Args:
            weight: Weight of this component in total reward
            min_reasoning_length: Minimum reasoning content length
            max_reasoning_length: Maximum reasoning content length
        """
        super().__init__(
            name="thinking_mode",
            description="Evaluates reasoning quality in thinking mode",
            weight=weight,
        )
        self.min_reasoning_length = min_reasoning_length
        self.max_reasoning_length = max_reasoning_length

    async def compute(
        self,
        trajectory: Any,
        environment: Any,
        **kwargs,
    ) -> float:
        """
        Compute thinking mode reward.
        
        Args:
            trajectory: Agent trajectory with reasoning_content
            environment: Environment context
            
        Returns:
            Reward score [0, 1]
        """
        # Extract reasoning content from trajectory
        reasoning_content = getattr(trajectory, "reasoning_content", "")
        
        if not reasoning_content:
            # If no reasoning content, penalize
            return 0.0

        reasoning_length = len(reasoning_content)
        
        # Check reasoning length constraints
        if reasoning_length < self.min_reasoning_length:
            return 0.1  # Too brief
        if reasoning_length > self.max_reasoning_length:
            return 0.7  # Too verbose but still valuable

        # Check for reasoning markers (penalty for low-quality reasoning)
        quality_indicators = [
            "First,", "Then,", "However,",
            "Therefore,", "In conclusion,",
            "Let me analyze", "Breaking down",
            "Considering", "We need to",
        ]
        
        # Check for structural markers
        has_structure = any(
            indicator.lower() in reasoning_content.lower()
            for indicator in quality_indicators
        )
        
        base_reward = 0.8 if has_structure else 0.6
        
        # Check for backtracking (good sign of reconsideration)
        backtrack_indicators = ["reconsider", "wait,", "actually", "let me rethink"]
        has_backtrack = any(
            indicator.lower() in reasoning_content.lower()
            for indicator in backtrack_indicators
        )
        
        if has_backtrack:
            base_reward += 0.1

        # Check for final confidence
        confidence_indicators = ["confident", "certain", "definitely", "clearly"]
        has_confidence = any(
            indicator.lower() in reasoning_content.lower()
            for indicator in confidence_indicators
        )
        
        if has_confidence:
            base_reward += 0.1

        return min(base_reward, 1.0)


class MultimodalConsistencyReward(RewardComponent):
    """
    Reward component for evaluating multimodal consistency.
    
    When Kimi-K2.5 processes visual content, this component evaluates:
    - Response acknowledges visual input
    - Visual-specific details are mentioned
    - Cross-modal reasoning is present
    """

    def __init__(
        self,
        weight: float = 0.2,
        require_visual_ack: bool = True,
        min_visual_references: int = 1,
    ):
        """
        Initialize MultimodalConsistencyReward.
        
        Args:
            weight: Weight of this component in total reward
            require_visual_ack: Require explicit visual acknowledgment
            min_visual_references: Minimum visual references required
        """
        super().__init__(
            name="multimodal_consistency",
            description="Evaluates multimodal reasoning consistency",
            weight=weight,
        )
        self.require_visual_ack = require_visual_ack
        self.min_visual_references = min_visual_references

    async def compute(
        self,
        trajectory: Any,
        environment: Any,
        **kwargs,
    ) -> float:
        """
        Compute multimodal consistency reward.
        
        Args:
            trajectory: Agent trajectory with multimodal context
            environment: Environment context
            
        Returns:
            Reward score [0, 1]
        """
        # Check if there's visual content in the input
        has_visual_input = kwargs.get("has_visual_input", False)
        
        if not has_visual_input:
            return 1.0  # Not applicable, give full marks

        response = getattr(trajectory, "response", "")
        
        if not response:
            return 0.0

        # Check for visual acknowledgment phrases
        visual_ack_phrases = [
            "i can see", "looking at", "in the image",
            "the video shows", "based on visual",
            "from the picture", "the scene shows",
        ]
        
        has_ack = any(
            phrase.lower() in response.lower()
            for phrase in visual_ack_phrases
        )
        
        if self.require_visual_ack and not has_ack:
            return 0.3

        # Count visual-specific references
        visual_ref_clusters = [
            "image", "picture", "photo", "visual", "screenshot",
            "video", "frame", "scene", "diagram", "chart",
            "shown", "displayed", "visible", "appear",
        ]
        
        visual_count = sum(
            1 for ref in visual_ref_clusters
            if ref.lower() in response.lower()
        )

        # Reward based on visual reference count
        if visual_count >= self.min_visual_references:
            return 1.0
        elif visual_count > 0:
            return 0.6 + (visual_count / self.min_visual_references) * 0.4
        else:
            return 0.3


class CodeExecutionReward(RewardComponent):
    """
    Reward component for evaluating code execution quality.
    
    Kimi-K2.5 excels at coding tasks with visual specifications.
    This component evaluates:
    - Code completeness and correctness
    - Code-following of visual specifications
    - Code explanation quality
    """

    def __init__(
        self,
        weight: float = 0.25,
        require_code_blocks: bool = True,
        require_explanation: bool = True,
    ):
        """
        Initialize CodeExecutionReward.
        
        Args:
            weight: Weight of this component in total reward
            require_code_blocks: Require code blocks
            require_explanation: Require code explanations
        """
        super().__init__(
            name="code_execution",
            description="Evaluates code execution quality",
            weight=weight,
        )
        self.require_code_blocks = require_code_blocks
        self.require_explanation = require_explanation

    async def compute(
        self,
        trajectory: Any,
        environment: Any,
        **kwargs,
    ) -> float:
        """
        Compute code execution reward.
        
        Args:
            trajectory: Agent trajectory with potential code
            environment: Environment context
            
        Returns:
            Reward score [0, 1]
        """
        response = getattr(trajectory, "response", "")
        
        if not response:
            return 0.0

        # Check for code blocks
        code_block_markers = ["```python", "```javascript", "```java", "```c++", "```"]
        has_code = any(marker in response for marker in code_block_markers)
        
        if self.require_code_blocks and not has_code:
            return 0.0

        # If not a coding task, give full marks
        is_coding_task = kwargs.get("is_coding_task", False)
        if not is_coding_task:
            return 1.0

        # Evaluate code quality indicators
        quality_indicators = [
            "function", "class", "def ", "import ",
            "return ", "async def", "let ", "const ",
        ]

        quality_score = sum(
            1 for indicator in quality_indicators
            if indicator in response
        ) / len(quality_indicators)

        # Check for code explanation
        has_explanation = any(
            exp in response.lower()
            for exp in [
                "here is the code", "the code above",
                "this function", "the solution",
            ]
        )

        if self.require_explanation:
            explanation_bonus = 0.3 if has_explanation else 0.0
        else:
            explanation_bonus = 0.0

        # Calculate final score
        if not self.require_code_blocks:
            # Don't penalize for lack of code if not required
            base_score = 1.0
        elif has_code:
            base_score = 0.5 + quality_score * 0.5
        else:
            base_score = 0.0

        return min(base_score + explanation_bonus, 1.0)


class LongContextReward(RewardComponent):
    """
    Reward component for evaluating long context handling.
    
    Kimi-K2.5 has 256K context window.
    This component evaluates:
    - Ability to maintain coherence over long conversations
    - Reference to earlier parts of conversation
    - Context management quality
    """

    def __init__(
        self,
        weight: float = 0.15,
        require_earlier_reference: bool = True,
        context_retention_threshold: float = 0.8,
    ):
        """
        Initialize LongContextReward.
        
        Args:
            weight: Weight of this component in total reward
            require_earlier_reference: Require referencing earlier context
            context_retention_threshold: Threshold for context retention
        """
        super().__init__(
            name="long_context",
            description="Evaluates long context handling",
            weight=weight,
        )
        self.require_earlier_reference = require_earlier_reference
        self.context_retention_threshold = context_retention_threshold

    async def compute(
        self,
        trajectory: Any,
        environment: Any,
        **kwargs,
    ) -> float:
        """
        Compute long context reward.
        
        Args:
            trajectory: Agent trajectory
            environment: Environment context
            
        Returns:
            Reward score [0, 1]
        """
        # Check conversation length
        conversation_turns = kwargs.get("conversation_turns", 1)
        
        # For short conversations, give full marks
        if conversation_turns < 5:
            return 1.0

        response = getattr(trajectory, "response", "")
        earlier_messages = kwargs.get("earlier_messages", [])

        if not earlier_messages:
            return 1.0

        # Check for references to earlier parts
        reference_indicators = [
            "earlier", "previously", "before",
            "as mentioned", "we discussed",
            "from the beginning", "remember",
            "as you said", "fitting your",
        ]

        has_earlier_reference = any(
            indicator.lower() in response.lower()
            for indicator in reference_indicators
        )

        if self.require_earlier_reference and not has_earlier_reference:
            return self.context_retention_threshold

        # Penalize for inconsistencies with earlier context
        consistency_score = 1.0
        for earlier_msg in earlier_messages[-3:]:  # Check last 3 messages
            ifearlier_msg.lower() in response.lower():
                consistency_score += 0.1  # Good: references specific earlier points

        return min(consistency_score, 1.0)


def create_kimi_k25_customer_service_reward(
    include_thinking: bool = True,
    include_multimodal: bool = False,
    include_code: bool = False,
    include_long_context: bool = True,
) -> MultiObjectiveReward:
    """
    Create a reward function optimized for Kimi-K2.5 customer service training.

    Args:
        include_thinking: Include thinking mode evaluation
        include_multimodal: Include multimodal consistency
        include_code: Include code execution evaluation
        include_long_context: Include long context handling

    Returns:
        MultiObjectiveReward with Kimi-K2.5 specific components
    """
    components = []

    if include_thinking:
        components.append(
            ThinkingModeReward(
                weight=0.25,
                min_reasoning_length=150,
                max_reasoning_length=5000,
            )
        )

    if include_multimodal:
        components.append(
            MultimodalConsistencyReward(
                weight=0.15,
                require_visual_ack=True,
                min_visual_references=1,
            )
        )

    if include_code:
        components.append(
            CodeExecutionReward(
                weight=0.20,
                require_code_blocks=True,
                require_explanation=True,
            )
        )

    if include_long_context:
        components.append(
            LongContextReward(
                weight=0.15,
                require_earlier_reference=True,
                context_retention_threshold=0.7,
            )
        )

    reward_function = MultiObjectiveReward(
        name="kimi_k25_customer_service",
        components=components,
    )

    return reward_function


def create_kimi_k25_conversational_reward(
    include_thinking: bool = True,
    include_multimodal: bool = True,
    include_code: bool = False,
    include_long_context: bool = True,
) -> MultiObjectiveReward:
    """
    Create a reward function for general Kimi-K2.5 conversational tasks.

    Args:
        include_thinking: Include thinking mode evaluation
        include_multimodal: Include multimodal consistency
        include_code: Include code execution evaluation
        include_long_context: Include long context handling

    Returns:
        MultiObjectiveReward with Kimi-K2.5 specific components
    """
    components = []

    if include_thinking:
        components.append(
            ThinkingModeReward(
                weight=0.30,
                min_reasoning_length=200,
                max_reasoning_length=6000,
            )
        )

    if include_multimodal:
        components.append(
            MultimodalConsistencyReward(
                weight=0.20,
                require_visual_ack=True,
                min_visual_references=2,
            )
        )

    if include_code:
        components.append(
            CodeExecutionReward(
                weight=0.20,
                require_code_blocks=False,
                require_explanation=True,
            )
        )

    if include_long_context:
        components.append(
            LongContextReward(
                weight=0.15,
                require_earlier_reference=True,
                context_retention_threshold=0.8,
            )
        )

    reward_function = MultiObjectiveReward(
        name="kimi_k25_conversational",
        components=components,
    )

    return reward_function