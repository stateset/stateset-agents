"""
Multimodal Environment for Kimi-K2.5 Models

This module provides a conversational environment tailored for Kimi-K2.5's
multimodal capabilities (text + image + video inputs).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MultimodalMessage:
    """A message supporting text, image, and video content"""

    role: str
    content: List[Dict[str, Any]] = field(default_factory=list)

    def add_text(self, text: str):
        self.content.append({"type": "text", "text": text})

    def add_image(self, image_path: str, image_format: str = "png"):
        self.content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path, "format": image_format},
            }
        )

    def add_video(self, video_path: str, video_format: str = "mp4"):
        self.content.append(
            {
                "type": "video_url",
                "video_url": {"url": video_path, "format": video_format},
            }
        )


@dataclass
class MultimodalScenario:
    """A conversational scenario with multimodal content"""

    scenario_id: str
    task_description: str
    user_input: MultimodalMessage
    expected_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"


class MultimodalConversationEnvironment:
    """
    Environment for multimodal conversational interactions.

    Supports text, image, and video inputs for Kimi-K2.5's native multimodality.
    """

    def __init__(
        self,
        scenarios: List[MultimodalScenario],
        max_turns: int = 5,
        enable_vision: bool = True,
        enable_video: bool = True,
    ):
        self.scenarios = scenarios
        self.max_turns = max_turns
        self.enable_vision = enable_vision
        self.enable_video = enable_video
        self.current_scenario: Optional[MultimodalScenario] = None
        self.current_turn = 0
        self.conversation_history: List[MultimodalMessage] = []

    async def reset(
        self,
        scenario_id: Optional[str] = None,
    ) -> MultimodalMessage:
        """Reset environment for a new conversation"""
        if scenario_id:
            self.current_scenario = next(
                (s for s in self.scenarios if s.scenario_id == scenario_id),
                self.scenarios[0],
            )
        else:
            self.current_scenario = self.scenarios[0]

        self.current_turn = 0
        self.conversation_history = []

        logger.info(f"Starting scenario: {self.current_scenario.scenario_id}")
        logger.info(f"Task: {self.current_scenario.task_description}")

        return self.current_scenario.user_input

    async def step(
        self,
        agent_response: str,
    ) -> tuple[MultimodalMessage, float, bool, Dict[str, Any]]:
        """
        Execute one conversational turn.

        Returns:
            user_input: The next user input (end turn if done)
            reward: Reward signal for this turn
            done: Whether the conversation is complete
            info: Additional information (metrics, state, etc.)
        """
        if not self.current_scenario:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self.current_turn += 1

        # Store agent response
        agent_message = MultimodalMessage(role="assistant")
        agent_message.add_text(agent_response)
        self.conversation_history.append(agent_message)

        # Check if episode is done
        done = self.current_turn >= self.max_turns

        # Compute reward
        reward = self._compute_reward(agent_response)

        # Get next user input (in a real implementation, this would be adaptive)
        if done:
            user_input = None
        else:
            # For training, we might have predefined follow-up inputs
            user_input = self._get_next_user_input()

        info = {
            "scenario_id": self.current_scenario.scenario_id,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "conversation_length": len(self.conversation_history),
        }

        return user_input, reward, done, info

    def _compute_reward(self, response: str) -> float:
        """Compute reward for agent response"""
        if not self.current_scenario:
            return 0.0

        if not self.current_scenario.expected_response:
            # If no expected response, use heuristics
            reward = self._heuristic_reward(response)
        else:
            # Compare with expected response
            reward = self._similarity_reward(
                response, self.current_scenario.expected_response
            )

        return reward

    def _heuristic_reward(self, response: str) -> float:
        """Heuristic reward when no ground truth is available"""
        reward = 0.0

        # Response length penalty (too short or too long)
        if 50 < len(response) < 500:
            reward += 0.3
        elif len(response) >= 500:
            reward += 0.1
        else:
            reward -= 0.2

        # Politeness
        polite_words = ["please", "thank", "appreciate", "happy to help"]
        if any(word in response.lower() for word in polite_words):
            reward += 0.2

        # Structure
        if response.count(".") >= 2:
            reward += 0.2

        # No repetition
        words = response.lower().split()
        if len(set(words)) / max(len(words), 1) > 0.7:
            reward += 0.3

        return max(min(reward, 1.0), -1.0)

    def _similarity_reward(self, response: str, expected: str) -> float:
        """Simple similarity-based reward"""
        # In practice, use more sophisticated metrics (BLEU, ROUGE, semantic similarity)
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())

        # Jaccard similarity
        intersection = response_words & expected_words
        union = response_words | expected_words

        if len(union) == 0:
            return 0.0

        similarity = len(intersection) / len(union)
        return similarity

    def _get_next_user_input(self) -> Optional[MultimodalMessage]:
        """Get the next user input for multi-turn conversations"""
        # In a real implementation, this would be based on the scenario context
        # For now, return None to end after one turn (single-turn mode)
        return None


# Pre-defined multimodal scenarios
VISUAL_QA_SCENARIOS = [
    MultimodalScenario(
        scenario_id="visual_qa_001",
        task_description="Answer questions about the provided image",
        user_input=_create_text_message(
            "Describe what you see in this image in detail."
        ),
        expected_response="A detailed description of the image content",
        difficulty="easy",
        metadata={"modality": "image", "task": "vision_qa"},
    ),
    MultimodalScenario(
        scenario_id="document_analysis_001",
        task_description="Analyze and extract information from a document image",
        user_input=_create_text_message(
            "Extract the key information from this document image. What is the main topic?"
        ),
        expected_response="Key extracted information from the document",
        difficulty="medium",
        metadata={"modality": "image", "task": "document_analysis"},
    ),
]


def _create_text_message(text: str) -> MultimodalMessage:
    """Create a text-only message"""
    msg = MultimodalMessage(role="user")
    msg.add_text(text)
    return msg


def create_multimodal_environment(
    task: str = "visual_qa",
    num_scenarios: int = 10,
    enable_vision: bool = True,
    enable_video: bool = False,
) -> MultimodalConversationEnvironment:
    """Create a multimodal environment for Kimi-K2.5"""

    if task == "visual_qa":
        scenarios = VISUAL_QA_SCENARIOS[:num_scenarios]
    else:
        scenarios = VISUAL_QA_SCENARIOS[:num_scenarios]

    environment = MultimodalConversationEnvironment(
        scenarios=scenarios,
        enable_vision=enable_vision,
        enable_video=enable_video,
    )

    return environment


# Add to core environment module
CONVERSATION_CONFIGS = {
    "customer_service": {
        "scenarios": VISUAL_QA_SCENARIOS,
        "max_turns": 3,
        "enable_vision": True,
    },
    "visual_qa": {
        "scenarios": VISUAL_QA_SCENARIOS,
        "max_turns": 1,
        "enable_vision": True,
    },
}
