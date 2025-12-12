"""
GymAgent - Specialized Agent for Gym Environments

Optimized agent for traditional RL tasks with numeric action spaces.
Extends MultiTurnAgent with gym-specific optimizations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..agent import MultiTurnAgent, AgentConfig


logger = logging.getLogger(__name__)


class GymAgent(MultiTurnAgent):
    """
    Agent specialized for Gym/Gymnasium environments.

    Key optimizations for gym tasks:
    1. Very short text generation (5-10 tokens) for efficiency
    2. Optimized prompt formatting for numeric actions
    3. Better error recovery for action parsing
    4. Compatible with existing GRPO trainer

    Args:
        config: AgentConfig with model parameters
        action_space_info: Optional dict with action space details
        **kwargs: Additional arguments passed to MultiTurnAgent

    Example:
        >>> from core.gym.agents import GymAgent
        >>> from core.agent import AgentConfig
        >>>
        >>> config = AgentConfig(
        ...     model_name="gpt2",
        ...     max_new_tokens=10,  # Very short for gym
        ...     temperature=0.7,
        ...     use_stub_model=False
        ... )
        >>> agent = GymAgent(config)
        >>> await agent.initialize()
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        action_space_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Apply gym-specific defaults if config not fully specified
        if config is None:
            config = AgentConfig()

        # Optimize for short generation (gym actions are typically 1-2 tokens)
        if config.max_new_tokens is None or config.max_new_tokens > 20:
            config.max_new_tokens = 10
            logger.info("Set max_new_tokens=10 for gym agent (short actions)")

        # Slightly higher temperature for exploration
        if config.temperature is None:
            config.temperature = 0.8

        # Enable sampling for exploration
        if config.do_sample is None:
            config.do_sample = True

        super().__init__(config, **kwargs)

        self.action_space_info = action_space_info or {}

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Generate action for gym environment.

        Optimized for:
        - Short generation (actions are typically 1-2 tokens)
        - Direct action output (minimal explanation)
        - Fast inference

        Args:
            messages: Conversation history (system prompt + observation)
            context: Optional context dict
            **kwargs: Additional generation parameters

        Returns:
            Agent response (action as text)
        """
        # Add gym-specific context to messages if needed
        if context and "observation_text" in context:
            # The observation is already in messages, just generate
            pass

        # Generate with parent method (handles all the model logic)
        response = await super().generate_response(messages, context, **kwargs)

        # Post-process: extract just the action if agent was verbose
        # This helps when agents output "I choose action 1" instead of just "1"
        response = response.strip()

        # If response is very short (1-3 chars), it's probably just the action
        if len(response) <= 3:
            return response

        # Otherwise return as-is, mapper will parse it
        return response

    def format_observation_for_prompt(self, observation_text: str) -> str:
        """
        Format gym observation for agent prompt.

        Args:
            observation_text: Processed observation text

        Returns:
            Formatted prompt text
        """
        return f"Current state:\n{observation_text}\n\nYour action:"

    def get_optimized_config_for_gym(self) -> Dict[str, Any]:
        """
        Get recommended config settings for gym environments.

        Returns:
            Dict of recommended settings
        """
        return {
            "max_new_tokens": 10,  # Very short for actions
            "temperature": 0.8,  # Moderate exploration
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,  # Enable exploration
            "repetition_penalty": 1.0,  # No penalty needed for short output
        }

    def __repr__(self) -> str:
        return (
            f"GymAgent(model={self.config.model_name}, "
            f"max_tokens={self.config.max_new_tokens}, "
            f"temp={self.config.temperature})"
        )


# Convenience function for creating pre-configured gym agents
def create_gym_agent(
    model_name: str = "gpt2",
    use_stub: bool = False,
    temperature: float = 0.8,
    **kwargs
) -> GymAgent:
    """
    Create a GymAgent with sensible defaults for gym environments.

    Args:
        model_name: HuggingFace model name (default: "gpt2" - fast and small)
        use_stub: Use stub model for testing (default: False)
        temperature: Sampling temperature (default: 0.8 for exploration)
        **kwargs: Additional AgentConfig parameters

    Returns:
        Initialized GymAgent (not yet initialized - call await agent.initialize())

    Example:
        >>> agent = create_gym_agent(model_name="gpt2", use_stub=False)
        >>> await agent.initialize()
    """
    config = AgentConfig(
        model_name=model_name,
        max_new_tokens=10,
        temperature=temperature,
        do_sample=True,
        use_stub_model=use_stub,
        **kwargs
    )

    return GymAgent(config)
