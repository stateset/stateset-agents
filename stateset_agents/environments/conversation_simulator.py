"""
Conversation Simulator for Sim-to-Real Transfer

A calibratable conversation simulation environment that can be
tuned to match real conversation distributions for better transfer.
"""

import asyncio
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.environment import Environment, EnvironmentState, EpisodeStatus
from ..core.trajectory import ConversationTurn, MultiTurnTrajectory
from ..training.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
    PersonaGenerator,
    ScenarioGenerator,
    UserPersona,
)

logger = logging.getLogger(__name__)

SIMULATOR_EXCEPTIONS = (OSError, RuntimeError, TypeError, ValueError)


@dataclass
class ConversationSimulatorConfig:
    """Configuration for the conversation simulator"""

    # Persona configuration
    personas: List[UserPersona] = field(default_factory=list)
    persona_sampling: str = "uniform"  # uniform, weighted, curriculum

    # Topic distribution
    topic_distribution: Dict[str, float] = field(
        default_factory=lambda: {
            "general": 0.3,
            "technical": 0.25,
            "customer_service": 0.25,
            "educational": 0.2,
        }
    )

    # Difficulty settings
    difficulty_range: Tuple[float, float] = (0.3, 0.9)
    use_curriculum: bool = True
    curriculum_steps: int = 5000

    # Response noise for robustness
    noise_level: float = 0.1
    typo_rate: float = 0.02
    truncation_rate: float = 0.05

    # Simulation backend
    llm_backend: str = "rule_based"  # rule_based, local, openai, anthropic
    llm_model: Optional[str] = None
    use_rule_based_fallback: bool = True

    # Calibration
    calibration_dataset_path: Optional[str] = None
    calibration_method: str = "mmd"  # mmd, kl, wasserstein

    # Episode settings
    max_turns: int = 20
    timeout_probability: float = 0.05
    early_exit_probability: float = 0.1

    # Reward settings
    reward_scale: float = 1.0
    penalize_repetition: bool = True
    penalize_irrelevance: bool = True


class UserSimulator:
    """
    Simulates user responses in conversations.

    Can use rule-based responses or LLM-generated responses
    based on the configured persona.
    """

    def __init__(
        self,
        persona: UserPersona,
        config: ConversationSimulatorConfig,
    ):
        self.persona = persona
        self.config = config
        self._response_templates = self._load_templates()
        self._llm_client = None

    def _load_templates(self) -> Dict[str, List[str]]:
        """Load response templates for rule-based simulation"""
        templates = {
            "greeting": [
                "Hi, I need help with something.",
                "Hello! I have a question.",
                "Hey, can you help me?",
            ],
            "clarification": [
                "What do you mean by that?",
                "Can you explain more?",
                "I don't understand, could you clarify?",
            ],
            "follow_up": [
                "Okay, but what about {topic}?",
                "That makes sense. What about {topic}?",
                "Got it. One more thing about {topic}...",
            ],
            "frustrated": [
                "This isn't helping at all.",
                "I've already tried that.",
                "That's not what I asked for.",
            ],
            "satisfied": [
                "That's exactly what I needed, thanks!",
                "Perfect, that helps a lot.",
                "Great, I think I understand now.",
            ],
            "confused": [
                "I'm still confused about this.",
                "That doesn't make sense to me.",
                "Can you try explaining it differently?",
            ],
            "acknowledgment": [
                "Okay.",
                "I see.",
                "Alright.",
                "Got it.",
            ],
        }

        # Adjust templates based on persona
        if self.persona.formality > 0.7:
            templates["greeting"] = [
                "Good day, I require assistance.",
                "Hello, I have an inquiry.",
                "Greetings, could you please help me?",
            ]
        elif self.persona.formality < 0.3:
            templates["greeting"] = [
                "yo, need some help here",
                "hey whats up, got a question",
                "hi can u help me out",
            ]

        return templates

    async def generate_response(
        self,
        history: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a user response based on conversation history.

        Args:
            history: List of {"role": ..., "content": ...} messages
            context: Additional context (scenario, turn number, etc.)

        Returns:
            Simulated user response
        """
        if self.config.llm_backend == "rule_based" or (
            self.config.use_rule_based_fallback and self._llm_client is None
        ):
            return self._generate_rule_based(history, context)

        try:
            return await self._generate_llm_response(history, context)
        except SIMULATOR_EXCEPTIONS as e:
            logger.warning(f"LLM generation failed: {e}, falling back to rule-based")
            return self._generate_rule_based(history, context)

    def _generate_rule_based(
        self,
        history: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> str:
        """Generate response using rule-based logic"""
        turn_count = len([m for m in history if m.get("role") == "user"])

        # First turn: greeting
        if turn_count == 0:
            response = random.choice(self._response_templates["greeting"])

        # Based on persona emotion
        elif self.persona.emotion_model == "frustrated" and random.random() < 0.4:
            response = random.choice(self._response_templates["frustrated"])

        elif self.persona.emotion_model == "confused" and random.random() < 0.3:
            response = random.choice(self._response_templates["confused"])

        # Check last assistant response
        elif history and history[-1].get("role") == "assistant":
            last_response = history[-1].get("content", "")

            # Ask for clarification if response is long and persona is not expert
            if len(last_response) > 500 and self.persona.expertise_level < 0.5:
                response = random.choice(self._response_templates["clarification"])

            # Follow-up question based on patience
            elif random.random() > self.persona.patience_level:
                topic = context.get("topic", "this")
                templates = self._response_templates["follow_up"]
                response = random.choice(templates).format(topic=topic)

            # Simple acknowledgment
            elif random.random() < 0.3:
                response = random.choice(self._response_templates["acknowledgment"])

            else:
                # Generate contextual response
                response = self._generate_contextual_response(history, context)

        else:
            response = self._generate_contextual_response(history, context)

        # Add noise if configured
        if self.config.noise_level > 0:
            response = self._add_noise(response)

        return response

    def _generate_contextual_response(
        self,
        history: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> str:
        """Generate a contextual response based on scenario"""
        scenario = context.get("scenario", {})
        topic = scenario.get("topic", "general")

        contextual_responses = {
            "customer_service": [
                "I ordered something last week and it hasn't arrived yet.",
                "I'm having an issue with my account.",
                "Can I get a refund for this?",
                "The product doesn't work as described.",
            ],
            "technical": [
                "I'm getting an error when I try to do this.",
                "The feature isn't working correctly.",
                "How do I configure this setting?",
                "I followed the instructions but it's not working.",
            ],
            "educational": [
                "Can you explain how this works?",
                "What's the difference between these two things?",
                "I'm trying to learn about this topic.",
                "Can you give me an example?",
            ],
            "general": [
                "I have a question about something.",
                "Can you help me understand this?",
                "I'm not sure how to proceed.",
                "What would you recommend?",
            ],
        }

        responses = contextual_responses.get(topic, contextual_responses["general"])
        return random.choice(responses)

    async def _generate_llm_response(
        self,
        history: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> str:
        """Generate response using LLM"""
        # Build system prompt from persona
        system_prompt = self.persona.to_system_prompt()

        # Add scenario context
        scenario = context.get("scenario", {})
        if scenario:
            system_prompt += f"\n\nScenario context: {scenario.get('context', '')}"
            system_prompt += f"\nYour goal: {scenario.get('user_goal', '')}"

        # This would call the LLM - placeholder for actual implementation
        # In practice, you'd integrate with OpenAI/Anthropic/local model
        raise NotImplementedError("LLM backend not implemented")

    def _add_noise(self, response: str) -> str:
        """Add realistic noise to response"""
        # Typos
        if random.random() < self.config.typo_rate:
            response = self._add_typo(response)

        # Truncation (user sends incomplete message)
        if random.random() < self.config.truncation_rate:
            words = response.split()
            if len(words) > 3:
                truncate_at = random.randint(len(words) // 2, len(words) - 1)
                response = " ".join(words[:truncate_at])

        return response

    def _add_typo(self, text: str) -> str:
        """Add a random typo"""
        if len(text) < 5:
            return text

        pos = random.randint(1, len(text) - 2)
        typo_type = random.choice(["swap", "delete", "duplicate"])

        if typo_type == "swap" and pos < len(text) - 1:
            text = text[:pos] + text[pos + 1] + text[pos] + text[pos + 2:]
        elif typo_type == "delete":
            text = text[:pos] + text[pos + 1:]
        elif typo_type == "duplicate":
            text = text[:pos] + text[pos] + text[pos:]

        return text

    def set_emotion(self, emotion: str) -> None:
        """Update persona emotion state"""
        self.persona.emotion_model = emotion

    def adjust_difficulty(self, difficulty: float) -> None:
        """Adjust persona traits based on difficulty"""
        self.persona.patience_level = 1.0 - difficulty * 0.7
        self.persona.traits["patience"] = self.persona.patience_level


class ConversationSimulator(Environment):
    """
    Calibratable conversation simulation environment.

    Features:
    - Domain randomization (personas, topics, styles)
    - LLM-based or rule-based user simulation
    - Calibration to real conversation distributions
    - Progressive difficulty adjustment
    - Sim-to-real gap measurement

    Example:
        >>> config = ConversationSimulatorConfig(...)
        >>> simulator = ConversationSimulator(config)
        >>>
        >>> state = await simulator.reset()
        >>> while not done:
        ...     action = agent.generate_response(state)
        ...     state, reward, done, info = await simulator.step(state, action)
    """

    def __init__(
        self,
        config: ConversationSimulatorConfig,
        domain_randomization: Optional[DomainRandomizationConfig] = None,
    ):
        super().__init__(max_turns=config.max_turns)

        self.config = config
        self.dr_config = domain_randomization or DomainRandomizationConfig(
            use_curriculum=config.use_curriculum,
            curriculum_steps=config.curriculum_steps,
            topics=list(config.topic_distribution.keys()),
            topic_weights=config.topic_distribution,
        )

        # Initialize components
        self.domain_randomizer = DomainRandomizer(self.dr_config)
        self.persona_generator = PersonaGenerator(self.dr_config)
        self.scenario_generator = ScenarioGenerator(self.dr_config)

        # Current episode state
        self._current_simulator: Optional[UserSimulator] = None
        self._current_scenario: Optional[Dict[str, Any]] = None
        self._episode_history: List[ConversationTurn] = []

        # Calibration data
        self._calibration_stats: Dict[str, Any] = {}
        self._is_calibrated = False

        # Training step counter
        self._training_step = 0

    async def reset(
        self,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> EnvironmentState:
        """
        Reset environment for new episode.

        Args:
            scenario: Optional pre-defined scenario

        Returns:
            Initial EnvironmentState
        """
        # Generate or use provided scenario
        if scenario is None:
            scenario = self.domain_randomizer.randomize_scenario()
        self._current_scenario = scenario

        # Get or create persona
        persona = scenario.get("persona")
        if persona is None:
            difficulty = scenario.get("difficulty", 0.5)
            persona = self.persona_generator.get_persona_for_difficulty(difficulty)

        # Create user simulator
        self._current_simulator = UserSimulator(persona, self.config)

        # Reset episode state
        self._episode_history = []
        episode_id = str(uuid.uuid4())

        # Generate initial user message
        initial_message = await self._current_simulator.generate_response(
            history=[],
            context={"scenario": scenario, "turn": 0},
        )

        initial_turn = ConversationTurn(
            role="user",
            content=initial_message,
            metadata={"simulated": True},
        )
        self._episode_history.append(initial_turn)

        # Create state
        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=1,
            status=EpisodeStatus.ONGOING,
            context={
                "scenario": scenario,
                "persona": persona.name,
                "difficulty": scenario.get("difficulty", 0.5),
                "topic": scenario.get("topic", "general"),
                "last_user_message": initial_message,
            },
            metadata={
                "persona_traits": persona.traits,
                "is_simulated": True,
            },
        )

        self.active_episodes[episode_id] = state
        self._training_step += 1

        return state

    async def step(
        self,
        state: EnvironmentState,
        action: ConversationTurn,
    ) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """
        Execute one conversation turn.

        Args:
            state: Current environment state
            action: Agent's response (ConversationTurn)

        Returns:
            Tuple of (new_state, reward, done, info)
        """
        # Record agent response
        self._episode_history.append(action)

        # Compute reward for agent response
        reward = self._compute_reward(action, state)

        # Check for early termination
        done = False
        info = {"reward_breakdown": {}}

        # Random early exit (user leaves)
        if random.random() < self.config.early_exit_probability:
            done = True
            info["termination_reason"] = "user_exit"

        # Timeout
        elif random.random() < self.config.timeout_probability:
            done = True
            info["termination_reason"] = "timeout"

        # Max turns reached
        elif state.turn_count >= self.max_turns:
            done = True
            info["termination_reason"] = "max_turns"

        # Check for conversation completion
        elif self._is_conversation_complete(state, action):
            done = True
            info["termination_reason"] = "completed"
            reward += 0.5 * self.config.reward_scale  # Bonus for completion

        if not done:
            # Generate user response
            history = [
                {"role": t.role, "content": t.content}
                for t in self._episode_history
            ]
            user_response = await self._current_simulator.generate_response(
                history=history,
                context={
                    "scenario": self._current_scenario,
                    "turn": state.turn_count,
                },
            )

            user_turn = ConversationTurn(
                role="user",
                content=user_response,
                metadata={"simulated": True},
            )
            self._episode_history.append(user_turn)

            info["user_response"] = user_response

        # Update state
        new_state = state.copy()
        new_state.turn_count += 1
        new_state.context["last_user_message"] = (
            info.get("user_response", "") if not done else ""
        )

        if done:
            new_state.status = EpisodeStatus.COMPLETED
            self.active_episodes.pop(state.episode_id, None)

        return new_state, reward, done, info

    def _compute_reward(
        self,
        action: ConversationTurn,
        state: EnvironmentState,
    ) -> float:
        """Compute reward for agent action"""
        reward = 0.0
        response = action.content

        # Base reward for responding
        reward += 0.1 * self.config.reward_scale

        # Length-based reward (prefer medium-length responses)
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            reward += 0.1 * self.config.reward_scale
        elif word_count < 5:
            reward -= 0.1 * self.config.reward_scale
        elif word_count > 200:
            reward -= 0.05 * self.config.reward_scale

        # Penalize repetition
        if self.config.penalize_repetition:
            history_text = " ".join([t.content for t in self._episode_history[:-1]])
            if response in history_text:
                reward -= 0.2 * self.config.reward_scale

        # Check relevance to topic
        if self.config.penalize_irrelevance:
            topic = state.context.get("topic", "")
            last_user = state.context.get("last_user_message", "")
            if not self._is_relevant(response, topic, last_user):
                reward -= 0.1 * self.config.reward_scale

        # Politeness bonus
        if any(word in response.lower() for word in ["please", "thank", "appreciate", "help"]):
            reward += 0.05 * self.config.reward_scale

        return reward

    def _is_relevant(
        self,
        response: str,
        topic: str,
        user_message: str,
    ) -> bool:
        """Check if response is relevant to topic and user message"""
        # Simple keyword overlap check
        response_words = set(response.lower().split())
        user_words = set(user_message.lower().split())

        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall"}
        response_words -= stopwords
        user_words -= stopwords

        # Check for overlap
        overlap = response_words & user_words
        return len(overlap) > 0 or len(response_words) > 5

    def _is_conversation_complete(
        self,
        state: EnvironmentState,
        action: ConversationTurn,
    ) -> bool:
        """Check if conversation goal is achieved"""
        response = action.content.lower()

        # Check for resolution indicators
        resolution_phrases = [
            "let me know if you have any other questions",
            "is there anything else",
            "hope this helps",
            "does that answer your question",
            "feel free to ask",
        ]

        for phrase in resolution_phrases:
            if phrase in response:
                return random.random() < 0.5  # 50% chance user is satisfied

        return False

    def randomize_persona(self) -> UserPersona:
        """Get a new random persona"""
        return self.persona_generator.generate_random_persona()

    def randomize_scenario(self) -> Dict[str, Any]:
        """Get a new random scenario"""
        return self.scenario_generator.curriculum_sample(self._training_step)

    async def calibrate(
        self,
        real_data: Any,  # ConversationDataset
        num_samples: int = 1000,
    ) -> Dict[str, float]:
        """
        Calibrate simulator to match real conversation data.

        Args:
            real_data: ConversationDataset with real conversations
            num_samples: Number of samples for calibration

        Returns:
            Calibration metrics
        """
        logger.info(f"Calibrating simulator with {num_samples} samples")

        # Collect statistics from real data
        real_stats = self._compute_dataset_stats(real_data)

        # Collect statistics from simulated data
        sim_trajectories = []
        for _ in range(min(num_samples, len(real_data))):
            state = await self.reset()
            trajectory = []

            for _ in range(self.max_turns):
                # Simulate with random agent responses
                response = self._generate_dummy_response()
                action = ConversationTurn(role="assistant", content=response)
                trajectory.append(action)

                new_state, _, done, _ = await self.step(state, action)
                if done:
                    break
                state = new_state

            sim_trajectories.append(trajectory)

        sim_stats = self._compute_trajectory_stats(sim_trajectories)

        # Compute calibration adjustments
        adjustments = self._compute_adjustments(real_stats, sim_stats)

        # Apply adjustments
        self._apply_calibration(adjustments)

        self._calibration_stats = {
            "real_stats": real_stats,
            "sim_stats": sim_stats,
            "adjustments": adjustments,
        }
        self._is_calibrated = True

        return self.compute_sim_real_gap(real_data)

    def _compute_dataset_stats(
        self,
        dataset: Any,
    ) -> Dict[str, float]:
        """Compute statistics from a dataset"""
        response_lengths = []
        turn_counts = []
        rewards = []

        for traj in dataset:
            turn_counts.append(len(traj.turns))
            for turn in traj.turns:
                if turn.role == "assistant":
                    response_lengths.append(len(turn.content.split()))
            if hasattr(traj, "total_reward"):
                rewards.append(traj.total_reward)

        return {
            "mean_response_length": np.mean(response_lengths) if response_lengths else 0,
            "std_response_length": np.std(response_lengths) if response_lengths else 0,
            "mean_turn_count": np.mean(turn_counts) if turn_counts else 0,
            "mean_reward": np.mean(rewards) if rewards else 0,
        }

    def _compute_trajectory_stats(
        self,
        trajectories: List[List[ConversationTurn]],
    ) -> Dict[str, float]:
        """Compute statistics from generated trajectories"""
        response_lengths = []
        turn_counts = []

        for traj in trajectories:
            turn_counts.append(len(traj))
            for turn in traj:
                if turn.role == "assistant":
                    response_lengths.append(len(turn.content.split()))

        return {
            "mean_response_length": np.mean(response_lengths) if response_lengths else 0,
            "std_response_length": np.std(response_lengths) if response_lengths else 0,
            "mean_turn_count": np.mean(turn_counts) if turn_counts else 0,
            "mean_reward": 0,  # Not computed for generated
        }

    def _compute_adjustments(
        self,
        real_stats: Dict[str, float],
        sim_stats: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute calibration adjustments"""
        return {
            "response_length_ratio": (
                real_stats["mean_response_length"] / (sim_stats["mean_response_length"] + 1e-6)
            ),
            "turn_count_ratio": (
                real_stats["mean_turn_count"] / (sim_stats["mean_turn_count"] + 1e-6)
            ),
        }

    def _apply_calibration(
        self,
        adjustments: Dict[str, float],
    ) -> None:
        """Apply calibration adjustments to simulator"""
        # Adjust max turns based on calibration
        turn_ratio = adjustments.get("turn_count_ratio", 1.0)
        self.max_turns = int(self.max_turns * min(max(turn_ratio, 0.5), 2.0))

        logger.info(f"Applied calibration: max_turns={self.max_turns}")

    def _generate_dummy_response(self) -> str:
        """Generate a dummy agent response for calibration"""
        responses = [
            "I understand. Let me help you with that.",
            "Thanks for reaching out. Here's what I can do.",
            "I see what you mean. Let me explain.",
            "Good question! Here's the answer.",
            "I appreciate you asking. Here's some information.",
        ]
        return random.choice(responses)

    def compute_sim_real_gap(
        self,
        real_data: Any,
    ) -> Dict[str, float]:
        """
        Compute metrics for sim-to-real gap.

        Args:
            real_data: ConversationDataset with real conversations

        Returns:
            Gap metrics dictionary
        """
        real_stats = self._compute_dataset_stats(real_data)
        sim_stats = self._calibration_stats.get("sim_stats", {})

        if not sim_stats:
            return {"error": "Calibration required first"}

        # Response length gap
        length_gap = abs(
            real_stats["mean_response_length"] - sim_stats["mean_response_length"]
        ) / (real_stats["mean_response_length"] + 1e-6)

        # Turn count gap
        turn_gap = abs(
            real_stats["mean_turn_count"] - sim_stats["mean_turn_count"]
        ) / (real_stats["mean_turn_count"] + 1e-6)

        return {
            "response_length_gap": length_gap,
            "turn_count_gap": turn_gap,
            "overall_gap": (length_gap + turn_gap) / 2,
            "is_calibrated": self._is_calibrated,
        }

    def clone(self) -> "ConversationSimulator":
        """Create a clone of this simulator for parallel rollouts"""
        cloned = ConversationSimulator(self.config, self.dr_config)
        cloned._calibration_stats = dict(self._calibration_stats)
        cloned._is_calibrated = self._is_calibrated
        cloned._training_step = self._training_step
        return cloned

    async def get_initial_prompt(
        self,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get system prompt for the scenario"""
        if scenario is None:
            scenario = self._current_scenario or {}

        topic = scenario.get("topic", "general")
        context = scenario.get("context", "")

        return f"""You are a helpful assistant engaged in a {topic} conversation.

Context: {context}

Please respond helpfully, accurately, and appropriately to the user's messages."""
