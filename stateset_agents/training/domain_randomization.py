"""
Domain Randomization for Conversational Agents

Provides utilities for randomizing conversation training environments
to improve sim-to-real transfer. Includes persona generation,
scenario variation, and curriculum learning.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UserPersona:
    """
    Defines a simulated user persona for conversation training.

    Personas encapsulate user behavior patterns, communication styles,
    and personality traits that affect how they interact in conversations.
    """

    name: str
    traits: Dict[str, float] = field(default_factory=dict)
    vocabulary_style: str = "neutral"  # formal, casual, technical, emotional
    response_patterns: List[str] = field(default_factory=list)
    emotion_model: str = "neutral"  # frustrated, happy, confused, impatient
    expertise_level: float = 0.5  # 0-1, domain expertise
    patience_level: float = 0.5  # 0-1, conversation patience
    verbosity: float = 0.5  # 0-1, response length preference
    formality: float = 0.5  # 0-1, language formality
    description: str = ""

    def to_system_prompt(self) -> str:
        """Convert persona to a system prompt for LLM-based simulation"""
        traits_desc = ", ".join([f"{k}: {v:.1f}" for k, v in self.traits.items()])

        prompt = f"""You are simulating a user named {self.name}.

Personality traits: {traits_desc}
Communication style: {self.vocabulary_style}
Emotional state: {self.emotion_model}
Expertise level: {self.expertise_level:.1%}
Patience: {self.patience_level:.1%}
Verbosity preference: {self.verbosity:.1%}
Formality: {self.formality:.1%}

{self.description}

Respond as this user would, matching their communication style and emotional state."""

        return prompt

    def adjust_trait(self, trait: str, delta: float) -> None:
        """Adjust a trait value, clamping to [0, 1]"""
        if trait in self.traits:
            self.traits[trait] = max(0.0, min(1.0, self.traits[trait] + delta))

    def copy(self) -> "UserPersona":
        """Create a copy of this persona"""
        return UserPersona(
            name=self.name,
            traits=dict(self.traits),
            vocabulary_style=self.vocabulary_style,
            response_patterns=list(self.response_patterns),
            emotion_model=self.emotion_model,
            expertise_level=self.expertise_level,
            patience_level=self.patience_level,
            verbosity=self.verbosity,
            formality=self.formality,
            description=self.description,
        )


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization"""

    # Persona randomization
    num_personas: int = 100
    persona_traits: List[str] = field(
        default_factory=lambda: [
            "patience",
            "expertise",
            "verbosity",
            "formality",
            "emotion_stability",
            "cooperativeness",
            "detail_orientation",
        ]
    )
    trait_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Topic randomization
    topics: List[str] = field(default_factory=list)
    topic_weights: Optional[Dict[str, float]] = None

    # Style randomization
    styles: List[str] = field(
        default_factory=lambda: [
            "formal",
            "casual",
            "technical",
            "emotional",
            "concise",
            "verbose",
        ]
    )
    style_weights: Optional[Dict[str, float]] = None

    # Emotion randomization
    emotions: List[str] = field(
        default_factory=lambda: [
            "neutral",
            "frustrated",
            "happy",
            "confused",
            "impatient",
            "curious",
            "skeptical",
        ]
    )
    emotion_weights: Optional[Dict[str, float]] = None

    # Difficulty curriculum
    use_curriculum: bool = True
    initial_difficulty: float = 0.3
    max_difficulty: float = 0.9
    curriculum_steps: int = 5000
    curriculum_schedule: str = "linear"  # linear, exponential, cosine

    # Noise parameters
    response_noise_level: float = 0.1
    typo_probability: float = 0.02
    truncation_probability: float = 0.05

    # Seed for reproducibility
    seed: Optional[int] = None


# Pre-defined persona templates
PERSONA_TEMPLATES = {
    "patient_expert": UserPersona(
        name="Expert User",
        traits={"patience": 0.9, "expertise": 0.9, "cooperativeness": 0.8},
        vocabulary_style="technical",
        emotion_model="neutral",
        expertise_level=0.9,
        patience_level=0.9,
        verbosity=0.6,
        formality=0.7,
        description="A knowledgeable user who communicates clearly and patiently.",
    ),
    "frustrated_novice": UserPersona(
        name="Frustrated Novice",
        traits={"patience": 0.2, "expertise": 0.2, "cooperativeness": 0.5},
        vocabulary_style="casual",
        emotion_model="frustrated",
        expertise_level=0.2,
        patience_level=0.2,
        verbosity=0.7,
        formality=0.3,
        description="A user who is new to the topic and getting frustrated.",
    ),
    "busy_professional": UserPersona(
        name="Busy Professional",
        traits={"patience": 0.4, "expertise": 0.7, "cooperativeness": 0.6},
        vocabulary_style="formal",
        emotion_model="impatient",
        expertise_level=0.7,
        patience_level=0.3,
        verbosity=0.3,
        formality=0.8,
        description="A professional with limited time who wants quick answers.",
    ),
    "curious_learner": UserPersona(
        name="Curious Learner",
        traits={"patience": 0.8, "expertise": 0.4, "cooperativeness": 0.9},
        vocabulary_style="casual",
        emotion_model="curious",
        expertise_level=0.4,
        patience_level=0.8,
        verbosity=0.8,
        formality=0.4,
        description="An eager learner who asks follow-up questions.",
    ),
    "skeptical_critic": UserPersona(
        name="Skeptical Critic",
        traits={"patience": 0.5, "expertise": 0.6, "cooperativeness": 0.4},
        vocabulary_style="formal",
        emotion_model="skeptical",
        expertise_level=0.6,
        patience_level=0.5,
        verbosity=0.6,
        formality=0.6,
        description="A user who questions responses and wants proof.",
    ),
}


class PersonaGenerator:
    """
    Generates diverse user personas for training.

    Supports random generation, template-based creation,
    and interpolation between personas.
    """

    def __init__(
        self,
        config: DomainRandomizationConfig,
        templates: Optional[Dict[str, UserPersona]] = None,
    ):
        self.config = config
        self.templates = templates or PERSONA_TEMPLATES

        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        # Pre-generate persona pool
        self._persona_pool: List[UserPersona] = []

    def generate_random_persona(self) -> UserPersona:
        """Generate a random persona with varied traits"""
        # Random traits
        traits = {}
        for trait in self.config.persona_traits:
            if trait in self.config.trait_ranges:
                low, high = self.config.trait_ranges[trait]
            else:
                low, high = 0.0, 1.0
            traits[trait] = random.uniform(low, high)

        # Random style
        if self.config.style_weights:
            styles = list(self.config.style_weights.keys())
            weights = list(self.config.style_weights.values())
            style = random.choices(styles, weights=weights, k=1)[0]
        else:
            style = random.choice(self.config.styles)

        # Random emotion
        if self.config.emotion_weights:
            emotions = list(self.config.emotion_weights.keys())
            weights = list(self.config.emotion_weights.values())
            emotion = random.choices(emotions, weights=weights, k=1)[0]
        else:
            emotion = random.choice(self.config.emotions)

        # Generate name
        names = [
            "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
            "Avery", "Reese", "Parker", "Drew", "Jamie", "Sam", "Charlie",
        ]
        name = f"{random.choice(names)}_{random.randint(1000, 9999)}"

        return UserPersona(
            name=name,
            traits=traits,
            vocabulary_style=style,
            emotion_model=emotion,
            expertise_level=traits.get("expertise", 0.5),
            patience_level=traits.get("patience", 0.5),
            verbosity=traits.get("verbosity", 0.5),
            formality=0.7 if style == "formal" else 0.3 if style == "casual" else 0.5,
        )

    def generate_persona_batch(self, n: int) -> List[UserPersona]:
        """Generate a batch of random personas"""
        return [self.generate_random_persona() for _ in range(n)]

    def from_template(
        self,
        template_name: str,
        variations: Optional[Dict[str, float]] = None,
    ) -> UserPersona:
        """Create persona from template with optional variations"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")

        persona = self.templates[template_name].copy()

        if variations:
            for trait, delta in variations.items():
                if trait in persona.traits:
                    persona.adjust_trait(trait, delta)

        return persona

    def interpolate_personas(
        self,
        persona1: UserPersona,
        persona2: UserPersona,
        alpha: float,
    ) -> UserPersona:
        """
        Interpolate between two personas.

        Args:
            persona1: First persona
            persona2: Second persona
            alpha: Interpolation factor (0 = persona1, 1 = persona2)

        Returns:
            Interpolated persona
        """
        alpha = max(0.0, min(1.0, alpha))

        # Interpolate traits
        traits = {}
        all_traits = set(persona1.traits.keys()) | set(persona2.traits.keys())
        for trait in all_traits:
            v1 = persona1.traits.get(trait, 0.5)
            v2 = persona2.traits.get(trait, 0.5)
            traits[trait] = (1 - alpha) * v1 + alpha * v2

        # Choose style based on alpha
        style = persona1.vocabulary_style if alpha < 0.5 else persona2.vocabulary_style
        emotion = persona1.emotion_model if alpha < 0.5 else persona2.emotion_model

        return UserPersona(
            name=f"Interpolated_{random.randint(1000, 9999)}",
            traits=traits,
            vocabulary_style=style,
            emotion_model=emotion,
            expertise_level=(1 - alpha) * persona1.expertise_level + alpha * persona2.expertise_level,
            patience_level=(1 - alpha) * persona1.patience_level + alpha * persona2.patience_level,
            verbosity=(1 - alpha) * persona1.verbosity + alpha * persona2.verbosity,
            formality=(1 - alpha) * persona1.formality + alpha * persona2.formality,
        )

    def get_persona_for_difficulty(
        self,
        difficulty: float,
    ) -> UserPersona:
        """Get a persona appropriate for a difficulty level"""
        difficulty = max(0.0, min(1.0, difficulty))

        # Low difficulty: patient, clear personas
        # High difficulty: impatient, confusing personas
        persona = self.generate_random_persona()

        # Adjust traits based on difficulty
        persona.patience_level = 1.0 - difficulty * 0.8
        persona.traits["patience"] = persona.patience_level
        persona.traits["cooperativeness"] = 1.0 - difficulty * 0.6

        # Higher difficulty = more emotional variety
        if difficulty > 0.7:
            persona.emotion_model = random.choice(["frustrated", "impatient", "skeptical"])
        elif difficulty > 0.4:
            persona.emotion_model = random.choice(["neutral", "confused", "curious"])
        else:
            persona.emotion_model = random.choice(["neutral", "happy", "curious"])

        return persona


class ScenarioGenerator:
    """
    Generates diverse conversation scenarios for training.

    Scenarios define the context, goals, and constraints of
    a conversation training episode.
    """

    def __init__(
        self,
        config: DomainRandomizationConfig,
        custom_scenarios: Optional[List[Dict[str, Any]]] = None,
    ):
        self.config = config
        self.custom_scenarios = custom_scenarios or []

        if config.seed is not None:
            random.seed(config.seed)

    def generate_scenario(
        self,
        difficulty: Optional[float] = None,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a random conversation scenario.

        Args:
            difficulty: Optional difficulty override
            topic: Optional topic override

        Returns:
            Scenario dictionary with context and constraints
        """
        # Select topic
        if topic is None:
            if self.config.topics:
                if self.config.topic_weights:
                    topics = list(self.config.topic_weights.keys())
                    weights = list(self.config.topic_weights.values())
                    topic = random.choices(topics, weights=weights, k=1)[0]
                else:
                    topic = random.choice(self.config.topics)
            else:
                topic = "general"

        # Determine difficulty
        if difficulty is None:
            difficulty = random.uniform(
                self.config.initial_difficulty,
                self.config.max_difficulty,
            )

        # Generate scenario components
        scenario = {
            "topic": topic,
            "difficulty": difficulty,
            "context": self._generate_context(topic),
            "user_goal": self._generate_goal(topic, difficulty),
            "constraints": self._generate_constraints(difficulty),
            "success_criteria": self._generate_success_criteria(topic),
            "max_turns": self._get_max_turns(difficulty),
        }

        return scenario

    def _generate_context(self, topic: str) -> str:
        """Generate context description for scenario"""
        contexts = {
            "customer_service": [
                "User is contacting support about a product issue.",
                "User needs help with an order or delivery.",
                "User wants to request a refund or exchange.",
                "User has a billing question.",
            ],
            "technical_support": [
                "User is experiencing a software bug.",
                "User needs help setting up a feature.",
                "User's system is not working as expected.",
                "User wants to understand how something works.",
            ],
            "general": [
                "User has a general question.",
                "User is looking for information.",
                "User wants advice or recommendations.",
                "User needs help with a task.",
            ],
        }

        topic_contexts = contexts.get(topic, contexts["general"])
        return random.choice(topic_contexts)

    def _generate_goal(self, topic: str, difficulty: float) -> str:
        """Generate user goal based on topic and difficulty"""
        simple_goals = [
            "Get a quick answer to a simple question",
            "Understand a basic concept",
            "Complete a straightforward task",
        ]

        complex_goals = [
            "Resolve a multi-step issue",
            "Understand a complex topic",
            "Complete a task with specific requirements",
            "Get help with an unusual edge case",
        ]

        if difficulty < 0.4:
            return random.choice(simple_goals)
        elif difficulty < 0.7:
            return random.choice(simple_goals + complex_goals)
        else:
            return random.choice(complex_goals)

    def _generate_constraints(self, difficulty: float) -> Dict[str, Any]:
        """Generate scenario constraints"""
        constraints = {
            "time_pressure": difficulty > 0.6,
            "limited_context": difficulty > 0.5,
            "emotional_user": difficulty > 0.7,
            "technical_jargon": difficulty > 0.4,
            "ambiguous_request": difficulty > 0.6,
        }
        return constraints

    def _generate_success_criteria(self, topic: str) -> List[str]:
        """Generate success criteria for the scenario"""
        common_criteria = [
            "User's question is answered",
            "Conversation is polite and professional",
            "Response is helpful and accurate",
        ]

        topic_criteria = {
            "customer_service": [
                "Issue is resolved or escalated appropriately",
                "Customer feels heard and valued",
            ],
            "technical_support": [
                "Technical issue is diagnosed",
                "Clear solution or workaround provided",
            ],
        }

        return common_criteria + topic_criteria.get(topic, [])

    def _get_max_turns(self, difficulty: float) -> int:
        """Get max turns based on difficulty"""
        base_turns = 5
        additional = int(difficulty * 15)
        return base_turns + additional

    def curriculum_sample(self, step: int) -> Dict[str, Any]:
        """
        Sample a scenario using curriculum learning.

        Difficulty increases over training steps.
        """
        if not self.config.use_curriculum:
            return self.generate_scenario()

        # Calculate difficulty based on curriculum schedule
        progress = min(step / self.config.curriculum_steps, 1.0)

        if self.config.curriculum_schedule == "linear":
            difficulty = (
                self.config.initial_difficulty
                + progress * (self.config.max_difficulty - self.config.initial_difficulty)
            )
        elif self.config.curriculum_schedule == "exponential":
            difficulty = self.config.initial_difficulty * (
                (self.config.max_difficulty / self.config.initial_difficulty) ** progress
            )
        elif self.config.curriculum_schedule == "cosine":
            difficulty = self.config.initial_difficulty + 0.5 * (
                self.config.max_difficulty - self.config.initial_difficulty
            ) * (1 - np.cos(np.pi * progress))
        else:
            difficulty = self.config.initial_difficulty

        return self.generate_scenario(difficulty=difficulty)


class DomainRandomizer:
    """
    Main class for applying domain randomization to conversation training.

    Coordinates persona generation, scenario variation, and response
    augmentation to create diverse training experiences.
    """

    def __init__(self, config: DomainRandomizationConfig):
        self.config = config
        self.persona_generator = PersonaGenerator(config)
        self.scenario_generator = ScenarioGenerator(config)
        self._step = 0

    def randomize_scenario(
        self,
        base_scenario: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply randomization to a scenario.

        Args:
            base_scenario: Optional base scenario to modify

        Returns:
            Randomized scenario
        """
        if base_scenario is None:
            scenario = self.scenario_generator.curriculum_sample(self._step)
        else:
            scenario = dict(base_scenario)

        # Add randomized persona
        difficulty = scenario.get("difficulty", 0.5)
        scenario["persona"] = self.persona_generator.get_persona_for_difficulty(
            difficulty
        )

        # Add noise parameters
        scenario["noise"] = {
            "response_noise": self.config.response_noise_level,
            "typo_probability": self.config.typo_probability * difficulty,
            "truncation_probability": self.config.truncation_probability * difficulty,
        }

        self._step += 1
        return scenario

    def add_response_noise(
        self,
        response: str,
        noise_level: float = None,
    ) -> str:
        """
        Add realistic noise to a response.

        Includes typos, truncation, and informal language.
        """
        if noise_level is None:
            noise_level = self.config.response_noise_level

        if noise_level <= 0:
            return response

        result = response

        # Typos
        if random.random() < self.config.typo_probability * noise_level:
            result = self._add_typo(result)

        # Truncation
        if random.random() < self.config.truncation_probability * noise_level:
            result = self._truncate(result)

        return result

    def _add_typo(self, text: str) -> str:
        """Add a random typo to text"""
        if len(text) < 5:
            return text

        typo_types = ["swap", "delete", "duplicate", "replace"]
        typo_type = random.choice(typo_types)

        pos = random.randint(1, len(text) - 2)

        if typo_type == "swap" and pos < len(text) - 1:
            text = text[:pos] + text[pos + 1] + text[pos] + text[pos + 2:]
        elif typo_type == "delete":
            text = text[:pos] + text[pos + 1:]
        elif typo_type == "duplicate":
            text = text[:pos] + text[pos] + text[pos:]
        elif typo_type == "replace":
            keyboard_neighbors = {
                "a": "sq", "s": "awd", "d": "sfe", "f": "dgr",
                "q": "wa", "w": "qes", "e": "wrd", "r": "etf",
            }
            char = text[pos].lower()
            if char in keyboard_neighbors:
                replacement = random.choice(keyboard_neighbors[char])
                text = text[:pos] + replacement + text[pos + 1:]

        return text

    def _truncate(self, text: str) -> str:
        """Randomly truncate text"""
        if len(text) < 20:
            return text

        # Truncate to 60-90% of original
        ratio = random.uniform(0.6, 0.9)
        truncate_pos = int(len(text) * ratio)

        # Try to truncate at word boundary
        space_pos = text.rfind(" ", 0, truncate_pos)
        if space_pos > truncate_pos * 0.5:
            truncate_pos = space_pos

        return text[:truncate_pos] + "..."

    def get_curriculum_parameters(self, step: int) -> Dict[str, Any]:
        """Get current curriculum parameters"""
        self._step = step
        scenario = self.scenario_generator.curriculum_sample(step)

        return {
            "difficulty": scenario["difficulty"],
            "step": step,
            "progress": min(step / self.config.curriculum_steps, 1.0),
            "schedule": self.config.curriculum_schedule,
        }

    def reset(self) -> None:
        """Reset the randomizer state"""
        self._step = 0
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
