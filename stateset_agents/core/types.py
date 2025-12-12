"""
Type definitions and protocols for the StateSet Agents framework.

This module provides comprehensive type hints and protocols to improve
type safety across the framework.
"""

from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import torch

# Generic type variables
T = TypeVar("T")
AgentType = TypeVar("AgentType", bound="BaseAgent")
EnvironmentType = TypeVar("EnvironmentType", bound="BaseEnvironment")
RewardType = TypeVar("RewardType", bound="BaseReward")


# Basic data types
JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
ConfigDict = Dict[str, Any]
MetadataDict = Dict[str, Any]


@dataclass
class ModelConfig:
    """Configuration for language models."""

    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    system_prompt: Optional[str] = None
    use_chat_template: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    device_map: Optional[str] = "auto"
    use_peft: bool = False
    peft_config: Optional[ConfigDict] = None


@dataclass
class TrainingConfig:
    """Configuration for training."""

    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "./outputs"
    use_wandb: bool = False
    wandb_project: str = "stateset-agents"
    seed: int = 42


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    user_message: str
    assistant_response: str
    reward: float
    metadata: MetadataDict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Trajectory:
    """A sequence of conversation turns."""

    turns: List[ConversationTurn]
    total_reward: float = 0.0
    metadata: MetadataDict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self._update_total_reward()

    def _update_total_reward(self):
        """Update total reward based on turns."""
        self.total_reward = sum(turn.reward for turn in self.turns)

    def add_turn(self, turn: ConversationTurn):
        """Add a turn to the trajectory."""
        self.turns.append(turn)
        self._update_total_reward()

    @property
    def average_reward(self) -> float:
        """Calculate average reward."""
        return self.total_reward / len(self.turns) if self.turns else 0.0


@dataclass
class RewardResult:
    """Result of reward computation."""

    score: float
    components: Dict[str, float]
    metadata: MetadataDict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Protocol definitions
class BaseAgent(Protocol):
    """Protocol for agent implementations."""

    async def initialize(self) -> None:
        """Initialize the agent."""
        ...

    async def generate_response(
        self,
        messages: Union[str, List[Dict[str, str]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response given messages."""
        ...


class BaseEnvironment(Protocol):
    """Protocol for environment implementations."""

    async def reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        ...

    async def step(self, action: str) -> Tuple[Dict[str, Any], float, bool]:
        """Take a step in the environment."""
        ...


class BaseReward(Protocol):
    """Protocol for reward function implementations."""

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward for given turns."""
        ...


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer implementations."""

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to tokens."""
        ...

    def decode(self, tokens: List[int], **kwargs) -> str:
        """Decode tokens to text."""
        ...

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Apply chat template to messages."""
        ...


class ModelProtocol(Protocol):
    """Protocol for model implementations."""

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate tokens from input."""
        ...


# Callback protocols
class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    async def on_episode_start(self, episode: int) -> None:
        """Called at the start of each episode."""
        ...

    async def on_episode_end(self, episode: int, reward: float) -> None:
        """Called at the end of each episode."""
        ...

    async def on_step(
        self, step: int, observation: Any, action: Any, reward: float
    ) -> None:
        """Called at each step."""
        ...


# Utility types
AgentFactory = Callable[[ModelConfig], BaseAgent]
EnvironmentFactory = Callable[[], BaseEnvironment]
RewardFactory = Callable[[float], BaseReward]

# Async types
AsyncAgentFactory = Callable[[ModelConfig], Awaitable[BaseAgent]]
AsyncEnvironmentFactory = Callable[[], Awaitable[BaseEnvironment]]
AsyncRewardFactory = Callable[[float], Awaitable[BaseReward]]


# Error types
class FrameworkError(Exception):
    """Base exception for framework errors."""

    pass


class AgentError(FrameworkError):
    """Error related to agent operations."""

    pass


class EnvironmentError(FrameworkError):
    """Error related to environment operations."""

    pass


class TrainingError(FrameworkError):
    """Error related to training operations."""

    pass


class ConfigurationError(FrameworkError):
    """Error related to configuration."""

    pass
