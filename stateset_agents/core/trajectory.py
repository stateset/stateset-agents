"""
Trajectory classes for multi-turn agent training

This module defines the data structures for representing agent interactions,
including single turns and complete multi-turn conversations.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union


@dataclass(init=False)
class ConversationTurn:
    """
    Represents a single turn in a conversation.

    Compatibility: accepts positional arguments in legacy form
    `(user_message: str, assistant_response: str, reward: float)` or
    the standard keyword form `role=..., content=...`.
    """

    role: Optional[str] = None  # "user", "assistant", "system", "tool"
    content: Optional[str] = None
    # Compatibility fields
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None
    reward: Optional[float] = None

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # For tool calls and responses
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None

    def __init__(self, *args, **kwargs):
        # Initialize defaults
        self.role = None
        self.content = None
        self.user_message = None
        self.assistant_response = None
        self.reward = None
        self.timestamp = datetime.now()
        self.metadata = {}
        self.turn_id = str(uuid.uuid4())
        self.tool_calls = None
        self.tool_results = None

        if args:
            if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
                # (A, B, [float]) — treat as legacy (user_message, assistant_response, reward)
                if len(args) >= 3 and isinstance(args[2], (int, float)):
                    self.user_message = args[0]
                    self.assistant_response = args[1]
                    self.reward = float(args[2])
                    self.role = "assistant"
                    self.content = self.assistant_response
                else:
                    # (role, content, [reward])
                    self.role = args[0]
                    self.content = args[1]
                    if len(args) >= 3 and isinstance(args[2], (int, float)):
                        self.reward = float(args[2])
            else:
                raise TypeError("Invalid positional arguments for ConversationTurn")

        # Apply keyword overrides
        for key in [
            "role",
            "content",
            "user_message",
            "assistant_response",
            "reward",
            "metadata",
            "tool_calls",
            "tool_results",
            "timestamp",
            "turn_id",
        ]:
            if key in kwargs and kwargs[key] is not None:
                setattr(self, key, kwargs[key])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        data = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "turn_id": self.turn_id,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
        }
        if self.user_message is not None:
            data["user_message"] = self.user_message
        if self.assistant_response is not None:
            data["assistant_response"] = self.assistant_response
        if self.reward is not None:
            data["reward"] = self.reward
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary"""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Trajectory:
    """
    Base trajectory class for single interactions
    """

    prompt: str = ""
    response: str = ""
    reward: float = 0.0
    # Explicit list of per-turn records for compatibility with tests
    turns: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # For GRPO training
    log_probs: Optional[List[float]] = None
    value_estimates: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "turns": self.turns,
            "metadata": self.metadata,
            "trajectory_id": self.trajectory_id,
            "log_probs": self.log_probs,
            "value_estimates": self.value_estimates,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create from dictionary"""
        return cls(**data)

    # Compatibility: accept add_turn(dict) used by some tests
    def add_turn(self, turn: Dict[str, Any]) -> None:
        self.turns.append(turn)

    # Convenience helpers used by some training utilities
    def get_prompt(self) -> str:
        """Return prompt text, deriving from first user turn if needed."""
        if self.prompt:
            return self.prompt
        for t in self.turns:
            if t.get("role") == "user" and t.get("content"):
                return str(t.get("content"))
        # Fallback: concatenate any user messages
        user_msgs = [str(t.get("content", "")) for t in self.turns if t.get("role") == "user"]
        return " ".join(user_msgs)

    def get_last_response(self) -> str:
        """Return last assistant response, falling back to response field."""
        for t in reversed(self.turns):
            if t.get("role") == "assistant" and t.get("content"):
                return str(t.get("content"))
        return self.response



@dataclass
class MultiTurnTrajectory:
    """
    Represents a complete multi-turn conversation trajectory
    """

    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: List[ConversationTurn] = field(default_factory=list)
    total_reward: float = 0.0
    # Backwards-compat: some tests/configs use `rewards=` instead of `turn_rewards=`.
    rewards: Optional[List[float]] = None
    turn_rewards: Optional[List[float]] = None  # Reward for each turn
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Environment state tracking
    initial_state: Optional[Dict[str, Any]] = None
    final_state: Optional[Dict[str, Any]] = None
    state_history: Optional[List[Dict[str, Any]]] = None

    # Training metadata
    episode_length: int = field(init=False)
    conversation_quality_score: Optional[float] = None
    task_completion_score: Optional[float] = None

    def __post_init__(self):
        self.episode_length = len(self.turns)
        if self.turn_rewards is None and self.rewards is not None:
            self.turn_rewards = list(self.rewards)

        if self.turn_rewards is None:
            # Distribute total reward evenly if no per-turn rewards
            self.turn_rewards = [
                self.total_reward / max(1, self.episode_length)
            ] * self.episode_length

        # Keep aliases in sync (tests access `traj.rewards`).
        self.turn_rewards = [float(r) for r in (self.turn_rewards or [])]
        self.rewards = self.turn_rewards

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation in standard chat format"""
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]

    @property
    def user_turns(self) -> List[ConversationTurn]:
        """Get only user turns"""
        return [turn for turn in self.turns if turn.role == "user"]

    @property
    def assistant_turns(self) -> List[ConversationTurn]:
        """Get only assistant turns"""
        return [turn for turn in self.turns if turn.role == "assistant"]

    def get_turn_by_id(self, turn_id: str) -> Optional[ConversationTurn]:
        """Get turn by ID"""
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def add_turn(self, turn: ConversationTurn, reward: Optional[float] = None):
        """Add a new turn to the trajectory"""
        self.turns.append(turn)
        self.episode_length = len(self.turns)
        # Determine reward for this turn
        if reward is None:
            reward = getattr(turn, "reward", None) or 0.0
        if self.turn_rewards is None:
            self.turn_rewards = []
        self.turn_rewards.append(float(reward))
        self.rewards = self.turn_rewards
        from decimal import Decimal

        self.total_reward = round(
            float(Decimal(str(self.total_reward)) + Decimal(str(reward))), 1
        )

    def get_context_up_to_turn(self, turn_index: int) -> List[Dict[str, str]]:
        """Get conversation context up to a specific turn"""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.turns[:turn_index]
        ]

    def compute_cumulative_reward(self) -> List[float]:
        """Compute cumulative rewards for each turn"""
        if not self.turn_rewards:
            return []

        cumulative = []
        total = 0
        for reward in self.turn_rewards:
            total += reward
            cumulative.append(total)
        return cumulative

    @property
    def average_reward(self) -> float:
        """Average reward per turn (0.0 if no turns)."""
        if self.episode_length == 0:
            return 0.0
        return float(self.total_reward) / float(self.episode_length)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "total_reward": self.total_reward,
            "rewards": self.rewards,
            "turn_rewards": self.turn_rewards,
            "metadata": self.metadata,
            "trajectory_id": self.trajectory_id,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "state_history": self.state_history,
            "episode_length": self.episode_length,
            "conversation_quality_score": self.conversation_quality_score,
            "task_completion_score": self.task_completion_score,
        }

    def to_messages(self) -> List[Dict[str, str]]:
        """Return turns in `{role, content}` message format."""
        return [
            {"role": (turn.role or ""), "content": (turn.content or "")}
            for turn in self.turns
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiTurnTrajectory":
        """Create from dictionary"""
        data = data.copy()
        if "turns" in data:
            data["turns"] = [ConversationTurn.from_dict(turn) for turn in data["turns"]]
        return cls(**data)

    def save(self, filepath: str):
        """Save trajectory to file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> "MultiTurnTrajectory":
        """Load trajectory from file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class TrajectoryGroup:
    """
    Group of trajectories for the same scenario (for GRPO training)
    """

    scenario_id: str = ""
    trajectories: List[Union[Trajectory, MultiTurnTrajectory]] = field(default_factory=list)
    scenario_metadata: Dict[str, Any] = field(default_factory=dict)
    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def rewards(self) -> List[float]:
        """Get all rewards in the group"""
        return [
            traj.total_reward if hasattr(traj, "total_reward") else traj.reward
            for traj in self.trajectories
        ]

    def add_trajectory(self, trajectory: Union[Trajectory, MultiTurnTrajectory]) -> None:
        """Append a trajectory to the group (compatibility helper)."""
        self.trajectories.append(trajectory)

    @property
    def reward_diversity(self) -> float:
        """Calculate reward diversity within the group"""
        import numpy as np

        rewards = self.rewards
        return np.std(rewards) if len(rewards) > 1 else 0.0

    def compute_advantages(self, baseline_method: str = "group_mean") -> List[float]:
        """Compute GRPO advantages for this group"""
        import numpy as np

        rewards = np.array(self.rewards)

        if baseline_method == "group_mean":
            baseline = rewards.mean()
        elif baseline_method == "group_median":
            baseline = np.median(rewards)
        else:
            baseline = rewards.mean()

        advantages = rewards - baseline
        return advantages.tolist()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "scenario_id": self.scenario_id,
            "trajectories": [traj.to_dict() for traj in self.trajectories],
            "scenario_metadata": self.scenario_metadata,
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryGroup":
        """Create from dictionary"""
        data = data.copy()
        if "trajectories" in data:
            trajectories = []
            for traj_data in data["trajectories"]:
                if "turns" in traj_data:
                    trajectories.append(MultiTurnTrajectory.from_dict(traj_data))
                else:
                    trajectories.append(Trajectory.from_dict(traj_data))
            data["trajectories"] = trajectories
        return cls(**data)


# Utility functions
def create_trajectory_from_conversation(
    conversation: List[Dict[str, str]],
    total_reward: float,
    turn_rewards: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MultiTurnTrajectory:
    """
    Create a MultiTurnTrajectory from a standard conversation format
    """
    turns = [
        ConversationTurn(
            role=msg["role"], content=msg["content"], metadata=msg.get("metadata", {})
        )
        for msg in conversation
    ]

    return MultiTurnTrajectory(
        turns=turns,
        total_reward=total_reward,
        turn_rewards=turn_rewards,
        metadata=metadata or {},
    )


def conversation_to_single_trajectory(
    conversation: List[Dict[str, str]],
    reward: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Trajectory:
    """
    Convert a conversation to a single trajectory for backwards compatibility
    """
    # Combine all user messages as prompt, all assistant messages as response
    user_messages = [msg["content"] for msg in conversation if msg["role"] == "user"]
    assistant_messages = [
        msg["content"] for msg in conversation if msg["role"] == "assistant"
    ]

    prompt = " ".join(user_messages)
    response = " ".join(assistant_messages)

    return Trajectory(
        prompt=prompt, response=response, reward=reward, metadata=metadata or {}
    )
