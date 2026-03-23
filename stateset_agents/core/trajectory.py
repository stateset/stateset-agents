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
from typing import Any


@dataclass(init=False)
class ConversationTurn:
    """
    Represents a single turn in a conversation.

    Compatibility: accepts positional arguments in legacy form
    `(user_message: str, assistant_response: str, reward: float)` or
    the standard keyword form `role=..., content=...`.
    """

    role: str | None = None  # "user", "assistant", "system", "tool"
    content: str | None = None
    # Compatibility fields
    user_message: str | None = None
    assistant_response: str | None = None
    reward: float | None = None

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # For tool calls and responses
    tool_calls: list[dict[str, Any]] | None = None
    tool_results: list[dict[str, Any]] | None = None

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
                valid_roles = {"user", "assistant", "system", "tool"}
                # (A, B, [float]) — treat as legacy (user_message, assistant_response, reward)
                if (
                    len(args) >= 3
                    and isinstance(args[2], (int, float))
                    and not isinstance(args[2], bool)
                ):
                    role_candidate = args[0].lower()
                    if role_candidate in valid_roles:
                        self.role = role_candidate
                        self.content = args[1]
                        self.reward = float(args[2])
                    else:
                        self.user_message = args[0]
                        self.assistant_response = args[1]
                        self.reward = float(args[2])
                        self.role = "assistant"
                        self.content = self.assistant_response
                else:
                    # (role, content, [reward])
                    self.role = (
                        args[0].lower() if args[0].lower() in valid_roles else args[0]
                    )
                    self.content = args[1]
                    if (
                        len(args) >= 3
                        and isinstance(args[2], (int, float))
                        and not isinstance(args[2], bool)
                    ):
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

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
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
    turns: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # For GRPO training
    log_probs: list[float] | None = None
    value_estimates: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        """Create from dictionary"""
        return cls(**data)

    # Compatibility: accept add_turn(dict) used by some tests
    def add_turn(self, turn: dict[str, Any]) -> None:
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
        user_msgs = [
            str(t.get("content", "")) for t in self.turns if t.get("role") == "user"
        ]
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
    turns: list[ConversationTurn] = field(default_factory=list)
    total_reward: float = 0.0
    # Backwards-compat: some tests/configs use `rewards=` instead of `turn_rewards=`.
    rewards: list[float] | None = None
    turn_rewards: list[float] | None = None  # Reward for each turn
    metadata: dict[str, Any] = field(default_factory=dict)

    # Environment state tracking
    initial_state: dict[str, Any] | None = None
    final_state: dict[str, Any] | None = None
    state_history: list[dict[str, Any]] | None = None

    # Training metadata
    episode_length: int = field(init=False)
    conversation_quality_score: float | None = None
    task_completion_score: float | None = None

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
    def conversation_history(self) -> list[dict[str, str]]:
        """Get conversation in standard chat format"""
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]

    @property
    def user_turns(self) -> list[ConversationTurn]:
        """Get only user turns"""
        return [turn for turn in self.turns if turn.role == "user"]

    @property
    def assistant_turns(self) -> list[ConversationTurn]:
        """Get only assistant turns"""
        return [turn for turn in self.turns if turn.role == "assistant"]

    def get_turn_by_id(self, turn_id: str) -> ConversationTurn | None:
        """Get turn by ID"""
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def add_turn(self, turn: ConversationTurn, reward: float | None = None):
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

        self.total_reward = round(
            float(Decimal(str(self.total_reward)) + Decimal(str(reward))), 1
        )

    def get_context_up_to_turn(self, turn_index: int) -> list[dict[str, str]]:
        """Get conversation context up to a specific turn"""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.turns[:turn_index]
        ]

    def compute_cumulative_reward(self) -> list[float]:
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

    def to_dict(self) -> dict[str, Any]:
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

    def to_messages(self) -> list[dict[str, str]]:
        """Return turns in `{role, content}` message format."""
        return [
            {"role": (turn.role or ""), "content": (turn.content or "")}
            for turn in self.turns
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiTurnTrajectory":
        """Create from dictionary"""
        data = data.copy()
        if "turns" in data:
            data["turns"] = [ConversationTurn.from_dict(turn) for turn in data["turns"]]
        data.pop("episode_length", None)
        return cls(**data)

    def save(self, filepath: str):
        """Save trajectory to file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> "MultiTurnTrajectory":
        """Load trajectory from file"""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class TrajectoryGroup:
    """
    Group of trajectories for the same scenario (for GRPO training)
    """

    scenario_id: str = ""
    trajectories: list[Trajectory | MultiTurnTrajectory] = field(
        default_factory=list
    )
    scenario_metadata: dict[str, Any] = field(default_factory=dict)
    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def rewards(self) -> list[float]:
        """Get all rewards in the group"""
        return [
            traj.total_reward if hasattr(traj, "total_reward") else traj.reward
            for traj in self.trajectories
        ]

    def add_trajectory(
        self, trajectory: Trajectory | MultiTurnTrajectory
    ) -> None:
        """Append a trajectory to the group (compatibility helper)."""
        self.trajectories.append(trajectory)

    @property
    def reward_diversity(self) -> float:
        """Calculate reward diversity within the group"""
        import numpy as np

        rewards = self.rewards
        return np.std(rewards) if len(rewards) > 1 else 0.0

    def compute_advantages(self, baseline_method: str = "group_mean") -> list[float]:
        """Compute GRPO advantages for this group"""
        import numpy as np

        rewards_list = self.rewards
        if not rewards_list:
            return []

        rewards = np.array(rewards_list, dtype=float)

        if baseline_method == "group_mean":
            baseline = rewards.mean()
        elif baseline_method == "group_median":
            baseline = np.median(rewards)
        else:
            baseline = rewards.mean()

        advantages = rewards - baseline
        return advantages.tolist()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "scenario_id": self.scenario_id,
            "trajectories": [traj.to_dict() for traj in self.trajectories],
            "scenario_metadata": self.scenario_metadata,
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryGroup":
        """Create a TrajectoryGroup from a serialized dictionary."""
        data = dict(data or {})
        raw_trajectories = data.get("trajectories") or []

        trajectories: list[Trajectory | MultiTurnTrajectory] = []
        for item in raw_trajectories:
            if isinstance(item, (Trajectory, MultiTurnTrajectory)):
                trajectories.append(item)
                continue
            if not isinstance(item, dict):
                continue
            # Multi-turn trajectories are typically keyed by `turns`.
            if "turns" in item:
                trajectories.append(MultiTurnTrajectory.from_dict(item))
            elif "prompt" in item or "response" in item:
                trajectories.append(Trajectory.from_dict(item))
            else:
                trajectories.append(MultiTurnTrajectory.from_dict(item))

        data["trajectories"] = trajectories
        return cls(**data)


# Utility functions
def create_trajectory_from_conversation(
    conversation: list[dict[str, str]],
    total_reward: float,
    turn_rewards: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
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
    conversation: list[dict[str, str]],
    reward: float,
    metadata: dict[str, Any] | None = None,
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
