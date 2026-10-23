"""
Conversation Dataset Utilities for Offline RL

Provides D4RL-style dataset loading, replay buffers, and embedding caching
for training offline RL algorithms on conversation data.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import random
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from stateset_agents.data.identity import (
    first_user_text,
    normalize_identity_text,
    prompt_identity,
)

try:
    import torch as _torch

    torch: Any = _torch
except ImportError:
    torch = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import Dataset as TorchDataset
else:
    try:
        from torch.utils.data import Dataset as TorchDataset
    except ImportError:
        TorchDataset = object


def _load_sentence_transformer_cls() -> Any:
    """Import ``SentenceTransformer`` lazily.

    The import costs over a second (it pulls in torch, datasets and sklearn),
    so it must not happen at module import time -- this module is reachable
    from the CLI import chain.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer


logger = logging.getLogger(__name__)

HF_PARSE_EXCEPTIONS = (AttributeError, KeyError, TypeError, ValueError)

#: ``TrajectoryData.metadata`` marker for a trajectory whose only reward is
#: its ``total_reward``, placed on the final turn (no per-turn rewards given).
REWARD_PLACEMENT_KEY = "reward_placement"
TERMINAL_REWARD = "terminal"
#: Seed used by :meth:`ConversationDataset.split` when none is given.
DEFAULT_SPLIT_SEED = 0


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for conversation datasets. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class ConversationDatasetConfig:
    """Configuration for conversation dataset loading.

    ``min_trajectory_length`` / ``max_trajectory_length`` (turns, inclusive)
    are enforced when a :class:`ConversationDataset` is built: trajectories
    outside the range are dropped and counted in a warning. ``max_turns`` is
    deprecated and ignored (it will be removed in v0.71.0); use
    ``max_trajectory_length``.
    """

    data_path: str = ""
    format: str = "jsonl"  # jsonl, json, parquet, hf_dataset
    max_turns: int = 20
    embedding_model: str = "all-MiniLM-L6-v2"
    normalize_rewards: bool = True
    reward_normalization_method: str = "standard"  # standard, minmax, percentile
    compute_returns: bool = True
    discount_factor: float = 0.99
    min_trajectory_length: int = 1
    max_trajectory_length: int | None = None

    def __post_init__(self) -> None:
        if self.max_turns != 20:
            warnings.warn(
                "ConversationDatasetConfig.max_turns is deprecated and ignored; "
                "use max_trajectory_length to drop longer trajectories. The "
                "field will be removed in v0.71.0.",
                DeprecationWarning,
                stacklevel=3,
            )


@dataclass
class ConversationTurnData:
    """Lightweight turn representation for datasets"""

    role: str
    content: str
    reward: float = 0.0
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryData:
    """Single conversation trajectory for offline RL"""

    trajectory_id: str
    turns: list[ConversationTurnData]
    total_reward: float
    turn_rewards: list[float]
    returns_to_go: list[float] | None = None
    is_simulated: bool = False
    domain_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_returns_to_go(self, gamma: float = 0.99) -> list[float]:
        """Compute returns-to-go for Decision Transformer"""
        returns = []
        running_return = 0.0
        for reward in reversed(self.turn_rewards):
            running_return = reward + gamma * running_return
            returns.append(running_return)
        self.returns_to_go = list(reversed(returns))
        return self.returns_to_go


class ConversationDataset(TorchDataset):
    """
    D4RL-style dataset for conversation trajectories.

    Supports loading from various formats and provides utilities for:
    - Batch sampling for offline RL training
    - Reward normalization
    - Quality filtering
    - Returns computation for Decision Transformer

    Example:
        >>> dataset = ConversationDataset.from_jsonl("conversations.jsonl")
        >>> batch = dataset.get_batch(batch_size=32)
        >>> high_quality = dataset.filter_by_quality(min_reward=0.7)
    """

    def __init__(
        self,
        trajectories: list[TrajectoryData],
        config: ConversationDatasetConfig | None = None,
        embedding_cache: Optional["EmbeddingCache"] = None,
    ):
        self.config = config or ConversationDatasetConfig()
        _require_finite_rewards(trajectories)
        self.trajectories = self._within_length_limits(trajectories)
        self.embedding_cache = embedding_cache
        self._embedding_dim: int | None = None
        self._statistics: dict[str, float] | None = None

        if self.config.normalize_rewards:
            self.normalize_rewards(self.config.reward_normalization_method)

        if self.config.compute_returns:
            self.compute_all_returns(self.config.discount_factor)

    def _within_length_limits(
        self, trajectories: list[TrajectoryData]
    ) -> list[TrajectoryData]:
        """Drop trajectories outside the configured turn-count range."""
        low = self.config.min_trajectory_length
        high = self.config.max_trajectory_length
        kept = [
            t
            for t in trajectories
            if len(t.turns) >= low and (high is None or len(t.turns) <= high)
        ]
        if len(kept) == len(trajectories):
            return trajectories
        logger.warning(
            "Dropped %d of %d trajectories outside the configured length range "
            "[%d, %s] turns.",
            len(trajectories) - len(kept),
            len(trajectories),
            low,
            "inf" if high is None else high,
        )
        return kept

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> TrajectoryData:
        return self.trajectories[idx]

    def __iter__(self) -> Iterator[TrajectoryData]:
        return iter(self.trajectories)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        config: ConversationDatasetConfig | None = None,
        **kwargs,
    ) -> "ConversationDataset":
        """
        Load dataset from JSONL file.

        Expected format per line:
        {
            "trajectory_id": "...",
            "turns": [{"role": "user", "content": "...", "reward": 0.0}, ...],
            "total_reward": 0.8,
            "metadata": {...}
        }

        Per-turn ``reward`` values are optional. When no turn carries a
        nonzero reward, ``total_reward`` is the trajectory's only signal and
        is placed on the final turn (as for Hugging Face rows), so batches,
        returns-to-go and policy transitions all see it.
        """
        config = config or ConversationDatasetConfig(data_path=path, format="jsonl")
        trajectories = []

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    trajectory = cls._parse_trajectory(data, line_num)
                    if trajectory:
                        trajectories.append(trajectory)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(trajectories)} trajectories from {path}")
        return cls(trajectories, config, **kwargs)

    @classmethod
    def from_json(
        cls,
        path: str,
        config: ConversationDatasetConfig | None = None,
        **kwargs,
    ) -> "ConversationDataset":
        """Load dataset from JSON file (list of trajectories)"""
        config = config or ConversationDatasetConfig(data_path=path, format="json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "trajectories" in data:
            data = data["trajectories"]

        trajectories = []
        for idx, item in enumerate(data):
            trajectory = cls._parse_trajectory(item, idx)
            if trajectory:
                trajectories.append(trajectory)

        logger.info(f"Loaded {len(trajectories)} trajectories from {path}")
        return cls(trajectories, config, **kwargs)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str = "train",
        config: ConversationDatasetConfig | None = None,
        conversation_column: str = "conversations",
        reward_column: str = "reward",
        **kwargs,
    ) -> "ConversationDataset":
        """
        Load dataset from HuggingFace Hub.

        Supports common conversation dataset formats like:
        - anthropic/hh-rlhf
        - OpenAssistant/oasst1
        - Custom conversation datasets
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HuggingFace datasets required. Install: pip install datasets"
            ) from exc

        config = config or ConversationDatasetConfig(
            data_path=dataset_name, format="hf_dataset"
        )

        # dataset_name is a caller-supplied config value (public HF dataset
        # repo id), not attacker-controlled input; pinning a fixed revision
        # would break support for arbitrary user-chosen datasets.
        hf_dataset = load_dataset(dataset_name, split=split)  # nosec: B615
        trajectories = []

        for idx, item in enumerate(hf_dataset):
            trajectory = cls._parse_hf_item(
                item, idx, conversation_column, reward_column
            )
            if trajectory:
                trajectories.append(trajectory)

        logger.info(
            f"Loaded {len(trajectories)} trajectories from HuggingFace: {dataset_name}"
        )
        return cls(trajectories, config, **kwargs)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: ConversationDatasetConfig | None = None,
        **kwargs,
    ) -> "ConversationDataset":
        """Create dataset from dictionary (useful for testing)"""
        config = config or ConversationDatasetConfig()
        trajectories = []

        for idx, item in enumerate(data.get("trajectories", [])):
            trajectory = cls._parse_trajectory(item, idx)
            if trajectory:
                trajectories.append(trajectory)

        return cls(trajectories, config, **kwargs)

    @staticmethod
    def _parse_trajectory(data: dict[str, Any], idx: int) -> TrajectoryData | None:
        """Parse a single trajectory from dictionary.

        Without per-turn rewards (none given, or all zero) ``total_reward`` is
        placed on the final turn and the trajectory is marked as carrying a
        terminal reward; otherwise the per-turn rewards are kept as given.
        """
        try:
            turns = []
            turn_rewards = []

            for turn_data in data.get("turns", []):
                turn = ConversationTurnData(
                    role=_canonical_role(
                        turn_data.get("role", turn_data.get("from", "user"))
                    ),
                    content=turn_data.get("content", ""),
                    reward=float(turn_data.get("reward", 0.0)),
                    timestamp=turn_data.get("timestamp"),
                    metadata=turn_data.get("metadata", {}),
                )
                turns.append(turn)
                turn_rewards.append(turn.reward)

            if not turns:
                return None

            total_reward = float(data.get("total_reward", sum(turn_rewards)))
            metadata = dict(data.get("metadata") or {})
            if not any(turn_rewards):
                turn_rewards = [0.0] * (len(turns) - 1) + [total_reward]
                turns[-1].reward = total_reward
                metadata[REWARD_PLACEMENT_KEY] = TERMINAL_REWARD

            return TrajectoryData(
                trajectory_id=data.get("trajectory_id", f"traj_{idx}"),
                turns=turns,
                total_reward=total_reward,
                turn_rewards=turn_rewards,
                is_simulated=data.get("is_simulated", False),
                domain_label=data.get("domain_label"),
                metadata=metadata,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse trajectory {idx}: {e}")
            return None

    @staticmethod
    def _parse_hf_item(
        item: dict[str, Any],
        idx: int,
        conversation_column: str,
        reward_column: str,
    ) -> TrajectoryData | None:
        """Parse HuggingFace dataset item"""
        try:
            turns = []
            conversations = item.get(conversation_column, [])

            # Handle different HF formats
            if isinstance(conversations, str):
                # Single string format (e.g., chosen/rejected)
                turns.append(
                    ConversationTurnData(role="assistant", content=conversations)
                )
            elif isinstance(conversations, list):
                for conv in conversations:
                    if isinstance(conv, dict):
                        turns.append(
                            ConversationTurnData(
                                role=_canonical_role(
                                    conv.get("role", conv.get("from", "user"))
                                ),
                                content=str(
                                    conv.get(
                                        "content",
                                        conv.get("value", conv.get("text", "")),
                                    )
                                ),
                            )
                        )
                    elif isinstance(conv, str):
                        turns.append(ConversationTurnData(role="user", content=conv))

            if not turns:
                return None

            reward = float(item.get(reward_column, 0.0))
            turn_rewards = [0.0] * (len(turns) - 1) + [reward]
            turns[-1].reward = reward

            return TrajectoryData(
                trajectory_id=f"hf_{idx}",
                turns=turns,
                total_reward=reward,
                turn_rewards=turn_rewards,
                metadata={
                    "source": "huggingface",
                    REWARD_PLACEMENT_KEY: TERMINAL_REWARD,
                },
            )
        except HF_PARSE_EXCEPTIONS as e:
            logger.warning(f"Failed to parse HF item {idx}: {e}")
            return None

    def get_batch(
        self,
        batch_size: int,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        Sample a batch of trajectories for training.

        Returns:
            Dictionary with keys:
            - trajectories: List[TrajectoryData]
            - states: Optional[torch.Tensor] - conversation state embeddings
            - actions: Optional[torch.Tensor] - response embeddings
            - rewards: torch.Tensor
            - returns_to_go: Optional[torch.Tensor]
            - dones: torch.Tensor
        """
        _require_torch()

        indices = random.sample(
            range(len(self.trajectories)), min(batch_size, len(self.trajectories))
        )
        batch_trajectories = [self.trajectories[i] for i in indices]

        rewards = []
        dones = []
        returns_to_go = []

        for traj in batch_trajectories:
            rewards.extend(traj.turn_rewards)
            dones.extend([False] * (len(traj.turns) - 1) + [True])
            if traj.returns_to_go:
                returns_to_go.extend(traj.returns_to_go)

        result = {
            "trajectories": batch_trajectories,
            "rewards": torch.FloatTensor(rewards),
            "dones": torch.FloatTensor(dones),
        }

        if returns_to_go:
            result["returns_to_go"] = torch.FloatTensor(returns_to_go)

        if include_embeddings and self.embedding_cache:
            states, actions = self._get_embeddings(batch_trajectories)
            result["states"] = states
            result["actions"] = actions

        return result

    def _get_embeddings(self, trajectories: list[TrajectoryData]) -> tuple[Any, Any]:
        """Get state and action embeddings for trajectories"""
        if not self.embedding_cache:
            raise ValueError("Embedding cache required for embeddings")

        states = []
        actions = []

        for traj in trajectories:
            # State is the conversation history up to each point
            history = ""
            for i, turn in enumerate(traj.turns):
                if turn.role in ("user", "system"):
                    history += f"{turn.role}: {turn.content}\n"
                    state_embedding = self.embedding_cache.get_or_compute(
                        f"{traj.trajectory_id}_state_{i}", history
                    )
                    states.append(state_embedding)
                elif turn.role == "assistant":
                    action_embedding = self.embedding_cache.get_or_compute(
                        f"{traj.trajectory_id}_action_{i}", turn.content
                    )
                    actions.append(action_embedding)
                    history += f"assistant: {turn.content}\n"

        # Empty batches keep the original empty lists instead of a tensor.
        state_batch: Any = torch.stack(states) if states else states
        action_batch: Any = torch.stack(actions) if actions else actions
        return state_batch, action_batch

    def get_statistics(self) -> dict[str, float]:
        """Compute dataset statistics"""
        if self._statistics is not None:
            return self._statistics

        rewards = [t.total_reward for t in self.trajectories]
        lengths = [len(t.turns) for t in self.trajectories]
        turn_rewards = []
        for t in self.trajectories:
            turn_rewards.extend(t.turn_rewards)

        self._statistics = {
            "num_trajectories": len(self.trajectories),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "std_length": float(np.std(lengths)) if lengths else 0.0,
            "total_turns": sum(lengths),
            "mean_turn_reward": float(np.mean(turn_rewards)) if turn_rewards else 0.0,
            "std_turn_reward": float(np.std(turn_rewards)) if turn_rewards else 0.0,
        }

        return self._statistics

    def normalize_rewards(self, method: str = "standard") -> None:
        """
        Normalize rewards across the dataset.

        Args:
            method: Normalization method
                - "standard": (r - mean) / std
                - "minmax": (r - min) / (max - min)
                - "percentile": clip to 5-95 percentile, then standard

        Only reward-bearing positions are normalized: every turn of a
        trajectory with per-turn rewards, but just the final turn of one
        whose sole reward is its terminal ``total_reward`` (its padding
        zeros stay zero, so a longer conversation is not penalized for its
        length). ``total_reward`` becomes the sum of the normalized rewards.
        """
        all_rewards: list[float] = []
        for t in self.trajectories:
            all_rewards.extend(t.turn_rewards[i] for i in _reward_positions(t))

        if not all_rewards:
            return

        reward_array = np.array(all_rewards)

        if method == "standard":
            mean = np.mean(reward_array)
            std = np.std(reward_array) + 1e-8

            def normalize(r: float) -> float:
                return float((r - mean) / std)

        elif method == "minmax":
            min_r: float = float(np.min(reward_array))
            max_r: float = float(np.max(reward_array))
            range_r = max_r - min_r + 1e-8

            def normalize(r: float) -> float:
                return float((r - min_r) / range_r)

        elif method == "percentile":
            p5 = np.percentile(reward_array, 5)
            p95 = np.percentile(reward_array, 95)
            clipped = np.clip(reward_array, p5, p95)
            mean = np.mean(clipped)
            std = np.std(clipped) + 1e-8

            def normalize(r: float) -> float:
                return float((np.clip(r, p5, p95) - mean) / std)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Apply normalization (raw totals stay available for raw-scale filters)
        for traj in self.trajectories:
            traj.metadata.setdefault("raw_total_reward", float(traj.total_reward))
            positions = _reward_positions(traj)
            traj.turn_rewards = [
                float(normalize(r)) if i in positions else float(r)
                for i, r in enumerate(traj.turn_rewards)
            ]
            traj.total_reward = sum(traj.turn_rewards)

        # Clear cached statistics
        self._statistics = None
        logger.info(f"Normalized rewards using {method} method")

    def compute_all_returns(self, gamma: float = 0.99) -> None:
        """Compute returns-to-go for all trajectories"""
        for traj in self.trajectories:
            traj.compute_returns_to_go(gamma)

    def filter_by_quality(self, min_reward: float) -> "ConversationDataset":
        """
        Filter trajectories by minimum total reward.

        Returns:
            New ConversationDataset with filtered trajectories
        """
        filtered = [
            t
            for t in self.trajectories
            if t.metadata.get("raw_total_reward", t.total_reward) >= min_reward
        ]
        logger.info(
            f"Filtered from {len(self.trajectories)} to {len(filtered)} "
            f"trajectories (min_reward={min_reward})"
        )
        return self._derive(filtered)

    def _derive(self, trajectories: list[TrajectoryData]) -> "ConversationDataset":
        """A subset sharing this dataset's already-processed trajectories.

        Normalization and returns are not re-applied: re-running them mutated
        the shared objects (changing the parent) and rescaled each subset by
        its own statistics.
        """
        config = dataclasses.replace(
            self.config, normalize_rewards=False, compute_returns=False
        )
        derived = ConversationDataset(trajectories, config, self.embedding_cache)
        derived.config = self.config
        return derived

    def filter_by_length(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> "ConversationDataset":
        """Filter trajectories by turn count"""
        filtered = self.trajectories

        if min_length is not None:
            filtered = [t for t in filtered if len(t.turns) >= min_length]

        if max_length is not None:
            filtered = [t for t in filtered if len(t.turns) <= max_length]

        return self._derive(filtered)

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> tuple["ConversationDataset", "ConversationDataset"]:
        """Split dataset into train and validation sets without leakage.

        Trajectories with the same normalized content (roles and NFKC-,
        case- and whitespace-normalized text), the same first user request
        (so rollouts of one prompt stay together), or a shared ``group_id`` /
        ``source_id`` in their metadata are linked transitively and always
        land in the same split; ``train_ratio`` applies to these groups. The
        request identity is :func:`stateset_agents.data.identity.prompt_identity`,
        the one ``split_finetuning_data`` and the eval ladder use.
        With ``shuffle`` the groups are ordered by a hash of ``seed`` and
        their content, so the split is reproducible and independent of
        input order; ``seed=None`` means :data:`DEFAULT_SPLIT_SEED`. Without
        ``shuffle`` groups keep their first-appearance order. When there are
        at least two groups and ``0 < train_ratio < 1``, both splits are
        non-empty.
        """
        groups = _linked_groups(self.trajectories)
        if shuffle:
            effective_seed = DEFAULT_SPLIT_SEED if seed is None else seed
            groups.sort(
                key=lambda group: hashlib.sha256(
                    f"{effective_seed}:{_group_digest(group)}".encode()
                ).hexdigest()
            )

        split_idx = int(len(groups) * train_ratio)
        if 0 < train_ratio < 1:
            if len(groups) >= 2:
                split_idx = min(len(groups) - 1, max(1, split_idx))
            else:
                split_idx = round(train_ratio)
        if len(groups) == 1 and len(self.trajectories) > 1:
            logger.warning(
                "All %d trajectories share content or a group/source id; they "
                "cannot be split without leakage and stay in one split.",
                len(self.trajectories),
            )
        train_trajectories = [t for group in groups[:split_idx] for t in group]
        val_trajectories = [t for group in groups[split_idx:] for t in group]

        return (
            self._derive(train_trajectories),
            self._derive(val_trajectories),
        )

    def policy_transitions(self) -> list[dict[str, Any]]:
        """Return ordered logged actions and their complete next observations.

        User/tool observations between assistant actions belong to the next state.
        Terminal rewards after the last assistant action stay on that transition.
        """
        transitions = []
        for trajectory in self.trajectories:
            messages = [
                {"role": turn.role, "content": turn.content}
                for turn in trajectory.turns
            ]
            positions = [
                i for i, turn in enumerate(trajectory.turns) if turn.role == "assistant"
            ]
            for offset, index in enumerate(positions):
                if index == 0:
                    raise ValueError(
                        "Offline assistant actions require a preceding observation"
                    )
                end = (
                    positions[offset + 1]
                    if offset + 1 < len(positions)
                    else len(messages)
                )
                transitions.append(
                    {
                        "transition_id": f"{trajectory.trajectory_id}:{index}",
                        "messages": messages[: index + 1],
                        "history": messages[:index],
                        "action": messages[index]["content"],
                        "next_history": messages[:end],
                        "reward": float(sum(trajectory.turn_rewards[index:end])),
                        "done": offset == len(positions) - 1,
                    }
                )
        if not transitions:
            raise ValueError(
                "Offline dataset requires at least one logged assistant action"
            )
        return transitions

    def to_policy_training_format(
        self, tokenizer: Any, max_length: int
    ) -> list[dict[str, Any]]:
        """Tokenize logged actions with only the current assistant action supervised.

        Overlong examples are rejected rather than truncating the action silently.
        Feature and token rows share the exact ``policy_transitions`` ordering.
        """
        from stateset_agents.core.chat_masking import last_assistant_token_mask

        if type(max_length) is not int or max_length < 2:
            raise ValueError("max_length must be an integer of at least two")
        rows = []
        for transition in self.policy_transitions():
            encoded = last_assistant_token_mask(tokenizer, transition["messages"])
            if encoded is None:
                raise ValueError(
                    "Tokenizer cannot faithfully mask logged assistant actions"
                )
            ids, mask = encoded
            if len(ids) > max_length or len(ids) < 2 or not any(mask[1:]):
                raise ValueError("Offline action is empty or exceeds max_length")
            rows.append(
                {
                    "transition_id": transition["transition_id"],
                    "input_ids": ids,
                    "labels": [
                        token if selected else -100
                        for token, selected in zip(ids, mask, strict=True)
                    ],
                }
            )
        return rows

    def to_offline_rl_format(self) -> dict[str, Any]:
        """Embed the real logged transitions, in policy-training row order."""
        _require_torch()
        embedding_cache = self.embedding_cache
        if embedding_cache is None:
            raise ValueError(
                "Embedding cache required for offline RL format. Initialize with EmbeddingCache."
            )

        def render(messages: list[dict[str, str]]) -> str:
            return "".join(
                f"{message['role']}: {message['content']}\n" for message in messages
            )

        states, actions, rewards, next_states, dones = [], [], [], [], []
        contexts, responses = [], []

        def embed(identity: str, suffix: str, text: str) -> np.ndarray:
            assert embedding_cache is not None
            digest = hashlib.sha256(text.encode()).hexdigest()
            value = embedding_cache.get_or_compute(
                f"{identity}_{suffix}_{digest}", text
            )
            array: np.ndarray = np.asarray(value.detach().cpu().numpy())
            return array

        for row in self.policy_transitions():
            identity = row["transition_id"]
            states.append(embed(identity, "state", render(row["history"])))
            actions.append(embed(identity, "action", row["action"]))
            next_states.append(embed(identity, "next", render(row["next_history"])))
            rewards.append(row["reward"])
            dones.append(float(row["done"]))
            contexts.append(row["history"])
            responses.append(row["action"])
        return {
            "states": np.asarray(states),
            "actions": np.asarray(actions),
            "rewards": np.asarray(rewards),
            "next_states": np.asarray(next_states),
            "dones": np.asarray(dones),
            "contexts": contexts,
            "responses": responses,
        }


def _require_finite_rewards(trajectories: list[TrajectoryData]) -> None:
    """Reject NaN or infinite rewards before they reach normalization.

    ``json.loads`` accepts ``NaN``/``Infinity``, and one such reward made the
    dataset's mean and standard deviation — and so every normalized reward —
    NaN, while ``filter_by_quality`` silently dropped the trajectory.
    """
    bad = [
        t.trajectory_id
        for t in trajectories
        if not math.isfinite(float(t.total_reward))
        or not all(math.isfinite(float(r)) for r in t.turn_rewards)
    ]
    if bad:
        shown = ", ".join(map(str, bad[:5]))
        raise ValueError(
            f"{len(bad)} trajectory(ies) carry a non-finite reward ({shown}); "
            "rewards must be finite numbers"
        )


def _reward_positions(trajectory: TrajectoryData) -> set[int]:
    """Turn indices that carry reward signal (see ``normalize_rewards``)."""
    if trajectory.metadata.get(REWARD_PLACEMENT_KEY) == TERMINAL_REWARD:
        return {len(trajectory.turn_rewards) - 1} if trajectory.turn_rewards else set()
    return set(range(len(trajectory.turn_rewards)))


def _normalized_text(text: str) -> str:
    return normalize_identity_text(text)


def _content_digest(trajectory: TrajectoryData) -> str:
    content = [(turn.role, _normalized_text(turn.content)) for turn in trajectory.turns]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def _identity_keys(trajectory: TrajectoryData) -> set[str]:
    keys = {"content:" + _content_digest(trajectory)}
    # Rollouts of one prompt (a GRPO group) differ in every reply but share
    # the request: they must land in the same split.
    request = first_user_text(
        [{"role": turn.role, "content": turn.content} for turn in trajectory.turns]
    )
    if request is not None:
        keys.add("prompt:" + prompt_identity(request))
    for field_name in ("group_id", "source_id"):
        value = trajectory.metadata.get(field_name)
        if isinstance(value, str) and value.strip():
            keys.add(f"{field_name}:{_normalized_text(value)}")
    return keys


def _linked_groups(trajectories: list[TrajectoryData]) -> list[list[TrajectoryData]]:
    """Union trajectories that share content or a group/source id."""
    parents = list(range(len(trajectories)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[str, int] = {}
    for index, trajectory in enumerate(trajectories):
        for key in sorted(_identity_keys(trajectory)):
            if key in owners:
                parents[root(index)] = root(owners[key])
            else:
                owners[key] = index
    groups: dict[int, list[TrajectoryData]] = {}
    for index, trajectory in enumerate(trajectories):
        groups.setdefault(root(index), []).append(trajectory)
    return list(groups.values())


def _group_digest(group: list[TrajectoryData]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(_content_digest(t) for t in group)).encode()
    ).hexdigest()


_ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
}


def _canonical_role(role: Any) -> str:
    """Map ShareGPT-style speaker names onto user/assistant/system roles."""
    name = str(role)
    return _ROLE_ALIASES.get(name.lower(), name)


class ConversationReplayBuffer:
    """
    Experience replay buffer optimized for conversation trajectories.

    Supports:
    - Uniform and prioritized sampling
    - Trajectory-level and turn-level sampling
    - Online addition during training
    """

    def __init__(
        self,
        capacity: int = 100000,
        priority: bool = False,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
    ):
        self.capacity = capacity
        self.priority = priority
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta

        self.trajectories: list[TrajectoryData] = []
        self.priorities: list[float] = []
        self._total_turns = 0

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def total_turns(self) -> int:
        return self._total_turns

    def add_trajectory(self, trajectory: TrajectoryData) -> None:
        """Add a trajectory to the buffer"""
        if len(self.trajectories) >= self.capacity:
            # Remove oldest trajectory
            removed = self.trajectories.pop(0)
            self._total_turns -= len(removed.turns)
            if self.priority:
                self.priorities.pop(0)

        self.trajectories.append(trajectory)
        self._total_turns += len(trajectory.turns)

        if self.priority:
            # New trajectories get max priority
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(max_priority)

    def add_trajectories(self, trajectories: list[TrajectoryData]) -> None:
        """Add multiple trajectories"""
        for traj in trajectories:
            self.add_trajectory(traj)

    def sample(
        self,
        batch_size: int,
        return_indices: bool = False,
    ) -> list[TrajectoryData] | tuple[list[TrajectoryData], list[int]]:
        """
        Sample trajectories from buffer.

        Args:
            batch_size: Number of trajectories to sample
            return_indices: Whether to return indices for priority updates

        Returns:
            List of trajectories (and optionally indices)
        """
        if len(self.trajectories) == 0:
            return ([], []) if return_indices else []

        batch_size = min(batch_size, len(self.trajectories))

        if self.priority:
            # Prioritized sampling
            priorities = np.array(self.priorities) ** self.priority_alpha
            probs = priorities / priorities.sum()
            indices = np.random.choice(
                len(self.trajectories), size=batch_size, replace=False, p=probs
            )
        else:
            # Uniform sampling
            indices = np.random.choice(
                len(self.trajectories), size=batch_size, replace=False
            )

        trajectories = [self.trajectories[i] for i in indices]

        if return_indices:
            return trajectories, list(indices)
        return trajectories

    def sample_turns(
        self,
        batch_size: int,
    ) -> list[tuple[TrajectoryData, int]]:
        """
        Sample individual turns uniformly across all trajectories.

        Returns:
            List of (trajectory, turn_index) tuples
        """
        if self._total_turns == 0:
            return []

        # Build index mapping
        turn_to_traj = []
        for traj_idx, traj in enumerate(self.trajectories):
            for turn_idx in range(len(traj.turns)):
                turn_to_traj.append((traj_idx, turn_idx))

        # Sample turns
        batch_size = min(batch_size, len(turn_to_traj))
        indices = np.random.choice(len(turn_to_traj), size=batch_size, replace=False)

        return [
            (self.trajectories[turn_to_traj[i][0]], turn_to_traj[i][1]) for i in indices
        ]

    def update_priorities(
        self,
        indices: list[int],
        priorities: list[float],
    ) -> None:
        """Update priorities for prioritized replay"""
        if not self.priority:
            return

        for idx, priority in zip(indices, priorities, strict=False):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6  # Avoid zero priority

    def clear(self) -> None:
        """Clear the buffer"""
        self.trajectories.clear()
        self.priorities.clear()
        self._total_turns = 0

    def get_statistics(self) -> dict[str, float]:
        """Get buffer statistics"""
        if not self.trajectories:
            return {"num_trajectories": 0, "total_turns": 0}

        rewards = [t.total_reward for t in self.trajectories]
        return {
            "num_trajectories": len(self.trajectories),
            "total_turns": self._total_turns,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "buffer_utilization": len(self.trajectories) / self.capacity,
        }


class EmbeddingCache:
    """
    Cache for conversation embeddings.

    Uses sentence transformers to embed conversation turns and
    caches results to avoid recomputation during training.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_size: int = 100000,
        device: str | None = None,
    ):
        if _load_sentence_transformer_cls() is None:
            raise ImportError(
                "sentence-transformers required for embedding cache. "
                "Install: pip install sentence-transformers"
            )

        self.model_name = embedding_model
        self.cache_size = cache_size
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []

        # Lazy load model
        self._model: SentenceTransformer | None = None
        self._device = device

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy load the embedding model"""
        if self._model is None:
            self._model = _load_sentence_transformer_cls()(self.model_name)
            if self._device:
                self._model = self._model.to(self._device)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        return int(self.model.get_sentence_embedding_dimension())

    def get_or_compute(self, key: str, text: str) -> Any:
        """
        Get cached embedding or compute and cache it.

        Args:
            key: Unique key for caching
            text: Text to embed

        Returns:
            Embedding tensor
        """
        if key in self._cache:
            # Move to end of access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        # Compute embedding
        embedding = self.model.encode(text, convert_to_tensor=True)

        # Cache with LRU eviction
        if len(self._cache) >= self.cache_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = embedding
        self._access_order.append(key)

        return embedding

    async def embed_conversation(
        self,
        turns: list[ConversationTurnData],
    ) -> Any:
        """
        Embed a full conversation asynchronously.

        Returns:
            Tensor of shape [num_turns, embedding_dim]
        """
        _require_torch()

        texts = [f"{turn.role}: {turn.content}" for turn in turns]

        # Run embedding in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.encode(texts, convert_to_tensor=True)
        )

        return embeddings

    def embed_batch(self, texts: list[str]) -> Any:
        """Embed a batch of texts"""
        return self.model.encode(texts, convert_to_tensor=True)

    def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "max_size": self.cache_size,
            "utilization": len(self._cache) / self.cache_size,
            "model": self.model_name,
        }
