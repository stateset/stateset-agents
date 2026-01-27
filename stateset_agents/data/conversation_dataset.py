"""
Conversation Dataset Utilities for Offline RL

Provides D4RL-style dataset loading, replay buffers, and embedding caching
for training offline RL algorithms on conversation data.
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    torch = None
    TorchDataset = object

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

HF_PARSE_EXCEPTIONS = (AttributeError, KeyError, TypeError, ValueError)


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for conversation datasets. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class ConversationDatasetConfig:
    """Configuration for conversation dataset loading"""

    data_path: str = ""
    format: str = "jsonl"  # jsonl, json, parquet, hf_dataset
    max_turns: int = 20
    embedding_model: str = "all-MiniLM-L6-v2"
    normalize_rewards: bool = True
    reward_normalization_method: str = "standard"  # standard, minmax, percentile
    compute_returns: bool = True
    discount_factor: float = 0.99
    min_trajectory_length: int = 1
    max_trajectory_length: Optional[int] = None


@dataclass
class ConversationTurnData:
    """Lightweight turn representation for datasets"""

    role: str
    content: str
    reward: float = 0.0
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryData:
    """Single conversation trajectory for offline RL"""

    trajectory_id: str
    turns: List[ConversationTurnData]
    total_reward: float
    turn_rewards: List[float]
    returns_to_go: Optional[List[float]] = None
    is_simulated: bool = False
    domain_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_returns_to_go(self, gamma: float = 0.99) -> List[float]:
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
        trajectories: List[TrajectoryData],
        config: Optional[ConversationDatasetConfig] = None,
        embedding_cache: Optional["EmbeddingCache"] = None,
    ):
        self.trajectories = trajectories
        self.config = config or ConversationDatasetConfig()
        self.embedding_cache = embedding_cache
        self._embedding_dim: Optional[int] = None
        self._statistics: Optional[Dict[str, float]] = None

        if self.config.normalize_rewards:
            self.normalize_rewards(self.config.reward_normalization_method)

        if self.config.compute_returns:
            self.compute_all_returns(self.config.discount_factor)

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
        config: Optional[ConversationDatasetConfig] = None,
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
        """
        config = config or ConversationDatasetConfig(data_path=path, format="jsonl")
        trajectories = []

        with open(path, "r", encoding="utf-8") as f:
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
        config: Optional[ConversationDatasetConfig] = None,
        **kwargs,
    ) -> "ConversationDataset":
        """Load dataset from JSON file (list of trajectories)"""
        config = config or ConversationDatasetConfig(data_path=path, format="json")

        with open(path, "r", encoding="utf-8") as f:
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
        config: Optional[ConversationDatasetConfig] = None,
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
        except ImportError:
            raise ImportError(
                "HuggingFace datasets required. Install: pip install datasets"
            )

        config = config or ConversationDatasetConfig(
            data_path=dataset_name, format="hf_dataset"
        )

        hf_dataset = load_dataset(dataset_name, split=split)
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
        data: Dict[str, Any],
        config: Optional[ConversationDatasetConfig] = None,
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
    def _parse_trajectory(
        data: Dict[str, Any], idx: int
    ) -> Optional[TrajectoryData]:
        """Parse a single trajectory from dictionary"""
        try:
            turns = []
            turn_rewards = []

            for turn_data in data.get("turns", []):
                turn = ConversationTurnData(
                    role=turn_data.get("role", "user"),
                    content=turn_data.get("content", ""),
                    reward=float(turn_data.get("reward", 0.0)),
                    timestamp=turn_data.get("timestamp"),
                    metadata=turn_data.get("metadata", {}),
                )
                turns.append(turn)
                turn_rewards.append(turn.reward)

            if not turns:
                return None

            return TrajectoryData(
                trajectory_id=data.get("trajectory_id", f"traj_{idx}"),
                turns=turns,
                total_reward=float(data.get("total_reward", sum(turn_rewards))),
                turn_rewards=turn_rewards,
                is_simulated=data.get("is_simulated", False),
                domain_label=data.get("domain_label"),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse trajectory {idx}: {e}")
            return None

    @staticmethod
    def _parse_hf_item(
        item: Dict[str, Any],
        idx: int,
        conversation_column: str,
        reward_column: str,
    ) -> Optional[TrajectoryData]:
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
                                role=conv.get("role", conv.get("from", "user")),
                                content=conv.get(
                                    "content", conv.get("value", conv.get("text", ""))
                                ),
                            )
                        )
                    elif isinstance(conv, str):
                        turns.append(
                            ConversationTurnData(role="user", content=conv)
                        )

            if not turns:
                return None

            reward = float(item.get(reward_column, 0.0))
            turn_rewards = [0.0] * (len(turns) - 1) + [reward]

            return TrajectoryData(
                trajectory_id=f"hf_{idx}",
                turns=turns,
                total_reward=reward,
                turn_rewards=turn_rewards,
                metadata={"source": "huggingface"},
            )
        except HF_PARSE_EXCEPTIONS as e:
            logger.warning(f"Failed to parse HF item {idx}: {e}")
            return None

    def get_batch(
        self,
        batch_size: int,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
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

        indices = random.sample(range(len(self.trajectories)), min(batch_size, len(self.trajectories)))
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

    def _get_embeddings(
        self, trajectories: List[TrajectoryData]
    ) -> Tuple[Any, Any]:
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

        if states:
            states = torch.stack(states)
        if actions:
            actions = torch.stack(actions)

        return states, actions

    def get_statistics(self) -> Dict[str, float]:
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
        """
        all_rewards = []
        for t in self.trajectories:
            all_rewards.extend(t.turn_rewards)

        if not all_rewards:
            return

        all_rewards = np.array(all_rewards)

        if method == "standard":
            mean = np.mean(all_rewards)
            std = np.std(all_rewards) + 1e-8
            normalize = lambda r: (r - mean) / std

        elif method == "minmax":
            min_r = np.min(all_rewards)
            max_r = np.max(all_rewards)
            range_r = max_r - min_r + 1e-8
            normalize = lambda r: (r - min_r) / range_r

        elif method == "percentile":
            p5 = np.percentile(all_rewards, 5)
            p95 = np.percentile(all_rewards, 95)
            clipped = np.clip(all_rewards, p5, p95)
            mean = np.mean(clipped)
            std = np.std(clipped) + 1e-8
            normalize = lambda r: (np.clip(r, p5, p95) - mean) / std

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Apply normalization
        for traj in self.trajectories:
            traj.turn_rewards = [float(normalize(r)) for r in traj.turn_rewards]
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
        filtered = [t for t in self.trajectories if t.total_reward >= min_reward]
        logger.info(
            f"Filtered from {len(self.trajectories)} to {len(filtered)} "
            f"trajectories (min_reward={min_reward})"
        )
        return ConversationDataset(
            filtered, self.config, self.embedding_cache
        )

    def filter_by_length(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> "ConversationDataset":
        """Filter trajectories by turn count"""
        filtered = self.trajectories

        if min_length is not None:
            filtered = [t for t in filtered if len(t.turns) >= min_length]

        if max_length is not None:
            filtered = [t for t in filtered if len(t.turns) <= max_length]

        return ConversationDataset(
            filtered, self.config, self.embedding_cache
        )

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple["ConversationDataset", "ConversationDataset"]:
        """Split dataset into train and validation sets"""
        trajectories = list(self.trajectories)

        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(trajectories)

        split_idx = int(len(trajectories) * train_ratio)
        train_trajectories = trajectories[:split_idx]
        val_trajectories = trajectories[split_idx:]

        return (
            ConversationDataset(train_trajectories, self.config, self.embedding_cache),
            ConversationDataset(val_trajectories, self.config, self.embedding_cache),
        )

    def to_offline_rl_format(self) -> Dict[str, np.ndarray]:
        """
        Convert to format expected by offline RL algorithms.

        Returns:
            Dictionary with numpy arrays for states, actions, rewards,
            next_states, and dones.
        """
        _require_torch()

        if not self.embedding_cache:
            raise ValueError(
                "Embedding cache required for offline RL format. "
                "Initialize with EmbeddingCache."
            )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for traj in self.trajectories:
            history = ""

            for i, turn in enumerate(traj.turns):
                if turn.role in ("user", "system"):
                    history += f"{turn.role}: {turn.content}\n"
                    state_emb = self.embedding_cache.get_or_compute(
                        f"{traj.trajectory_id}_s_{i}", history
                    )

                elif turn.role == "assistant":
                    action_emb = self.embedding_cache.get_or_compute(
                        f"{traj.trajectory_id}_a_{i}", turn.content
                    )

                    # Get next state (or same state if last turn)
                    next_history = history + f"assistant: {turn.content}\n"
                    next_state_emb = self.embedding_cache.get_or_compute(
                        f"{traj.trajectory_id}_ns_{i}", next_history
                    )

                    is_done = i == len(traj.turns) - 1

                    states.append(state_emb.numpy())
                    actions.append(action_emb.numpy())
                    rewards.append(traj.turn_rewards[i] if i < len(traj.turn_rewards) else 0.0)
                    next_states.append(next_state_emb.numpy())
                    dones.append(float(is_done))

                    history = next_history

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_states": np.array(next_states),
            "dones": np.array(dones),
        }


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

        self.trajectories: List[TrajectoryData] = []
        self.priorities: List[float] = []
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

    def add_trajectories(self, trajectories: List[TrajectoryData]) -> None:
        """Add multiple trajectories"""
        for traj in trajectories:
            self.add_trajectory(traj)

    def sample(
        self,
        batch_size: int,
        return_indices: bool = False,
    ) -> Union[List[TrajectoryData], Tuple[List[TrajectoryData], List[int]]]:
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
    ) -> List[Tuple[TrajectoryData, int]]:
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
            (self.trajectories[turn_to_traj[i][0]], turn_to_traj[i][1])
            for i in indices
        ]

    def update_priorities(
        self,
        indices: List[int],
        priorities: List[float],
    ) -> None:
        """Update priorities for prioritized replay"""
        if not self.priority:
            return

        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6  # Avoid zero priority

    def clear(self) -> None:
        """Clear the buffer"""
        self.trajectories.clear()
        self.priorities.clear()
        self._total_turns = 0

    def get_statistics(self) -> Dict[str, float]:
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
        device: Optional[str] = None,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers required for embedding cache. "
                "Install: pip install sentence-transformers"
            )

        self.model_name = embedding_model
        self.cache_size = cache_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []

        # Lazy load model
        self._model: Optional[SentenceTransformer] = None
        self._device = device

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model"""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            if self._device:
                self._model = self._model.to(self._device)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

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
        turns: List[ConversationTurnData],
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

    def embed_batch(self, texts: List[str]) -> Any:
        """Embed a batch of texts"""
        return self.model.encode(texts, convert_to_tensor=True)

    def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "max_size": self.cache_size,
            "utilization": len(self._cache) / self.cache_size,
            "model": self.model_name,
        }
