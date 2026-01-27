"""
Continual learning utilities for GRPO training.

Supports:
- Experience replay with reservoir or FIFO storage.
- Learning without Forgetting (LwF) via KL to a frozen reference model.
- Elastic Weight Consolidation (EWC) regularization.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from stateset_agents.core.trajectory import TrajectoryGroup

from .loss_computation import _prepare_inputs_and_labels

logger = logging.getLogger(__name__)

CONTINUAL_EXCEPTIONS = (AttributeError, RuntimeError, TypeError, ValueError)


class ContinualLearningStrategy(str, Enum):
    """High-level continual learning strategy selection."""

    NONE = "none"
    REPLAY = "replay"
    LWF = "lwf"
    EWC = "ewc"
    REPLAY_LWF = "replay_lwf"
    REPLAY_EWC = "replay_ewc"
    REPLAY_LWF_EWC = "replay_lwf_ewc"


@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning components."""

    strategy: str = ContinualLearningStrategy.NONE.value

    # Replay settings
    replay_buffer_size: int = 2000
    replay_ratio: float = 0.5
    replay_min_size: int = 100
    replay_sampling: str = "uniform"  # "uniform", "recent", "reward", "balanced"
    replay_storage: str = "reservoir"  # "reservoir", "fifo"

    # LwF settings (KL to reference model)
    continual_kl_beta: float = 0.0

    # EWC settings
    ewc_lambda: float = 0.0
    ewc_num_samples: int = 64
    ewc_decay: float = 0.9

    # Task boundary extraction
    task_id_key: str = "task_id"

    def uses_replay(self) -> bool:
        return "replay" in self.strategy and self.replay_buffer_size > 0

    def uses_lwf(self) -> bool:
        return "lwf" in self.strategy

    def uses_ewc(self) -> bool:
        return "ewc" in self.strategy and self.ewc_lambda > 0.0

    def enabled(self) -> bool:
        return self.strategy != ContinualLearningStrategy.NONE.value


@dataclass
class ReplayEntry:
    """Stored trajectory group with metadata for replay."""

    group: TrajectoryGroup
    task_id: Optional[str]
    reward: float
    sequence: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryReplayBuffer:
    """Replay buffer for trajectory groups."""

    def __init__(
        self,
        max_size: int = 2000,
        storage_strategy: str = "reservoir",
        sampling_strategy: str = "uniform",
        seed: Optional[int] = None,
    ):
        self.max_size = max(1, int(max_size))
        self.storage_strategy = storage_strategy
        self.sampling_strategy = sampling_strategy
        self._rng = random.Random(seed)
        self._entries: List[ReplayEntry] = []
        self._seen = 0
        self._sequence = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._seen = 0
        self._sequence = 0

    def add_groups(
        self,
        groups: Iterable[TrajectoryGroup],
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        for group in groups:
            reward = _mean_reward(group)
            entry = ReplayEntry(
                group=group,
                task_id=task_id,
                reward=reward,
                sequence=self._sequence,
                metadata=metadata or {},
            )
            self._sequence += 1
            self._seen += 1
            self._insert_entry(entry)

    def _insert_entry(self, entry: ReplayEntry) -> None:
        if len(self._entries) < self.max_size:
            self._entries.append(entry)
            return

        if self.storage_strategy == "fifo":
            self._entries.pop(0)
            self._entries.append(entry)
            return

        # Reservoir sampling (default)
        idx = self._rng.randint(0, self._seen - 1)
        if idx < self.max_size:
            self._entries[idx] = entry

    def sample_groups(
        self,
        count: int,
        task_id: Optional[str] = None,
        sampling_strategy: Optional[str] = None,
    ) -> List[TrajectoryGroup]:
        if count <= 0 or not self._entries:
            return []

        candidates = self._entries
        if task_id is not None:
            candidates = [entry for entry in candidates if entry.task_id == task_id]

        if not candidates:
            return []

        strategy = sampling_strategy or self.sampling_strategy
        count = min(int(count), len(candidates))

        if strategy == "recent":
            ordered = sorted(candidates, key=lambda e: e.sequence, reverse=True)
            return [entry.group for entry in ordered[:count]]
        if strategy == "reward":
            weights = [max(entry.reward, 0.0) + 1e-3 for entry in candidates]
            if not any(weights):
                return [entry.group for entry in self._rng.sample(candidates, count)]
            chosen = self._rng.choices(candidates, weights=weights, k=count)
            return [entry.group for entry in chosen]
        if strategy == "balanced":
            by_task: Dict[Optional[str], List[ReplayEntry]] = {}
            for entry in candidates:
                by_task.setdefault(entry.task_id, []).append(entry)
            task_ids = [tid for tid in by_task if tid is not None]
            if not task_ids:
                return [entry.group for entry in self._rng.sample(candidates, count)]
            per_task = max(1, count // len(task_ids))
            selected: List[ReplayEntry] = []
            for tid in task_ids:
                entries = by_task.get(tid, [])
                if not entries:
                    continue
                take = min(per_task, len(entries))
                selected.extend(self._rng.sample(entries, take))
            remaining = count - len(selected)
            if remaining > 0:
                pool = [entry for entry in candidates if entry not in selected]
                if pool:
                    remaining = min(remaining, len(pool))
                    selected.extend(self._rng.sample(pool, remaining))
            return [entry.group for entry in selected[:count]]

        # Uniform without replacement
        return [entry.group for entry in self._rng.sample(candidates, count)]

    def stats(self) -> Dict[str, Any]:
        rewards = [entry.reward for entry in self._entries]
        return {
            "size": len(self._entries),
            "seen": self._seen,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "storage_strategy": self.storage_strategy,
            "sampling_strategy": self.sampling_strategy,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the buffer state."""
        return {
            "max_size": self.max_size,
            "storage_strategy": self.storage_strategy,
            "sampling_strategy": self.sampling_strategy,
            "seen": self._seen,
            "sequence": self._sequence,
            "entries": [
                {
                    "group": entry.group.to_dict(),
                    "task_id": entry.task_id,
                    "reward": entry.reward,
                    "sequence": entry.sequence,
                    "metadata": entry.metadata,
                }
                for entry in self._entries
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore buffer state from a snapshot."""
        if not isinstance(state, dict):
            return

        max_size = state.get("max_size")
        if isinstance(max_size, int) and max_size > 0:
            self.max_size = max_size
        storage_strategy = state.get("storage_strategy")
        if isinstance(storage_strategy, str) and storage_strategy:
            self.storage_strategy = storage_strategy
        sampling_strategy = state.get("sampling_strategy")
        if isinstance(sampling_strategy, str) and sampling_strategy:
            self.sampling_strategy = sampling_strategy

        entries: List[ReplayEntry] = []
        for item in state.get("entries", []) or []:
            if not isinstance(item, dict):
                continue
            group_data = item.get("group")
            if isinstance(group_data, TrajectoryGroup):
                group = group_data
            elif isinstance(group_data, dict):
                group = TrajectoryGroup.from_dict(group_data)
            else:
                continue
            reward_value = item.get("reward")
            entries.append(
                ReplayEntry(
                    group=group,
                    task_id=item.get("task_id"),
                    reward=float(reward_value) if reward_value is not None else 0.0,
                    sequence=int(item.get("sequence", 0)),
                    metadata=item.get("metadata") or {},
                )
            )

        if len(entries) > self.max_size:
            entries = entries[-self.max_size :]

        self._entries = entries
        seen = state.get("seen")
        if isinstance(seen, int):
            self._seen = max(seen, len(entries))
        else:
            self._seen = len(entries)
        sequence = state.get("sequence")
        if isinstance(sequence, int):
            self._sequence = sequence
        else:
            sequences = [entry.sequence for entry in entries]
            self._sequence = (max(sequences) + 1) if sequences else 0


class ContinualLearningManager:
    """Coordinator for replay, LwF, and EWC."""

    def __init__(
        self,
        config: ContinualLearningConfig,
        training_config: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.training_config = training_config
        self._rng = random.Random(seed)
        self.buffer = TrajectoryReplayBuffer(
            max_size=config.replay_buffer_size,
            storage_strategy=config.replay_storage,
            sampling_strategy=config.replay_sampling,
            seed=seed,
        )
        self.reference_model = None
        self._ewc_fisher: Dict[str, Any] = {}
        self._ewc_params: Dict[str, Any] = {}

        if self.config.uses_ewc() and torch is None:
            logger.warning("EWC requested but PyTorch is unavailable; disabling EWC.")
            self.config.ewc_lambda = 0.0

    @classmethod
    def from_training_config(cls, training_config: Any) -> "ContinualLearningManager":
        strategy = _coerce_str(
            getattr(training_config, "continual_strategy", None), "none"
        )
        if strategy not in {
            ContinualLearningStrategy.NONE.value,
            ContinualLearningStrategy.REPLAY.value,
            ContinualLearningStrategy.LWF.value,
            ContinualLearningStrategy.EWC.value,
            ContinualLearningStrategy.REPLAY_LWF.value,
            ContinualLearningStrategy.REPLAY_EWC.value,
            ContinualLearningStrategy.REPLAY_LWF_EWC.value,
        }:
            strategy = "none"
        config = ContinualLearningConfig(
            strategy=str(strategy),
            replay_buffer_size=_coerce_int(
                getattr(training_config, "replay_buffer_size", None), 2000
            ),
            replay_ratio=_coerce_float(
                getattr(training_config, "replay_ratio", None), 0.5
            ),
            replay_min_size=_coerce_int(
                getattr(training_config, "replay_min_size", None), 100
            ),
            replay_sampling=_coerce_str(
                getattr(training_config, "replay_sampling", None), "uniform"
            ),
            replay_storage=_coerce_str(
                getattr(training_config, "replay_storage", None), "reservoir"
            ),
            continual_kl_beta=_coerce_float(
                getattr(training_config, "continual_kl_beta", None), 0.0
            ),
            ewc_lambda=_coerce_float(
                getattr(training_config, "ewc_lambda", None), 0.0
            ),
            ewc_num_samples=_coerce_int(
                getattr(training_config, "ewc_num_samples", None), 64
            ),
            ewc_decay=_coerce_float(
                getattr(training_config, "ewc_decay", None), 0.9
            ),
            task_id_key=_coerce_str(
                getattr(training_config, "task_id_key", None), "task_id"
            ),
        )
        seed = getattr(training_config, "seed", None)
        return cls(config=config, training_config=training_config, seed=seed)

    @property
    def enabled(self) -> bool:
        return self.config.enabled()

    def should_replay(self) -> bool:
        if not self.config.uses_replay():
            return False
        if self.buffer.size < self.config.replay_min_size:
            return False
        return self.config.replay_ratio > 0.0

    def add_trajectory_groups(
        self,
        groups: Sequence[TrajectoryGroup],
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not (self.config.uses_replay() or self.config.uses_ewc()):
            return
        self.buffer.add_groups(groups, task_id=task_id, metadata=metadata)

    def sample_replay_groups(
        self, new_group_count: int, task_id: Optional[str] = None
    ) -> List[TrajectoryGroup]:
        if not self.should_replay():
            return []
        if new_group_count <= 0:
            return []
        expected = max(0.0, new_group_count * self.config.replay_ratio)
        target = int(expected)
        if self._rng.random() < expected - target:
            target += 1
        if target <= 0:
            return []
        return self.buffer.sample_groups(target, task_id=task_id)

    def get_effective_beta(self, base_beta: float) -> float:
        if self.config.uses_lwf() and self.reference_model is not None:
            return max(base_beta, self.config.continual_kl_beta)
        return base_beta

    def on_task_end(
        self,
        agent: Any,
        task_id: Optional[str] = None,
        recent_groups: Optional[Sequence[TrajectoryGroup]] = None,
    ) -> None:
        if self.config.uses_lwf():
            self._snapshot_reference_model(agent)
        if self.config.uses_ewc():
            self._update_ewc(agent, task_id=task_id, recent_groups=recent_groups)

    def _snapshot_reference_model(self, agent: Any) -> None:
        try:
            self.reference_model = deepcopy(agent.model).eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        except CONTINUAL_EXCEPTIONS as exc:
            logger.warning("Failed to snapshot reference model: %s", exc)
            self.reference_model = None

    def _update_ewc(
        self,
        agent: Any,
        task_id: Optional[str] = None,
        recent_groups: Optional[Sequence[TrajectoryGroup]] = None,
    ) -> None:
        if torch is None:
            return

        groups = list(recent_groups) if recent_groups else []
        if not groups:
            groups = self.buffer.sample_groups(
                self.config.ewc_num_samples, task_id=task_id
            )
        trajectories = _flatten_trajectories(groups)
        if not trajectories:
            logger.debug("No trajectories available for EWC estimation.")
            return

        fisher, params = self._estimate_fisher(agent, trajectories)
        if not fisher:
            return

        if not self._ewc_fisher:
            self._ewc_fisher = fisher
            self._ewc_params = params
            return

        decay = float(self.config.ewc_decay)
        for name, value in fisher.items():
            if name in self._ewc_fisher:
                self._ewc_fisher[name] = self._ewc_fisher[name] * decay + value
            else:
                self._ewc_fisher[name] = value
            if name in params:
                self._ewc_params[name] = params[name]

    def _estimate_fisher(
        self, agent: Any, trajectories: Sequence[Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if torch is None:
            return {}, {}

        model = agent.model
        was_training = model.training
        model.eval()

        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not named_params:
            return {}, {}

        fisher: Dict[str, Any] = {
            name: torch.zeros_like(param) for name, param in named_params
        }
        params: Dict[str, Any] = {
            name: param.detach().clone() for name, param in named_params
        }

        max_samples = min(len(trajectories), int(self.config.ewc_num_samples))
        if max_samples <= 0:
            if was_training:
                model.train()
            return {}, {}

        for idx, trajectory in enumerate(trajectories[:max_samples]):
            try:
                inputs, labels = _prepare_inputs_and_labels(
                    trajectory, agent, self.training_config or {}
                )
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                grads = torch.autograd.grad(
                    loss,
                    [param for _, param in named_params],
                    retain_graph=False,
                    allow_unused=True,
                )
                for (name, _), grad in zip(named_params, grads):
                    if grad is not None:
                        fisher[name] += grad.detach() ** 2
            except CONTINUAL_EXCEPTIONS as exc:
                logger.debug("Skipping EWC sample %s: %s", idx, exc)
                continue

        for name in fisher:
            fisher[name] = fisher[name] / float(max_samples)

        if was_training:
            model.train()

        return fisher, params

    def compute_ewc_penalty(self, agent: Any) -> Optional[Any]:
        if not self.config.uses_ewc() or not self._ewc_fisher:
            return None
        if torch is None:
            return None

        loss = None
        for name, param in agent.model.named_parameters():
            if name not in self._ewc_fisher:
                continue
            fisher = self._ewc_fisher[name]
            mean = self._ewc_params.get(name)
            if mean is None:
                continue
            penalty = fisher * (param - mean) ** 2
            term = penalty.sum()
            loss = term if loss is None else loss + term

        if loss is None:
            return None
        return 0.5 * float(self.config.ewc_lambda) * loss

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable snapshot of continual learning state."""
        return {
            "config": asdict(self.config),
            "buffer": self.buffer.state_dict(),
            "ewc_fisher": self._ewc_fisher,
            "ewc_params": self._ewc_params,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore continual learning state from a snapshot."""
        if not isinstance(state, dict):
            return

        config_state = state.get("config")
        if isinstance(config_state, dict):
            for key, value in config_state.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        buffer_state = state.get("buffer")
        if isinstance(buffer_state, dict):
            self.buffer.load_state_dict(buffer_state)

        ewc_fisher = state.get("ewc_fisher")
        self._ewc_fisher = ewc_fisher if isinstance(ewc_fisher, dict) else {}
        ewc_params = state.get("ewc_params")
        self._ewc_params = ewc_params if isinstance(ewc_params, dict) else {}


def _flatten_trajectories(
    groups: Sequence[TrajectoryGroup],
) -> List[Any]:
    trajectories: List[Any] = []
    for group in groups:
        trajectories.extend(list(group.trajectories))
    return trajectories


def _mean_reward(group: TrajectoryGroup) -> float:
    rewards = group.rewards
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _coerce_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default
