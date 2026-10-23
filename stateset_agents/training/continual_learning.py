"""
Continual learning utilities for GRPO training.

Supports:
- Experience replay with reservoir or FIFO storage.
- Learning without Forgetting (LwF) via KL to a frozen reference model.
- Elastic Weight Consolidation (EWC) regularization.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, cast

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = cast(Any, None)

from stateset_agents.core.trajectory import TrajectoryGroup
from stateset_agents.exceptions import ATTRIBUTE_VALUE_EXCEPTIONS

from .loss_computation import _prepare_inputs_and_labels

logger = logging.getLogger(__name__)

CONTINUAL_EXCEPTIONS = ATTRIBUTE_VALUE_EXCEPTIONS


def _torch_available() -> bool:
    """Whether the optional torch import succeeded (checked at call time)."""
    return torch is not None


def _finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _restore_private_rng(state: Any, fallback: random.Random) -> random.Random:
    """Validate a private generator without changing the live stream."""
    if state is None:
        return deepcopy(fallback)
    restored = random.Random(0)
    try:
        if (
            not isinstance(state, (list, tuple))
            or len(state) != 3
            or type(state[0]) is not int
            or (state[2] is not None and not _finite_number(state[2]))
        ):
            raise ValueError("Invalid RNG structure")
        # JSON round trips turn tuples into lists.
        restored.setstate((state[0], tuple(state[1]), state[2]))
    except (TypeError, ValueError, IndexError, OverflowError) as error:
        raise ValueError("Invalid continual learning RNG state") from error
    return restored


def _cpu_tensor_state(values: dict[str, Any]) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def _finite_tensors(values: Iterable[Any]) -> bool:
    """Reduce finite checks per device, avoiding a host sync per parameter."""
    checks: dict[Any, list[Any]] = {}
    for value in values:
        checks.setdefault(value.device, []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(items).all()) for items in checks.values())


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
    task_id: str | None
    reward: float
    sequence: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TrajectoryReplayBuffer:
    """Replay buffer for trajectory groups."""

    def __init__(
        self,
        max_size: int = 2000,
        storage_strategy: str = "reservoir",
        sampling_strategy: str = "uniform",
        seed: int | None = None,
    ):
        self.max_size = max(1, int(max_size))
        self.storage_strategy = storage_strategy
        self.sampling_strategy = sampling_strategy
        self._rng = random.Random(seed)
        self._entries: list[ReplayEntry] = []
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
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
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
        task_id: str | None = None,
        sampling_strategy: str | None = None,
    ) -> list[TrajectoryGroup]:
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
            by_task: dict[str | None, list[ReplayEntry]] = {}
            for entry in candidates:
                by_task.setdefault(entry.task_id, []).append(entry)
            task_ids = [tid for tid in by_task if tid is not None]
            if not task_ids:
                return [entry.group for entry in self._rng.sample(candidates, count)]
            # Random task order: with fewer samples than tasks the truncation
            # below would otherwise always keep the first-seen tasks.
            self._rng.shuffle(task_ids)
            per_task = max(1, count // len(task_ids))
            selected: list[ReplayEntry] = []
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

    def stats(self) -> dict[str, Any]:
        rewards = [entry.reward for entry in self._entries]
        return {
            "size": len(self._entries),
            "seen": self._seen,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "storage_strategy": self.storage_strategy,
            "sampling_strategy": self.sampling_strategy,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the buffer state."""
        return {
            "schema_version": 1,
            "rng_state": self._rng.getstate(),
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

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore buffer state from a snapshot."""
        if not isinstance(state, dict):
            raise ValueError("Invalid replay checkpoint")
        if "schema_version" in state and (
            type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state.get("rng_state") is None
        ):
            raise ValueError("Invalid replay checkpoint schema or missing RNG")
        modern = "schema_version" in state
        if modern:
            _validate_replay_state(state)
        rng = _restore_private_rng(state.get("rng_state"), self._rng)
        max_size = state.get("max_size")
        if type(max_size) is not int or max_size < 1:
            max_size = self.max_size
        storage_strategy = state.get("storage_strategy")
        if not isinstance(storage_strategy, str) or not storage_strategy:
            storage_strategy = self.storage_strategy
        sampling_strategy = state.get("sampling_strategy")
        if not isinstance(sampling_strategy, str) or not sampling_strategy:
            sampling_strategy = self.sampling_strategy

        entries: list[ReplayEntry] = []
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

        if len(entries) > max_size:
            entries = entries[-max_size:]

        self.max_size = max_size
        self.storage_strategy = storage_strategy
        self.sampling_strategy = sampling_strategy
        self._rng = rng
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
        training_config: Any | None = None,
        seed: int | None = None,
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
        self.reference_model: Any | None = None
        self._ewc_fisher: dict[str, Any] = {}
        self._ewc_params: dict[str, Any] = {}
        self.last_fisher_estimate: dict[str, int] = {}
        self._task_boundary: dict[str, Any] | None = None
        self._ddp_restore_failed = False

        if self.config.uses_ewc() and not _torch_available():
            logger.warning("EWC requested but PyTorch is unavailable; disabling EWC.")
            self.config.ewc_lambda = 0.0

    @classmethod
    def from_training_config(cls, training_config: Any) -> ContinualLearningManager:
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
            ewc_lambda=_coerce_float(getattr(training_config, "ewc_lambda", None), 0.0),
            ewc_num_samples=_coerce_int(
                getattr(training_config, "ewc_num_samples", None), 64
            ),
            ewc_decay=_coerce_float(getattr(training_config, "ewc_decay", None), 0.9),
            task_id_key=_coerce_str(
                getattr(training_config, "task_id_key", None), "task_id"
            ),
        )
        seed = _coerce_seed(getattr(training_config, "seed", None))
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
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not (self.config.uses_replay() or self.config.uses_ewc()):
            return
        self.buffer.add_groups(groups, task_id=task_id, metadata=metadata)

    def sample_replay_groups(
        self, new_group_count: int, task_id: str | None = None
    ) -> list[TrajectoryGroup]:
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
        task_id: str | None = None,
        recent_groups: Sequence[TrajectoryGroup] | None = None,
    ) -> None:
        if _torch_available():
            from torch.nn.parallel import DistributedDataParallel

            if isinstance(agent.model, DistributedDataParallel):
                from .ddp_continual import consolidate_ddp_task

                consolidate_ddp_task(self, agent, task_id, recent_groups)
                return
        if self.config.uses_lwf():
            self._snapshot_reference_model(agent)
        if self.config.uses_ewc():
            self._update_ewc(agent, task_id=task_id, recent_groups=recent_groups)
        self._task_boundary = None

    def _snapshot_reference_model(self, agent: Any) -> None:
        model = agent.model
        if _torch_available():
            from torch.nn.parallel import DistributedDataParallel

            if isinstance(model, DistributedDataParallel):
                model = model.module
        reference = deepcopy(model).eval()
        for param in reference.parameters():
            param.requires_grad = False
        self.reference_model = reference

    def _update_ewc(
        self,
        agent: Any,
        task_id: str | None = None,
        recent_groups: Sequence[TrajectoryGroup] | None = None,
    ) -> None:
        if not _torch_available():
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

        self._consolidate_fisher(fisher, params)

    def _consolidate_fisher(
        self, fisher: dict[str, Any], params: dict[str, Any]
    ) -> None:
        """Prepare finite task anchors before replacing prior EWC memory."""
        decay = float(self.config.ewc_decay)
        next_fisher = dict(self._ewc_fisher)
        next_params = dict(self._ewc_params)
        for name, value in fisher.items():
            if name in self._ewc_fisher:
                previous = self._ewc_fisher[name].to(
                    device=value.device, dtype=value.dtype
                )
                next_fisher[name] = previous * decay + value
            else:
                next_fisher[name] = value
            if name in params:
                next_params[name] = params[name]
        if not _finite_tensors([*next_fisher.values(), *next_params.values()]):
            raise ValueError("EWC consolidation produced nonfinite task memory")
        self._ewc_fisher = next_fisher
        self._ewc_params = next_params

    def _estimate_fisher(
        self, agent: Any, trajectories: Sequence[Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Average squared gradients of accepted trajectory losses.

        The sample cap bounds attempts. Failed/unscored samples contribute
        neither a squared gradient nor a denominator. Existing optimizer
        gradients and each module's train/eval mode remain unchanged.
        """
        report = dict.fromkeys(
            (
                "attempted",
                "accepted",
                "skipped",
                "sample_errors",
                "invalid_loss",
                "invalid_gradients",
                "no_supervised_tokens",
                "failed",
            ),
            0,
        )
        self.last_fisher_estimate = report
        if not _torch_available():
            return {}, {}

        model = agent.model
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not named_params:
            return {}, {}
        max_samples = min(len(trajectories), int(self.config.ewc_num_samples))
        if max_samples <= 0:
            return {}, {}
        if any(not param.is_floating_point() for _, param in named_params):
            raise ValueError("EWC requires real floating-point trainable parameters")
        if not _finite_tensors(param.detach() for _, param in named_params):
            report["failed"] = 1
            raise ValueError("EWC requires finite policy parameters")
        fisher: dict[str, Any] = {
            name: torch.zeros_like(
                param,
                dtype=torch.float64 if param.dtype == torch.float64 else torch.float32,
            )
            for name, param in named_params
        }
        params: dict[str, Any] = {
            name: param.detach().to(dtype=fisher[name].dtype).clone()
            for name, param in named_params
        }

        def skip(reason: str) -> None:
            report["skipped"] += 1
            report[reason] += 1

        modes = [(child, child.training) for child in model.modules()]
        try:
            with torch.enable_grad():
                model.eval()
                for idx, trajectory in enumerate(trajectories[:max_samples]):
                    report["attempted"] += 1
                    try:
                        inputs, labels = _prepare_inputs_and_labels(
                            trajectory, agent, self.training_config or {}
                        )
                        if (
                            labels is None
                            or not torch.is_tensor(labels)
                            or labels.ndim < 1
                            or not bool(labels[..., 1:].ne(-100).any())
                        ):
                            skip("no_supervised_tokens")
                            continue
                        loss = model(**inputs, labels=labels).loss
                        if (
                            not torch.is_tensor(loss)
                            or loss.ndim != 0
                            or not loss.is_floating_point()
                            or not loss.requires_grad
                            or not bool(torch.isfinite(loss))
                        ):
                            skip("invalid_loss")
                            continue
                        grads = torch.autograd.grad(
                            loss,
                            [param for _, param in named_params],
                            retain_graph=False,
                            allow_unused=True,
                        )
                        if not any(grad is not None for grad in grads) or any(
                            grad is not None
                            and (
                                grad.layout != torch.strided
                                or not grad.is_floating_point()
                            )
                            for grad in grads
                        ):
                            skip("invalid_gradients")
                            continue
                        squared = {
                            name: grad.detach().to(dtype=fisher[name].dtype).square()
                            for (name, _), grad in zip(named_params, grads, strict=True)
                            if grad is not None
                        }
                        if not _finite_tensors(squared.values()):
                            skip("invalid_gradients")
                            continue
                        count = report["accepted"] + 1
                        # An online mean avoids overflowing a finite mean's sum.
                        for name, value in fisher.items():
                            if name in squared:
                                value.lerp_(squared[name], 1.0 / count)
                            else:
                                value.mul_(1.0 - 1.0 / count)
                        report["accepted"] = count
                    except CONTINUAL_EXCEPTIONS as exc:
                        oom = getattr(torch, "OutOfMemoryError", None)
                        if oom is None:
                            oom = getattr(torch.cuda, "OutOfMemoryError", None)
                        if isinstance(oom, type) and isinstance(exc, oom):
                            raise
                        skip("sample_errors")
                        logger.debug("Skipping EWC sample %s: %s", idx, exc)
        except BaseException:
            report["failed"] = 1
            raise
        finally:
            for child, mode in modes:
                child.training = mode
        if report["skipped"]:
            logger.warning(
                "EWC accepted %s/%s trajectories; skipped %s invalid examples",
                report["accepted"],
                report["attempted"],
                report["skipped"],
            )
        return (fisher, params) if report["accepted"] else ({}, {})

    def compute_ewc_penalty(self, agent: Any) -> Any | None:
        if not self.config.uses_ewc() or not self._ewc_fisher:
            return None
        if not _torch_available():
            return None

        loss = None
        for name, param in agent.model.named_parameters():
            if name not in self._ewc_fisher:
                continue
            fisher = self._ewc_fisher[name].to(device=param.device)
            mean = self._ewc_params.get(name)
            if mean is None:
                continue
            penalty = fisher * (param - mean.to(device=param.device)) ** 2
            term = penalty.sum()
            loss = term if loss is None else loss + term

        if loss is None:
            return None
        return 0.5 * float(self.config.ewc_lambda) * loss

    def state_dict(self, *, include_reference: bool = True) -> dict[str, Any]:
        """Return a serializable snapshot of continual learning state."""
        config = asdict(self.config)
        if isinstance(config["strategy"], ContinualLearningStrategy):
            config["strategy"] = config["strategy"].value
        return {
            "schema_version": 1,
            "task_boundary": deepcopy(self._task_boundary),
            "rng_state": self._rng.getstate(),
            "config": config,
            "buffer": self.buffer.state_dict(),
            "ewc_fisher": _cpu_tensor_state(self._ewc_fisher),
            "ewc_params": _cpu_tensor_state(self._ewc_params),
            "reference_state": (
                _cpu_tensor_state(self.reference_model.state_dict())
                if include_reference and self.reference_model is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any], *, model: Any = None) -> None:
        """Validate then restore private RNGs, replay, reference weights and EWC.

        ``model`` supplies the architecture/device for a saved reference when
        this manager does not already have one. No live fields change on a
        validation failure.
        """
        if not isinstance(state, dict):
            raise ValueError("Invalid continual learning checkpoint")
        boundary = state.get("task_boundary")
        if boundary is not None and (
            not isinstance(boundary, dict)
            or set(boundary) != {"task_id", "global_step", "buffer_sequence"}
            or (
                boundary["task_id"] is not None
                and not isinstance(boundary["task_id"], str)
            )
            or any(
                type(boundary[name]) is not int or not 0 <= boundary[name] < 2**63
                for name in ("global_step", "buffer_sequence")
            )
        ):
            raise ValueError("Invalid continual task-boundary receipt")
        if "schema_version" in state and (
            type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or not {
                "rng_state",
                "reference_state",
                "buffer",
                "config",
                "ewc_fisher",
                "ewc_params",
            }
            <= set(state)
            or state["rng_state"] is None
        ):
            raise ValueError("Invalid continual learning checkpoint schema")
        rng = _restore_private_rng(state.get("rng_state"), self._rng)
        config = deepcopy(self.config)
        config_state = state.get("config")
        if "schema_version" in state and (
            not isinstance(config_state, dict)
            or set(config_state) != set(asdict(config))
        ):
            raise ValueError("Incomplete continual checkpoint configuration")
        if isinstance(config_state, dict):
            for key, value in config_state.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        if "schema_version" in state:
            _validate_continual_config(config)
        if config.uses_lwf() and "reference_state" not in state:
            raise ValueError("Legacy LwF checkpoint has no frozen reference state")

        buffer = TrajectoryReplayBuffer(
            config.replay_buffer_size,
            config.replay_storage,
            config.replay_sampling,
            seed=0,
        )
        buffer._rng = deepcopy(self.buffer._rng)
        buffer_state = state.get("buffer")
        if "schema_version" in state and (
            not isinstance(buffer_state, dict) or "schema_version" not in buffer_state
        ):
            raise ValueError("Incomplete continual replay checkpoint")
        if isinstance(buffer_state, dict):
            buffer.load_state_dict(buffer_state)
        if boundary is not None and boundary["buffer_sequence"] > buffer._sequence:
            raise ValueError("Task-boundary receipt exceeds replay history")
        template = model if model is not None else self.reference_model
        reference_state = state.get("reference_state")
        reference = None
        if reference_state is not None:
            if template is None or not _torch_available():
                raise ValueError("A model template is required to restore LwF weights")
            _validate_tensor_state(reference_state, template.state_dict())
            reference = deepcopy(template).eval()
            reference.load_state_dict(reference_state, strict=True)
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
        fisher = state.get("ewc_fisher", {})
        means = state.get("ewc_params", {})
        if not isinstance(fisher, dict) or not isinstance(means, dict):
            raise ValueError("Invalid EWC checkpoint state")
        _validate_tensor_state(fisher, means)
        _validate_tensor_state(means, fisher)
        parameters = dict(template.named_parameters()) if template is not None else {}
        for name, value in fisher.items():
            if not value.is_floating_point() or bool((value < 0).any()):
                raise ValueError("EWC Fisher values must be nonnegative")
            if parameters and (
                name not in parameters or value.shape != parameters[name].shape
            ):
                raise ValueError("EWC checkpoint parameters differ from policy")
        self.config = config
        self.buffer = buffer
        self._rng = rng
        self.reference_model = reference
        self._ewc_fisher = _place_tensor_state(fisher, parameters)
        self._ewc_params = _place_tensor_state(means, parameters)
        self._task_boundary = deepcopy(boundary)
        self._ddp_restore_failed = False


def _place_tensor_state(
    values: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    return {
        name: value.detach()
        .to(device=parameters[name].device if name in parameters else value.device)
        .clone()
        for name, value in values.items()
    }


def _validate_tensor_state(saved: Any, expected: dict[str, Any]) -> None:
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("Continual checkpoint tensor names differ")
    for name, value in saved.items():
        other = expected[name]
        if (
            not _torch_available()
            or not torch.is_tensor(value)
            or not torch.is_tensor(other)
            or value.layout != torch.strided
            or value.shape != other.shape
            or value.dtype != other.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid continual checkpoint tensor: {name}")


def _validate_continual_config(config: ContinualLearningConfig) -> None:
    if not isinstance(config.strategy, str) or config.strategy not in {
        value.value for value in ContinualLearningStrategy
    }:
        raise ValueError("Invalid continual checkpoint strategy")
    if config.replay_storage not in (
        "reservoir",
        "fifo",
    ) or config.replay_sampling not in ("uniform", "recent", "reward", "balanced"):
        raise ValueError("Invalid continual replay configuration")
    for name in ("replay_buffer_size", "replay_min_size", "ewc_num_samples"):
        value = getattr(config, name)
        if type(value) is not int or not 0 <= value <= 2**63 - 1:
            raise ValueError("Invalid continual checkpoint count")
    for name in ("replay_ratio", "continual_kl_beta", "ewc_lambda", "ewc_decay"):
        value = getattr(config, name)
        if not _finite_number(value) or value < 0:
            raise ValueError("Invalid continual checkpoint coefficient")
    if config.ewc_decay > 1 or not isinstance(config.task_id_key, str):
        raise ValueError("Invalid continual checkpoint configuration")


def _validate_replay_state(state: dict[str, Any]) -> None:
    required = {
        "max_size",
        "storage_strategy",
        "sampling_strategy",
        "seen",
        "sequence",
        "entries",
        "rng_state",
        "schema_version",
    }
    if not required <= set(state):
        raise ValueError("Incomplete replay checkpoint")
    for name in ("max_size", "seen", "sequence"):
        value = state[name]
        if type(value) is not int or not 0 <= value <= 2**63 - 1:
            raise ValueError("Invalid replay checkpoint counter")
    if state["max_size"] < 1 or state["storage_strategy"] not in ("fifo", "reservoir"):
        raise ValueError("Invalid replay checkpoint storage")
    if state["sampling_strategy"] not in ("uniform", "recent", "reward", "balanced"):
        raise ValueError("Invalid replay checkpoint sampling")
    entries = state["entries"]
    if not isinstance(entries, list) or len(entries) > min(
        state["max_size"], state["seen"]
    ):
        raise ValueError("Invalid replay checkpoint entries")
    sequences = set()
    for entry in entries:
        if not isinstance(entry, dict) or not {
            "group",
            "task_id",
            "reward",
            "sequence",
            "metadata",
        } <= set(entry):
            raise ValueError("Incomplete replay checkpoint entry")
        sequence = entry["sequence"]
        if (
            type(sequence) is not int
            or not 0 <= sequence < state["sequence"]
            or sequence in sequences
        ):
            raise ValueError("Invalid replay checkpoint sequence")
        sequences.add(sequence)
        reward = entry["reward"]
        if (
            not _finite_number(reward)
            or not isinstance(entry["group"], dict)
            or not isinstance(entry["metadata"], dict)
            or (entry["task_id"] is not None and not isinstance(entry["task_id"], str))
        ):
            raise ValueError("Invalid replay checkpoint entry values")


def _flatten_trajectories(
    groups: Sequence[TrajectoryGroup],
) -> list[Any]:
    trajectories: list[Any] = []
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


def _coerce_seed(value: Any) -> int | None:
    """Coerce arbitrary seed values into a stable Optional[int].

    `random.Random()` accepts a broad range of seed types, but non-standard
    objects trigger a deprecation warning (hash-based seeding). For configs
    coming from mocks or dynamic sources, we prefer an explicit int-or-None.
    """
    if value is None:
        return None
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
            return None
    return None


def _coerce_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default
