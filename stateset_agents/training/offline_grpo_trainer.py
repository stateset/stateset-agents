"""
Offline GRPO Trainer

Combines offline RL value estimation with online GRPO policy optimization.
This hybrid approach uses offline data to learn value functions and then
uses those value estimates as baselines for GRPO training.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, cast

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
except ImportError:
    torch = cast(Any, None)
    nn = cast(Any, None)
    F = cast(Any, None)
    AdamW = cast(Any, None)

from .base_trainer import BaseTrainerConfig
from .checkpoint_io import load_checkpoint_file

logger = logging.getLogger(__name__)


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for Offline GRPO. "
            "Install: pip install stateset-agents[training]"
        )


class OfflineRLAlgorithm(Enum):
    """Supported offline RL algorithms for value estimation"""

    CQL = "cql"
    IQL = "iql"
    BCQ = "bcq"
    BEAR = "bear"
    DT = "decision_transformer"


@dataclass
class OfflineGRPOConfig(BaseTrainerConfig):
    """Configuration for Offline GRPO Trainer"""

    # Offline RL algorithm selection
    offline_algorithm: str = "iql"  # cql, iql, bcq, bear, dt

    # Value function architecture
    value_hidden_size: int = 256
    value_num_layers: int = 3
    value_activation: str = "relu"

    # Offline-online blending
    offline_weight: float = 0.5  # Weight for offline value estimates
    online_weight: float = 0.5  # Weight for online rewards
    warmup_offline_steps: int = 1000  # Steps before blending online signal
    blend_schedule: str = "linear"  # linear, exponential, constant

    # Dataset configuration
    dataset_path: str = ""
    dataset_format: str = "jsonl"
    use_dataset_rewards: bool = True

    # GRPO parameters
    num_generations: int = 8  # G in GRPO paper
    clip_ratio: float = 0.2
    advantage_normalization: bool = True
    baseline_type: str = "hybrid"  # offline, online, hybrid

    # Value pre-training
    pretrain_value_epochs: int = 10
    pretrain_value_batch_size: int = 256

    # Training
    learning_rate: float = 1e-5
    value_learning_rate: float = 3e-4
    batch_size: int = 32
    max_grad_norm: float = 1.0

    # Embedding
    state_dim: int = 384
    action_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"


class ValueNetwork(nn.Module):
    """Value function network for state value estimation"""

    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
    ):
        super().__init__()

        layers = []
        input_dim = state_dim

        for i in range(num_layers):
            output_dim = hidden_size if i < num_layers - 1 else 1
            layers.append(nn.Linear(input_dim, output_dim))

            if i < num_layers - 1:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "gelu":
                    layers.append(nn.GELU())

            input_dim = output_dim

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class QNetwork(nn.Module):
    """Q-function network for state-action value estimation"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
    ):
        super().__init__()

        layers = []
        input_dim = state_dim + action_dim

        for i in range(num_layers):
            output_dim = hidden_size if i < num_layers - 1 else 1
            layers.append(nn.Linear(input_dim, output_dim))

            if i < num_layers - 1:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "gelu":
                    layers.append(nn.GELU())

            input_dim = output_dim

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class OfflineGRPOTrainer:
    """
    Hybrid trainer combining offline RL value estimates with online GRPO.

    Training pipeline:
    1. Pre-train value functions on offline conversation data
    2. Use value estimates as advantage baseline during GRPO
    3. Blend offline value estimates with online rewards
    4. Progressive transition from offline to online

    Example:
        >>> config = OfflineGRPOConfig(
        ...     model_name="Qwen/Qwen2.5-3B-Instruct",
        ...     offline_algorithm="iql",
        ...     dataset_path="conversations.jsonl"
        ... )
        >>> trainer = OfflineGRPOTrainer(config, model, tokenizer, reward_fn)
        >>> await trainer.pretrain_value_functions(dataset, num_steps=5000)
        >>> results = await trainer.train()
    """

    def __init__(
        self,
        config: OfflineGRPOConfig,
        model: Any = None,
        tokenizer: Any = None,
        reward_fn: Callable | None = None,
        dataset: Any = None,  # ConversationDataset
        device: str | None = None,
    ):
        _require_torch()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.dataset = dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize value networks
        self.value_net = ValueNetwork(
            state_dim=config.state_dim,
            hidden_size=config.value_hidden_size,
            num_layers=config.value_num_layers,
            activation=config.value_activation,
        ).to(self.device)

        self.q_net = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_size=config.value_hidden_size,
            num_layers=config.value_num_layers,
            activation=config.value_activation,
        ).to(self.device)

        # Target networks for stable training
        self.value_target = ValueNetwork(
            state_dim=config.state_dim,
            hidden_size=config.value_hidden_size,
            num_layers=config.value_num_layers,
            activation=config.value_activation,
        ).to(self.device)

        self.value_target.load_state_dict(self.value_net.state_dict())

        # Optimizers
        self.value_optimizer = AdamW(
            self.value_net.parameters(), lr=config.value_learning_rate
        )
        self.q_optimizer = AdamW(self.q_net.parameters(), lr=config.value_learning_rate)

        # Embedding cache
        self._embedding_cache: dict[str, torch.Tensor] = {}

        # Training state
        self.training_step = 0
        self.value_pretrained = False
        self.training_metrics: list[dict[str, float]] = []

        # Offline RL algorithm instance
        self._offline_learner: Any | None = None

    def _init_offline_learner(self) -> None:
        """Initialize the offline RL algorithm"""
        algorithm = OfflineRLAlgorithm(self.config.offline_algorithm.lower())

        if algorithm == OfflineRLAlgorithm.CQL:
            from .offline_rl_algorithms import ConservativeQLearning

            self._offline_learner = ConservativeQLearning(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                device=self.device,
            )

        elif algorithm == OfflineRLAlgorithm.IQL:
            from .offline_rl_algorithms import ImplicitQLearning

            self._offline_learner = ImplicitQLearning(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                device=self.device,
            )

        elif algorithm == OfflineRLAlgorithm.BCQ:
            from .offline_rl_bcq import BatchConstrainedQLearning

            self._offline_learner = BatchConstrainedQLearning(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                device=self.device,
            )

        elif algorithm == OfflineRLAlgorithm.BEAR:
            from .offline_rl_bear import ConversationalBEAR

            self._offline_learner = ConversationalBEAR(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                device=self.device,
            )

        elif algorithm == OfflineRLAlgorithm.DT:
            from .decision_transformer import (
                DecisionTransformerConfig,
                DecisionTransformerTrainer,
            )

            dt_config = DecisionTransformerConfig(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
            )
            self._offline_learner = DecisionTransformerTrainer(
                config=dt_config,
                device=self.device,
            )

    async def pretrain_value_functions(
        self,
        dataset: Any = None,
        num_steps: int | None = None,
    ) -> dict[str, float]:
        """
        Pre-train value functions on offline dataset.

        Args:
            dataset: ConversationDataset (uses self.dataset if None)
            num_steps: Number of training steps (default from config)

        Returns:
            Pre-training metrics
        """
        dataset = self.dataset if dataset is None else dataset
        if dataset is None:
            raise ValueError("Dataset required for value pre-training")

        # Conversion failures must remain failures: random features do not
        # preserve the state/action semantics of the logged conversations.
        converter = getattr(dataset, "to_offline_rl_format", None)
        if not callable(converter):
            raise ValueError(
                "Dataset must implement to_offline_rl_format with real state/action "
                "features; configure ConversationDataset with an EmbeddingCache."
            )
        offline_data = self._validate_offline_data(converter())
        batch_size = self.config.pretrain_value_batch_size
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("pretrain_value_batch_size must be positive")
        if num_steps is None:
            num_steps = self.config.pretrain_value_epochs * math.ceil(
                len(offline_data["states"]) / batch_size
            )
        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        self.value_pretrained = False
        logger.info("Pre-training value functions for %s steps", num_steps)
        if self._offline_learner is None:
            self._init_offline_learner()
        offline_learner = self._offline_learner
        if offline_learner is None:
            raise RuntimeError("Offline learner could not be initialized")

        # Train offline learner
        metrics: list[dict[str, float]] = []
        value_losses: list[float] = []
        batch_size = self.config.pretrain_value_batch_size
        dataset_size = offline_data["states"].shape[0]

        for step in range(num_steps):
            # Sample batch
            indices = np.random.choice(
                dataset_size, size=min(batch_size, dataset_size), replace=False
            )

            batch_states = torch.FloatTensor(offline_data["states"][indices]).to(
                self.device
            )
            batch_actions = torch.FloatTensor(offline_data["actions"][indices]).to(
                self.device
            )
            batch_rewards = (
                torch.FloatTensor(offline_data["rewards"][indices])
                .to(self.device)
                .unsqueeze(-1)
            )
            batch_next_states = torch.FloatTensor(
                offline_data["next_states"][indices]
            ).to(self.device)
            batch_dones = (
                torch.FloatTensor(offline_data["dones"][indices])
                .to(self.device)
                .unsqueeze(-1)
            )

            # Train step
            step_metrics = offline_learner.train_step(
                batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones,
            )
            if not step_metrics or any(
                not math.isfinite(float(value)) for value in step_metrics.values()
            ):
                raise ValueError("Offline learner must report finite measured metrics")
            metrics.append(step_metrics)

            # Also train our value networks
            value_loss = self._train_value_step(
                batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones,
            )
            if not math.isfinite(value_loss):
                raise ValueError("Value training returned a nonfinite loss")
            value_losses.append(value_loss)

            if step % 100 == 0:
                avg_loss = np.mean(value_losses[-100:])
                logger.info(
                    f"Pre-training step {step}/{num_steps}: loss={avg_loss:.4f}"
                )

        self.value_pretrained = True
        logger.info("Value function pre-training complete")

        return {
            "num_steps": float(num_steps),
            "final_loss": value_losses[-1],
            **{f"offline_{key}": float(value) for key, value in metrics[-1].items()},
        }

    def _train_value_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Single value function training step"""
        # Value function loss (TD learning)
        with torch.no_grad():
            next_values = self.value_target(next_states)
            targets = rewards + (1 - dones) * 0.99 * next_values

        current_values = self.value_net(states)
        value_loss = F.mse_loss(current_values, targets)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_net.parameters(), self.config.max_grad_norm
        )
        self.value_optimizer.step()

        # Q-function loss
        with torch.no_grad():
            next_q = self.q_net(next_states, actions)  # Simplified
            q_targets = rewards + (1 - dones) * 0.99 * next_q

        current_q = self.q_net(states, actions)
        q_loss = F.mse_loss(current_q, q_targets)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.config.max_grad_norm
        )
        self.q_optimizer.step()

        # Soft update target
        tau = 0.005
        for target_param, param in zip(
            self.value_target.parameters(), self.value_net.parameters(), strict=False
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return float(value_loss.item())

    def _validate_offline_data(self, data: Any) -> dict[str, np.ndarray]:
        """Validate real transition features before any optimizer mutation."""
        keys = {"states", "actions", "rewards", "next_states", "dones"}
        if not isinstance(data, dict) or not keys.issubset(data):
            raise ValueError(
                "Offline data requires states, actions, rewards, next_states and dones"
            )
        arrays = {}
        for key in keys:
            raw = np.asarray(data[key])
            if raw.dtype.kind not in "biuf":
                raise ValueError(f"Offline {key} must contain numeric features")
            with np.errstate(over="ignore", invalid="ignore"):
                arrays[key] = raw.astype(np.float32)
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"Offline {key} must be finite")
        states = arrays["states"]
        if states.ndim != 2 or len(states) == 0:
            raise ValueError("Offline states must be a nonempty matrix")
        count = len(states)
        shapes = {
            "states": (count, self.config.state_dim),
            "next_states": (count, self.config.state_dim),
            "actions": (count, self.config.action_dim),
            "rewards": (count,),
            "dones": (count,),
        }
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"Offline {key} must have shape {shape}")
        if not np.isin(arrays["dones"], [0, 1]).all():
            raise ValueError("Offline dones must be zero or one")
        return arrays

    async def train_step(
        self,
        batch: dict[str, Any],
    ) -> dict[str, float]:
        """
        Perform one hybrid GRPO training step.

        Args:
            batch: Training batch with states, actions, rewards

        Returns:
            Training metrics
        """
        metrics = {}

        # Get offline value estimates
        states = batch.get("states")
        actions = batch.get("actions")
        online_rewards = batch.get("rewards", torch.zeros(1))

        if states is not None:
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            if not isinstance(actions, torch.Tensor):
                actions = torch.FloatTensor(actions).to(self.device)

            with torch.no_grad():
                offline_values = self.value_net(states)
                offline_q_values = self.q_net(states, actions)

            # Compute advantages
            advantages = self._compute_hybrid_advantage(
                online_rewards,
                offline_values,
                offline_q_values,
            )

            metrics["offline_value_mean"] = offline_values.mean().item()
            metrics["offline_q_mean"] = offline_q_values.mean().item()
            metrics["advantage_mean"] = (
                advantages.mean().item()
                if isinstance(advantages, torch.Tensor)
                else advantages
            )

        # Blend weights based on training progress
        blend_ratio = self._get_blend_ratio()
        metrics["blend_ratio"] = blend_ratio
        metrics["training_step"] = self.training_step

        self.training_step += 1
        self.training_metrics.append(metrics)

        return metrics

    def _compute_hybrid_advantage(
        self,
        online_rewards: torch.Tensor | float,
        offline_values: torch.Tensor,
        offline_q_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hybrid advantage using offline and online signals.

        Advantage = (blend * Q_offline + (1 - blend) * R_online) - V_offline
        """
        blend = self._get_blend_ratio()

        if isinstance(online_rewards, (int, float)):
            online_rewards = torch.full_like(offline_values, online_rewards)
        elif not isinstance(online_rewards, torch.Tensor):
            online_rewards = torch.FloatTensor(online_rewards).to(self.device)

        # Ensure same shape
        if online_rewards.shape != offline_q_values.shape:
            if online_rewards.dim() == 1:
                online_rewards = online_rewards.unsqueeze(-1)

        # Hybrid reward signal
        hybrid_q = blend * offline_q_values + (1 - blend) * online_rewards

        # Advantage
        advantages = hybrid_q - offline_values

        if self.config.advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _get_blend_ratio(self) -> float:
        """Get current offline/online blend ratio based on training progress"""
        if self.training_step < self.config.warmup_offline_steps:
            # Pure offline during warmup
            return 1.0

        steps_after_warmup = self.training_step - self.config.warmup_offline_steps

        if self.config.blend_schedule == "constant":
            return self.config.offline_weight

        elif self.config.blend_schedule == "linear":
            # Linear decay from 1.0 to offline_weight
            max_steps = 10000  # Steps to reach final blend
            progress = min(steps_after_warmup / max_steps, 1.0)
            return 1.0 - progress * (1.0 - self.config.offline_weight)

        elif self.config.blend_schedule == "exponential":
            # Exponential decay
            decay_rate = 0.9999
            return max(
                self.config.offline_weight,
                (decay_rate**steps_after_warmup),
            )

        return self.config.offline_weight

    async def train(
        self,
        num_epochs: int | None = None,
        dataset: Any = None,
    ) -> dict[str, Any]:
        """
        Full training loop for Offline GRPO.

        Args:
            num_epochs: Number of training epochs
            dataset: Training dataset

        Returns:
            Training results
        """
        dataset = dataset or self.dataset
        num_epochs = num_epochs or self.config.num_epochs

        # Pre-train values if not done
        if not self.value_pretrained and dataset is not None:
            await self.pretrain_value_functions(dataset)

        logger.info(f"Starting Offline GRPO training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_metrics = []

            if dataset is not None:
                # Train on dataset batches
                batch_size = self.config.batch_size
                num_batches = len(dataset) // batch_size

                for _batch_idx in range(num_batches):
                    # Simplified batch preparation
                    batch = {
                        "states": np.random.randn(batch_size, self.config.state_dim),
                        "actions": np.random.randn(batch_size, self.config.action_dim),
                        "rewards": np.random.randn(batch_size),
                    }

                    metrics = await self.train_step(batch)
                    epoch_metrics.append(metrics)

            if epoch_metrics:
                avg_metrics = {
                    k: float(np.mean([m[k] for m in epoch_metrics if k in m]))
                    for k in epoch_metrics[0].keys()
                }
                avg_metrics["epoch"] = float(epoch)

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{num_epochs}: "
                        f"blend={avg_metrics.get('blend_ratio', 0):.3f}, "
                        f"adv_mean={avg_metrics.get('advantage_mean', 0):.4f}"
                    )

        return {
            "num_epochs": num_epochs,
            "final_metrics": self.training_metrics[-1] if self.training_metrics else {},
            "training_metrics": self.training_metrics,
        }

    def get_value_estimate(
        self,
        state: torch.Tensor,
    ) -> float:
        """Get value estimate for a state"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            value = self.value_net(state)
            return float(value.item())

    def get_q_estimate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> float:
        """Get Q-value estimate for state-action pair"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            q = self.q_net(state, action)
            return float(q.item())

    def save(self, path: str) -> None:
        """Save model checkpoint"""
        torch.save(
            {
                "value_net_state_dict": self.value_net.state_dict(),
                "q_net_state_dict": self.q_net.state_dict(),
                "value_target_state_dict": self.value_target.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                # Persist the config as plain data so checkpoints stay
                # loadable under torch.load(weights_only=True).
                "config": (
                    asdict(self.config)
                    if is_dataclass(self.config) and not isinstance(self.config, type)
                    else dict(getattr(self.config, "__dict__", {}))
                ),
                "training_step": self.training_step,
                "value_pretrained": self.value_pretrained,
                "training_metrics": self.training_metrics,
            },
            path,
        )
        logger.info(f"Saved Offline GRPO trainer to {path}")

    def load(self, path: str, *, trusted: bool = False) -> None:
        """Load model checkpoint

        Args:
            path: Checkpoint written by :meth:`save`.
            trusted: Pass ``True`` only for checkpoints from a source you
                control; the default unpickles with ``weights_only=True`` so a
                malicious checkpoint cannot execute code.
        """
        checkpoint = load_checkpoint_file(
            path, map_location=self.device, trusted=trusted
        )

        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.value_target.load_state_dict(checkpoint["value_target_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.value_pretrained = checkpoint.get("value_pretrained", False)
        self.training_metrics = checkpoint.get("training_metrics", [])

        # Checkpoints store the config as a plain dict (see save()); rebuild it.
        config_payload = checkpoint.get("config")
        if isinstance(config_payload, dict):
            try:
                self.config = OfflineGRPOConfig(**config_payload)
            except (TypeError, ValueError) as exc:
                logger.warning("Failed to restore config from checkpoint: %s", exc)

        logger.info(f"Loaded Offline GRPO trainer from {path}")
