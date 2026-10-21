"""
Offline value learning and IQL-weighted language-model policy extraction.

The historical OfflineGRPOTrainer name is retained for compatibility. Its
train() method fits real logged assistant actions with detached IQL weights;
hybrid-baseline helpers do not turn this into an on-policy GRPO objective.
"""

import hashlib
import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
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

from ...training.base_trainer import BaseTrainerConfig
from ...training.callbacks import (
    notify_episode_end,
    notify_step_end,
    notify_training_end,
    notify_training_start,
)
from ...training.checkpoint_io import load_checkpoint_file
from ...training.checkpoint_publication import checkpoint_write
from ...training.runtime_state import capture_rng, restore_rng

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

    # Append new fields to preserve positional configuration compatibility.
    policy_temperature: float = 3.0
    policy_max_weight: float = 100.0

    def __post_init__(self) -> None:
        """Keep the offline batch size after common alias normalization."""
        batch_size = self.batch_size
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        super().__post_init__()
        self.batch_size = batch_size


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
    Offline value learning and IQL-weighted logged-action policy extraction.

    Training pipeline:
    1. Pre-train value functions on offline conversation data
    2. Fit assistant-token likelihoods with detached, bounded IQL weights
    3. Save policy, optimizer, value learner, RNG and pending accumulation state

    Hybrid-baseline helpers are available separately; ``train`` does not apply
    a signed hybrid objective or perform online rollouts.

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
        config.validate_for_training()

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
        self.policy_optimizer: Any | None = None
        self.callbacks: list[Any] = []
        self.current_epoch = 0
        self.optimizer_updates = 0
        self._grad_accum_step = 0
        self._grad_accum_actions = 0
        self._grad_accum_normalizer = 0
        self.dataset_sha256: str | None = None
        self._epoch_order: list[int] | None = None
        self._batch_offset = 0
        self._rng = np.random.default_rng(config.seed)

    def _init_offline_learner(self) -> None:
        """Initialize the offline RL algorithm"""
        algorithm = OfflineRLAlgorithm(self.config.offline_algorithm.lower())

        if algorithm == OfflineRLAlgorithm.CQL:
            from .algorithms import ConservativeQLearning

            self._offline_learner = ConservativeQLearning(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                device=self.device,
            )

        elif algorithm == OfflineRLAlgorithm.IQL:
            from .algorithms import ImplicitQLearning, IQLConfig

            self._offline_learner = ImplicitQLearning(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                config=IQLConfig(
                    hidden_size=self.config.value_hidden_size,
                    num_layers=self.config.value_num_layers,
                    learning_rate=self.config.value_learning_rate,
                    temperature=self.config.policy_temperature,
                ),
                device=self.device,
            )

        elif algorithm == OfflineRLAlgorithm.BCQ:
            from .bcq import BatchConstrainedQLearning

            self._offline_learner = BatchConstrainedQLearning(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                device=self.device,
            )

        elif algorithm == OfflineRLAlgorithm.BEAR:
            from .bear import ConversationalBEAR

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
            indices = self._rng.choice(
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

        # Q-function loss. Bootstrap from V(s'): the logged action belongs to
        # s, so Q(s', a) would score an action never taken in s'.
        q_targets = targets

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

    def _validate_offline_data(self, data: Any) -> dict[str, Any]:
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
        text: dict[str, Any] = {}
        if "contexts" in data or "responses" in data:
            contexts, responses = data.get("contexts"), data.get("responses")
            if (
                not isinstance(contexts, (list, tuple))
                or not isinstance(responses, (list, tuple))
                or len(contexts) != count
                or len(responses) != count
            ):
                raise ValueError(
                    "Offline contexts and responses must have one entry per transition"
                )
            text = {"contexts": list(contexts), "responses": list(responses)}
        return {**arrays, **text}

    def _logged_policy_rows(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Mask logged message contexts using the canonical assistant boundary."""
        from stateset_agents.core.chat_masking import last_assistant_token_mask

        if "contexts" not in data:
            raise ValueError(
                "Dataset must implement to_policy_training_format with logged tokens "
                "or provide aligned message contexts and responses"
            )
        rows = []
        limit = self.config.max_prompt_length + self.config.max_completion_length
        for context, response in zip(data["contexts"], data["responses"], strict=True):
            if (
                not isinstance(context, list)
                or not context
                or not isinstance(response, str)
                or not response
                or any(
                    not isinstance(message, dict)
                    or message.get("role")
                    not in ("system", "user", "assistant", "tool")
                    or not isinstance(message.get("content"), str)
                    for message in context
                )
            ):
                raise ValueError(
                    "Logged contexts must contain typed chat messages and a response"
                )
            if getattr(self.tokenizer, "chat_template", None):
                encoded = last_assistant_token_mask(
                    self.tokenizer,
                    [*context, {"role": "assistant", "content": response}],
                )
            else:
                # Models without a chat template use an explicit plain-text
                # serialization. Encode the action separately so no prompt
                # token can acquire an assistant label through a BPE merge.
                roles = {
                    "system": "System",
                    "user": "User",
                    "assistant": "Assistant",
                    "tool": "Tool",
                }
                prompt = (
                    "".join(f"{roles[m['role']]}: {m['content']}\n" for m in context)
                    + "Assistant: "
                )
                prefix = list(
                    self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                )
                action = list(
                    self.tokenizer(response, add_special_tokens=False)["input_ids"]
                )
                eos = getattr(self.tokenizer, "eos_token_id", None)
                if eos is not None:
                    if type(eos) is not int or eos < 0:
                        raise ValueError("Offline policy requires an integer EOS token")
                    action.append(eos)
                encoded = (
                    prefix + action,
                    [False] * len(prefix) + [True] * len(action),
                )
            if encoded is None:
                raise ValueError(
                    "Tokenizer cannot faithfully mask logged assistant actions"
                )
            ids, mask = encoded
            if len(ids) > limit or len(ids) < 2 or not any(mask[1:]):
                raise ValueError("Offline action is empty or exceeds max_length")
            rows.append(
                {
                    "input_ids": ids,
                    "labels": [
                        token if keep else -100
                        for token, keep in zip(ids, mask, strict=True)
                    ],
                }
            )
        return rows

    def _setup_policy_optimizer(self) -> None:
        """Require an initialized trainable language model before policy extraction."""
        if self.config.offline_algorithm.lower() != "iql":
            raise ValueError(
                "Language-model policy extraction currently supports IQL only"
            )
        if self.model is None or not callable(getattr(self.model, "parameters", None)):
            raise ValueError(
                "Offline policy training requires an initialized language model"
            )
        for name in ("policy_temperature", "policy_max_weight"):
            value = getattr(self.config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.policy_optimizer is None:
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            if not parameters:
                raise ValueError("Offline policy has no trainable parameters")
            self.policy_optimizer = AdamW(
                parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                weight_decay=self.config.weight_decay,
            )

    async def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Extract an IQL policy using bounded advantage-weighted logged-token NLL.

        Only labels belonging to the current assistant action are supervised.
        This is offline advantage-weighted behavior cloning, not on-policy GRPO.
        """
        self._setup_policy_optimizer()
        if not self.value_pretrained or self._offline_learner is None:
            raise ValueError(
                "Pretrain IQL value functions before extracting the policy"
            )
        parameter = next(self.model.parameters())
        device = parameter.device
        ids = torch.as_tensor(batch["input_ids"], device=device)
        labels = torch.as_tensor(batch["labels"], device=device)
        if ids.dtype not in (torch.int32, torch.int64) or labels.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Policy token IDs and labels must be integers")
        if (
            ids.ndim != 2
            or ids.shape != labels.shape
            or ids.shape[1] < 2
            or torch.any(ids < 0)
        ):
            raise ValueError("Invalid policy token batch")
        mask = labels[:, 1:] != -100
        if not mask.any(dim=1).all() or torch.any((labels != -100) & (labels != ids)):
            raise ValueError(
                "Every action must have faithful supervised assistant tokens"
            )
        attention = torch.as_tensor(
            batch.get("attention_mask", torch.ones_like(ids)), device=device
        )
        if attention.shape != ids.shape or torch.any(
            (attention != 0) & (attention != 1)
        ):
            raise ValueError("Invalid policy attention mask")
        if torch.any((labels != -100) & (attention == 0)):
            raise ValueError("Padding cannot be supervised")
        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        if states.shape != (len(ids), self.config.state_dim) or actions.shape != (
            len(ids),
            self.config.action_dim,
        ):
            raise ValueError("Policy features do not align with token rows")
        if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
            raise ValueError("Policy features must be finite")
        learner = self._offline_learner
        with torch.no_grad():
            values = learner.value_net(states)
            q_values = torch.minimum(
                learner.q1(states, actions), learner.q2(states, actions)
            )
            advantages = (q_values - values).reshape(-1)
            if not torch.isfinite(advantages).all():
                raise ValueError("IQL advantages must be finite")
            weights = torch.exp(
                (advantages / self.config.policy_temperature).clamp(
                    max=math.log(self.config.policy_max_weight)
                )
            ).to(device)
        if not weights.any():
            raise ValueError("IQL policy weights have no usable learning signal")
        self.model.train()
        optimizer = self.policy_optimizer
        if optimizer is None:
            raise RuntimeError("Offline policy optimizer was not initialized")
        if self._grad_accum_step == 0:
            optimizer.zero_grad(set_to_none=True)
        logits = self.model(input_ids=ids.long(), attention_mask=attention).logits
        token_losses = F.cross_entropy(
            logits[:, :-1, :].float().transpose(1, 2),
            labels[:, 1:].long(),
            ignore_index=-100,
            reduction="none",
        )
        per_action = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
        loss = (weights * per_action).mean()
        if not torch.isfinite(loss):
            raise ValueError("Offline policy loss must be finite")
        accumulation = self.config.gradient_accumulation_steps
        if type(accumulation) is not int or accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self._grad_accum_step == 0:
            batch_size = self.config.batch_size
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError("batch_size must be positive")
            self._grad_accum_normalizer = batch_size * accumulation
        # Accumulate action sums at a fixed scale, then normalize by the actual
        # action count at commit. Averaging microbatch means overweights tails.
        (loss * (len(ids) / self._grad_accum_normalizer)).backward()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not any(p.grad is not None for p in parameters):
            raise ValueError("Offline policy loss did not produce gradients")
        self._grad_accum_step += 1
        self._grad_accum_actions += len(ids)
        committed = self._grad_accum_step == accumulation
        grad_norm = self._commit_policy_update() if committed else 0.0
        self.training_step += 1
        metrics = {
            "policy_loss": float(loss.detach()),
            "nll": float(per_action.mean().detach()),
            "advantage_mean": float(advantages.mean()),
            "policy_weight_mean": float(weights.mean()),
            "grad_norm": float(grad_norm),
            "optimizer_updates": float(committed),
            "training_step": float(self.training_step),
            "supervised_tokens": float(mask.sum()),
        }
        self.training_metrics.append(metrics)
        await notify_step_end(self.callbacks, step=self.training_step, metrics=metrics)
        return metrics

    def _commit_policy_update(self) -> float:
        optimizer = self.policy_optimizer
        if (
            optimizer is None
            or self._grad_accum_step < 1
            or self._grad_accum_actions < 1
            or self._grad_accum_normalizer < 1
        ):
            raise RuntimeError("Cannot commit an empty offline policy update")
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        factor = self._grad_accum_normalizer / self._grad_accum_actions
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
        try:
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            self.optimizer_updates += 1
            return float(norm)
        finally:
            optimizer.zero_grad(set_to_none=True)
            self._grad_accum_step = 0
            self._grad_accum_actions = 0
            self._grad_accum_normalizer = 0

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
        self, num_epochs: int | None = None, dataset: Any = None
    ) -> dict[str, Any]:
        """Fit the language-model policy to aligned, real logged IQL transitions."""
        dataset = self.dataset if dataset is None else dataset
        epochs = self.config.num_epochs if num_epochs is None else num_epochs
        if type(epochs) is not int or epochs < 1:
            raise ValueError("num_epochs must be a positive integer")
        batch_size = self.config.batch_size
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._setup_policy_optimizer()
        features = self._validate_offline_data(dataset.to_offline_rl_format())
        converter = getattr(dataset, "to_policy_training_format", None)
        rows = (
            converter(
                self.tokenizer,
                self.config.max_prompt_length + self.config.max_completion_length,
            )
            if callable(converter)
            else self._logged_policy_rows(features)
        )
        if not rows or len(rows) != len(features["states"]):
            raise ValueError("Logged policy tokens and transition features must align")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Policy rows must be dictionaries")
            ids, labels = row.get("input_ids"), row.get("labels")
            if (
                not isinstance(ids, list)
                or not isinstance(labels, list)
                or len(ids) != len(labels)
                or len(ids) < 2
                or any(type(token) is not int or token < 0 for token in ids)
                or any(
                    type(label) is not int or label not in (-100, token)
                    for token, label in zip(ids, labels, strict=True)
                )
                or not any(label != -100 for label in labels[1:])
            ):
                raise ValueError(
                    "Policy rows require faithful integer tokens and nonempty assistant labels"
                )
        digest = hashlib.sha256(
            json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        )
        for name in sorted({"states", "actions", "rewards", "next_states", "dones"}):
            digest.update(name.encode())
            digest.update(str(features[name].shape).encode())
            digest.update(features[name].tobytes())
        dataset_sha256 = digest.hexdigest()
        if self.dataset_sha256 is not None and self.dataset_sha256 != dataset_sha256:
            raise ValueError("Offline training dataset differs from the checkpoint")
        self.dataset_sha256 = dataset_sha256
        if self._epoch_order is not None and (
            sorted(self._epoch_order) != list(range(len(rows)))
            or not 0 <= self._batch_offset <= len(rows)
        ):
            raise ValueError("Invalid offline checkpoint batch cursor")
        if not self.value_pretrained:
            await self.pretrain_value_functions(
                SimpleNamespace(to_offline_rl_format=lambda: features)
            )
        await notify_training_start(self.callbacks, trainer=self, config=self.config)
        for epoch in range(self.current_epoch, epochs):
            if self._epoch_order is None:
                self._epoch_order = self._rng.permutation(len(rows)).tolist()
                self._batch_offset = 0
            ordering = self._epoch_order
            metrics = self.training_metrics[-1] if self.training_metrics else {}
            for start in range(self._batch_offset, len(rows), batch_size):
                indices = ordering[start : start + batch_size]
                selected = [rows[int(index)] for index in indices]
                width = max(len(row["input_ids"]) for row in selected)
                pad = getattr(self.tokenizer, "pad_token_id", None)
                if pad is None:
                    pad = getattr(self.tokenizer, "eos_token_id", None)
                if type(pad) is not int or pad < 0:
                    raise ValueError("Tokenizer requires a valid padding or EOS token")
                ids = np.full((len(selected), width), pad, dtype=np.int64)
                labels = np.full_like(ids, -100)
                attention = np.zeros_like(ids)
                for index, row in enumerate(selected):
                    length = len(row["input_ids"])
                    ids[index, :length] = row["input_ids"]
                    labels[index, :length] = row["labels"]
                    attention[index, :length] = 1
                previous_step = self.training_step
                self._batch_offset = min(len(rows), start + batch_size)
                try:
                    metrics = await self.train_step(
                        {
                            "input_ids": ids,
                            "labels": labels,
                            "attention_mask": attention,
                            "states": features["states"][indices],
                            "actions": features["actions"][indices],
                        }
                    )
                except BaseException:
                    if self.training_step == previous_step:
                        self._batch_offset = start
                    raise
            if self._grad_accum_step:
                grad_norm = self._commit_policy_update()
                metrics = {
                    **metrics,
                    "grad_norm": grad_norm,
                    "optimizer_updates": 1.0,
                }
                if self.training_metrics:
                    self.training_metrics[-1] = metrics
            self.current_epoch = epoch + 1
            self._epoch_order = None
            self._batch_offset = 0
            await notify_episode_end(self.callbacks, episode=epoch, metrics=metrics)
        result = {
            "num_epochs": epochs,
            "optimizer_updates": self.optimizer_updates,
            "final_metrics": self.training_metrics[-1],
            "training_metrics": self.training_metrics,
        }
        await notify_training_end(self.callbacks, metrics=result)
        return result

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
                "policy_optimizer_state_dict": (
                    self.policy_optimizer.state_dict()
                    if self.policy_optimizer is not None
                    else None
                ),
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
                "current_epoch": self.current_epoch,
                "dataset_sha256": self.dataset_sha256,
                "epoch_order": self._epoch_order,
                "batch_offset": self._batch_offset,
                "optimizer_updates": self.optimizer_updates,
                "grad_accum_step": self._grad_accum_step,
                "policy_accumulation": {
                    "schema_version": 1,
                    "action_count": self._grad_accum_actions,
                    "normalizer": self._grad_accum_normalizer,
                },
                "gradients": (
                    {
                        name: parameter.grad
                        for name, parameter in self.model.named_parameters()
                    }
                    if self.model is not None
                    else {}
                ),
                "policy_state_dict": (
                    self.model.state_dict() if self.model is not None else None
                ),
                "iql_state": self._iql_state(),
                "rng": capture_rng(torch),
                "sampler_rng": self._rng.bit_generator.state,
            },
            path,
        )
        logger.info(f"Saved Offline GRPO trainer to {path}")

    def _iql_state(self) -> dict[str, Any] | None:
        if (
            self._offline_learner is None
            or self.config.offline_algorithm.lower() != "iql"
        ):
            return None
        return {
            name: getattr(self._offline_learner, name).state_dict()
            for name in (
                "q1",
                "q2",
                "value_net",
                "value_target",
                "q_optimizer",
                "value_optimizer",
            )
        }

    def save_checkpoint(self, path: str) -> None:
        """Atomically publish reusable policy weights and complete offline runtime state."""
        if self.model is None or self.policy_optimizer is None:
            raise ValueError("Cannot publish an offline policy without optimizer state")
        with checkpoint_write(Path(path)) as staging:
            self.save(str(staging / "training_state.pt"))
            if callable(getattr(self.model, "save_pretrained", None)):
                self.model.save_pretrained(staging)
            else:
                torch.save(self.model.state_dict(), staging / "pytorch_model.bin")
            if callable(getattr(self.tokenizer, "save_pretrained", None)):
                self.tokenizer.save_pretrained(staging)

    def load(
        self, path: str, *, trusted: bool = False, require_policy: bool = False
    ) -> None:
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

        saved_config = checkpoint.get("config", {})
        for name in (
            "model_name",
            "model_revision",
            "state_dim",
            "action_dim",
            "offline_algorithm",
        ):
            if saved_config.get(name) != getattr(self.config, name):
                raise ValueError(f"Offline checkpoint configuration differs: {name}")
        accumulation_state = self._checkpoint_accumulation(checkpoint)
        policy_state = checkpoint.get("policy_state_dict")
        if require_policy and policy_state is None:
            raise ValueError("Checkpoint has no resumable language-model policy")
        if policy_state is not None:
            self._setup_policy_optimizer()
            if (
                checkpoint.get("policy_optimizer_state_dict") is None
                or checkpoint.get("iql_state") is None
            ):
                raise ValueError(
                    "Offline policy checkpoint is missing optimizer or IQL state"
                )
            if self._offline_learner is None:
                self._init_offline_learner()
            if set(checkpoint["iql_state"]) != {
                "q1",
                "q2",
                "value_net",
                "value_target",
                "q_optimizer",
                "value_optimizer",
            }:
                raise ValueError("Offline learner checkpoint components are incomplete")
            self.model.load_state_dict(policy_state, strict=True)
            optimizer = self.policy_optimizer
            if optimizer is None:
                raise RuntimeError("Offline policy optimizer was not initialized")
            optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
            for name, state in checkpoint["iql_state"].items():
                if name not in {
                    "q1",
                    "q2",
                    "value_net",
                    "value_target",
                    "q_optimizer",
                    "value_optimizer",
                }:
                    raise ValueError("Unknown offline learner checkpoint component")
                getattr(self._offline_learner, name).load_state_dict(state)

        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.value_target.load_state_dict(checkpoint["value_target_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        policy_state = checkpoint.get("policy_optimizer_state_dict")
        if policy_state is not None and self.model is not None:
            self.policy_optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=float(getattr(self.config, "weight_decay", 0.0)),
            )
            self.policy_optimizer.load_state_dict(policy_state)
        self.training_step = checkpoint.get("training_step", 0)
        self.value_pretrained = checkpoint.get("value_pretrained", False)
        self.training_metrics = checkpoint.get("training_metrics", [])
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.dataset_sha256 = checkpoint.get("dataset_sha256")
        self._epoch_order = checkpoint.get("epoch_order")
        self._batch_offset = checkpoint.get("batch_offset", 0)
        self.optimizer_updates = checkpoint.get("optimizer_updates", 0)
        (
            self._grad_accum_step,
            self._grad_accum_actions,
            self._grad_accum_normalizer,
        ) = accumulation_state
        if self.model is not None:
            for name, parameter in self.model.named_parameters():
                gradient = checkpoint.get("gradients", {}).get(name)
                parameter.grad = (
                    None if gradient is None else gradient.to(parameter.device)
                )
        if "rng" in checkpoint:
            restore_rng(checkpoint["rng"], torch)
            self._rng.bit_generator.state = checkpoint["sampler_rng"]

        # Checkpoints store the config as a plain dict (see save()); rebuild it.
        config_payload = checkpoint.get("config")
        if isinstance(config_payload, dict):
            try:
                self.config = OfflineGRPOConfig(**config_payload)
            except (TypeError, ValueError) as exc:
                logger.warning("Failed to restore config from checkpoint: %s", exc)

        logger.info(f"Loaded Offline GRPO trainer from {path}")

    @staticmethod
    def _checkpoint_accumulation(checkpoint: dict[str, Any]) -> tuple[int, int, int]:
        """Validate weighted pending gradients before mutating model state."""
        steps = checkpoint.get("grad_accum_step", 0)
        if type(steps) is not int or steps < 0:
            raise ValueError("Invalid offline checkpoint accumulation step")
        state = checkpoint.get("policy_accumulation")
        if state is None:
            if steps:
                raise ValueError(
                    "Legacy offline checkpoint has pending gradients without action "
                    "counts; resume from a completed optimizer update instead"
                )
            return 0, 0, 0
        if (
            not isinstance(state, dict)
            or set(state) != {"schema_version", "action_count", "normalizer"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
        ):
            raise ValueError("Invalid offline checkpoint policy accumulation schema")
        actions, normalizer = state["action_count"], state["normalizer"]
        accumulation = checkpoint.get("config", {}).get("gradient_accumulation_steps")
        if (
            type(actions) is not int
            or type(normalizer) is not int
            or type(accumulation) is not int
            or not 0 <= steps < accumulation
            or actions < steps
            or (steps == 0 and (actions != 0 or normalizer != 0))
            or (steps > 0 and normalizer < 1)
        ):
            raise ValueError("Invalid offline checkpoint action accumulation state")
        if steps:
            gradients = checkpoint.get("gradients")
            if not isinstance(gradients, dict) or not any(
                value is not None for value in gradients.values()
            ):
                raise ValueError(
                    "Offline checkpoint is missing pending policy gradients"
                )
        return steps, actions, normalizer
