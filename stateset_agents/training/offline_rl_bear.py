"""
Bootstrapping Error Accumulation Reduction (BEAR) for Offline RL

BEAR uses support matching via Maximum Mean Discrepancy (MMD) to
constrain the learned policy to be close to the behavior policy.

Reference: Kumar et al. "Stabilizing Off-Policy Q-Learning via
Bootstrapping Error Reduction" (NeurIPS 2019)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.distributions import Normal
except ImportError:
    torch = None
    nn = None
    F = None
    Adam = None
    Normal = None

logger = logging.getLogger(__name__)


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for BEAR. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class BEARConfig:
    """Configuration for BEAR algorithm"""

    # Network architecture
    hidden_size: int = 256
    num_layers: int = 3
    activation: str = "relu"

    # Actor network
    actor_hidden_size: int = 256
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    # BEAR-specific parameters
    kernel_type: str = "laplacian"  # "laplacian" or "gaussian"
    mmd_sigma: float = 20.0  # Kernel bandwidth
    lagrange_threshold: float = 0.05  # Target MMD threshold
    num_samples_for_mmd: int = 10  # Samples for MMD estimation
    use_automatic_alpha: bool = True  # Auto-tune Lagrange multiplier

    # Training parameters
    learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    discount_factor: float = 0.99
    soft_target_tau: float = 0.005
    target_update_frequency: int = 2

    # Optimization
    batch_size: int = 256
    max_grad_norm: float = 1.0
    warmup_steps: int = 40000  # Steps before using Lagrange multiplier

    # Logging
    log_frequency: int = 100


class MMDKernel:
    """
    Maximum Mean Discrepancy (MMD) kernel for distribution matching.

    MMD measures the distance between two distributions by comparing
    their embeddings in a reproducing kernel Hilbert space (RKHS).
    """

    def __init__(self, kernel_type: str = "laplacian", sigma: float = 20.0):
        self.kernel_type = kernel_type
        self.sigma = sigma

    def gaussian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """
        Gaussian (RBF) kernel: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

        Args:
            x: [batch, n_x, dim]
            y: [batch, n_y, dim]
            sigma: Kernel bandwidth

        Returns:
            kernel_values: [batch, n_x, n_y]
        """
        # Compute squared distances
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)  # [batch, n_x, 1]
        yy = torch.sum(y ** 2, dim=-1, keepdim=True)  # [batch, n_y, 1]
        xy = torch.bmm(x, y.transpose(-2, -1))  # [batch, n_x, n_y]

        distances = xx - 2 * xy + yy.transpose(-2, -1)
        return torch.exp(-distances / (2 * sigma ** 2))

    def laplacian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """
        Laplacian kernel: k(x, y) = exp(-||x - y|| / sigma)

        Args:
            x: [batch, n_x, dim]
            y: [batch, n_y, dim]
            sigma: Kernel bandwidth

        Returns:
            kernel_values: [batch, n_x, n_y]
        """
        # Compute L2 distances
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        yy = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy = torch.bmm(x, y.transpose(-2, -1))

        distances = torch.sqrt(
            torch.clamp(xx - 2 * xy + yy.transpose(-2, -1), min=1e-10)
        )
        return torch.exp(-distances / sigma)

    def compute_mmd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MMD between two sets of samples.

        MMD^2 = E[k(x, x')] + E[k(y, y')] - 2 * E[k(x, y)]

        Args:
            x: [batch, n_x, dim] - samples from distribution P
            y: [batch, n_y, dim] - samples from distribution Q

        Returns:
            mmd: [batch] - MMD values
        """
        if self.kernel_type == "gaussian":
            kernel = self.gaussian_kernel
        elif self.kernel_type == "laplacian":
            kernel = self.laplacian_kernel
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        k_xx = kernel(x, x, self.sigma)
        k_yy = kernel(y, y, self.sigma)
        k_xy = kernel(x, y, self.sigma)

        # Compute MMD^2 (unbiased estimator would exclude diagonal)
        n_x = x.shape[1]
        n_y = y.shape[1]

        mmd = (
            k_xx.sum(dim=(-2, -1)) / (n_x * n_x)
            + k_yy.sum(dim=(-2, -1)) / (n_y * n_y)
            - 2 * k_xy.sum(dim=(-2, -1)) / (n_x * n_y)
        )

        return mmd


class BEARQNetwork(nn.Module):
    """Q-function network for BEAR"""

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

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class BEARActor(nn.Module):
    """
    Gaussian actor network for BEAR.

    Outputs a diagonal Gaussian distribution over actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        # Shared layers
        layers = []
        input_dim = state_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        self.shared = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mean and log_std for action distribution.

        Args:
            state: [batch, state_dim]

        Returns:
            mean: [batch, action_dim]
            log_std: [batch, action_dim]
        """
        h = self.shared(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.

        Args:
            state: [batch, state_dim]
            num_samples: Number of samples per state

        Returns:
            actions: [batch, num_samples, action_dim]
            log_probs: [batch, num_samples]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        # Expand for multiple samples
        batch_size = state.shape[0]
        mean = mean.unsqueeze(1).repeat(1, num_samples, 1)
        std = std.unsqueeze(1).repeat(1, num_samples, 1)

        # Sample
        dist = Normal(mean, std)
        raw_actions = dist.rsample()
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)

        return raw_actions, log_probs

    def get_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probability of action under current policy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1, keepdim=True)


class ConversationalBEAR:
    """
    BEAR (Bootstrapping Error Accumulation Reduction) for offline RL.

    BEAR constrains the policy to stay close to the behavior policy by
    minimizing the Maximum Mean Discrepancy (MMD) between the learned
    policy and the data distribution.

    Key features:
    - Support matching via MMD (not density matching)
    - Lagrangian constraint optimization
    - Works with continuous action spaces

    Example:
        >>> bear = ConversationalBEAR(state_dim=384, action_dim=384)
        >>> for batch in dataset:
        ...     metrics = bear.train_step(batch)
        >>> action = bear.select_action(state)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[BEARConfig] = None,
        device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or BEARConfig()
        self.device = device

        # MMD kernel
        self.mmd_kernel = MMDKernel(
            kernel_type=self.config.kernel_type,
            sigma=self.config.mmd_sigma,
        )

        # Actor network
        self.actor = BEARActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.actor_hidden_size,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(device)

        # Q-networks (ensemble)
        self.q1 = BEARQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
        ).to(device)

        self.q2 = BEARQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
        ).to(device)

        # Target networks
        self.q1_target = BEARQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
        ).to(device)

        self.q2_target = BEARQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
        ).to(device)

        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Lagrange multiplier for MMD constraint
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        # Optimizers
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.config.learning_rate,
        )
        self.alpha_optimizer = Adam(
            [self.log_alpha], lr=self.config.alpha_learning_rate
        )

        self.training_step = 0

    def _compute_mmd_loss(
        self,
        policy_actions: torch.Tensor,
        data_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MMD loss between policy and data actions.

        Args:
            policy_actions: [batch, n_samples, action_dim] - samples from policy
            data_actions: [batch, 1, action_dim] - actions from dataset

        Returns:
            mmd_loss: [batch] - MMD values
        """
        # Expand data actions for comparison
        data_actions_expanded = data_actions.expand(
            -1, self.config.num_samples_for_mmd, -1
        )

        mmd = self.mmd_kernel.compute_mmd(policy_actions, data_actions_expanded)
        return mmd

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Q-function loss"""
        with torch.no_grad():
            # Sample next actions from policy
            next_actions, _ = self.actor.sample(
                next_states, num_samples=self.config.num_samples_for_mmd
            )

            # Get Q-values for all samples
            batch_size = next_states.shape[0]
            next_states_expanded = next_states.unsqueeze(1).repeat(
                1, self.config.num_samples_for_mmd, 1
            )
            next_states_flat = next_states_expanded.view(
                batch_size * self.config.num_samples_for_mmd, -1
            )
            next_actions_flat = next_actions.view(
                batch_size * self.config.num_samples_for_mmd, -1
            )

            q1_next = self.q1_target(next_states_flat, next_actions_flat)
            q2_next = self.q2_target(next_states_flat, next_actions_flat)
            q_next = torch.min(q1_next, q2_next)

            # Take max over samples
            q_next = q_next.view(batch_size, self.config.num_samples_for_mmd)
            q_next = q_next.max(dim=1, keepdim=True)[0]

            # Bellman target
            target_q = rewards + (1 - dones) * self.config.discount_factor * q_next

        # Current Q-values
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        # MSE loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_loss": q_loss.item(),
            "q1_mean": current_q1.mean().item(),
            "q2_mean": current_q2.mean().item(),
        }

        return q_loss, metrics

    def _compute_actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute actor loss with MMD constraint.

        Loss = -Q(s, a) + alpha * (MMD - threshold)
        """
        # Sample actions from policy
        policy_actions, log_probs = self.actor.sample(
            states, num_samples=self.config.num_samples_for_mmd
        )

        # Compute Q-values for policy actions
        batch_size = states.shape[0]
        states_expanded = states.unsqueeze(1).repeat(
            1, self.config.num_samples_for_mmd, 1
        )
        states_flat = states_expanded.view(
            batch_size * self.config.num_samples_for_mmd, -1
        )
        actions_flat = policy_actions.view(
            batch_size * self.config.num_samples_for_mmd, -1
        )

        q1 = self.q1(states_flat, actions_flat)
        q2 = self.q2(states_flat, actions_flat)
        q_values = torch.min(q1, q2).view(batch_size, self.config.num_samples_for_mmd)

        # Policy objective: maximize Q
        q_loss = -q_values.mean()

        # MMD constraint
        data_actions = actions.unsqueeze(1)  # [batch, 1, action_dim]
        mmd = self._compute_mmd_loss(policy_actions, data_actions)
        mmd_loss = mmd.mean()

        # Lagrangian
        alpha = torch.exp(self.log_alpha).detach()

        # Only apply constraint after warmup
        if self.training_step >= self.config.warmup_steps:
            actor_loss = q_loss + alpha * mmd_loss
        else:
            actor_loss = q_loss

        metrics = {
            "actor_q_loss": q_loss.item(),
            "mmd_loss": mmd_loss.item(),
            "actor_total_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "log_prob_mean": log_probs.mean().item(),
        }

        return actor_loss, mmd_loss, metrics

    def _update_alpha(self, mmd_loss: torch.Tensor) -> Dict[str, float]:
        """Update Lagrange multiplier for MMD constraint"""
        if not self.config.use_automatic_alpha:
            return {"alpha_loss": 0.0}

        if self.training_step < self.config.warmup_steps:
            return {"alpha_loss": 0.0}

        # alpha_loss = alpha * (mmd - threshold)
        alpha_loss = torch.exp(self.log_alpha) * (
            mmd_loss.detach() - self.config.lagrange_threshold
        )

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Clamp log_alpha to prevent instability
        with torch.no_grad():
            self.log_alpha.clamp_(-10, 10)

        return {"alpha_loss": alpha_loss.item()}

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one BEAR training step.

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size, 1]
            next_states: [batch_size, state_dim]
            dones: [batch_size, 1]

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        # 1. Update Q-functions
        self.q_optimizer.zero_grad()
        q_loss, q_metrics = self._compute_q_loss(
            states, actions, rewards, next_states, dones
        )
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.max_grad_norm,
        )
        self.q_optimizer.step()
        metrics.update(q_metrics)

        # 2. Update actor
        self.actor_optimizer.zero_grad()
        actor_loss, mmd_loss, actor_metrics = self._compute_actor_loss(states, actions)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm
        )
        self.actor_optimizer.step()
        metrics.update(actor_metrics)

        # 3. Update Lagrange multiplier
        alpha_metrics = self._update_alpha(mmd_loss)
        metrics.update(alpha_metrics)

        # 4. Update target networks
        if self.training_step % self.config.target_update_frequency == 0:
            self._soft_update_targets()

        self.training_step += 1
        metrics["training_step"] = self.training_step

        return metrics

    def _soft_update_targets(self):
        """Soft update target networks"""
        tau = self.config.soft_target_tau

        for target_param, param in zip(
            self.q1_target.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

        for target_param, param in zip(
            self.q2_target.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def select_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select action using BEAR policy.

        Args:
            state: [batch_size, state_dim] or [state_dim]
            deterministic: If True, use mean action

        Returns:
            Selected action
        """
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True

        with torch.no_grad():
            mean, log_std = self.actor(state)

            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                action = dist.sample()

        if squeeze:
            action = action.squeeze(0)

        return action

    def get_q_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> float:
        """Get Q-value for state-action pair"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)

            q1 = self.q1(state, action)
            q2 = self.q2(state, action)
            return torch.min(q1, q2).item()

    def save(self, path: str) -> None:
        """Save model checkpoint"""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha,
                "config": self.config,
                "training_step": self.training_step,
            },
            path,
        )
        logger.info(f"Saved BEAR model to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        self.log_alpha = checkpoint["log_alpha"]
        self.training_step = checkpoint.get("training_step", 0)

        logger.info(f"Loaded BEAR model from {path}")


class BEARTrainer:
    """
    Trainer wrapper for BEAR algorithm.

    Handles dataset loading, training loops, and evaluation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[BEARConfig] = None,
        device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.bear = ConversationalBEAR(state_dim, action_dim, config, device)
        self.config = config or BEARConfig()
        self.device = device
        self.training_metrics: List[Dict[str, float]] = []

    def train(
        self,
        dataset: Dict[str, np.ndarray],
        num_epochs: int = 100,
        batch_size: int = 256,
    ) -> List[Dict[str, float]]:
        """
        Train BEAR on offline dataset.

        Args:
            dataset: Dictionary with keys 'states', 'actions', 'rewards',
                    'next_states', 'dones'
            num_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            List of training metrics per epoch
        """
        states = torch.FloatTensor(dataset["states"]).to(self.device)
        actions = torch.FloatTensor(dataset["actions"]).to(self.device)
        rewards = torch.FloatTensor(dataset["rewards"]).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(dataset["next_states"]).to(self.device)
        dones = torch.FloatTensor(dataset["dones"]).unsqueeze(-1).to(self.device)

        dataset_size = states.shape[0]
        num_batches = dataset_size // batch_size

        logger.info(
            f"Training BEAR on {dataset_size} samples for {num_epochs} epochs"
        )

        for epoch in range(num_epochs):
            indices = torch.randperm(dataset_size)
            epoch_metrics = []

            for batch_idx in range(num_batches):
                batch_indices = indices[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_dones = dones[batch_indices]

                metrics = self.bear.train_step(
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_dones,
                )
                epoch_metrics.append(metrics)

            # Average metrics for epoch
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            avg_metrics["epoch"] = epoch
            self.training_metrics.append(avg_metrics)

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs}: "
                    + ", ".join(
                        [
                            f"{k}={v:.4f}"
                            for k, v in avg_metrics.items()
                            if k not in ("epoch", "training_step")
                        ]
                    )
                )

        return self.training_metrics

    def save(self, path: str) -> None:
        """Save trained model"""
        self.bear.save(path)

    def load(self, path: str) -> None:
        """Load trained model"""
        self.bear.load(path)
