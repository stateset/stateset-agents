"""
Bootstrapping Error Accumulation Reduction (BEAR) for Offline RL

BEAR uses support matching via Maximum Mean Discrepancy (MMD) to
constrain the learned policy to be close to the behavior policy.

Reference: Kumar et al. "Stabilizing Off-Policy Q-Learning via
Bootstrapping Error Reduction" (NeurIPS 2019)
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np

from .checkpoint_io import load_checkpoint_file
from .offline_rl_batches import transition_batch, transition_epochs
from .offline_rl_targets import (
    critic_network,
    restore_optimizer,
    sampled_twin_target,
    update_target_networks,
    validate_target_parameters,
)
from .rl_losses import td_regression
from .trainer_utils import clip_parameter_gradients

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.distributions import Normal as _Normal
    from torch.optim import Adam as _Adam

    torch: Any = _torch
    nn: Any = _nn
    F: Any = _F
    Normal: Any = _Normal
    Adam: Any = _Adam
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

    def __post_init__(self) -> None:
        """Require a nonempty proposal set and a convex target update."""
        validate_target_parameters(
            self.discount_factor,
            self.soft_target_tau,
            self.target_update_frequency,
            self.num_samples_for_mmd,
        )


class _GaussianTail(torch.autograd.Function):
    """Recover tiny-bandwidth slopes after the forward kernel rounds to zero."""

    generate_vmap_rule = True

    @staticmethod
    def forward(difference, scaled, exponent, sigma):
        return exponent.exp()

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, scaled, exponent, sigma = inputs
        ctx.save_for_backward(scaled, exponent)
        ctx.sigma = sigma

    @staticmethod
    def backward(ctx, gradient):
        scaled, exponent = ctx.saved_tensors
        upstream = gradient.unsqueeze(-1)
        magnitude = (
            exponent.unsqueeze(-1)
            + scaled.abs().log()
            + upstream.abs().log()
            - math.log(ctx.sigma)
        ).exp()
        return -upstream.sign() * scaled.sign() * magnitude, None, None, None


class MMDKernel:
    """
    Maximum Mean Discrepancy (MMD) kernel for distribution matching.

    MMD measures the distance between two distributions by comparing
    their embeddings in a reproducing kernel Hilbert space (RKHS).
    """

    def __init__(self, kernel_type: str = "laplacian", sigma: float = 20.0):
        self.kernel_type = kernel_type
        self.sigma = sigma

    @staticmethod
    def _distances(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("BEAR kernel bandwidth must be finite and positive")
        if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0]:
            raise ValueError(
                "BEAR samples must be [batch, samples, features] with matching batches"
            )
        if not x.shape[1] or not y.shape[1]:
            raise ValueError("BEAR requires nonempty sample sets")
        if x.dtype in (torch.float16, torch.bfloat16):
            x, y = x.float(), y.float()
        # Pairwise subtraction preserves small distances under a common offset.
        # The direct norm also supplies a finite subgradient at coincident points.
        return torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")

    @staticmethod
    def _gaussian_parts(
        x: torch.Tensor, y: torch.Tensor, sigma: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a Gaussian kernel into a zero/one anchor and its residual."""
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian bandwidth must be finite and positive")
        if (
            x.ndim != 3
            or y.ndim != 3
            or x.shape[0] != y.shape[0]
            or x.shape[2] != y.shape[2]
            or not x.shape[2]
            or min(x.shape[1], y.shape[1]) == 0
        ):
            raise ValueError(
                "Gaussian kernel needs nonempty [batch, samples, features] arrays of matching width"
            )
        device = "cpu" if x.device.type == "mps" else x.device
        x = x.to(device=device, dtype=torch.float64)
        y = y.to(device=device, dtype=torch.float64)
        # At large bandwidths opposite-sign coordinates can overflow before
        # division, even though their normalized difference is small.
        if sigma >= 1.0:
            x, y, sigma = x * 0.5, y * 0.5, sigma * 0.5
        chunk = max(1, (1 << 18) // max(1, y.numel()))
        anchors, residuals = [], []
        for rows in x.split(chunk, dim=1):
            difference = rows.unsqueeze(2) - y.unsqueeze(1)
            # Beyond 64 bandwidths even dividing the Gaussian derivative by
            # the smallest binary64 bandwidth cannot recover a nonzero result.
            far = (difference.detach().abs() > 64 * sigma).any(dim=-1)
            scaled = difference.masked_fill(far.unsqueeze(-1), 0) / sigma
            exponent = -0.5 * scaled.square().sum(dim=-1)
            exponent = exponent.masked_fill(far, -math.inf)
            near = exponent > -math.log(2)
            residual = torch.where(near, exponent.expm1(), exponent.exp())
            if sigma < 1:
                # The inverse bandwidth can rescue a tiny derivative. Recover
                # its magnitude before rounding the product with the adjoint.
                tail = _GaussianTail.apply(
                    difference, scaled.detach(), exponent.detach(), sigma
                )
                residual = torch.where(exponent < -680, tail, residual)
            anchors.append(near)
            residuals.append(residual)
        return torch.cat(anchors, dim=1), torch.cat(residuals, dim=1)

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
        near, residual = self._gaussian_parts(x, y, sigma)
        dtype = torch.promote_types(
            torch.promote_types(x.dtype, y.dtype), torch.float32
        )
        return (near + residual).to(device=x.device, dtype=dtype)

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
        distances = self._distances(x, y, sigma)
        return torch.exp(-distances / sigma)

    def _gaussian_means(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split each kernel into a zero/one anchor and a small residual."""
        near, residual = self._gaussian_parts(x, y, self.sigma)
        return near.to(residual.dtype).mean((-2, -1)), residual.mean((-2, -1))

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
            dtype = torch.promote_types(
                torch.promote_types(x.dtype, y.dtype), torch.float32
            )
            xx, yy, xy = (
                self._gaussian_means(a, b) for a, b in ((x, x), (y, y), (x, y))
            )
            anchor = (xx[0] - xy[0]) + (yy[0] - xy[0])
            residual = (xx[1] - xy[1]) + (yy[1] - xy[1])
            return (anchor + residual).clamp_min(0).to(device=x.device, dtype=dtype)
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

            input_dim = output_dim

        self.network = critic_network(layers, activation)
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

        for _i in range(num_layers):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        config: BEARConfig | None = None,
        device: str = (
            "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        ),
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

        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

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
        # Repeating the single observed action leaves its empirical distribution
        # unchanged. Using it once avoids two redundant quadratic kernel matrices.
        return self.mmd_kernel.compute_mmd(policy_actions, data_actions)

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Q-function loss"""
        target_q = sampled_twin_target(
            rewards,
            dones,
            next_states,
            self.config.discount_factor,
            lambda states: self.actor_target.sample(
                states, self.config.num_samples_for_mmd
            )[0],
            self.q1_target,
            self.q2_target,
        )

        # Current Q-values
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        # MSE loss
        q1_loss = td_regression(current_q1, target_q)
        q2_loss = td_regression(current_q2, target_q)
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
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

    def _update_alpha(self, mmd_loss: torch.Tensor) -> dict[str, float]:
        """Update Lagrange multiplier for MMD constraint"""
        if not self.config.use_automatic_alpha:
            return {"alpha_loss": 0.0}

        if self.training_step < self.config.warmup_steps:
            return {"alpha_loss": 0.0}

        # The optimizer minimizes this loss. A violated constraint must raise alpha.
        alpha_loss = -torch.exp(self.log_alpha) * (
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
    ) -> dict[str, float]:
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
        rewards, dones = transition_batch(
            states,
            actions,
            rewards,
            next_states,
            dones,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discount=self.config.discount_factor,
        )
        metrics = {}

        # 1. Update Q-functions
        self.q_optimizer.zero_grad()
        q_loss, q_metrics = self._compute_q_loss(
            states, actions, rewards, next_states, dones
        )
        q_loss.backward()
        clip_parameter_gradients(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.max_grad_norm,
        )
        self.q_optimizer.step()
        metrics.update(q_metrics)

        # 2. Update actor
        self.actor_optimizer.zero_grad()
        actor_loss, mmd_loss, actor_metrics = self._compute_actor_loss(states, actions)
        actor_loss.backward()
        clip_parameter_gradients(self.actor.parameters(), self.config.max_grad_norm)
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

    def _soft_update_targets(self) -> None:
        """Update the policy and both critics from this completed optimizer step."""
        update_target_networks(
            (
                (self.actor_target, self.actor),
                (self.q1_target, self.q1),
                (self.q2_target, self.q2),
            ),
            self.config.soft_target_tau,
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
            return float(torch.min(q1, q2).item())

    def save(self, path: str) -> None:
        """Save model checkpoint"""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha,
                # Persist the config as plain data so checkpoints stay
                # loadable under torch.load(weights_only=True).
                "config": (
                    asdict(self.config)
                    if is_dataclass(self.config) and not isinstance(self.config, type)
                    else dict(getattr(self.config, "__dict__", {}))
                ),
                "training_step": self.training_step,
            },
            path,
        )
        logger.info(f"Saved BEAR model to {path}")

    def load(self, path: str, *, trusted: bool = False) -> None:
        """Load model checkpoint

        Args:
            path: Checkpoint written by :meth:`save` with matching network architecture.
            trusted: Pass ``True`` only for checkpoints from a source you
                control; the default unpickles with ``weights_only=True`` so a
                malicious checkpoint cannot execute code.
        """
        checkpoint = load_checkpoint_file(
            path, map_location=self.device, trusted=trusted
        )

        config = checkpoint.get("config")
        if isinstance(config, BEARConfig):
            config = asdict(config)
        if isinstance(config, dict):
            restored_config = BEARConfig(**config)
            if restored_config.activation != self.config.activation:
                for critic in (self.q1, self.q2, self.q1_target, self.q2_target):
                    linear_layers = (
                        layer
                        for layer in critic.network
                        if isinstance(layer, nn.Linear)
                    )
                    critic.network = critic_network(
                        linear_layers, restored_config.activation
                    )
            self.config = restored_config

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(
            checkpoint.get("actor_target_state_dict", checkpoint["actor_state_dict"])
        )
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])
        self.training_step = checkpoint.get("training_step", 0)

        for actor in (self.actor, self.actor_target):
            actor.log_std_min = self.config.log_std_min
            actor.log_std_max = self.config.log_std_max
        self.mmd_kernel = MMDKernel(self.config.kernel_type, self.config.mmd_sigma)
        restore_optimizer(
            self.q_optimizer, checkpoint.get("q_optimizer"), self.config.learning_rate
        )
        restore_optimizer(
            self.actor_optimizer,
            checkpoint.get("actor_optimizer"),
            self.config.actor_learning_rate,
        )
        restore_optimizer(
            self.alpha_optimizer,
            checkpoint.get("alpha_optimizer"),
            self.config.alpha_learning_rate,
        )

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
        config: BEARConfig | None = None,
        device: str = (
            "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        ),
    ):
        _require_torch()

        self.bear = ConversationalBEAR(state_dim, action_dim, config, device)
        self.config = config or BEARConfig()
        self.device = device
        self.training_metrics: list[dict[str, float]] = []

    def train(
        self,
        dataset: dict[str, np.ndarray],
        num_epochs: int = 100,
        batch_size: int = 256,
    ) -> list[dict[str, float]]:
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
        self.training_metrics.extend(
            transition_epochs(
                dataset,
                self.bear.train_step,
                discount=self.bear.config.discount_factor,
                algorithm="bear",
                state_dim=self.bear.state_dim,
                action_dim=self.bear.action_dim,
                device=self.device,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
        )
        return self.training_metrics

    def save(self, path: str) -> None:
        """Save trained model"""
        self.bear.save(path)

    def load(self, path: str) -> None:
        """Load trained model"""
        self.bear.load(path)
        self.config = self.bear.config
