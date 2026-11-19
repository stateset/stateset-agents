"""
Offline Reinforcement Learning Algorithms

Implements Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL)
for learning from fixed datasets without online interaction.
"""

from __future__ import annotations

import decimal
import logging
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from .checkpoint_io import load_checkpoint_file
from .offline_rl_batches import transition_batch, transition_epochs
from .offline_rl_targets import (
    successor_target,
    update_target_networks,
    validate_target_parameters,
)
from .rl_losses import td_regression
from .runtime_state import capture_rng, restore_rng
from .trainer_utils import clip_parameter_gradients

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.optim import Adam as _Adam

    torch: Any = _torch
    nn: Any = _nn
    F: Any = _F
    Adam: Any = _Adam
except ImportError:
    torch = None
    nn = None
    F = None
    Adam = None

logger = logging.getLogger(__name__)


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for offline RL. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class CQLConfig:
    """Configuration for Conservative Q-Learning"""

    # Q-function parameters
    hidden_size: int = 256
    num_layers: int = 3
    activation: str = "relu"

    # CQL-specific parameters
    cql_alpha: float = 1.0  # Weight for CQL regularization term
    min_q_weight: float = 5.0  # Legacy field; cql_alpha sets the fixed penalty weight
    temperature: float = 1.0  # Temperature for CQL penalty
    with_lagrange: bool = False  # Use Lagrangian formulation
    lagrange_threshold: float = 10.0  # Target for Lagrangian

    # Training parameters
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    target_update_frequency: int = 2
    soft_target_tau: float = 0.005

    # Optimization
    batch_size: int = 256
    num_random_actions: int = 10
    max_grad_norm: float = 1.0

    # Logging
    log_frequency: int = 100

    def __post_init__(self) -> None:
        """Require a well-defined soft maximum over a nonempty proposal set."""
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("CQL temperature must be finite and positive")
        validate_target_parameters(
            self.discount_factor,
            self.soft_target_tau,
            self.target_update_frequency,
            self.num_random_actions,
        )


@dataclass
class IQLConfig:
    """Configuration for Implicit Q-Learning"""

    # Network parameters
    hidden_size: int = 256
    num_layers: int = 3
    activation: str = "relu"

    # IQL-specific parameters
    expectile: float = (
        0.7  # Expectile for value learning (0.5 = mean, >0.5 = upper tail)
    )
    temperature: float = 3.0  # Temperature for advantage weighting
    clip_score: float | None = 100.0  # Clip importance weights

    # Training parameters
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    soft_target_tau: float = 0.005

    # Optimization
    batch_size: int = 256
    max_grad_norm: float = 1.0

    # Logging
    log_frequency: int = 100

    def __post_init__(self) -> None:
        """Require nonnegative expectile weights and a valid extraction temperature."""
        validate_target_parameters(self.discount_factor, self.soft_target_tau)
        if not 0 <= self.expectile <= 1:
            raise ValueError("IQL expectile must be between zero and one")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("IQL temperature must be finite and positive")
        if self.clip_score is not None and (
            not math.isfinite(self.clip_score) or self.clip_score < 0
        ):
            raise ValueError("IQL clip_score must be nonnegative and finite")


class QNetwork(nn.Module):
    """Q-function neural network"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network
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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for state-action pairs.

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]

        Returns:
            Q-values: [batch_size, 1]
        """
        x = torch.cat([states, actions], dim=-1)
        return self.network(x)


class ValueNetwork(nn.Module):
    """Value function network for IQL"""

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
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute value for states"""
        return self.network(states)


class _LargeTemperaturePartition(torch.autograd.Function):
    """Resolve a log partition when the temperature approaches the dtype limit."""

    @staticmethod
    def forward(q, baseline, tau, normalize):
        # This rare path must preserve both a difference larger than the dtype
        # maximum and a tiny residual after subtracting log N or the baseline.
        # In this regime every finite exponent q/tau has magnitude below 1024.
        shape = q.select(1, 0).shape
        rows = q.movedim(1, -1).reshape(-1, q.shape[1]).tolist()
        offsets = (
            [0.0] * len(rows)
            if baseline is None
            else baseline.expand(shape).reshape(-1).tolist()
        )
        info = torch.finfo(q.dtype)
        # Resolve one output subnormal after scaling the logarithm by tau,
        # independently of the calling application's decimal context.
        precision = (
            math.ceil(math.log10(tau) - math.log10(info.tiny) - math.log10(info.eps))
            + 24
        )
        with decimal.localcontext(decimal.Context(prec=precision)) as ctx:
            ctx.traps[decimal.InvalidOperation] = False
            temperature = decimal.Decimal.from_float(float(tau))
            values = []
            for row, offset in zip(rows, offsets, strict=True):
                total = sum(
                    (decimal.Decimal.from_float(x) / temperature).exp() for x in row
                )
                if normalize:
                    total /= len(row)
                value = temperature * total.ln() - decimal.Decimal.from_float(offset)
                values.append(float(value))
        return q.new_tensor(values).reshape(shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, baseline, ctx.tau, _ = inputs
        ctx.has_baseline = baseline is not None
        ctx.save_for_backward(q)
        ctx.save_for_forward(q)

    @staticmethod
    def _probabilities(q, tau):
        # The two factors fit the device dtype even when tau does not.
        scale = torch.finfo(q.dtype).max
        return ((q / scale) * (scale / tau)).softmax(dim=1)

    @staticmethod
    def backward(ctx, gradient):
        (q,) = ctx.saved_tensors
        probabilities = _LargeTemperaturePartition._probabilities(q, ctx.tau)
        return (
            gradient.unsqueeze(1) * probabilities,
            -gradient if ctx.has_baseline else None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, tangent, baseline_tangent, _tau, _normalize):
        (q,) = ctx.saved_tensors
        result = (_LargeTemperaturePartition._probabilities(q, ctx.tau) * tangent).sum(
            dim=1
        )
        return result if baseline_tangent is None else result - baseline_tangent


def _small_signed_log_mean(q: Any, tau: float, backup: Any) -> Any:
    """Retain the even term when small, opposite logits cancel in the mean."""
    # The ordinary graph already has the softmax derivative. Correct its
    # forward value only where adding the maximum canceled most of the result.
    values = q.detach()
    magnitude = values.abs().amax(dim=1)
    use = (
        (magnitude > 0)
        & (magnitude <= tau / 2)
        & (backup.detach().abs() <= magnitude * math.sqrt(torch.finfo(q.dtype).eps))
    )
    values = torch.where(use.unsqueeze(1), values, 0.0)
    z = values / tau
    half = z / 2
    cutoff = torch.finfo(q.dtype).eps / 2
    sinh_half = torch.where(half.abs() < cutoff, half, half.sinh())

    def ratio_at_zero(numerator, denominator):
        # Both analytic ratios round to one here, even for subnormal inputs.
        return torch.where(denominator.abs() < cutoff, 1.0, numerator / denominator)

    # exp(z)-1 = sinh(z) + 2*sinh(z/2)^2. Keep the two means
    # separate and put the factor tau into each term before the reduction.
    odd = values * ratio_at_zero(z.sinh(), z)
    even = values * ratio_at_zero(sinh_half, half) * sinh_half
    u = odd.mean(dim=1) + even.mean(dim=1)
    z_mean = u / tau
    stable = u * ratio_at_zero(z_mean.log1p(), z_mean)
    return torch.where(use, (backup - backup.detach()) + stable, backup)


class _SubnormalPartition(torch.autograd.Function):
    """Scaled log partition for temperatures below the normal float64 range."""

    @staticmethod
    def forward(centered, tau, normalize):
        partition = torch.logsumexp(centered / tau, dim=1)
        if normalize:
            partition = partition - math.log(centered.shape[1])
        return partition * tau

    @staticmethod
    def setup_context(ctx, inputs, output):
        centered, ctx.tau, _ = inputs
        ctx.save_for_backward(centered)
        ctx.save_for_forward(centered)

    @staticmethod
    def backward(ctx, gradient):
        # Analytically cancel the temperature before rounding. An intermediate
        # tau * probability can underflow even for a pair of tied maxima.
        (centered,) = ctx.saved_tensors
        probabilities = (centered / ctx.tau).softmax(dim=1)
        return gradient.unsqueeze(1) * probabilities, None, None

    @staticmethod
    def jvp(ctx, tangent, _tau, _normalize):
        (centered,) = ctx.saved_tensors
        return ((centered / ctx.tau).softmax(dim=1) * tangent).sum(dim=1)


class ConservativeQLearning:
    """
    Conservative Q-Learning (CQL) for offline RL.

    CQL adds a regularization term that penalizes Q-values for
    out-of-distribution actions, preventing overestimation.

    Reference: Kumar et al. "Conservative Q-Learning for Offline
    Reinforcement Learning" (NeurIPS 2020)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: CQLConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or CQLConfig()
        self.device = device

        # Q-networks (double Q-learning)
        self.q1 = QNetwork(
            state_dim,
            action_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
        ).to(device)

        self.q2 = QNetwork(
            state_dim,
            action_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
        ).to(device)

        # Target networks
        self.q1_target = QNetwork(
            state_dim,
            action_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
        ).to(device)

        self.q2_target = QNetwork(
            state_dim,
            action_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
        ).to(device)

        # Copy parameters to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=self.config.learning_rate)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=self.config.learning_rate)

        # Lagrange multiplier for automatic tuning
        if self.config.with_lagrange:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.learning_rate)

        self.training_step = 0

    def _bellman_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Clipped double-Q Bellman target from next-state actions.

        Uses the dataset's next actions when given (SARSA-style backup);
        otherwise a soft maximum over the uniform action proposals the CQL
        penalty uses, ``tau * (logsumexp(Q(s', a_j) / tau) - log N)``. The
        current action is never reused: ``Q(s', a_t)`` values repeating a_t.
        """

        def evaluate(
            states: torch.Tensor, actions: torch.Tensor | None = None
        ) -> torch.Tensor:
            if actions is not None:
                return torch.minimum(
                    self.q1_target(states, actions), self.q2_target(states, actions)
                )
            n = int(self.config.num_random_actions)
            proposals = states.new_empty(states.shape[0] * n, self.action_dim).uniform_(
                -1, 1
            )
            repeated = states.repeat_interleave(n, dim=0)
            q = torch.minimum(
                self.q1_target(repeated, proposals), self.q2_target(repeated, proposals)
            )
            return self._soft_max(q.reshape(states.shape[0], n, 1), normalize=True)

        features = () if next_actions is None else (next_actions,)
        return successor_target(
            rewards,
            dones,
            next_states,
            self.config.discount_factor,
            evaluate,
            *features,
        )

    def _soft_max(
        self, q: torch.Tensor, *, normalize: bool, baseline: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the soft maximum relative to a per-state baseline, if given."""
        tau = self.config.temperature
        result_dtype = torch.promote_types(q.dtype, torch.float32)
        limits = torch.finfo(result_dtype)
        if tau > limits.max / 1024:
            return _LargeTemperaturePartition.apply(
                q.to(result_dtype), baseline, tau, normalize
            )
        dtype = torch.float64 if tau < limits.tiny else result_dtype
        q = q.to(dtype)
        maximum = q.amax(dim=1, keepdim=True).detach()
        centered = q - maximum
        if tau < torch.finfo(torch.float64).tiny:
            correction = _SubnormalPartition.apply(centered, tau, normalize)
        else:
            scaled = centered / tau
            partition = torch.logsumexp(scaled, dim=1)
            if normalize:
                # At high temperature logsumexp rounds to log N before their
                # difference is taken. log1p(mean(expm1)) retains that difference.
                delta = scaled.expm1().mean(dim=1)
                partition = torch.where(
                    delta >= -0.5,
                    delta.clamp_min(-0.5).log1p(),
                    partition - math.log(q.shape[1]),
                )
            correction = tau * partition
        offset = maximum.squeeze(1)
        if baseline is not None:
            # Subtract before adding the partition so a large common Q offset
            # cannot round away the entire conservative penalty.
            offset = offset - baseline
        result = offset + correction
        if normalize and baseline is None:
            result = _small_signed_log_mean(q, tau, result)
        return result.to(result_dtype)

    def _compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q1_values: torch.Tensor,
        q2_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CQL regularization loss.

        The key idea: penalize Q-values for random/unseen actions
        while preserving Q-values for dataset actions.
        """
        batch_size = states.shape[0]

        # Sample random actions
        random_actions = (
            torch.FloatTensor(
                batch_size * self.config.num_random_actions,
                self.action_dim,
            )
            .uniform_(-1, 1)
            .to(self.device)
        )

        # Repeat states for random actions
        repeated_states = states.unsqueeze(1).repeat(
            1, self.config.num_random_actions, 1
        )
        repeated_states = repeated_states.view(
            batch_size * self.config.num_random_actions, -1
        )

        # Q-values for random actions
        q1_random = self.q1(repeated_states, random_actions)
        q2_random = self.q2(repeated_states, random_actions)

        # Reshape back
        q1_random = q1_random.view(batch_size, self.config.num_random_actions, 1)
        q2_random = q2_random.view(batch_size, self.config.num_random_actions, 1)

        return (
            self._soft_max(q1_random, normalize=False, baseline=q1_values).mean(),
            self._soft_max(q2_random, normalize=False, baseline=q2_values).mean(),
        )

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_actions: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Perform one training step.

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size, 1]
            next_states: [batch_size, state_dim]
            dones: [batch_size, 1]
            next_actions: Optional dataset actions taken in ``next_states``

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
            next_actions=next_actions,
        )
        target_q = self._bellman_target(rewards, next_states, dones, next_actions)

        # Current Q-values
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        # Bellman loss
        bellman_loss_1 = td_regression(current_q1, target_q)
        bellman_loss_2 = td_regression(current_q2, target_q)

        # CQL regularization
        cql_loss_1, cql_loss_2 = self._compute_cql_loss(
            states, actions, current_q1, current_q2
        )
        cql_loss = cql_loss_1 + cql_loss_2

        # Total loss
        if self.config.with_lagrange:
            alpha_tensor = torch.exp(self.log_alpha)
            alpha_loss = -alpha_tensor * (
                cql_loss.detach() - self.config.lagrange_threshold
            )
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = float(alpha_tensor.detach().item())
        else:
            alpha = float(self.config.cql_alpha)

        total_loss_1 = bellman_loss_1 + alpha * cql_loss_1
        total_loss_2 = bellman_loss_2 + alpha * cql_loss_2
        total_loss = total_loss_1 + total_loss_2

        # Optimize both Q networks from a single backward pass to avoid
        # in-place optimizer updates invalidating the autograd graph.
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        total_loss.backward()
        clip_parameter_gradients(self.q2.parameters(), self.config.max_grad_norm)
        clip_parameter_gradients(self.q1.parameters(), self.config.max_grad_norm)
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update target networks
        if self.training_step % self.config.target_update_frequency == 0:
            self._soft_update_target_networks()

        self.training_step += 1

        return {
            "bellman_loss": (bellman_loss_1 + bellman_loss_2).item() / 2,
            "cql_loss": cql_loss.item(),
            "total_loss": total_loss.item() / 2,
            "q1_mean": current_q1.mean().item(),
            "q2_mean": current_q2.mean().item(),
        }

    def _soft_update_target_networks(self) -> None:
        """Interpolate target parameters after the learner update."""
        update_target_networks(
            ((self.q1_target, self.q1), (self.q2_target, self.q2)),
            self.config.soft_target_tau,
        )

    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Get Q-value for state-action pair"""
        with torch.no_grad():
            q1 = self.q1(state.unsqueeze(0), action.unsqueeze(0))
            q2 = self.q2(state.unsqueeze(0), action.unsqueeze(0))
            return float(torch.min(q1, q2).item())


class ImplicitQLearning:
    """
    Implicit Q-Learning (IQL) for offline RL.

    IQL uses expectile regression to learn value functions without
    explicitly learning a policy, avoiding distribution shift issues.

    Reference: Kostrikov et al. "Offline Reinforcement Learning with
    Implicit Q-Learning" (ICLR 2022)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: IQLConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or IQLConfig()
        self.device = device

        # Q-networks (double Q)
        self.q1 = QNetwork(
            state_dim,
            action_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
        ).to(device)

        self.q2 = QNetwork(
            state_dim,
            action_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
        ).to(device)

        # Value network
        self.value_net = ValueNetwork(
            state_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
        ).to(device)

        # Target value network
        self.value_target = ValueNetwork(
            state_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
        ).to(device)

        self.value_target.load_state_dict(self.value_net.state_dict())

        # Optimizers
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.config.learning_rate,
        )
        self.value_optimizer = Adam(
            self.value_net.parameters(), lr=self.config.learning_rate
        )

        self.training_step = 0

    def _expectile_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        expectile: float,
    ) -> torch.Tensor:
        """
        Expectile regression loss.

        At expectile 0.5 the loss equals half the MSE.
        When expectile > 0.5, focuses on upper tail of distribution.
        """
        dtype = torch.promote_types(torch.float32, torch.result_type(predicted, target))
        diff = target.to(dtype) - predicted.to(dtype)
        if diff.numel() == 0:
            raise ValueError("IQL expectile loss requires a nonempty batch")
        upper = torch.scalar_tensor(expectile, dtype=dtype, device=diff.device)
        lower = torch.scalar_tensor(1 - expectile, dtype=dtype, device=diff.device)
        weight = torch.where(diff > 0, upper, lower)
        # Apply both scales before the final product. Squaring a residual or
        # summing unscaled squares can overflow despite a representable mean.
        return ((weight * diff) * (diff / diff.numel())).sum()

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        """
        Perform one IQL training step.

        IQL training involves three steps:
        1. Update value function with expectile regression
        2. Update Q-functions with Bellman backup using value targets
        3. Policy extraction is implicit through advantage weighting
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
        # Build the detached backup and validate row alignment before either optimizer moves.
        critic_target = successor_target(
            rewards, dones, next_states, self.config.discount_factor, self.value_target
        )
        # Step 1: Update value function
        with torch.no_grad():
            # Target Q-values for value learning
            target_q1 = self.q1(states, actions)
            target_q2 = self.q2(states, actions)
            target_q = torch.min(target_q1, target_q2)

        current_v = self.value_net(states)
        value_loss = self._expectile_loss(current_v, target_q, self.config.expectile)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        clip_parameter_gradients(self.value_net.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()

        # Step 2: Update Q-functions

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        q_loss = td_regression(current_q1, critic_target) + td_regression(
            current_q2, critic_target
        )

        self.q_optimizer.zero_grad()
        q_loss.backward()
        clip_parameter_gradients(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.max_grad_norm,
        )
        self.q_optimizer.step()

        # Step 3: Soft update target network
        self._soft_update_target_network()

        self.training_step += 1

        return {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "value_mean": current_v.mean().item(),
            "q_mean": current_q1.mean().item(),
        }

    def _soft_update_target_network(self) -> None:
        """Interpolate target parameters after the learner update."""
        update_target_networks(
            ((self.value_target, self.value_net),), self.config.soft_target_tau
        )

    def get_advantage(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """
        Get advantage for state-action pair.

        IQL uses advantages for implicit policy extraction.
        """
        with torch.no_grad():
            q = torch.min(
                self.q1(state.unsqueeze(0), action.unsqueeze(0)),
                self.q2(state.unsqueeze(0), action.unsqueeze(0)),
            )
            v = self.value_net(state.unsqueeze(0))
            # Subtract the two represented critic values at binary64 precision.
            # Opposite finite float32 values can have a difference beyond float32.
            return float(q.item()) - float(v.item())

    def get_policy_weight(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """
        Get policy weight using advantage-weighted regression.

        This provides an implicit policy for action selection.
        """
        advantage = self.get_advantage(state, action)

        # Advantage-weighted probability
        if not (math.isfinite(self.config.temperature) and self.config.temperature > 0):
            raise ValueError("IQL temperature must be positive and finite")
        log_weight = advantage / self.config.temperature
        cap = self.config.clip_score
        if cap is not None:
            if not math.isfinite(cap) or cap < 0:
                raise ValueError("IQL clip_score must be nonnegative and finite")
            if cap == 0 or log_weight >= math.log(cap):
                return float(cap)
        return math.exp(log_weight)


def learner_checkpoint(
    learner: ConservativeQLearning | ImplicitQLearning,
) -> dict[str, Any]:
    """Capture the trainable and frozen networks, optimizers and target cadence."""
    return {
        "version": 1,
        "algorithm": "cql" if isinstance(learner, ConservativeQLearning) else "iql",
        "state_dim": learner.state_dim,
        "action_dim": learner.action_dim,
        "config": asdict(learner.config),
        "learner_state": {
            k: v.state_dict()
            for k, v in vars(learner).items()
            if isinstance(v, nn.Module)
        },
        "optimizers": {
            k: v.state_dict()
            for k, v in vars(learner).items()
            if isinstance(v, torch.optim.Optimizer)
        },
        "log_alpha": (
            learner.log_alpha.detach().clone()
            if isinstance(learner, ConservativeQLearning)
            and learner.config.with_lagrange
            else None
        ),
        "training_step": learner.training_step,
        # The current-state bank is sampled on CPU. Successor proposals
        # are sampled on the target's device, including the Metal backend.
        "mps_proposal_rng": (
            torch.mps.get_rng_state()
            if isinstance(learner, ConservativeQLearning)
            and torch.device(learner.device).type == "mps"
            else None
        ),
    }


def restore_learner_checkpoint(
    saved: dict[str, Any],
    algorithm: str,
    state_dim: int,
    action_dim: int,
    device: str,
) -> ConservativeQLearning | ImplicitQLearning:
    """Build and validate a replacement learner before publishing its state."""
    if (
        type(saved.get("version")) is not int
        or saved["version"] != 1
        or saved.get("algorithm") != algorithm
        or (saved.get("state_dim"), saved.get("action_dim")) != (state_dim, action_dim)
    ):
        raise ValueError(
            "Offline learner checkpoint version, algorithm or dimensions differ"
        )
    step = saved.get("training_step")
    if type(step) is not int or step < 0:
        raise ValueError("Invalid offline learner checkpoint step")
    config = saved.get("config")
    kind = CQLConfig if algorithm == "cql" else IQLConfig
    if not isinstance(config, dict) or set(config) != {
        field.name for field in fields(kind)
    }:
        raise ValueError("Incomplete offline learner checkpoint configuration")
    # Linear layers initialize on CPU. Loading a snapshot must not itself draw
    # from the training stream, including when validation rejects a checkpoint.
    with torch.random.fork_rng(devices=[]):
        learner: ConservativeQLearning | ImplicitQLearning
        if algorithm == "cql":
            learner = ConservativeQLearning(
                state_dim, action_dim, CQLConfig(**config), device
            )
        elif algorithm == "iql":
            learner = ImplicitQLearning(
                state_dim, action_dim, IQLConfig(**config), device
            )
        else:
            raise ValueError("Unknown offline learner checkpoint algorithm")
    for key, component_type in (
        ("learner_state", nn.Module),
        ("optimizers", torch.optim.Optimizer),
    ):
        components = {
            k: v for k, v in vars(learner).items() if isinstance(v, component_type)
        }
        values = saved.get(key)
        if not isinstance(values, dict) or values.keys() != components.keys():
            raise ValueError(f"Incomplete offline learner checkpoint {key}")
        for name, component in components.items():
            component.load_state_dict(values[name])
    alpha = saved.get("log_alpha")
    if isinstance(learner, ConservativeQLearning) and learner.config.with_lagrange:
        if (
            alpha is None
            or not torch.is_tensor(alpha)
            or alpha.shape != learner.log_alpha.shape
        ):
            raise ValueError("Invalid CQL checkpoint multiplier")
        with torch.no_grad():
            learner.log_alpha.copy_(alpha)
    elif alpha is not None:
        raise ValueError("Unexpected CQL checkpoint multiplier")
    learner.training_step = step
    mps_rng = saved.get("mps_proposal_rng")
    if mps_rng is not None and torch.device(device).type == "mps":
        torch.mps.set_rng_state(mps_rng.cpu())
    return learner


class OfflineRLTrainer:
    """
    Unified trainer for offline RL algorithms.

    Handles data loading, training loops, and evaluation for CQL and IQL.
    """

    def __init__(
        self,
        algorithm: str,  # "cql" or "iql"
        state_dim: int,
        action_dim: int,
        config: CQLConfig | IQLConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.algorithm = algorithm.lower()
        self.device = device

        if self.algorithm == "cql":
            cql_config = config if isinstance(config, CQLConfig) else None
            self.learner: ConservativeQLearning | ImplicitQLearning = (
                ConservativeQLearning(state_dim, action_dim, cql_config, device)
            )
        elif self.algorithm == "iql":
            iql_config = config if isinstance(config, IQLConfig) else None
            self.learner = ImplicitQLearning(state_dim, action_dim, iql_config, device)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'cql' or 'iql'")

        self.training_metrics: list[dict[str, float]] = []

    def train(
        self,
        dataset: dict[str, np.ndarray],
        num_epochs: int = 100,
        batch_size: int = 256,
    ) -> list[dict[str, float]]:
        """
        Train on offline dataset.

        Args:
            dataset: Aligned states, actions, rewards, next_states, and dones.
                CQL also accepts logged next_actions for its Bellman target.
                Rewards and terminal flags may be vectors or single columns.
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            List of training metrics per epoch
        """
        self.training_metrics.extend(
            transition_epochs(
                dataset,
                self.learner.train_step,
                discount=self.learner.config.discount_factor,
                algorithm=self.algorithm,
                state_dim=self.learner.state_dim,
                action_dim=self.learner.action_dim,
                device=self.device,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
        )
        return self.training_metrics

    def save(self, path: str) -> None:
        """Save a completed learner step, optimizer history and sampling streams."""
        torch.save(
            {
                **learner_checkpoint(self.learner),
                "rng": capture_rng(torch),
                "training_metrics": self.training_metrics,
            },
            path,
        )
        logger.info(f"Saved {self.algorithm.upper()} model to {path}")

    def load(self, path: str, *, trusted: bool = False) -> None:
        """Load trained model

        Args:
            path: Checkpoint written by :meth:`save`.
            trusted: Pass ``True`` only for checkpoints from a source you
                control; the default unpickles with ``weights_only=True`` so a
                malicious checkpoint cannot execute code.
        """
        checkpoint = load_checkpoint_file(
            path, map_location=self.device, trusted=trusted
        )

        if checkpoint.get("algorithm") != self.algorithm:
            raise ValueError("Offline checkpoint algorithm differs")
        if "version" in checkpoint:
            from .ddp_checkpoint_state import _validate_rng

            if "rng" not in checkpoint:
                raise ValueError("Offline checkpoint is missing the proposal RNG state")
            _validate_rng(checkpoint["rng"], torch)
            restored = restore_learner_checkpoint(
                checkpoint,
                self.algorithm,
                self.learner.state_dim,
                self.learner.action_dim,
                self.device,
            )
            restore_rng(checkpoint["rng"], torch)
        else:
            # A weights-only checkpoint has no optimizer history, clock or RNG.
            # Use fresh optimizer state even when loading on a previously used trainer.
            with torch.random.fork_rng(devices=[]):
                restored = OfflineRLTrainer(
                    self.algorithm,
                    self.learner.state_dim,
                    self.learner.action_dim,
                    self.learner.config,
                    self.device,
                ).learner
            for name, module in vars(restored).items():
                if isinstance(module, nn.Module):
                    module.load_state_dict(checkpoint["learner_state"][name])
        self.learner = restored
        self.training_metrics = checkpoint.get("training_metrics", [])

        logger.info(f"Loaded {self.algorithm.upper()} model from {path}")
