"""
Offline Reinforcement Learning Algorithms

Implements Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL)
for learning from fixed datasets without online interaction.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
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
    min_q_weight: float = 5.0  # Weight for min Q value
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


@dataclass
class IQLConfig:
    """Configuration for Implicit Q-Learning"""

    # Network parameters
    hidden_size: int = 256
    num_layers: int = 3
    activation: str = "relu"

    # IQL-specific parameters
    expectile: float = 0.7  # Expectile for value learning (0.5 = mean, >0.5 = upper tail)
    temperature: float = 3.0  # Temperature for advantage weighting
    clip_score: Optional[float] = 100.0  # Clip importance weights

    # Training parameters
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    soft_target_tau: float = 0.005

    # Optimization
    batch_size: int = 256
    max_grad_norm: float = 1.0

    # Logging
    log_frequency: int = 100


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
        config: Optional[CQLConfig] = None,
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

    def _compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q1_values: torch.Tensor,
        q2_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CQL regularization loss.

        The key idea: penalize Q-values for random/unseen actions
        while preserving Q-values for dataset actions.
        """
        batch_size = states.shape[0]

        # Sample random actions
        random_actions = torch.FloatTensor(
            batch_size * self.config.num_random_actions,
            self.action_dim,
        ).uniform_(-1, 1).to(self.device)

        # Repeat states for random actions
        repeated_states = states.unsqueeze(1).repeat(1, self.config.num_random_actions, 1)
        repeated_states = repeated_states.view(batch_size * self.config.num_random_actions, -1)

        # Q-values for random actions
        q1_random = self.q1(repeated_states, random_actions)
        q2_random = self.q2(repeated_states, random_actions)

        # Reshape back
        q1_random = q1_random.view(batch_size, self.config.num_random_actions, 1)
        q2_random = q2_random.view(batch_size, self.config.num_random_actions, 1)

        # Logsumexp for soft maximum
        q1_logsumexp = torch.logsumexp(q1_random / self.config.temperature, dim=1)
        q2_logsumexp = torch.logsumexp(q2_random / self.config.temperature, dim=1)

        # CQL loss: penalize high Q-values for random actions
        # and encourage correct Q-values for dataset actions
        cql_loss_1 = (q1_logsumexp - q1_values).mean()
        cql_loss_2 = (q2_logsumexp - q2_values).mean()

        return cql_loss_1, cql_loss_2

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            rewards: [batch_size, 1]
            next_states: [batch_size, state_dim]
            dones: [batch_size, 1]

        Returns:
            Dictionary of training metrics
        """
        # Compute target Q-values
        with torch.no_grad():
            # For next states, we need next actions
            # In offline RL, we typically use dataset actions or policy actions
            # Here we'll use a simplified approach with current actions
            next_q1 = self.q1_target(next_states, actions)
            next_q2 = self.q2_target(next_states, actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.config.discount_factor * next_q

        # Current Q-values
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        # Bellman loss
        bellman_loss_1 = F.mse_loss(current_q1, target_q)
        bellman_loss_2 = F.mse_loss(current_q2, target_q)

        # CQL regularization
        cql_loss_1, cql_loss_2 = self._compute_cql_loss(
            states, actions, current_q1, current_q2
        )
        cql_loss = cql_loss_1 + cql_loss_2

        # Total loss
        if self.config.with_lagrange:
            alpha_tensor = torch.exp(self.log_alpha)
            alpha_loss = -alpha_tensor * (cql_loss.detach() - self.config.lagrange_threshold)
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
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.max_grad_norm)
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

    def _soft_update_target_networks(self):
        """Soft update of target networks"""
        tau = self.config.soft_target_tau

        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Get Q-value for state-action pair"""
        with torch.no_grad():
            q1 = self.q1(state.unsqueeze(0), action.unsqueeze(0))
            q2 = self.q2(state.unsqueeze(0), action.unsqueeze(0))
            return torch.min(q1, q2).item()


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
        config: Optional[IQLConfig] = None,
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
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.config.learning_rate)

        self.training_step = 0

    def _expectile_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        expectile: float,
    ) -> torch.Tensor:
        """
        Expectile regression loss.

        When expectile = 0.5, this is MSE loss.
        When expectile > 0.5, focuses on upper tail of distribution.
        """
        diff = target - predicted
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * (diff ** 2)).mean()

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one IQL training step.

        IQL training involves three steps:
        1. Update value function with expectile regression
        2. Update Q-functions with Bellman backup using value targets
        3. Policy extraction is implicit through advantage weighting
        """
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
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()

        # Step 2: Update Q-functions
        with torch.no_grad():
            next_v = self.value_target(next_states)
            target_q = rewards + (1 - dones) * self.config.discount_factor * next_v

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
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

    def _soft_update_target_network(self):
        """Soft update of target value network"""
        tau = self.config.soft_target_tau

        for target_param, param in zip(self.value_target.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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
            advantage = q - v
            return advantage.item()

    def get_policy_weight(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """
        Get policy weight using advantage-weighted regression.

        This provides an implicit policy for action selection.
        """
        advantage = self.get_advantage(state, action)

        # Advantage-weighted probability
        weight = torch.exp(advantage / self.config.temperature)

        if self.config.clip_score is not None:
            weight = torch.clamp(weight, max=self.config.clip_score)

        return weight.item()


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
        config: Optional[Union[CQLConfig, IQLConfig]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.algorithm = algorithm.lower()
        self.device = device

        if self.algorithm == "cql":
            self.learner = ConservativeQLearning(state_dim, action_dim, config, device)
        elif self.algorithm == "iql":
            self.learner = ImplicitQLearning(state_dim, action_dim, config, device)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'cql' or 'iql'")

        self.training_metrics: List[Dict[str, float]] = []

    def train(
        self,
        dataset: Dict[str, np.ndarray],
        num_epochs: int = 100,
        batch_size: int = 256,
    ) -> List[Dict[str, float]]:
        """
        Train on offline dataset.

        Args:
            dataset: Dictionary with keys 'states', 'actions', 'rewards',
                    'next_states', 'dones'
            num_epochs: Number of training epochs
            batch_size: Batch size for training

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

        logger.info(f"Training {self.algorithm.upper()} on {dataset_size} samples for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            epoch_metrics = []

            for batch_idx in range(num_batches):
                batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_dones = dones[batch_indices]

                metrics = self.learner.train_step(
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_dones,
                )

                epoch_metrics.append(metrics)

            # Average metrics for epoch
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics]) for key in epoch_metrics[0].keys()
            }
            avg_metrics["epoch"] = epoch

            self.training_metrics.append(avg_metrics)

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs}: "
                    + ", ".join([f"{k}={v:.4f}" for k, v in avg_metrics.items() if k != "epoch"])
                )

        return self.training_metrics

    def save(self, path: str) -> None:
        """Save trained model"""
        torch.save(
            {
                "algorithm": self.algorithm,
                "learner_state": {
                    k: v.state_dict() if hasattr(v, "state_dict") else v
                    for k, v in self.learner.__dict__.items()
                    if isinstance(v, nn.Module)
                },
                "training_metrics": self.training_metrics,
            },
            path,
        )
        logger.info(f"Saved {self.algorithm.upper()} model to {path}")

    def load(self, path: str) -> None:
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)

        for name, module in self.learner.__dict__.items():
            if isinstance(module, nn.Module) and name in checkpoint["learner_state"]:
                module.load_state_dict(checkpoint["learner_state"][name])

        logger.info(f"Loaded {self.algorithm.upper()} model from {path}")
