"""
Batch-Constrained Q-Learning (BCQ) for Offline RL

BCQ constrains the action space to prevent out-of-distribution actions
by using a generative model to learn the behavior policy.

Reference: Fujimoto et al. "Off-Policy Deep Reinforcement Learning
without Exploration" (ICML 2019)
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
            "PyTorch is required for BCQ. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class BCQConfig:
    """Configuration for Batch-Constrained Q-Learning"""

    # Network architecture
    hidden_size: int = 256
    num_layers: int = 3
    activation: str = "relu"

    # VAE parameters
    latent_dim: int = 64
    vae_hidden_size: int = 256

    # BCQ-specific parameters
    action_threshold: float = 0.3  # Threshold for action perturbation
    phi: float = 0.05  # Perturbation scale
    lmbda: float = 0.75  # Soft clipping parameter for double Q-learning

    # Training parameters
    learning_rate: float = 3e-4
    vae_learning_rate: float = 3e-4
    discount_factor: float = 0.99
    soft_target_tau: float = 0.005
    target_update_frequency: int = 2

    # Optimization
    batch_size: int = 256
    max_grad_norm: float = 1.0
    num_action_samples: int = 10  # Number of actions to sample from VAE

    # Logging
    log_frequency: int = 100


class ConversationalVAE(nn.Module):
    """
    Variational Autoencoder for learning conversation action distributions.

    Encodes conversation state into a latent space and decodes to generate
    actions (response embeddings) that are likely under the behavior policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: state + action -> latent
        encoder_layers = []
        input_dim = state_dim + action_dim
        for i in range(num_layers):
            output_dim = hidden_size
            encoder_layers.append(nn.Linear(input_dim, output_dim))
            encoder_layers.append(nn.ReLU())
            input_dim = output_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.mean_layer = nn.Linear(hidden_size, latent_dim)
        self.log_var_layer = nn.Linear(hidden_size, latent_dim)

        # Decoder: state + latent -> action
        decoder_layers = []
        input_dim = state_dim + latent_dim
        for i in range(num_layers):
            output_dim = hidden_size if i < num_layers - 1 else action_dim
            decoder_layers.append(nn.Linear(input_dim, output_dim))
            if i < num_layers - 1:
                decoder_layers.append(nn.ReLU())
            input_dim = output_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def encode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state-action pair to latent distribution parameters"""
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        return mean, log_var

    def reparameterize(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent + state to action"""
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, sample, decode.

        Returns:
            reconstructed_action, mean, log_var
        """
        mean, log_var = self.encode(state, action)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(state, z)
        return reconstructed, mean, log_var

    def sample(
        self,
        state: torch.Tensor,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Sample actions from the learned behavior distribution.

        Args:
            state: [batch_size, state_dim]
            num_samples: Number of actions to sample per state

        Returns:
            actions: [batch_size, num_samples, action_dim]
        """
        batch_size = state.shape[0]

        # Expand state for multiple samples
        state_expanded = state.unsqueeze(1).repeat(1, num_samples, 1)
        state_flat = state_expanded.view(batch_size * num_samples, -1)

        # Sample from prior
        z = torch.randn(batch_size * num_samples, self.latent_dim, device=state.device)

        # Decode
        actions = self.decode(state_flat, z)
        actions = actions.view(batch_size, num_samples, -1)

        return actions


class PerturbationNetwork(nn.Module):
    """
    Network that learns small perturbations to VAE-generated actions.

    The perturbation is constrained to keep actions within the
    support of the behavior policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        phi: float = 0.05,
    ):
        super().__init__()

        self.phi = phi
        self.action_dim = action_dim

        layers = []
        input_dim = state_dim + action_dim

        for i in range(num_layers):
            output_dim = hidden_size if i < num_layers - 1 else action_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            input_dim = output_dim

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perturbation and return perturbed action.

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]

        Returns:
            perturbed_action: [batch_size, action_dim]
        """
        x = torch.cat([state, action], dim=-1)
        perturbation = self.network(x)
        # Constrain perturbation with tanh and scale by phi
        perturbation = self.phi * torch.tanh(perturbation)
        return action + perturbation


class BCQQNetwork(nn.Module):
    """Q-function network for BCQ"""

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
        """Compute Q-value for state-action pair"""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class BatchConstrainedQLearning:
    """
    Batch-Constrained Q-Learning (BCQ) for offline RL.

    BCQ addresses the extrapolation error in offline RL by:
    1. Learning a generative model (VAE) of the behavior policy
    2. Only considering actions similar to the data distribution
    3. Using a perturbation network for fine-grained action selection

    Example:
        >>> bcq = BatchConstrainedQLearning(state_dim=384, action_dim=384)
        >>> for batch in dataset:
        ...     metrics = bcq.train_step(batch)
        >>> action = bcq.select_action(state)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[BCQConfig] = None,
        device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or BCQConfig()
        self.device = device

        # VAE for behavior cloning
        self.vae = ConversationalVAE(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=self.config.latent_dim,
            hidden_size=self.config.vae_hidden_size,
        ).to(device)

        # Perturbation network
        self.perturbation = PerturbationNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            phi=self.config.phi,
        ).to(device)

        # Q-networks (double Q-learning)
        self.q1 = BCQQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
        ).to(device)

        self.q2 = BCQQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
        ).to(device)

        # Target networks
        self.q1_target = BCQQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
        ).to(device)

        self.q2_target = BCQQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
        ).to(device)

        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.vae_optimizer = Adam(
            self.vae.parameters(), lr=self.config.vae_learning_rate
        )
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.config.learning_rate,
        )
        self.perturbation_optimizer = Adam(
            self.perturbation.parameters(), lr=self.config.learning_rate
        )

        self.training_step = 0

    def _compute_vae_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute VAE reconstruction + KL loss"""
        reconstructed, mean, log_var = self.vae(states, actions)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, actions)

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        total_loss = recon_loss + 0.5 * kl_loss

        metrics = {
            "vae_recon_loss": recon_loss.item(),
            "vae_kl_loss": kl_loss.item(),
            "vae_total_loss": total_loss.item(),
        }

        return total_loss, metrics

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Q-function loss with BCQ action selection"""
        with torch.no_grad():
            # Sample actions from VAE for next states
            next_actions_vae = self.vae.sample(
                next_states, self.config.num_action_samples
            )

            # Apply perturbation
            batch_size = next_states.shape[0]
            next_states_expanded = next_states.unsqueeze(1).repeat(
                1, self.config.num_action_samples, 1
            )
            next_states_flat = next_states_expanded.view(
                batch_size * self.config.num_action_samples, -1
            )
            next_actions_flat = next_actions_vae.view(
                batch_size * self.config.num_action_samples, -1
            )

            next_actions_perturbed = self.perturbation(
                next_states_flat, next_actions_flat
            )

            # Compute Q-values for all sampled actions
            q1_next = self.q1_target(next_states_flat, next_actions_perturbed)
            q2_next = self.q2_target(next_states_flat, next_actions_perturbed)

            # Weighted combination (lmbda for uncertainty)
            q_next = self.config.lmbda * torch.min(q1_next, q2_next) + (
                1 - self.config.lmbda
            ) * torch.max(q1_next, q2_next)

            # Reshape and take max over sampled actions
            q_next = q_next.view(batch_size, self.config.num_action_samples)
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

    def _compute_perturbation_loss(
        self,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute perturbation network loss (maximize Q)"""
        # Sample actions from VAE
        with torch.no_grad():
            sampled_actions = self.vae.sample(states, num_samples=1).squeeze(1)

        # Apply perturbation
        perturbed_actions = self.perturbation(states, sampled_actions)

        # Maximize Q-value
        q_value = self.q1(states, perturbed_actions)
        perturbation_loss = -q_value.mean()

        metrics = {
            "perturbation_loss": perturbation_loss.item(),
            "perturbed_q_mean": q_value.mean().item(),
        }

        return perturbation_loss, metrics

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one BCQ training step.

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

        # 1. Train VAE
        self.vae_optimizer.zero_grad()
        vae_loss, vae_metrics = self._compute_vae_loss(states, actions)
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.vae.parameters(), self.config.max_grad_norm
        )
        self.vae_optimizer.step()
        metrics.update(vae_metrics)

        # 2. Train Q-functions
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

        # 3. Train perturbation network
        self.perturbation_optimizer.zero_grad()
        perturbation_loss, perturbation_metrics = self._compute_perturbation_loss(
            states
        )
        perturbation_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.perturbation.parameters(), self.config.max_grad_norm
        )
        self.perturbation_optimizer.step()
        metrics.update(perturbation_metrics)

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
        num_samples: int = None,
    ) -> torch.Tensor:
        """
        Select action using BCQ policy.

        Args:
            state: [batch_size, state_dim] or [state_dim]
            num_samples: Number of actions to sample (default from config)

        Returns:
            Selected action [batch_size, action_dim] or [action_dim]
        """
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True

        num_samples = num_samples or self.config.num_action_samples

        with torch.no_grad():
            # Sample from VAE
            sampled_actions = self.vae.sample(state, num_samples)

            # Apply perturbation
            batch_size = state.shape[0]
            state_expanded = state.unsqueeze(1).repeat(1, num_samples, 1)
            state_flat = state_expanded.view(batch_size * num_samples, -1)
            actions_flat = sampled_actions.view(batch_size * num_samples, -1)

            perturbed_actions = self.perturbation(state_flat, actions_flat)

            # Select action with highest Q-value
            q1 = self.q1(state_flat, perturbed_actions)
            q1 = q1.view(batch_size, num_samples)

            best_idx = q1.argmax(dim=1)

            perturbed_actions = perturbed_actions.view(batch_size, num_samples, -1)
            best_actions = perturbed_actions[
                torch.arange(batch_size, device=state.device), best_idx
            ]

        if squeeze:
            best_actions = best_actions.squeeze(0)

        return best_actions

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
                "vae_state_dict": self.vae.state_dict(),
                "perturbation_state_dict": self.perturbation.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "config": self.config,
                "training_step": self.training_step,
            },
            path,
        )
        logger.info(f"Saved BCQ model to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.vae.load_state_dict(checkpoint["vae_state_dict"])
        self.perturbation.load_state_dict(checkpoint["perturbation_state_dict"])
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)

        logger.info(f"Loaded BCQ model from {path}")


class BCQTrainer:
    """
    Trainer wrapper for BCQ algorithm.

    Handles dataset loading, training loops, and evaluation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[BCQConfig] = None,
        device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.bcq = BatchConstrainedQLearning(state_dim, action_dim, config, device)
        self.config = config or BCQConfig()
        self.device = device
        self.training_metrics: List[Dict[str, float]] = []

    def train(
        self,
        dataset: Dict[str, np.ndarray],
        num_epochs: int = 100,
        batch_size: int = 256,
    ) -> List[Dict[str, float]]:
        """
        Train BCQ on offline dataset.

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
            f"Training BCQ on {dataset_size} samples for {num_epochs} epochs"
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

                metrics = self.bcq.train_step(
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
        self.bcq.save(path)

    def load(self, path: str) -> None:
        """Load trained model"""
        self.bcq.load(path)
