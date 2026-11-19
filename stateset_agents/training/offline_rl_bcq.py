"""
Batch-Constrained Q-Learning (BCQ) for Offline RL

BCQ constrains the action space to prevent out-of-distribution actions
by using a generative model to learn the behavior policy.

Reference: Fujimoto et al. "Off-Policy Deep Reinforcement Learning
without Exploration" (ICML 2019)
"""

from __future__ import annotations

import copy
import logging
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

    def __post_init__(self) -> None:
        """Require a nonempty proposal set and a convex target update."""
        validate_target_parameters(
            self.discount_factor,
            self.soft_target_tau,
            self.target_update_frequency,
            self.num_action_samples,
            self.lmbda,
        )


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
        for _i in range(num_layers):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        z = torch.randn(
            batch_size * num_samples,
            self.latent_dim,
            device=state.device,
            dtype=state.dtype,
        )

        # Decode
        actions = self.decode(state_flat, z)
        actions = actions.view(batch_size, num_samples, -1)

        return actions


class PerturbationNetwork(nn.Module):
    """
    Network that learns small perturbations to VAE-generated actions.

    Each action coordinate stays within phi of its VAE-generated proposal.
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
        config: BCQConfig | None = None,
        device: str = (
            "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        ),
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

        self.perturbation_target = copy.deepcopy(self.perturbation).requires_grad_(
            False
        )
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

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
    ) -> tuple[torch.Tensor, dict[str, float]]:
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Q-function loss with BCQ action selection"""
        target_q = sampled_twin_target(
            rewards,
            dones,
            next_states,
            self.config.discount_factor,
            lambda states: self.vae.sample(states, self.config.num_action_samples),
            self.q1_target,
            self.q2_target,
            min_weight=self.config.lmbda,
            perturb=self.perturbation_target,
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

    def _compute_perturbation_loss(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
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
    ) -> dict[str, float]:
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

        # 1. Train VAE
        self.vae_optimizer.zero_grad()
        vae_loss, vae_metrics = self._compute_vae_loss(states, actions)
        vae_loss.backward()
        clip_parameter_gradients(self.vae.parameters(), self.config.max_grad_norm)
        self.vae_optimizer.step()
        metrics.update(vae_metrics)

        # 2. Train Q-functions
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

        # 3. Train perturbation network
        self.perturbation_optimizer.zero_grad()
        perturbation_loss, perturbation_metrics = self._compute_perturbation_loss(
            states
        )
        perturbation_loss.backward()
        clip_parameter_gradients(
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

    def _soft_update_targets(self) -> None:
        """Update the policy and both critics from this completed optimizer step."""
        update_target_networks(
            (
                (self.perturbation_target, self.perturbation),
                (self.q1_target, self.q1),
                (self.q2_target, self.q2),
            ),
            self.config.soft_target_tau,
        )

    def select_action(
        self,
        state: torch.Tensor,
        num_samples: int | None = None,
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
            return float(torch.min(q1, q2).item())

    def save(self, path: str) -> None:
        """Save model checkpoint"""
        torch.save(
            {
                "vae_state_dict": self.vae.state_dict(),
                "perturbation_state_dict": self.perturbation.state_dict(),
                "perturbation_target_state_dict": self.perturbation_target.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "vae_optimizer": self.vae_optimizer.state_dict(),
                "perturbation_optimizer": self.perturbation_optimizer.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
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
        logger.info(f"Saved BCQ model to {path}")

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
        if isinstance(config, BCQConfig):
            config = asdict(config)
        if isinstance(config, dict):
            restored_config = BCQConfig(**config)
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

        self.vae.load_state_dict(checkpoint["vae_state_dict"])
        self.perturbation.load_state_dict(checkpoint["perturbation_state_dict"])
        self.perturbation_target.load_state_dict(
            checkpoint.get(
                "perturbation_target_state_dict", checkpoint["perturbation_state_dict"]
            )
        )
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)

        self.perturbation.phi = self.perturbation_target.phi = self.config.phi
        restore_optimizer(
            self.q_optimizer, checkpoint.get("q_optimizer"), self.config.learning_rate
        )
        restore_optimizer(
            self.vae_optimizer,
            checkpoint.get("vae_optimizer"),
            self.config.vae_learning_rate,
        )
        restore_optimizer(
            self.perturbation_optimizer,
            checkpoint.get("perturbation_optimizer"),
            self.config.learning_rate,
        )

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
        config: BCQConfig | None = None,
        device: str = (
            "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        ),
    ):
        _require_torch()

        self.bcq = BatchConstrainedQLearning(state_dim, action_dim, config, device)
        self.config = config or BCQConfig()
        self.device = device
        self.training_metrics: list[dict[str, float]] = []

    def train(
        self,
        dataset: dict[str, np.ndarray],
        num_epochs: int = 100,
        batch_size: int = 256,
    ) -> list[dict[str, float]]:
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
        self.training_metrics.extend(
            transition_epochs(
                dataset,
                self.bcq.train_step,
                discount=self.bcq.config.discount_factor,
                algorithm="bcq",
                state_dim=self.bcq.state_dim,
                action_dim=self.bcq.action_dim,
                device=self.device,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
        )
        return self.training_metrics

    def save(self, path: str) -> None:
        """Save trained model"""
        self.bcq.save(path)

    def load(self, path: str) -> None:
        """Load trained model"""
        self.bcq.load(path)
        self.config = self.bcq.config
