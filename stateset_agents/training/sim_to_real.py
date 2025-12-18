"""
Sim-to-Real Transfer for Conversational Agents

Provides system identification, simulator calibration, and progressive
transfer techniques for bridging the gap between simulated and real
conversation environments.
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
            "PyTorch is required for sim-to-real transfer. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class SimToRealConfig:
    """Configuration for sim-to-real transfer"""

    # System identification
    user_model_type: str = "learned"  # learned, rule_based, llm
    user_model_hidden_size: int = 256
    user_model_num_layers: int = 3

    # Progressive transfer schedule
    transfer_schedule: str = "cosine"  # linear, exponential, cosine, step
    initial_sim_ratio: float = 1.0  # Start with 100% simulation
    final_sim_ratio: float = 0.1  # End with 10% simulation
    transfer_steps: int = 10000
    warmup_steps: int = 1000

    # Domain adaptation
    adaptation_method: str = "dann"  # dann, mmd, coral, none
    domain_critic_hidden: int = 128
    gradient_reversal_lambda: float = 1.0

    # Calibration
    calibration_frequency: int = 1000
    calibration_threshold: float = 0.1

    # Monitoring
    gap_monitoring_frequency: int = 500
    early_stop_gap: float = 0.05

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 32

    # Embedding
    state_dim: int = 384
    action_dim: int = 384


class UserBehaviorModel(nn.Module):
    """
    Learns to predict user behavior from real conversation data.

    This model captures patterns in how real users respond, including:
    - Response style and length
    - Follow-up question patterns
    - Emotional reactions
    - Topic preferences
    """

    def __init__(
        self,
        state_dim: int = 384,
        action_dim: int = 384,
        hidden_size: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        _require_torch()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Encoder for conversation state
        encoder_layers = []
        input_dim = state_dim + action_dim  # State + agent response
        for i in range(num_layers):
            output_dim = hidden_size
            encoder_layers.append(nn.Linear(input_dim, output_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.1))
            input_dim = output_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # User response predictor (embedding space)
        self.response_predictor = nn.Linear(hidden_size, action_dim)

        # Response length predictor
        self.length_predictor = nn.Linear(hidden_size, 1)

        # Emotion predictor
        self.emotion_predictor = nn.Linear(hidden_size, 7)  # 7 emotion classes

        # Continuation probability
        self.continuation_predictor = nn.Linear(hidden_size, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        state: torch.Tensor,
        agent_response: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict user behavior given conversation state and agent response.

        Args:
            state: [batch, state_dim] - conversation state embedding
            agent_response: [batch, action_dim] - agent response embedding

        Returns:
            Dictionary with predictions:
            - response: [batch, action_dim] - predicted user response embedding
            - length: [batch, 1] - predicted response length
            - emotion: [batch, 7] - emotion distribution
            - continuation: [batch, 1] - probability of continuing conversation
        """
        x = torch.cat([state, agent_response], dim=-1)
        h = self.encoder(x)

        return {
            "response": self.response_predictor(h),
            "length": F.softplus(self.length_predictor(h)),
            "emotion": F.softmax(self.emotion_predictor(h), dim=-1),
            "continuation": torch.sigmoid(self.continuation_predictor(h)),
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for user behavior prediction"""
        losses = {}

        # Response embedding loss
        if "response" in targets:
            losses["response"] = F.mse_loss(
                predictions["response"], targets["response"]
            )

        # Length prediction loss
        if "length" in targets:
            losses["length"] = F.mse_loss(
                predictions["length"], targets["length"]
            )

        # Emotion classification loss
        if "emotion" in targets:
            losses["emotion"] = F.cross_entropy(
                predictions["emotion"], targets["emotion"]
            )

        # Continuation prediction loss
        if "continuation" in targets:
            losses["continuation"] = F.binary_cross_entropy(
                predictions["continuation"], targets["continuation"]
            )

        total_loss = sum(losses.values())
        metrics = {k: v.item() for k, v in losses.items()}
        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics

    def predict_user_response(
        self,
        state: torch.Tensor,
        agent_response: torch.Tensor,
    ) -> torch.Tensor:
        """Get predicted user response embedding"""
        with torch.no_grad():
            predictions = self.forward(state, agent_response)
            return predictions["response"]

    def compute_user_likelihood(
        self,
        state: torch.Tensor,
        agent_response: torch.Tensor,
        actual_response: torch.Tensor,
    ) -> float:
        """Compute likelihood of actual user response under the model"""
        with torch.no_grad():
            predictions = self.forward(state, agent_response)
            predicted = predictions["response"]

            # Use negative MSE as proxy for likelihood
            mse = F.mse_loss(predicted, actual_response)
            likelihood = torch.exp(-mse)
            return likelihood.item()


class DomainAdaptationModule(nn.Module):
    """
    Domain adaptation for bridging sim-to-real gap.

    Supports multiple adaptation methods:
    - DANN: Domain-Adversarial Neural Network
    - MMD: Maximum Mean Discrepancy
    - CORAL: Correlation Alignment
    """

    def __init__(self, config: SimToRealConfig):
        super().__init__()
        _require_torch()

        self.config = config
        self.method = config.adaptation_method

        # Domain discriminator for DANN
        if self.method == "dann":
            self.domain_critic = nn.Sequential(
                nn.Linear(config.state_dim, config.domain_critic_hidden),
                nn.ReLU(),
                nn.Linear(config.domain_critic_hidden, config.domain_critic_hidden),
                nn.ReLU(),
                nn.Linear(config.domain_critic_hidden, 1),
            )

    def domain_adversarial_loss(
        self,
        sim_features: torch.Tensor,
        real_features: torch.Tensor,
        lambda_: float = None,
    ) -> torch.Tensor:
        """
        Compute domain adversarial loss (DANN).

        The gradient reversal layer encourages features that
        confuse the domain discriminator.
        """
        lambda_ = lambda_ or self.config.gradient_reversal_lambda

        # Domain labels: 0 for sim, 1 for real
        sim_labels = torch.zeros(sim_features.shape[0], 1, device=sim_features.device)
        real_labels = torch.ones(real_features.shape[0], 1, device=real_features.device)

        # Discriminator predictions
        sim_preds = torch.sigmoid(self.domain_critic(sim_features))
        real_preds = torch.sigmoid(self.domain_critic(real_features))

        # Binary cross entropy loss
        sim_loss = F.binary_cross_entropy(sim_preds, sim_labels)
        real_loss = F.binary_cross_entropy(real_preds, real_labels)

        return lambda_ * (sim_loss + real_loss)

    def compute_mmd(
        self,
        sim_dist: torch.Tensor,
        real_dist: torch.Tensor,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy"""
        def rbf_kernel(x, y, sigma):
            xx = torch.sum(x ** 2, dim=-1, keepdim=True)
            yy = torch.sum(y ** 2, dim=-1, keepdim=True)
            xy = torch.mm(x, y.t())
            distances = xx - 2 * xy + yy.t()
            return torch.exp(-distances / (2 * sigma ** 2))

        k_ss = rbf_kernel(sim_dist, sim_dist, sigma)
        k_rr = rbf_kernel(real_dist, real_dist, sigma)
        k_sr = rbf_kernel(sim_dist, real_dist, sigma)

        n_s = sim_dist.shape[0]
        n_r = real_dist.shape[0]

        mmd = (k_ss.sum() / (n_s * n_s) +
               k_rr.sum() / (n_r * n_r) -
               2 * k_sr.sum() / (n_s * n_r))

        return mmd

    def compute_coral_loss(
        self,
        sim_features: torch.Tensor,
        real_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CORAL (Correlation Alignment) loss"""
        # Compute covariance matrices
        d = sim_features.shape[1]

        sim_mean = sim_features.mean(dim=0, keepdim=True)
        real_mean = real_features.mean(dim=0, keepdim=True)

        sim_centered = sim_features - sim_mean
        real_centered = real_features - real_mean

        sim_cov = torch.mm(sim_centered.t(), sim_centered) / (sim_features.shape[0] - 1)
        real_cov = torch.mm(real_centered.t(), real_centered) / (real_features.shape[0] - 1)

        # Frobenius norm of covariance difference
        coral_loss = torch.norm(sim_cov - real_cov, p="fro") ** 2 / (4 * d * d)

        return coral_loss

    def compute_adaptation_loss(
        self,
        sim_features: torch.Tensor,
        real_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute domain adaptation loss based on configured method"""
        if self.method == "dann":
            return self.domain_adversarial_loss(sim_features, real_features)
        elif self.method == "mmd":
            return self.compute_mmd(sim_features, real_features)
        elif self.method == "coral":
            return self.compute_coral_loss(sim_features, real_features)
        else:
            return torch.tensor(0.0, device=sim_features.device)


class SimToRealTransfer:
    """
    Orchestrates progressive transfer from simulation to real conversations.

    Pipeline:
    1. System identification: Learn user behavior model from real data
    2. Simulator calibration: Adjust simulator to match real distributions
    3. Progressive transfer: Gradually blend real data into training
    4. Continuous monitoring: Track sim-to-real gap metrics

    Example:
        >>> transfer = SimToRealTransfer(config, simulator, real_dataset)
        >>> user_model = await transfer.identify_user_model(real_data, num_steps=5000)
        >>> await transfer.calibrate_simulator(user_model)
        >>>
        >>> for step in range(10000):
        ...     batch = await transfer.get_mixed_batch(batch_size=32, step=step)
        ...     metrics = await trainer.train_step(batch)
    """

    def __init__(
        self,
        config: SimToRealConfig,
        simulator: Any = None,  # ConversationSimulator
        real_dataset: Any = None,  # ConversationDataset
        device: str = None,
    ):
        _require_torch()

        self.config = config
        self.simulator = simulator
        self.real_dataset = real_dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.user_model = UserBehaviorModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_size=config.user_model_hidden_size,
            num_layers=config.user_model_num_layers,
        ).to(self.device)

        self.domain_adapter = DomainAdaptationModule(config).to(self.device)

        # Optimizer
        self.optimizer = Adam(
            list(self.user_model.parameters()) +
            list(self.domain_adapter.parameters()),
            lr=config.learning_rate,
        )

        # Training state
        self.training_step = 0
        self.gap_history: List[Dict[str, float]] = []
        self.is_calibrated = False

    async def identify_user_model(
        self,
        real_data: Any = None,
        num_steps: int = 5000,
    ) -> UserBehaviorModel:
        """
        Train user behavior model on real conversation data.

        Args:
            real_data: ConversationDataset (uses self.real_dataset if None)
            num_steps: Number of training steps

        Returns:
            Trained UserBehaviorModel
        """
        real_data = real_data or self.real_dataset
        if real_data is None:
            raise ValueError("Real data required for user model identification")

        logger.info(f"Identifying user model for {num_steps} steps")

        # Prepare training data
        training_data = self._prepare_user_model_data(real_data)

        for step in range(num_steps):
            # Sample batch
            batch = self._sample_user_model_batch(training_data, self.config.batch_size)

            # Forward pass
            predictions = self.user_model(batch["states"], batch["agent_responses"])

            # Compute loss
            loss, metrics = self.user_model.compute_loss(predictions, batch["targets"])

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 500 == 0:
                logger.info(f"User model step {step}/{num_steps}: loss={metrics['total_loss']:.4f}")

        logger.info("User model identification complete")
        return self.user_model

    def _prepare_user_model_data(
        self,
        dataset: Any,
    ) -> List[Dict[str, Any]]:
        """Prepare data for user model training"""
        training_data = []

        for traj in dataset:
            for i in range(len(traj.turns) - 1):
                turn = traj.turns[i]
                next_turn = traj.turns[i + 1]

                # State: conversation up to current turn
                # Agent response: current assistant turn
                # Target: next user turn
                if turn.role == "assistant" and next_turn.role == "user":
                    training_data.append({
                        "state": np.random.randn(self.config.state_dim).astype(np.float32),  # Placeholder
                        "agent_response": np.random.randn(self.config.action_dim).astype(np.float32),
                        "user_response": np.random.randn(self.config.action_dim).astype(np.float32),
                        "length": len(next_turn.content.split()),
                        "continuation": 1.0 if i < len(traj.turns) - 2 else 0.0,
                    })

        return training_data

    def _sample_user_model_batch(
        self,
        data: List[Dict[str, Any]],
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch for user model training"""
        indices = np.random.choice(len(data), size=min(batch_size, len(data)), replace=False)
        samples = [data[i] for i in indices]

        states = torch.FloatTensor([s["state"] for s in samples]).to(self.device)
        agent_responses = torch.FloatTensor([s["agent_response"] for s in samples]).to(self.device)
        user_responses = torch.FloatTensor([s["user_response"] for s in samples]).to(self.device)
        lengths = torch.FloatTensor([[s["length"]] for s in samples]).to(self.device)
        continuations = torch.FloatTensor([[s["continuation"]] for s in samples]).to(self.device)

        return {
            "states": states,
            "agent_responses": agent_responses,
            "targets": {
                "response": user_responses,
                "length": lengths,
                "continuation": continuations,
            },
        }

    async def calibrate_simulator(
        self,
        user_model: UserBehaviorModel = None,
    ) -> Dict[str, float]:
        """
        Calibrate simulator using learned user model.

        Args:
            user_model: Trained user behavior model

        Returns:
            Calibration metrics
        """
        user_model = user_model or self.user_model

        if self.simulator is None:
            raise ValueError("Simulator required for calibration")

        logger.info("Calibrating simulator with user model")

        # Calibrate simulator with real data statistics
        if self.real_dataset is not None:
            metrics = await self.simulator.calibrate(self.real_dataset)
        else:
            metrics = {}

        self.is_calibrated = True
        return metrics

    def get_calibration_metrics(self) -> Dict[str, float]:
        """Get current calibration metrics"""
        if not self.is_calibrated:
            return {"error": "Not calibrated"}

        return {
            "is_calibrated": self.is_calibrated,
            "gap_history_length": len(self.gap_history),
            "latest_gap": self.gap_history[-1] if self.gap_history else {},
        }

    def get_sim_real_ratio(self, step: int) -> Tuple[float, float]:
        """
        Get current sim/real data ratio based on transfer schedule.

        Args:
            step: Current training step

        Returns:
            Tuple of (sim_ratio, real_ratio)
        """
        # Warmup period: pure simulation
        if step < self.config.warmup_steps:
            return 1.0, 0.0

        effective_step = step - self.config.warmup_steps
        progress = min(effective_step / self.config.transfer_steps, 1.0)

        if self.config.transfer_schedule == "linear":
            sim_ratio = self.config.initial_sim_ratio - progress * (
                self.config.initial_sim_ratio - self.config.final_sim_ratio
            )

        elif self.config.transfer_schedule == "exponential":
            decay_rate = 0.9999
            sim_ratio = max(
                self.config.final_sim_ratio,
                self.config.initial_sim_ratio * (decay_rate ** effective_step),
            )

        elif self.config.transfer_schedule == "cosine":
            sim_ratio = self.config.final_sim_ratio + 0.5 * (
                self.config.initial_sim_ratio - self.config.final_sim_ratio
            ) * (1 + math.cos(math.pi * progress))

        elif self.config.transfer_schedule == "step":
            # Step function: drop at specific points
            if progress < 0.33:
                sim_ratio = 1.0
            elif progress < 0.66:
                sim_ratio = 0.5
            else:
                sim_ratio = self.config.final_sim_ratio

        else:
            sim_ratio = self.config.initial_sim_ratio

        real_ratio = 1.0 - sim_ratio
        return sim_ratio, real_ratio

    async def get_mixed_batch(
        self,
        batch_size: int,
        step: int,
    ) -> Dict[str, Any]:
        """
        Get a mixed batch of simulated and real data.

        Args:
            batch_size: Total batch size
            step: Current training step

        Returns:
            Mixed batch dictionary
        """
        sim_ratio, real_ratio = self.get_sim_real_ratio(step)

        sim_size = int(batch_size * sim_ratio)
        real_size = batch_size - sim_size

        batch = {
            "sim_data": [],
            "real_data": [],
            "sim_ratio": sim_ratio,
            "real_ratio": real_ratio,
            "step": step,
        }

        # Get simulated data
        if sim_size > 0 and self.simulator is not None:
            for _ in range(sim_size):
                # Generate simulated trajectory
                state = await self.simulator.reset()
                batch["sim_data"].append({
                    "state": state,
                    "is_simulated": True,
                })

        # Get real data
        if real_size > 0 and self.real_dataset is not None:
            indices = np.random.choice(
                len(self.real_dataset), size=min(real_size, len(self.real_dataset)), replace=False
            )
            for idx in indices:
                traj = self.real_dataset[idx]
                batch["real_data"].append({
                    "trajectory": traj,
                    "is_simulated": False,
                })

        self.training_step = step
        return batch

    def compute_transfer_gap(self) -> Dict[str, float]:
        """Compute current sim-to-real transfer gap"""
        if self.simulator is None or self.real_dataset is None:
            return {"error": "Simulator and real data required"}

        gap_metrics = self.simulator.compute_sim_real_gap(self.real_dataset)
        self.gap_history.append(gap_metrics)

        return gap_metrics

    def should_stop_transfer(self) -> bool:
        """Check if transfer gap is sufficiently small to stop"""
        if not self.gap_history:
            return False

        latest_gap = self.gap_history[-1].get("overall_gap", 1.0)
        return latest_gap < self.config.early_stop_gap

    def save(self, path: str) -> None:
        """Save transfer state"""
        torch.save(
            {
                "user_model_state_dict": self.user_model.state_dict(),
                "domain_adapter_state_dict": self.domain_adapter.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "training_step": self.training_step,
                "gap_history": self.gap_history,
                "is_calibrated": self.is_calibrated,
            },
            path,
        )
        logger.info(f"Saved sim-to-real transfer state to {path}")

    def load(self, path: str) -> None:
        """Load transfer state"""
        checkpoint = torch.load(path, map_location=self.device)

        self.user_model.load_state_dict(checkpoint["user_model_state_dict"])
        self.domain_adapter.load_state_dict(checkpoint["domain_adapter_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.gap_history = checkpoint.get("gap_history", [])
        self.is_calibrated = checkpoint.get("is_calibrated", False)

        logger.info(f"Loaded sim-to-real transfer state from {path}")
