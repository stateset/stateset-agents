"""
Neural Reward Model Training for GRPO Agent Framework

This module implements sophisticated neural reward models that learn from
trajectory data rather than using hand-crafted features, embodying the
principles of the "Bitter Lesson" - computation over human knowledge.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from stateset_agents.core.reward import RewardFunction, RewardResult
from stateset_agents.core.trajectory import Trajectory

try:
    from utils.cache import CacheService  # optional utility
except Exception:  # pragma: no cover
    CacheService = None  # type: ignore
from utils.monitoring import MonitoringService

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryData:
    """Data structure for trajectory-based learning"""

    id: str
    prompt: str
    response: str
    raw_reward: float
    learned_reward: float
    computational_cost: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "raw_reward": self.raw_reward,
            "learned_reward": self.learned_reward,
            "computational_cost": self.computational_cost,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory, reward: float) -> "TrajectoryData":
        """Create from framework Trajectory object"""
        return cls(
            id=str(uuid.uuid4()),
            prompt=trajectory.get_prompt(),
            response=trajectory.get_last_response(),
            raw_reward=reward,
            learned_reward=0.0,
            computational_cost=len(trajectory.get_prompt())
            + len(trajectory.get_last_response()),
            timestamp=datetime.now(),
            metadata=trajectory.metadata,
        )


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory data"""

    def __init__(self, trajectories: List[TrajectoryData], tokenizer=None):
        self.trajectories = trajectories
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # Simple text embedding (in practice, use proper tokenization)
        prompt_emb = self._embed_text(traj.prompt)
        response_emb = self._embed_text(traj.response)

        return {
            "prompt_embedding": prompt_emb,
            "response_embedding": response_emb,
            "reward": torch.tensor(traj.raw_reward, dtype=torch.float32),
            "metadata": traj.metadata,
        }

    def _embed_text(self, text: str, max_length: int = 128) -> torch.Tensor:
        """Simple text embedding using hash-based method"""
        # In practice, use proper tokenization and embeddings
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to tensor
        embedding = (
            torch.from_numpy(
                np.frombuffer(hash_bytes, dtype=np.uint8)[:max_length]
            ).float()
            / 255.0
        )

        # Pad if necessary
        if len(embedding) < max_length:
            padding = torch.zeros(max_length - len(embedding))
            embedding = torch.cat([embedding, padding])

        return embedding


class NeuralRewardModel(nn.Module):
    """
    Neural reward model that learns from trajectory data
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(
            embedding_dim * 2, hidden_dim
        )  # concat prompt + response

        # Hidden layers
        layers = []
        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU() if activation == "relu" else nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self, prompt_emb: torch.Tensor, response_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the reward model

        Args:
            prompt_emb: Prompt embeddings [batch_size, embedding_dim]
            response_emb: Response embeddings [batch_size, embedding_dim]

        Returns:
            Reward scores [batch_size, 1]
        """
        # Concatenate prompt and response embeddings
        combined = torch.cat([prompt_emb, response_emb], dim=-1)

        # Project to hidden dimension
        hidden = self.input_proj(combined)

        # Pass through hidden layers
        hidden = self.hidden_layers(hidden)

        # Output projection
        reward = self.output_proj(hidden)

        return reward

    def predict_reward(self, prompt: str, response: str) -> float:
        """Predict reward for a single prompt-response pair"""
        # Simple embedding (in practice, use proper tokenization)
        prompt_emb = self._embed_text(prompt).unsqueeze(0)
        response_emb = self._embed_text(response).unsqueeze(0)

        with torch.no_grad():
            reward = self.forward(prompt_emb, response_emb)
            return reward.item()

    def _embed_text(self, text: str, max_length: int = 128) -> torch.Tensor:
        """Simple text embedding"""
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        embedding = (
            torch.from_numpy(
                np.frombuffer(hash_bytes, dtype=np.uint8)[:max_length]
            ).float()
            / 255.0
        )

        if len(embedding) < max_length:
            padding = torch.zeros(max_length - len(embedding))
            embedding = torch.cat([embedding, padding])

        return embedding


class NeuralRewardTrainer:
    """
    Trainer for neural reward models
    """

    def __init__(
        self,
        model: Optional[NeuralRewardModel] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        per_device_train_batch_size: Optional[int] = None,
        max_epochs: int = 100,
        patience: int = 10,
        device: str = "cpu",
        cache_service: Optional[CacheService] = None,
        monitoring_service: Optional[MonitoringService] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralRewardTrainer")

        self.model = model or NeuralRewardModel()
        self.learning_rate = learning_rate
        # Accept HF-style alias per_device_train_batch_size
        self.batch_size = per_device_train_batch_size or batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Services
        self.cache = cache_service
        self.monitoring = monitoring_service

        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Experience replay
        self.replay_buffer = deque(maxlen=10000)

    def add_trajectory(self, trajectory_data: TrajectoryData):
        """Add trajectory to replay buffer"""
        self.replay_buffer.append(trajectory_data)

    def add_trajectories(self, trajectories: List[TrajectoryData]):
        """Add multiple trajectories to replay buffer"""
        self.replay_buffer.extend(trajectories)

    def train(
        self,
        train_trajectories: List[TrajectoryData],
        val_trajectories: Optional[List[TrajectoryData]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the neural reward model

        Args:
            train_trajectories: Training trajectory data
            val_trajectories: Validation trajectory data
            verbose: Whether to print training progress

        Returns:
            Training results and metrics
        """
        if not train_trajectories:
            raise ValueError("No training trajectories provided")

        # Create datasets
        train_dataset = TrajectoryDataset(train_trajectories)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if val_trajectories:
            val_dataset = TrajectoryDataset(val_trajectories)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{self.max_epochs}, Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

            # Monitor metrics
            if self.monitoring:
                asyncio.create_task(
                    self.monitoring.log_metric(
                        "neural_reward.train_loss",
                        train_loss,
                        tags={"epoch": epoch + 1},
                    )
                )
                if val_loss is not None:
                    asyncio.create_task(
                        self.monitoring.log_metric(
                            "neural_reward.val_loss",
                            val_loss,
                            tags={"epoch": epoch + 1},
                        )
                    )

        # Final results
        results = {
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "best_val_loss": self.best_val_loss if val_trajectories else None,
            "epochs_trained": len(self.train_losses),
            "early_stopped": self.patience_counter >= self.patience,
        }

        return results

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Move to device
            prompt_emb = batch["prompt_embedding"].to(self.device)
            response_emb = batch["response_embedding"].to(self.device)
            rewards = batch["reward"].to(self.device)

            # Forward pass
            predicted_rewards = self.model(prompt_emb, response_emb).squeeze()
            loss = self.criterion(predicted_rewards, rewards)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                prompt_emb = batch["prompt_embedding"].to(self.device)
                response_emb = batch["response_embedding"].to(self.device)
                rewards = batch["reward"].to(self.device)

                # Forward pass
                predicted_rewards = self.model(prompt_emb, response_emb).squeeze()
                loss = self.criterion(predicted_rewards, rewards)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train_from_replay_buffer(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Train from replay buffer"""
        if len(self.replay_buffer) < sample_size:
            sample_size = len(self.replay_buffer)

        if sample_size == 0:
            return {"error": "No trajectories in replay buffer"}

        # Sample trajectories
        indices = np.random.choice(len(self.replay_buffer), sample_size, replace=False)
        trajectories = [self.replay_buffer[i] for i in indices]

        # Split into train/val
        split_idx = int(0.8 * len(trajectories))
        train_trajectories = trajectories[:split_idx]
        val_trajectories = trajectories[split_idx:]

        # Train
        return self.train(train_trajectories, val_trajectories, verbose=False)

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
                "model_config": {
                    "embedding_dim": self.model.embedding_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "num_layers": self.model.num_layers,
                },
            },
            path,
        )

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return {
            "replay_buffer_size": len(self.replay_buffer),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.train_losses),
            "current_lr": self.optimizer.param_groups[0]["lr"],
        }


class NeuralRewardFunction(RewardFunction):
    """
    Reward function using a trained neural reward model
    """

    def __init__(
        self,
        model: Optional[NeuralRewardModel] = None,
        trainer: Optional[NeuralRewardTrainer] = None,
        weight: float = 1.0,
        update_frequency: int = 100,
        min_trajectories: int = 50,
        fallback_reward: float = 0.5,
    ):
        super().__init__(weight=weight)

        self.model = model or NeuralRewardModel()
        self.trainer = trainer or NeuralRewardTrainer(self.model)
        self.update_frequency = update_frequency
        self.min_trajectories = min_trajectories
        self.fallback_reward = fallback_reward

        # Tracking
        self.evaluation_count = 0
        self.update_count = 0
        self.last_update_performance = {}

    async def compute_reward(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """
        Compute reward using neural model

        Args:
            turns: List of conversation turns
            context: Optional context information

        Returns:
            RewardResult with neural reward score
        """
        try:
            # Extract prompt and response
            prompt = self._extract_prompt(turns)
            response = self._extract_response(turns)

            # Get neural reward
            reward = self.model.predict_reward(prompt, response)

            # Normalize to [0, 1]
            reward = max(0.0, min(1.0, reward))

            self.evaluation_count += 1

            # Periodic model updates
            if (
                self.evaluation_count % self.update_frequency == 0
                and len(self.trainer.replay_buffer) >= self.min_trajectories
            ):
                await self._update_model()

            return RewardResult(
                score=reward,
                breakdown={
                    "neural_reward": reward,
                    "model_updates": self.update_count,
                    "evaluation_count": self.evaluation_count,
                    "replay_buffer_size": len(self.trainer.replay_buffer),
                },
            )

        except Exception as e:
            logger.error(f"Neural reward computation failed: {e}")
            return RewardResult(
                score=self.fallback_reward,
                breakdown={
                    "neural_reward": self.fallback_reward,
                    "fallback_used": True,
                    "error": str(e),
                },
            )

    async def _update_model(self):
        """Update the neural model from replay buffer"""
        try:
            # Train from replay buffer
            results = self.trainer.train_from_replay_buffer()
            self.last_update_performance = results
            self.update_count += 1

            logger.info(
                f"Neural reward model updated. Train loss: {results.get('final_train_loss', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Neural model update failed: {e}")

    def add_trajectory_feedback(
        self,
        turns: List[Dict[str, Any]],
        reward: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Add trajectory feedback for model training"""
        prompt = self._extract_prompt(turns)
        response = self._extract_response(turns)

        trajectory_data = TrajectoryData(
            id=str(uuid.uuid4()),
            prompt=prompt,
            response=response,
            raw_reward=reward,
            learned_reward=0.0,
            computational_cost=len(prompt) + len(response),
            timestamp=datetime.now(),
            metadata=context or {},
        )

        self.trainer.add_trajectory(trajectory_data)

    def _extract_prompt(self, turns: List[Dict[str, Any]]) -> str:
        """Extract prompt from conversation turns"""
        user_turns = [t for t in turns if t.get("role") == "user"]
        if user_turns:
            return user_turns[0].get("content", "")
        return ""

    def _extract_response(self, turns: List[Dict[str, Any]]) -> str:
        """Extract response from conversation turns"""
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]
        if assistant_turns:
            return assistant_turns[-1].get("content", "")
        return ""

    def get_model_info(self) -> Dict[str, Any]:
        """Get neural model information"""
        return {
            "model_type": "neural_reward",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "evaluation_count": self.evaluation_count,
            "update_count": self.update_count,
            "last_update_performance": self.last_update_performance,
            "trainer_metrics": self.trainer.get_metrics(),
        }


# Convenience functions
def create_neural_reward_function(
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    learning_rate: float = 1e-4,
    weight: float = 1.0,
    **kwargs,
) -> NeuralRewardFunction:
    """Create a neural reward function with default configuration"""
    model = NeuralRewardModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    trainer = NeuralRewardTrainer(model=model, learning_rate=learning_rate, **kwargs)

    return NeuralRewardFunction(model=model, trainer=trainer, weight=weight)


def create_large_neural_reward_function(
    weight: float = 1.0, **kwargs
) -> NeuralRewardFunction:
    """Create a large neural reward function for complex tasks"""
    return create_neural_reward_function(
        embedding_dim=256, hidden_dim=512, learning_rate=5e-5, weight=weight, **kwargs
    )
