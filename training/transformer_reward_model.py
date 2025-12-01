"""
Production-Grade Transformer-Based Reward Model

This module implements state-of-the-art neural reward models using transformer
embeddings for accurate reward prediction in conversational AI.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModel,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    TORCH_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False

from stateset_agents.core.reward import RewardFunction, RewardResult
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class RewardTrainingConfig:
    """Configuration for reward model training"""

    # Model configuration
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    hidden_dim: int = 768
    num_layers: int = 2
    dropout: float = 0.1

    # Training configuration
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Early stopping
    patience: int = 3
    min_delta: float = 1e-4

    # Data configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Device configuration
    device: str = "auto"  # auto, cuda, cpu

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RewardExample:
    """Training example for reward model"""

    prompt: str
    response: str
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_conversation_turns(
        cls,
        turns: List[ConversationTurn],
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RewardExample":
        """Create from conversation turns"""
        # Extract prompt (all user turns concatenated)
        prompt_parts = [t.content for t in turns if t.role == "user"]
        prompt = " ".join(prompt_parts)

        # Extract response (last assistant turn)
        response_parts = [t.content for t in turns if t.role == "assistant"]
        response = response_parts[-1] if response_parts else ""

        return cls(
            prompt=prompt,
            response=response,
            reward=reward,
            metadata=metadata or {},
        )


class RewardDataset(Dataset):
    """PyTorch Dataset for reward model training"""

    def __init__(
        self,
        examples: List[RewardExample],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Tokenize prompt and response
        prompt_encoding = self.tokenizer(
            example.prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        response_encoding = self.tokenizer(
            example.response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "prompt_input_ids": prompt_encoding["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_encoding["attention_mask"].squeeze(0),
            "response_input_ids": response_encoding["input_ids"].squeeze(0),
            "response_attention_mask": response_encoding["attention_mask"].squeeze(0),
            "reward": torch.tensor(example.reward, dtype=torch.float32),
            "metadata": example.metadata,
        }


class TransformerRewardModel(nn.Module):
    """
    Transformer-based reward model using pre-trained language models
    """

    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for TransformerRewardModel. "
                "Install with: pip install transformers"
            )

        # Load pre-trained transformer
        self.prompt_encoder = AutoModel.from_pretrained(base_model_name)
        self.response_encoder = AutoModel.from_pretrained(base_model_name)

        # Get embedding dimension from model
        embedding_dim = self.prompt_encoder.config.hidden_size

        # Reward head
        layers = []
        input_dim = embedding_dim * 2  # Concatenate prompt + response embeddings

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))

        self.reward_head = nn.Sequential(*layers)

        # Initialize reward head
        self._init_reward_head()

    def _init_reward_head(self):
        """Initialize reward head weights"""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the reward model

        Args:
            prompt_input_ids: Input IDs for prompt [batch_size, seq_len]
            prompt_attention_mask: Attention mask for prompt
            response_input_ids: Input IDs for response
            response_attention_mask: Attention mask for response

        Returns:
            Reward scores [batch_size, 1]
        """
        # Encode prompt
        prompt_outputs = self.prompt_encoder(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
        )
        prompt_embedding = self._mean_pooling(
            prompt_outputs.last_hidden_state, prompt_attention_mask
        )

        # Encode response
        response_outputs = self.response_encoder(
            input_ids=response_input_ids,
            attention_mask=response_attention_mask,
        )
        response_embedding = self._mean_pooling(
            response_outputs.last_hidden_state, response_attention_mask
        )

        # Concatenate embeddings
        combined = torch.cat([prompt_embedding, response_embedding], dim=-1)

        # Pass through reward head
        reward = self.reward_head(combined)

        return reward

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling of token embeddings"""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def freeze_encoders(self):
        """Freeze encoder weights (train only reward head)"""
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.response_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze encoder weights (train entire model)"""
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        for param in self.response_encoder.parameters():
            param.requires_grad = True


class TransformerRewardTrainer:
    """
    Trainer for transformer-based reward models
    """

    def __init__(
        self,
        config: Optional[RewardTrainingConfig] = None,
        model: Optional[TransformerRewardModel] = None,
    ):
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers are required. "
                "Install with: pip install torch transformers"
            )

        self.config = config or RewardTrainingConfig()
        self.device = torch.device(self.config.device)

        # Initialize model
        if model is None:
            self.model = TransformerRewardModel(
                base_model_name=self.config.base_model,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
            )
        else:
            self.model = model

        self.model.to(self.device)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(
        self,
        train_examples: List[RewardExample],
        val_examples: Optional[List[RewardExample]] = None,
        freeze_encoders: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the reward model

        Args:
            train_examples: Training examples
            val_examples: Validation examples (optional)
            freeze_encoders: If True, only train reward head
            verbose: Print training progress

        Returns:
            Training metrics and results
        """
        if not train_examples:
            raise ValueError("No training examples provided")

        # Optionally freeze encoders
        if freeze_encoders:
            self.model.freeze_encoders()
            if verbose:
                print("Training with frozen encoders (reward head only)")

        # Create datasets
        train_dataset = RewardDataset(
            train_examples, self.tokenizer, self.config.max_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if val_examples:
            val_dataset = RewardDataset(
                val_examples, self.tokenizer, self.config.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Initialize scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Logging
            if verbose:
                msg = f"Epoch {epoch + 1}/{self.config.num_epochs} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)

        return {
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "best_val_loss": self.best_val_loss if val_examples else None,
            "epochs_trained": len(self.train_losses),
            "early_stopped": self.patience_counter >= self.config.patience,
        }

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Move to device
            prompt_input_ids = batch["prompt_input_ids"].to(self.device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(self.device)
            response_input_ids = batch["response_input_ids"].to(self.device)
            response_attention_mask = batch["response_attention_mask"].to(self.device)
            rewards = batch["reward"].to(self.device)

            # Forward pass
            predicted_rewards = self.model(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                response_input_ids=response_input_ids,
                response_attention_mask=response_attention_mask,
            ).squeeze()

            # Compute loss
            loss = F.mse_loss(predicted_rewards, rewards)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                prompt_input_ids = batch["prompt_input_ids"].to(self.device)
                prompt_attention_mask = batch["prompt_attention_mask"].to(self.device)
                response_input_ids = batch["response_input_ids"].to(self.device)
                response_attention_mask = batch["response_attention_mask"].to(
                    self.device
                )
                rewards = batch["reward"].to(self.device)

                # Forward pass
                predicted_rewards = self.model(
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    response_input_ids=response_input_ids,
                    response_attention_mask=response_attention_mask,
                ).squeeze()

                # Compute loss
                loss = F.mse_loss(predicted_rewards, rewards)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def predict(self, prompt: str, response: str) -> float:
        """Predict reward for a single prompt-response pair"""
        self.model.eval()

        # Tokenize
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        response_encoding = self.tokenizer(
            response,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Predict
        with torch.no_grad():
            reward = self.model(
                prompt_input_ids=prompt_encoding["input_ids"],
                prompt_attention_mask=prompt_encoding["attention_mask"],
                response_input_ids=response_encoding["input_ids"],
                response_attention_mask=response_encoding["attention_mask"],
            )

        return reward.item()

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_state_dict") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Checkpoint loaded from {path}")


class LearnedRewardFunction(RewardFunction):
    """
    Reward function using a trained transformer reward model
    """

    def __init__(
        self,
        trainer: TransformerRewardTrainer,
        weight: float = 1.0,
        normalize: bool = True,
        cache_predictions: bool = True,
    ):
        super().__init__(weight=weight, name="LearnedRewardFunction")
        self.trainer = trainer
        self.normalize = normalize
        self.cache_predictions = cache_predictions
        self._prediction_cache = {}

    async def compute_reward(
        self,
        turns: List[ConversationTurn],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardResult:
        """Compute reward using learned model"""
        try:
            # Extract prompt and response
            prompt = " ".join(t.content for t in turns if t.role == "user")
            response_turns = [t.content for t in turns if t.role == "assistant"]
            response = response_turns[-1] if response_turns else ""

            # Check cache
            cache_key = f"{prompt}||{response}"
            if self.cache_predictions and cache_key in self._prediction_cache:
                reward = self._prediction_cache[cache_key]
            else:
                # Predict reward
                reward = self.trainer.predict(prompt, response)

                # Normalize to [0, 1] if requested
                if self.normalize:
                    reward = torch.sigmoid(torch.tensor(reward)).item()

                # Cache prediction
                if self.cache_predictions:
                    self._prediction_cache[cache_key] = reward

            return RewardResult(
                score=reward,
                breakdown={"learned_reward": reward},
                metadata={
                    "model_type": "transformer",
                    "base_model": self.trainer.config.base_model,
                },
            )

        except Exception as e:
            logger.error(f"Learned reward computation failed: {e}")
            return RewardResult(
                score=0.5,
                breakdown={"learned_reward": 0.5, "fallback": True},
                metadata={"error": str(e)},
            )


# Convenience functions


def create_transformer_reward_function(
    checkpoint_path: Optional[str] = None,
    config: Optional[RewardTrainingConfig] = None,
    weight: float = 1.0,
) -> LearnedRewardFunction:
    """Create a learned reward function from a checkpoint"""
    trainer = TransformerRewardTrainer(config=config)

    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)

    return LearnedRewardFunction(trainer=trainer, weight=weight)
