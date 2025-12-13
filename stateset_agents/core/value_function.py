"""
Value Function for GRPO Training

This module provides value function implementations for computing advantages
using Generalized Advantage Estimation (GAE) in GRPO training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


def _require_torch() -> Any:
    """Ensure torch is available"""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for value function training. "
            "Install with: pip install stateset-agents[training]"
        )
    return torch


class ValueHead(nn.Module):
    """
    Value function head that can be attached to a language model.

    This implements a simple MLP that takes the last hidden state
    and outputs a scalar value prediction.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            hidden_size: Size of the model's hidden states
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        _require_torch()

        self.dropout = nn.Dropout(dropout)

        # Value head layers
        if use_layer_norm:
            self.norm = nn.LayerNorm(hidden_size)
        else:
            self.norm = nn.Identity()

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute value prediction from hidden states.

        Args:
            hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]

        Returns:
            Value predictions [batch_size, seq_len, 1]
        """
        # Apply normalization and dropout
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Compute value
        return self.value_head(hidden_states)


class ValueFunction:
    """
    Value function for advantage estimation in GRPO.

    This class manages value predictions and advantage computation
    using Generalized Advantage Estimation (GAE).
    """

    def __init__(
        self,
        model: Any,
        value_head: Optional[ValueHead] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
    ):
        """
        Args:
            model: The language model (with .model attribute)
            value_head: Optional value head. If None, will create one
            gamma: Discount factor for future rewards
            gae_lambda: Lambda parameter for GAE
            normalize_advantages: Whether to normalize advantages
        """
        _require_torch()

        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        # Initialize or use provided value head
        if value_head is None:
            hidden_size = self._get_hidden_size()
            self.value_head = ValueHead(hidden_size)

            # Move to same device as model
            if hasattr(model, 'device'):
                self.value_head = self.value_head.to(model.device)
        else:
            self.value_head = value_head

        # Optimizer for value function
        self.optimizer = torch.optim.Adam(
            self.value_head.parameters(),
            lr=1e-4,
        )

        logger.info(
            f"ValueFunction initialized: gamma={gamma}, "
            f"gae_lambda={gae_lambda}, normalize={normalize_advantages}"
        )

    def _get_hidden_size(self) -> int:
        """Get hidden size from the model"""
        config = None

        model_dict = getattr(self.model, "__dict__", {})
        if isinstance(model_dict, dict) and "config" in model_dict:
            config = model_dict.get("config")
        else:
            try:
                config = getattr(self.model, "config")
            except Exception:
                config = None

        candidates: List[Any] = []
        if config is not None:
            config_dict = getattr(config, "__dict__", {})
            if isinstance(config_dict, dict):
                if "hidden_size" in config_dict:
                    candidates.append(config_dict.get("hidden_size"))
                if "d_model" in config_dict:
                    candidates.append(config_dict.get("d_model"))
            else:
                for attr in ("hidden_size", "d_model"):
                    try:
                        candidates.append(getattr(config, attr))
                    except Exception:
                        continue

        for candidate in candidates:
            try:
                hidden_size = int(candidate)
            except (TypeError, ValueError):
                continue
            if hidden_size > 0:
                return hidden_size

        # Default fallback
        return 768

    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute value predictions for input sequences.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Value predictions [batch_size, seq_len]
        """
        _require_torch()

        with torch.no_grad():
            # Get hidden states from model
            if hasattr(self.model, 'model'):
                # For transformers models
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
            else:
                # For models that output hidden states directly
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

        # Compute values
        values = self.value_head(hidden_states).squeeze(-1)

        return values

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        dones: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards for each timestep
            values: Value predictions [seq_len]
            dones: Optional list of done flags

        Returns:
            Tuple of (advantages, returns)
        """
        _require_torch()

        if dones is None:
            dones = [False] * (len(rewards) - 1) + [True]

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=values.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=values.device)

        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards_tensor)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards_tensor[t] + self.gamma * next_value * (1 - dones_tensor[t]) - values[t]

            # GAE: A_t = δ_t + γλ * A_{t+1}
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones_tensor[t]) * last_advantage
            last_advantage = advantages[t]

        # Store unnormalized advantages for returns calculation
        unnormalized_advantages = advantages.clone()

        # Normalize advantages if requested
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute returns using unnormalized advantages: R_t = A_t + V(s_t)
        returns = unnormalized_advantages + values

        return advantages, returns

    def compute_grpo_advantages(
        self,
        group_rewards: List[float],
        baseline_type: str = "group_mean",
    ) -> torch.Tensor:
        """
        Compute GRPO-style advantages (group-relative).

        Args:
            group_rewards: Rewards for trajectories in the group
            baseline_type: Type of baseline ('group_mean', 'group_median', or 'learned')

        Returns:
            Advantages tensor
        """
        _require_torch()

        rewards = torch.tensor(group_rewards, dtype=torch.float32)

        # Compute baseline
        if baseline_type == "group_mean":
            baseline = rewards.mean()
        elif baseline_type == "group_median":
            baseline = rewards.median()
        elif baseline_type == "learned":
            # Use value function as baseline (if available)
            baseline = rewards.mean()  # Fallback
        else:
            baseline = rewards.mean()

        # Compute advantages
        advantages = rewards - baseline

        # Normalize
        if self.normalize_advantages and len(advantages) > 1:
            std = advantages.std()
            if std > 0:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        return advantages

    def update_value_function(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        returns: torch.Tensor,
        num_epochs: int = 5,
    ) -> float:
        """
        Update the value function using supervised learning.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            returns: Target returns [batch_size, seq_len]
            num_epochs: Number of update epochs

        Returns:
            Average value loss
        """
        _require_torch()

        total_loss = 0.0

        self.value_head.train()

        for epoch in range(num_epochs):
            # Get hidden states
            with torch.no_grad():
                if hasattr(self.model, 'model'):
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-1]
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

            # Compute value predictions
            value_preds = self.value_head(hidden_states).squeeze(-1)

            # Compute value loss (MSE)
            if attention_mask is not None:
                # Mask out padding tokens
                mask = attention_mask.bool()
                value_loss = ((value_preds - returns) ** 2 * mask).sum() / mask.sum()
            else:
                value_loss = F.mse_loss(value_preds, returns)

            # Backward pass
            self.optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
            self.optimizer.step()

            total_loss += value_loss.item()

        self.value_head.eval()

        avg_loss = total_loss / num_epochs
        logger.debug(f"Value function updated: avg_loss={avg_loss:.4f}")

        return avg_loss

    def save(self, path: str):
        """Save value function state"""
        _require_torch()
        torch.save({
            'value_head_state_dict': self.value_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
        }, path)
        logger.info(f"Value function saved to {path}")

    def load(self, path: str):
        """Load value function state"""
        _require_torch()
        checkpoint = torch.load(path)
        self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.gamma = checkpoint['gamma']
        self.gae_lambda = checkpoint['gae_lambda']
        logger.info(f"Value function loaded from {path}")


def create_value_function(
    model: Any,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    **kwargs
) -> ValueFunction:
    """
    Convenience function to create a value function.

    Args:
        model: The language model
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        **kwargs: Additional arguments

    Returns:
        ValueFunction instance
    """
    return ValueFunction(
        model=model,
        gamma=gamma,
        gae_lambda=gae_lambda,
        **kwargs
    )
