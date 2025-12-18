"""
Decision Transformer for Offline RL

Casts RL as a sequence modeling problem, conditioning on
desired returns to generate actions.

Reference: Chen et al. "Decision Transformer: Reinforcement Learning
via Sequence Modeling" (NeurIPS 2021)
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
except ImportError:
    torch = None
    nn = None
    F = None
    AdamW = None
    LambdaLR = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for Decision Transformer. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class DecisionTransformerConfig:
    """Configuration for Decision Transformer"""

    # Architecture
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    activation: str = "gelu"

    # Sequence configuration
    max_episode_length: int = 100  # Maximum turns per episode
    max_context_length: int = 20  # K in paper - context window

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 0.25

    # Optimization
    batch_size: int = 64
    num_epochs: int = 100

    # Conversation-specific
    use_conversation_embeddings: bool = True
    conversation_encoder: str = "all-MiniLM-L6-v2"
    state_dim: int = 384  # Embedding dimension
    action_dim: int = 384  # Same as state for response embeddings

    # Action generation
    action_tanh: bool = False  # Apply tanh to action output

    # Logging
    log_frequency: int = 100


class ConversationEmbedder(nn.Module):
    """
    Embeds conversation turns into fixed-dimensional vectors.

    Uses sentence transformers for text encoding and learns
    additional projection layers for the transformer.
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        output_dim: int = 512,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers required. "
                "Install: pip install sentence-transformers"
            )

        self.encoder = SentenceTransformer(encoder_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.output_dim = output_dim

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed

        Returns:
            embeddings: [batch_size, output_dim]
        """
        with torch.no_grad():
            base_embeddings = self.encoder.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )

        projected = self.projection(base_embeddings)
        return projected

    def embed_turn(self, role: str, content: str) -> torch.Tensor:
        """Embed a single conversation turn"""
        text = f"{role}: {content}"
        return self.forward([text]).squeeze(0)

    def embed_trajectory(
        self,
        turns: List[Dict[str, str]],
    ) -> torch.Tensor:
        """
        Embed all turns in a trajectory.

        Args:
            turns: List of {"role": ..., "content": ...} dicts

        Returns:
            embeddings: [num_turns, output_dim]
        """
        texts = [f"{t['role']}: {t['content']}" for t in turns]
        return self.forward(texts)


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer for Decision Transformer.

    Similar to GPT-2 attention but handles the interleaved
    (return, state, action) sequence structure.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal attention.

        Args:
            x: [batch, seq_len, n_embd]

        Returns:
            output: [batch, seq_len, n_embd]
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Combine heads
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    """Single transformer block for Decision Transformer"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for offline RL via sequence modeling.

    The model takes as input a sequence of (return-to-go, state, action)
    tuples and predicts the next action conditioned on the desired return.

    Architecture:
    - Separate embedding layers for returns, states, and actions
    - Learned timestep embeddings
    - GPT-style transformer with causal attention
    - Action prediction head

    Example:
        >>> dt = DecisionTransformer(config)
        >>> actions = dt(states, actions, returns_to_go, timesteps)
    """

    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        _require_torch()

        self.config = config
        self.n_embd = config.n_embd
        self.max_length = config.max_context_length

        # Embeddings
        self.embed_return = nn.Linear(1, config.n_embd)
        self.embed_state = nn.Linear(config.state_dim, config.n_embd)
        self.embed_action = nn.Linear(config.action_dim, config.n_embd)
        self.embed_timestep = nn.Embedding(
            config.max_episode_length, config.n_embd
        )

        # Layer norm for embeddings
        self.embed_ln = nn.LayerNorm(config.n_embd)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        # Sequence length is 3 * max_context_length (return, state, action triples)
        max_seq_len = 3 * config.max_context_length
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.n_embd, config.n_head, max_seq_len, config.dropout
                )
                for _ in range(config.n_layer)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Action prediction head
        self.predict_action = nn.Sequential(
            nn.Linear(config.n_embd, config.action_dim),
        )

        # Optional state and return prediction for auxiliary losses
        self.predict_state = nn.Linear(config.n_embd, config.state_dim)
        self.predict_return = nn.Linear(config.n_embd, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Decision Transformer.

        Args:
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim]
            returns_to_go: [batch, seq_len, 1]
            timesteps: [batch, seq_len]
            attention_mask: [batch, seq_len] (optional)

        Returns:
            action_preds: [batch, seq_len, action_dim]
            state_preds: [batch, seq_len, state_dim]
            return_preds: [batch, seq_len, 1]
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add timestep embeddings to each modality
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings

        # Interleave into sequence: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Shape: [batch, 3 * seq_len, n_embd]
        stacked = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings], dim=2
        )  # [batch, seq_len, 3, n_embd]
        stacked = stacked.view(batch_size, 3 * seq_len, self.n_embd)

        # Apply embedding layer norm and dropout
        stacked = self.embed_ln(stacked)
        stacked = self.drop(stacked)

        # Create attention mask if needed
        if attention_mask is not None:
            # Expand mask for interleaved sequence
            attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3)
            attention_mask = attention_mask.view(batch_size, 3 * seq_len)

        # Transformer blocks
        hidden = stacked
        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.ln_f(hidden)

        # Reshape back to separate modalities
        hidden = hidden.view(batch_size, seq_len, 3, self.n_embd)

        # Get predictions for each modality
        # Action prediction comes from state position (index 1)
        action_preds = self.predict_action(hidden[:, :, 1])
        # State prediction comes from action position (index 2)
        state_preds = self.predict_state(hidden[:, :, 2])
        # Return prediction comes from return position (index 0)
        return_preds = self.predict_return(hidden[:, :, 0])

        if self.config.action_tanh:
            action_preds = torch.tanh(action_preds)

        return action_preds, state_preds, return_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get action prediction for the last timestep.

        Args:
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim]
            returns_to_go: [batch, seq_len, 1]
            timesteps: [batch, seq_len]

        Returns:
            action: [batch, action_dim] - action for last timestep
        """
        action_preds, _, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[:, -1]


class DecisionTransformerTrainer:
    """
    Trainer for Decision Transformer on conversation data.

    Handles:
    - Sequence construction from trajectories
    - Training loop with warmup
    - Action generation conditioned on target return
    """

    def __init__(
        self,
        config: DecisionTransformerConfig,
        device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.config = config
        self.device = device

        # Initialize model
        self.model = DecisionTransformer(config).to(device)

        # Initialize embedder if using conversation embeddings
        if config.use_conversation_embeddings:
            self.embedder = ConversationEmbedder(
                encoder_name=config.conversation_encoder,
                embedding_dim=384,  # Default for MiniLM
                output_dim=config.state_dim,
            ).to(device)
        else:
            self.embedder = None

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler with warmup
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            return 1.0

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.training_step = 0
        self.training_metrics: List[Dict[str, float]] = []

    def _prepare_batch(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Prepare a batch of trajectories for training.

        Args:
            trajectories: List of trajectory dictionaries with:
                - states: [seq_len, state_dim]
                - actions: [seq_len, action_dim]
                - returns_to_go: [seq_len]
                - timesteps: [seq_len]

        Returns:
            Tuple of (states, actions, returns_to_go, timesteps, mask)
        """
        batch_size = len(trajectories)
        max_len = self.config.max_context_length

        # Initialize tensors
        states = torch.zeros(
            batch_size, max_len, self.config.state_dim, device=self.device
        )
        actions = torch.zeros(
            batch_size, max_len, self.config.action_dim, device=self.device
        )
        returns_to_go = torch.zeros(batch_size, max_len, 1, device=self.device)
        timesteps = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros(batch_size, max_len, device=self.device)

        for i, traj in enumerate(trajectories):
            traj_len = min(len(traj["states"]), max_len)

            states[i, :traj_len] = torch.tensor(
                traj["states"][:traj_len], device=self.device
            )
            actions[i, :traj_len] = torch.tensor(
                traj["actions"][:traj_len], device=self.device
            )
            returns_to_go[i, :traj_len, 0] = torch.tensor(
                traj["returns_to_go"][:traj_len], device=self.device
            )
            timesteps[i, :traj_len] = torch.tensor(
                traj["timesteps"][:traj_len], device=self.device
            )
            mask[i, :traj_len] = 1.0

        return states, actions, returns_to_go, timesteps, mask

    def train_step(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            trajectories: Batch of trajectory dictionaries

        Returns:
            Training metrics
        """
        self.model.train()

        # Prepare batch
        states, actions, returns_to_go, timesteps, mask = self._prepare_batch(
            trajectories
        )

        # Forward pass
        action_preds, state_preds, return_preds = self.model(
            states, actions, returns_to_go, timesteps
        )

        # Action prediction loss (main objective)
        action_loss = F.mse_loss(action_preds, actions, reduction="none")
        action_loss = (action_loss * mask.unsqueeze(-1)).sum() / mask.sum()

        # Optional auxiliary losses
        state_loss = F.mse_loss(state_preds[:, :-1], states[:, 1:], reduction="none")
        state_mask = mask[:, 1:].unsqueeze(-1)
        state_loss = (state_loss * state_mask).sum() / state_mask.sum() if state_mask.sum() > 0 else torch.tensor(0.0)

        # Total loss
        loss = action_loss + 0.1 * state_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()

        self.training_step += 1

        metrics = {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "state_loss": state_loss.item() if isinstance(state_loss, torch.Tensor) else state_loss,
            "lr": self.scheduler.get_last_lr()[0],
            "training_step": self.training_step,
        }

        return metrics

    async def train(
        self,
        dataset: Any,  # ConversationDataset
        num_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train Decision Transformer on conversation dataset.

        Args:
            dataset: ConversationDataset with trajectories
            num_epochs: Number of epochs (default from config)

        Returns:
            Training results
        """
        num_epochs = num_epochs or self.config.num_epochs

        logger.info(
            f"Training Decision Transformer for {num_epochs} epochs "
            f"on {len(dataset)} trajectories"
        )

        # Convert dataset to training format
        training_data = self._prepare_dataset(dataset)

        for epoch in range(num_epochs):
            epoch_metrics = []

            # Shuffle data
            indices = np.random.permutation(len(training_data))
            num_batches = len(training_data) // self.config.batch_size

            for batch_idx in range(num_batches):
                batch_indices = indices[
                    batch_idx * self.config.batch_size : (batch_idx + 1) * self.config.batch_size
                ]
                batch = [training_data[i] for i in batch_indices]

                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)

            # Average epoch metrics
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            avg_metrics["epoch"] = epoch
            self.training_metrics.append(avg_metrics)

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs}: "
                    f"loss={avg_metrics['loss']:.4f}, "
                    f"action_loss={avg_metrics['action_loss']:.4f}"
                )

        return {
            "final_loss": self.training_metrics[-1]["loss"],
            "num_epochs": num_epochs,
            "training_metrics": self.training_metrics,
        }

    def _prepare_dataset(
        self,
        dataset: Any,
    ) -> List[Dict[str, Any]]:
        """Convert ConversationDataset to training format"""
        training_data = []

        for traj in dataset:
            # Embed turns if using conversation embeddings
            if self.embedder is not None:
                states = []
                actions = []

                state_history = ""
                for i, turn in enumerate(traj.turns):
                    if turn.role in ("user", "system"):
                        state_history += f"{turn.role}: {turn.content}\n"
                        with torch.no_grad():
                            state_emb = self.embedder.forward([state_history])
                        states.append(state_emb.squeeze(0).cpu().numpy())
                    elif turn.role == "assistant":
                        with torch.no_grad():
                            action_emb = self.embedder.forward([turn.content])
                        actions.append(action_emb.squeeze(0).cpu().numpy())
                        state_history += f"assistant: {turn.content}\n"
            else:
                # Assume pre-embedded
                states = traj.metadata.get("state_embeddings", [])
                actions = traj.metadata.get("action_embeddings", [])

            if len(states) == 0 or len(actions) == 0:
                continue

            # Ensure same length
            min_len = min(len(states), len(actions))
            states = states[:min_len]
            actions = actions[:min_len]

            # Compute returns-to-go if not present
            if traj.returns_to_go:
                rtg = traj.returns_to_go[:min_len]
            else:
                rtg = traj.compute_returns_to_go(self.config.discount_factor)[:min_len] if hasattr(traj, 'compute_returns_to_go') else [0.0] * min_len

            training_data.append({
                "states": np.array(states),
                "actions": np.array(actions),
                "returns_to_go": np.array(rtg),
                "timesteps": np.arange(min_len),
            })

        return training_data

    def generate_response(
        self,
        history: List[Dict[str, str]],
        target_return: float,
        past_actions: Optional[List[torch.Tensor]] = None,
        past_returns: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Generate a response embedding conditioned on target return.

        Args:
            history: Conversation history [{"role": ..., "content": ...}, ...]
            target_return: Desired return (e.g., 0.9 for high quality)
            past_actions: Previous action embeddings
            past_returns: Previous returns

        Returns:
            action_embedding: [action_dim] - embedding for response
        """
        self.model.eval()

        with torch.no_grad():
            # Embed current state
            if self.embedder is not None:
                state_text = "\n".join(
                    [f"{t['role']}: {t['content']}" for t in history]
                )
                state = self.embedder.forward([state_text]).unsqueeze(0)
            else:
                raise ValueError("Embedder required for response generation")

            # Build context
            context_len = len(past_actions) + 1 if past_actions else 1

            states = torch.zeros(
                1, context_len, self.config.state_dim, device=self.device
            )
            actions = torch.zeros(
                1, context_len, self.config.action_dim, device=self.device
            )
            returns_to_go = torch.zeros(
                1, context_len, 1, device=self.device
            )
            timesteps = torch.arange(context_len, device=self.device).unsqueeze(0)

            # Fill in past context
            if past_actions:
                for i, (act, ret) in enumerate(zip(past_actions, past_returns or [])):
                    actions[0, i] = act
                    if past_returns:
                        returns_to_go[0, i, 0] = ret

            # Current state and target return
            states[0, -1] = state.squeeze(0)
            returns_to_go[0, -1, 0] = target_return

            # Get action prediction
            action = self.model.get_action(states, actions, returns_to_go, timesteps)

            return action.squeeze(0)

    def save(self, path: str) -> None:
        """Save model checkpoint"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
                "training_step": self.training_step,
                "training_metrics": self.training_metrics,
            },
            path,
        )
        logger.info(f"Saved Decision Transformer to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.training_metrics = checkpoint.get("training_metrics", [])

        logger.info(f"Loaded Decision Transformer from {path}")


# Discount factor for returns computation
DEFAULT_DISCOUNT = 0.99
