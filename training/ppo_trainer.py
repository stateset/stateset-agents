"""
Proximal Policy Optimization (PPO) Trainer for StateSet Agents

This module provides a clean, efficient PPO implementation that serves as a baseline
for comparing more advanced algorithms like GRPO, GSPO, DAPO, and VAPO.

PPO is the foundation algorithm that most modern RL-from-feedback methods build upon.
This implementation includes:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function training
- Entropy bonus for exploration
- Adaptive KL penalty (optional)

Reference: https://arxiv.org/abs/1707.06347
"""

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import framework components
from .config import TrainingConfig

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("NumPy not available")

# Lazy import transformers
_transformers_ppo_loaded = False
AutoModelForCausalLM = None
AutoTokenizer = None
get_cosine_schedule_with_warmup = None


def _load_transformers_ppo():
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_ppo_loaded, AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup

    if _transformers_ppo_loaded:
        return True

    try:
        from transformers import (
            AutoModelForCausalLM as _AutoModelForCausalLM,
            AutoTokenizer as _AutoTokenizer,
            get_cosine_schedule_with_warmup as _get_cosine,
        )
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        get_cosine_schedule_with_warmup = _get_cosine
        _transformers_ppo_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


@dataclass
class PPOConfig(TrainingConfig):
    """
    Configuration for PPO training.

    PPO provides a good baseline for RL from human/AI feedback tasks.
    """

    # Model
    model_name: str = "gpt2"

    # PPO-specific hyperparameters
    clip_eps: float = 0.2  # Clipping epsilon (standard PPO)
    value_clip_eps: float = 0.2  # Value function clipping
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda

    # KL penalty (optional, use 0 to disable)
    beta: float = 0.0  # KL penalty coefficient
    target_kl: Optional[float] = None  # Target KL for early stopping
    use_adaptive_kl: bool = False  # Adaptive KL penalty

    # Training parameters
    num_ppo_epochs: int = 4  # Inner PPO epochs per batch
    mini_batch_size: int = 4  # Mini-batch size

    # Value function
    value_coef: float = 0.5  # Value loss coefficient
    value_hidden_size: int = 256  # Value head hidden size

    # Entropy bonus
    entropy_coef: float = 0.01  # Entropy bonus coefficient

    # Generation
    num_generations: int = 4  # Generations per prompt
    max_prompt_length: int = 256
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Model optimization
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit: bool = False
    use_4bit: bool = False

    # Reference model (for KL computation)
    use_reference_model: bool = True


class PPOValueHead(nn.Module):
    """
    Value prediction head for PPO.

    Takes hidden states from the language model and outputs value estimates.
    """

    def __init__(
        self,
        hidden_size: int,
        value_hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_size, value_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(value_hidden_size, value_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(value_hidden_size, 1),
        )

        # Initialize with small weights
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            values: [batch, seq_len, 1]
        """
        return self.net(hidden_states)


class AdaptiveKLController:
    """
    Adaptive KL penalty controller.

    Adjusts the KL penalty coefficient to keep KL divergence near target.
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        target_kl: float = 0.01,
        horizon: int = 10000,
    ):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> float:
        """Update KL coefficient based on current KL divergence."""
        proportional_error = (current_kl - self.target_kl) / self.target_kl
        mult = 1.0 + proportional_error * n_steps / self.horizon

        self.kl_coef = self.kl_coef * mult

        # Clamp to reasonable range
        self.kl_coef = max(0.001, min(self.kl_coef, 100.0))

        return self.kl_coef


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: [batch, seq_len] rewards for each token
        values: [batch, seq_len] value estimates
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: [batch, seq_len]
        returns: [batch, seq_len]
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device

    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Compute advantages in reverse order
    last_gae = torch.zeros(batch_size, device=device)
    last_value = torch.zeros(batch_size, device=device)

    for t in reversed(range(seq_len)):
        # TD error
        delta = rewards[:, t] + gamma * last_value - values[:, t]

        # GAE
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[:, t] = last_gae

        # Update for next iteration
        last_value = values[:, t]

    # Returns = advantages + values
    returns = advantages + values

    return advantages, returns


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer

    A clean, reference implementation of PPO for language model fine-tuning.
    This serves as a baseline for comparing more advanced algorithms.

    Features:
    - Clipped surrogate objective for stable policy updates
    - Generalized Advantage Estimation (GAE)
    - Value function training with clipping
    - Entropy bonus for exploration
    - Optional adaptive KL penalty

    Reference: https://arxiv.org/abs/1707.06347
    """

    def __init__(
        self,
        config: PPOConfig,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        ref_model: Optional[Any] = None,
    ):
        # Ensure transformers is loaded
        _load_transformers_ppo()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ref_model = ref_model
        self.device = next(model.parameters()).device

        # Get hidden size from model
        if hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", 768)
        else:
            hidden_size = 768

        # Initialize value head
        self.value_head = PPOValueHead(
            hidden_size=hidden_size,
            value_hidden_size=config.value_hidden_size,
        ).to(self.device)

        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=config.learning_rate * 2,  # Value head often needs higher LR
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        # Schedulers
        total_steps = config.num_episodes * config.num_epochs

        if get_cosine_schedule_with_warmup is not None:
            warmup_steps = int(total_steps * config.warmup_ratio)
            self.policy_scheduler = get_cosine_schedule_with_warmup(
                self.policy_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            self.value_scheduler = get_cosine_schedule_with_warmup(
                self.value_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.policy_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.policy_optimizer, factor=1.0, total_iters=total_steps
            )
            self.value_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.value_optimizer, factor=1.0, total_iters=total_steps
            )

        # Adaptive KL controller
        if config.use_adaptive_kl and config.target_kl:
            self.kl_controller = AdaptiveKLController(
                init_kl_coef=config.beta,
                target_kl=config.target_kl,
            )
        else:
            self.kl_controller = None

        # Metrics tracking
        self.metrics_history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "average_reward": [],
            "clip_fraction": [],
            "approx_kl": [],
        }

        self.global_step = 0

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden states from model."""
        with torch.set_grad_enabled(self.model.training):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return outputs.hidden_states[-1]

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities.

        Returns:
            log_probs: [batch, seq_len-1]
            entropy: [batch, seq_len-1]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Log probabilities
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        probs = F.softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs_all.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Entropy
        entropy = -(probs * log_probs_all).sum(dim=-1)

        return token_log_probs, entropy

    def compute_kl_divergence(
        self,
        current_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL divergence between current and reference policy."""
        kl = current_log_probs - reference_log_probs

        if mask is not None:
            kl = kl * mask
            return kl.sum() / mask.sum().clamp(min=1)

        return kl.mean()

    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PPO clipped surrogate loss.

        Args:
            log_probs: Current policy log probs
            old_log_probs: Old policy log probs (from rollout)
            advantages: Computed advantages
            mask: Token mask

        Returns:
            loss: PPO loss
            clip_fraction: Fraction of clipped ratios
        """
        # Importance ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped ratio
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.config.clip_eps,
            1 + self.config.clip_eps,
        )

        # Surrogate losses
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        # Take minimum (pessimistic)
        loss = -torch.min(surr1, surr2)

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        # Clip fraction for logging
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_eps).float()
            if mask is not None:
                clip_fraction = (clip_fraction * mask).sum() / mask.sum().clamp(min=1)
            else:
                clip_fraction = clip_fraction.mean()

        return loss, clip_fraction

    def value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute clipped value loss.

        Args:
            values: Current value estimates
            old_values: Old value estimates (from rollout)
            returns: Computed returns
            mask: Token mask
        """
        # Clipped values
        clipped_values = old_values + torch.clamp(
            values - old_values,
            -self.config.value_clip_eps,
            self.config.value_clip_eps,
        )

        # Value losses
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (clipped_values - returns) ** 2

        # Take maximum (pessimistic)
        loss = 0.5 * torch.max(vf_loss1, vf_loss2)

        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)

        return loss.mean()

    async def train_step(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
    ) -> Dict[str, float]:
        """
        Execute one PPO training step.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            rewards: List of reward scores

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.value_head.train()

        # Tokenize inputs
        full_texts = [p + c for p, c in zip(prompts, completions)]
        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length + self.config.max_completion_length,
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # Create response mask (1 for completion tokens)
        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1)

        response_mask = torch.zeros_like(input_ids, dtype=torch.float)
        for i, prompt_len in enumerate(prompt_lengths):
            response_mask[i, prompt_len:] = 1.0

        # Get initial values and log probs (for old policy)
        with torch.no_grad():
            old_log_probs, _ = self.compute_log_probs(input_ids, attention_mask)
            hidden_states = self.get_hidden_states(input_ids, attention_mask)
            old_values = self.value_head(hidden_states).squeeze(-1)[:, :-1]

        # Convert rewards to tensor and broadcast to tokens
        rewards_tensor = torch.tensor(rewards, device=self.device).unsqueeze(1)
        token_rewards = response_mask[:, 1:] * rewards_tensor

        # Compute GAE
        advantages, returns = compute_gae(
            token_rewards,
            old_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0

        response_mask_shift = response_mask[:, 1:]

        for _ in range(self.config.num_ppo_epochs):
            # Forward pass
            log_probs, entropy = self.compute_log_probs(input_ids, attention_mask)
            hidden_states = self.get_hidden_states(input_ids, attention_mask)
            values = self.value_head(hidden_states).squeeze(-1)[:, :-1]

            # Policy loss
            policy_loss, clip_fraction = self.ppo_loss(
                log_probs, old_log_probs, advantages, response_mask_shift
            )

            # Value loss
            vf_loss = self.value_loss(
                values, old_values, returns, response_mask_shift
            )

            # Entropy bonus
            if response_mask_shift is not None:
                entropy_bonus = (entropy * response_mask_shift).sum() / response_mask_shift.sum().clamp(min=1)
            else:
                entropy_bonus = entropy.mean()

            # Total loss
            total_loss = (
                policy_loss
                + self.config.value_coef * vf_loss
                - self.config.entropy_coef * entropy_bonus
            )

            # KL penalty
            if self.config.beta > 0 and self.ref_model is not None:
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                    ref_logits = ref_outputs.logits[:, :-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs.gather(
                        dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)

                kl = self.compute_kl_divergence(log_probs, ref_token_log_probs, response_mask_shift)

                kl_coef = self.config.beta
                if self.kl_controller:
                    kl_coef = self.kl_controller.update(kl.item(), 1)

                total_loss = total_loss + kl_coef * kl

            # Backward pass
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.value_head.parameters(), self.config.max_grad_norm
            )

            self.policy_optimizer.step()
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += vf_loss.item()
            total_entropy += entropy_bonus.item()
            total_clip_fraction += clip_fraction.item()

        # Average over epochs
        num_epochs = self.config.num_ppo_epochs
        avg_policy_loss = total_policy_loss / num_epochs
        avg_value_loss = total_value_loss / num_epochs
        avg_entropy = total_entropy / num_epochs
        avg_clip_fraction = total_clip_fraction / num_epochs

        # Update schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()

        self.global_step += 1

        metrics = {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "clip_fraction": avg_clip_fraction,
            "average_reward": sum(rewards) / len(rewards),
        }

        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        return metrics

    async def train(
        self,
        prompts: List[str],
        num_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run full PPO training loop.

        Args:
            prompts: List of training prompts
            num_episodes: Number of training episodes

        Returns:
            Training summary
        """
        num_episodes = num_episodes or self.config.num_episodes

        logger.info(f"Starting PPO training for {num_episodes} episodes")

        for episode in range(num_episodes):
            # Sample prompts for this episode
            episode_prompts = prompts[:self.config.num_generations]

            # Generate completions (simplified - would use model generation in practice)
            completions = ["This is a sample completion."] * len(episode_prompts)

            # Compute rewards
            rewards = []
            for p, c in zip(episode_prompts, completions):
                try:
                    if asyncio.iscoroutinefunction(self.reward_fn):
                        reward = await self.reward_fn(p, c)
                    else:
                        reward = self.reward_fn(p, c)
                    rewards.append(float(reward))
                except Exception as e:
                    logger.warning(f"Reward computation failed: {e}")
                    rewards.append(0.0)

            # Training step
            metrics = await self.train_step(episode_prompts, completions, rewards)

            if episode % self.config.logging_steps == 0:
                logger.info(
                    f"Episode {episode}/{num_episodes} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Reward: {metrics['average_reward']:.4f}"
                )

        return {
            "final_metrics": metrics,
            "metrics_history": self.metrics_history,
            "total_episodes": num_episodes,
        }

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.output_dir,
                f"ppo-checkpoint-{self.global_step}",
            )

        os.makedirs(path, exist_ok=True)

        # Save model
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save value head
        torch.save(
            self.value_head.state_dict(),
            os.path.join(path, "value_head.pt"),
        )

        # Save training state
        torch.save({
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "policy_scheduler": self.policy_scheduler.state_dict(),
            "value_scheduler": self.value_scheduler.state_dict(),
            "global_step": self.global_step,
            "metrics_history": self.metrics_history,
            "config": self.config,
        }, os.path.join(path, "training_state.pt"))

        logger.info(f"Checkpoint saved to {path}")
        return path


# Convenience function for quick training
async def train_ppo(
    model_name: str,
    prompts: List[str],
    reward_fn: Callable[[str, str], float],
    num_episodes: int = 100,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick PPO training with minimal configuration.

    Args:
        model_name: HuggingFace model name
        prompts: Training prompts
        reward_fn: Reward function
        num_episodes: Number of training episodes
        **kwargs: Additional config parameters

    Returns:
        Training results
    """
    _load_transformers_ppo()

    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("transformers is required for PPO training")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Create config
    config = PPOConfig(
        model_name=model_name,
        num_episodes=num_episodes,
        **kwargs,
    )

    # Create trainer
    trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
    )

    # Train
    return await trainer.train(prompts, num_episodes)
