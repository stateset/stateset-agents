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
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import objectives, rl_losses
from .config import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REWARD_FN_EXCEPTIONS = (RuntimeError, TypeError, ValueError)

try:
    import numpy as _np

    np: Any = _np
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
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import get_cosine_schedule_with_warmup as _get_cosine

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
    target_kl: float | None = None  # Target KL for early stopping
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
        proportional_error = (current_kl - self.target_kl) / max(self.target_kl, 1e-8)
        proportional_error = max(-0.2, min(0.2, proportional_error))
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
) -> tuple[torch.Tensor, torch.Tensor]:
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


def encode_prompt_completion_rows(
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    max_prompt_length: int,
    max_completion_length: int,
    completion_ids: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Right-padded ``prompt + completion`` rows with a completion mask.

    Prompt and completion are tokenized separately so their boundary is exact:
    the prompt keeps its last ``max_prompt_length`` tokens and the completion
    its first ``max_completion_length``. Tokenizing ``prompt + completion`` as
    one string and measuring a separately truncated prompt instead puts an
    over-long prompt's tail inside the "completion". ``completion_ids`` (the
    sampled token ids) are used verbatim when given. Rows whose completion is
    empty after truncation are dropped.

    Returns ``(input_ids, attention_mask, completion_mask, kept_indices)``;
    ``completion_mask`` is in the unshifted token frame.
    """
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")
    if completion_ids is not None and len(completion_ids) != len(prompts):
        raise ValueError("completion_ids must have one entry per prompt")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0

    rows: list[tuple[list[int], list[int]]] = []
    kept: list[int] = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        prompt_ids = [int(t) for t in tokenizer(prompt)["input_ids"]]
        if max_prompt_length > 0:
            prompt_ids = prompt_ids[-max_prompt_length:]
        if completion_ids is not None:
            response_ids = [int(t) for t in completion_ids[i]]
        else:
            response_ids = [
                int(t)
                for t in tokenizer(completion, add_special_tokens=False)["input_ids"]
            ]
        response_ids = response_ids[:max_completion_length]
        if not response_ids:
            continue
        rows.append((prompt_ids, response_ids))
        kept.append(i)

    width = max((len(p) + len(c) for p, c in rows), default=0)
    input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    completion_mask = torch.zeros((len(rows), width), dtype=torch.float)
    for row, (prompt_ids, response_ids) in enumerate(rows):
        ids = prompt_ids + response_ids
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        completion_mask[row, len(prompt_ids) : len(ids)] = 1.0
    return input_ids, attention_mask, completion_mask, kept


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
        ref_model: Any | None = None,
    ):
        # Ensure transformers is loaded
        _load_transformers_ppo()

        self.config = config
        # PPO clipped surrogate as one declarative PolicyObjective; the KL
        # penalty is added separately in train_step (adaptive coefficient).
        self._objective = objectives.resolve_objective(
            config,
            "ppo",
            max_completion_length=int(config.max_completion_length),
            supported_ratios=("token", "sequence", "sequence_token"),
            clip_low=float(config.clip_eps),
            clip_high=float(config.clip_eps),
            kl="none",
        )
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        if config.beta > 0 and ref_model is None:
            if not getattr(config, "use_reference_model", True):
                raise ValueError(
                    "beta > 0 needs a reference model; pass ref_model or set "
                    "use_reference_model=True"
                )
            # Snapshot the initial policy so the KL penalty is not skipped.
            import copy

            ref_model = copy.deepcopy(model)
        if ref_model is not None:
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad_(False)
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
        self.kl_controller: AdaptiveKLController | None
        if config.use_adaptive_kl and config.target_kl:
            self.kl_controller = AdaptiveKLController(
                init_kl_coef=config.beta,
                target_kl=config.target_kl,
            )
        else:
            self.kl_controller = None

        # Metrics tracking
        self.metrics_history: dict[str, list[float]] = {
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """k3 estimate of KL(pi_cur || pi_ref) over response tokens.

        Replaces the naive ``log pi - log pi_ref`` mean, whose gradient has
        zero expectation and therefore never pulled the policy toward the
        reference.
        """
        if mask is None:
            return rl_losses.k3_kl(current_log_probs, reference_log_probs)
        return rl_losses.k3_kl(current_log_probs, reference_log_probs, mask)

    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PPO clipped surrogate via ``objectives.policy_loss``.

        Returns ``(loss, clip_fraction)``; the log-ratio is clamped before
        exponentiating so a wildly off-policy token cannot make the loss
        non-finite.
        """
        if mask is None:
            mask = torch.ones_like(log_probs)
        result = objectives.policy_loss(
            logp_cur=log_probs,
            mask=mask,
            advantages=advantages,
            objective=self._objective,
            logp_old=old_log_probs,
        )
        clip_fraction = torch.tensor(
            result.metrics["clip_fraction"], device=log_probs.device
        )
        return result.loss, clip_fraction

    def value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor | None = None,
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

    @staticmethod
    def build_response_mask(
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        padding_side: str = "right",
    ) -> torch.Tensor:
        """1.0 on completion tokens of each row; prompt and padding are 0."""
        response_mask = torch.zeros_like(attention_mask, dtype=torch.float)
        seq_len = attention_mask.shape[1]
        for i, prompt_len in enumerate(prompt_lengths.tolist()):
            pad = (
                seq_len - int(attention_mask[i].sum()) if padding_side == "left" else 0
            )
            response_mask[i, pad + int(prompt_len) :] = 1.0
        return response_mask * attention_mask.to(response_mask.dtype)

    def compute_episode_advantages(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        old_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE for one scalar reward per episode, in the shifted action frame.

        The reward sits on each row's last response action (broadcasting it to
        every token would inflate returns with response length). Values outside
        the mask are zeroed so padding after a response neither bootstraps nor
        earns reward, and advantages are normalized over response actions only.
        """
        idx = torch.arange(action_mask.shape[1], device=action_mask.device)
        last = torch.where(action_mask > 0, idx, -1).max(dim=1).values
        token_rewards = torch.zeros_like(action_mask)
        rows = torch.nonzero(last >= 0).flatten()
        token_rewards[rows, last[rows]] = rewards.to(token_rewards.dtype)[rows]

        advantages, returns = compute_gae(
            token_rewards,
            old_values * action_mask,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        valid = advantages[action_mask > 0]
        if valid.numel() > 1:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-8)
        return advantages * action_mask, returns

    async def train_step(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        *,
        completion_ids: list[list[int]] | None = None,
    ) -> dict[str, float]:
        """
        Execute one PPO training step.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            rewards: List of reward scores
            completion_ids: Sampled completion token ids; when given the
                policy is scored on exactly these tokens

        Returns:
            Dictionary of metrics
        """
        if len(rewards) != len(prompts):
            raise ValueError("rewards must have one entry per prompt")
        input_ids, attention_mask, response_mask, kept = encode_prompt_completion_rows(
            self.tokenizer,
            prompts,
            completions,
            int(self.config.max_prompt_length),
            int(self.config.max_completion_length),
            completion_ids,
        )
        if not kept:
            logger.warning("PPO step skipped: no completion has any tokens")
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "skipped_rows": float(len(prompts)),
            }
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        response_mask = response_mask.to(self.device)
        kept_rewards = [float(rewards[i]) for i in kept]

        self.model.train()
        self.value_head.train()

        # Get initial values and log probs (for old policy)
        with torch.no_grad():
            old_log_probs, _ = self.compute_log_probs(input_ids, attention_mask)
            hidden_states = self.get_hidden_states(input_ids, attention_mask)
            old_values = self.value_head(hidden_states).squeeze(-1)[:, :-1]

        rewards_tensor = torch.tensor(
            kept_rewards, device=self.device, dtype=torch.float
        )
        response_mask_shift = response_mask[:, 1:]
        advantages, returns = self.compute_episode_advantages(
            rewards_tensor, response_mask_shift, old_values
        )

        # PPO epochs, each a shuffled pass over mini-batches of rows
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        batch_size = input_ids.shape[0]
        mini_batch_size = int(getattr(self.config, "mini_batch_size", 0) or 0)
        if mini_batch_size <= 0:
            mini_batch_size = batch_size

        for _ in range(self.config.num_ppo_epochs):
            order = torch.randperm(batch_size, device=self.device)
            for idx in order.split(mini_batch_size):
                mb_ids = input_ids[idx]
                mb_attention = attention_mask[idx]
                mb_mask = response_mask_shift[idx]

                # Forward pass
                log_probs, entropy = self.compute_log_probs(mb_ids, mb_attention)
                hidden_states = self.get_hidden_states(mb_ids, mb_attention)
                values = self.value_head(hidden_states).squeeze(-1)[:, :-1]

                # Policy loss
                policy_loss, clip_fraction = self.ppo_loss(
                    log_probs, old_log_probs[idx], advantages[idx], mb_mask
                )

                # Value loss
                vf_loss = self.value_loss(
                    values, old_values[idx], returns[idx], mb_mask
                )

                # Entropy bonus
                entropy_bonus = (entropy * mb_mask).sum() / mb_mask.sum().clamp(min=1)

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_coef * vf_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                # KL penalty
                if self.config.beta > 0 and self.ref_model is not None:
                    with torch.no_grad():
                        ref_outputs = self.ref_model(
                            input_ids=mb_ids, attention_mask=mb_attention
                        )
                        ref_logits = ref_outputs.logits[:, :-1, :]
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        ref_token_log_probs = ref_log_probs.gather(
                            dim=-1, index=mb_ids[:, 1:].unsqueeze(-1)
                        ).squeeze(-1)

                    kl = self.compute_kl_divergence(
                        log_probs, ref_token_log_probs, mb_mask
                    )

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
                num_updates += 1

        # Average over updates
        num_updates = max(1, num_updates)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_clip_fraction = total_clip_fraction / num_updates

        # Update schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()

        self.global_step += 1

        metrics = {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "clip_fraction": avg_clip_fraction,
            "average_reward": sum(kept_rewards) / len(kept_rewards),
            "skipped_rows": float(len(prompts) - len(kept)),
        }

        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        return metrics

    def generate_completion_ids(self, prompts: list[str]) -> list[list[int]]:
        """Sample one completion per prompt from the policy, as token ids.

        Sampling runs in inference mode (eval, KV cache on; the previous mode
        is restored). Each completion is cut after its first EOS token, which
        is kept as the final action.
        """
        from stateset_agents.core.generation_mode import inference_mode

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        temperature = float(getattr(self.config, "temperature", 0.0) or 0.0)
        sample_kwargs: dict[str, Any] = {"do_sample": temperature > 0}
        if temperature > 0:
            sample_kwargs["temperature"] = temperature
            sample_kwargs["top_p"] = float(getattr(self.config, "top_p", 1.0))

        results: list[list[int]] = []
        with inference_mode(self.model), torch.no_grad():
            for prompt in prompts:
                prompt_ids = [int(t) for t in self.tokenizer(prompt)["input_ids"]]
                if self.config.max_prompt_length > 0:
                    prompt_ids = prompt_ids[-int(self.config.max_prompt_length) :]
                if not prompt_ids:
                    prompt_ids = [int(eos_id if eos_id is not None else 0)]
                input_ids = torch.tensor([prompt_ids], device=self.device)
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=int(self.config.max_completion_length),
                    pad_token_id=pad_id,
                    **sample_kwargs,
                )
                ids = [int(t) for t in output[0][len(prompt_ids) :].tolist()]
                if eos_id is not None and eos_id in ids:
                    ids = ids[: ids.index(eos_id) + 1]
                results.append(ids)
        return results

    async def train(
        self,
        prompts: list[str],
        num_episodes: int | None = None,
    ) -> dict[str, Any]:
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

        if not prompts:
            raise ValueError("PPO training needs at least one prompt")
        prompts_per_episode = min(
            len(prompts), max(1, int(self.config.per_device_train_batch_size))
        )
        samples_per_prompt = max(1, int(self.config.num_generations))

        for episode in range(num_episodes):
            # Cycle through the prompts; sample each one several times.
            start = episode * prompts_per_episode
            episode_prompts = [
                prompts[(start + k) % len(prompts)]
                for k in range(prompts_per_episode)
                for _ in range(samples_per_prompt)
            ]

            # Sample completions from the current policy
            completion_ids = self.generate_completion_ids(episode_prompts)
            completions = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in completion_ids
            ]

            # Compute rewards
            rewards = []
            for p, c in zip(episode_prompts, completions, strict=False):
                try:
                    if asyncio.iscoroutinefunction(self.reward_fn):
                        reward = await self.reward_fn(p, c)
                    else:
                        reward = self.reward_fn(p, c)
                    rewards.append(float(reward))
                except REWARD_FN_EXCEPTIONS as e:
                    logger.warning(f"Reward computation failed: {e}")
                    rewards.append(0.0)

            # Training step
            metrics = await self.train_step(
                episode_prompts, completions, rewards, completion_ids=completion_ids
            )

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

    def save_checkpoint(self, path: str | None = None) -> str:
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
        torch.save(
            {
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "policy_scheduler": self.policy_scheduler.state_dict(),
                "value_scheduler": self.value_scheduler.state_dict(),
                "global_step": self.global_step,
                "metrics_history": self.metrics_history,
                "config": self.config,
            },
            os.path.join(path, "training_state.pt"),
        )

        logger.info(f"Checkpoint saved to {path}")
        return path


# Convenience function for quick training
async def train_ppo(
    model_name: str,
    prompts: list[str],
    reward_fn: Callable[[str, str], float],
    num_episodes: int = 100,
    **kwargs,
) -> dict[str, Any]:
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
