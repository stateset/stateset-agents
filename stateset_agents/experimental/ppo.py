"""
Proximal Policy Optimization (PPO) Trainer for StateSet Agents

This module provides token-exact online PPO and reference text-batch updates.
Online sampling retains behavior likelihoods and critic values; the external
text interface cannot certify caller-supplied behavior-policy provenance.

PPO is the foundation algorithm that most modern RL-from-feedback methods build upon.
This implementation includes:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function training
- Entropy bonus for exploration
- Adaptive KL penalty (optional)

Reference: https://arxiv.org/abs/1707.06347
"""

import copy
import logging
import math
import os
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training import objectives, rl_losses
from ..training.config import TrainingConfig
from ..training.runtime_state import TrainingRuntime, validate_training_config
from ..training.trainer_utils import apply_dropout_setting

logger = logging.getLogger(__name__)


@contextmanager
def _deterministic_scoring(*models: nn.Module | None) -> Iterator[None]:
    """Disable stochastic layers without disabling gradients; restore all modes."""
    modes = {
        module: module.training
        for model in models
        if model is not None
        for module in model.modules()
    }
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield
    finally:
        # Restore mixed submodule modes as well as the root's flag.
        for module, training in modes.items():
            module.training = training


try:
    import numpy as _np

    np: Any = _np
except ImportError:
    np = None
    logger.warning("NumPy not available")

# Lazy import transformers
AutoModelForCausalLM: Any | None = None
AutoTokenizer: Any | None = None
get_cosine_schedule_with_warmup: Any | None = None


def _load_transformers_ppo():
    """Lazily load transformers to avoid import-time errors."""
    global AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup

    if AutoModelForCausalLM is not None:
        return True

    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import get_cosine_schedule_with_warmup as _get_cosine

        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        get_cosine_schedule_with_warmup = _get_cosine
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
    gradient_accumulation_steps: int = 1
    ppo_prompt_sha256: str | None = None

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
    top_p: float = 1.0
    top_k: int = 0

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

    def validate_for_training(self) -> None:
        """Validate PPO estimator and critic settings before learner mutation."""
        super().validate_for_training()
        for name in ("num_ppo_epochs", "mini_batch_size", "value_hidden_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite in [0, 1]")
        for name in ("clip_eps", "value_clip_eps", "value_coef"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.target_kl is not None and (
            type(self.target_kl) not in (int, float)
            or not math.isfinite(self.target_kl)
            or self.target_kl <= 0
        ):
            raise ValueError("target_kl must be finite and positive")
        if self.use_adaptive_kl and (self.target_kl is None or self.beta <= 0):
            raise ValueError("Adaptive PPO KL requires target_kl and a positive beta")


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
        values: torch.Tensor = self.net(hidden_states)
        return values


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

    def propose(self, current_kl: float, n_steps: int) -> float:
        """Calculate a coefficient without changing controller state."""
        if not math.isfinite(current_kl):
            raise FloatingPointError("PPO KL divergence must remain finite")
        proportional_error = (current_kl - self.target_kl) / self.target_kl
        mult = 1.0 + proportional_error * n_steps / self.horizon
        proposed = self.kl_coef * mult
        if not math.isfinite(proposed):
            raise FloatingPointError("PPO KL coefficient must remain finite")
        return max(0.001, min(proposed, 100.0))

    def update(self, current_kl: float, n_steps: int) -> float:
        """Update KL coefficient based on current KL divergence."""
        self.kl_coef = self.propose(current_kl, n_steps)
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


class PPOTrainer(TrainingRuntime):
    """
    Proximal Policy Optimization (PPO) Trainer

    Token-exact online updates and externally supplied text-batch updates.
    ``train_step`` requires callers to establish behavior-policy provenance.

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

        validate_training_config(config, PPOConfig)
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
        apply_dropout_setting(config, model)
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden: torch.Tensor = outputs.hidden_states[-1]
        return last_hidden

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
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
        shift_logits = logits[:, :-1, :].float().contiguous() / temperature
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
        # rl_losses is backend-agnostic (Any in, Any out); torch in, torch out.
        kl: torch.Tensor = rl_losses.k3_kl(current_log_probs, reference_log_probs, mask)
        return kl

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
        advantages = rl_losses.whiten(advantages, action_mask)
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

        This lower-level API scores caller-supplied text. It does not establish
        that completions were sampled from the policy or retain exact sampled
        token boundaries. Ambiguous prompt boundaries and truncated responses
        are rejected; callers remain responsible for sampling provenance.
        Scoring disables dropout while retaining gradients. A failure after an
        optimizer starts quarantines the runtime until a complete checkpoint
        reload; it does not roll back already committed optimizer updates.
        """
        if getattr(self, "_ppo_online_active", False):
            raise RuntimeError("PPO online operation is already active")
        self._ppo_update_in_flight = False
        try:
            with _deterministic_scoring(self.model, self.value_head, self.ref_model):
                return await self._train_text_step(
                    prompts, completions, rewards, completion_ids=completion_ids
                )
        except BaseException:
            if self._ppo_update_in_flight:
                self._runtime_checkpoint_poisoned = True
            raise
        finally:
            self._ppo_update_in_flight = False

    async def _train_text_step(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        *,
        completion_ids: list[list[int]] | None = None,
    ) -> dict[str, float]:
        """Validate and update a text batch with deterministic model scoring."""
        validate_training_config(self.config, PPOConfig)
        if self.config.beta > 0 and self.ref_model is None:
            raise ValueError("PPO beta > 0 requires a reference model")
        if (
            type(self.config.num_ppo_epochs) is not int
            or self.config.num_ppo_epochs < 1
        ):
            raise ValueError("num_ppo_epochs must be a positive integer")
        if (
            not prompts
            or len(prompts) != len(completions)
            or len(prompts) != len(rewards)
        ):
            raise ValueError(
                "PPO requires equal nonempty prompt, completion and reward batches"
            )
        if any(not isinstance(value, str) for value in (*prompts, *completions)):
            raise TypeError("PPO prompts and completions must be strings")
        try:
            finite_rewards = [float(value) for value in rewards]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PPO rewards must be finite numbers") from exc
        if any(isinstance(value, bool) for value in rewards) or not all(
            math.isfinite(value) for value in finite_rewards
        ):
            raise ValueError("PPO rewards must be finite numbers")
        rewards_tensor = torch.tensor(
            finite_rewards, device=self.device, dtype=torch.float
        )
        if not torch.isfinite(rewards_tensor).all():
            raise ValueError("PPO rewards must remain finite in the learner dtype")

        # One extra token detects an overlong prompt without allocating an
        # unbounded prompt batch. Never label a truncated prompt's tail as a
        # response merely because the full text has a larger token budget.
        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length + 1,
        )
        prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1)
        if (prompt_lengths > self.config.max_prompt_length).any():
            raise ValueError(
                "PPO prompts exceed max_prompt_length; truncate them explicitly"
            )

        if completion_ids is not None:
            if len(completion_ids) != len(prompts) or any(
                not isinstance(ids, list)
                or not ids
                or len(ids) > self.config.max_completion_length
                or any(type(token) is not int or token < 0 for token in ids)
                for ids in completion_ids
            ):
                raise ValueError(
                    "completion_ids must contain a nonempty integer action per prompt "
                    "within max_completion_length"
                )
            if (prompt_lengths == 0).any():
                raise ValueError("PPO requires a nonempty prompt for every response")
            ids, attention, mask, kept = encode_prompt_completion_rows(
                self.tokenizer,
                prompts,
                completions,
                self.config.max_prompt_length,
                self.config.max_completion_length,
                completion_ids,
            )
            if kept != list(range(len(prompts))):
                raise ValueError("PPO cannot discard rewarded completion rows")
            return self._update_token_batch(
                ids.to(self.device),
                attention.to(self.device),
                mask.to(self.device),
                rewards_tensor,
            )

        # Tokenize inputs
        full_texts = [p + c for p, c in zip(prompts, completions, strict=True)]
        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length
            + self.config.max_completion_length
            + 1,
        )

        # A BPE merge or an injected special token can change the separately
        # tokenized prompt. Length alone cannot locate the completion boundary.
        for row in range(len(prompts)):
            prompt_ids = prompt_encodings["input_ids"][row][
                prompt_encodings["attention_mask"][row].bool()
            ]
            full_ids = encodings["input_ids"][row][
                encodings["attention_mask"][row].bool()
            ]
            if prompt_ids.numel() == 0 or not torch.equal(
                prompt_ids, full_ids[: prompt_ids.numel()]
            ):
                raise ValueError(
                    "PPO prompt token boundary is ambiguous in combined text"
                )
            if (
                full_ids.numel() - prompt_ids.numel()
                > self.config.max_completion_length
            ):
                raise ValueError(
                    "PPO completions exceed max_completion_length; truncated rewards are invalid"
                )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        response_mask = self.build_response_mask(
            attention_mask,
            prompt_lengths,
            getattr(self.tokenizer, "padding_side", "right"),
        )
        if not (response_mask[:, 1:].sum(dim=1) > 0).all():
            raise ValueError("PPO requires at least one response action in every row")

        return self._update_token_batch(
            input_ids, attention_mask, response_mask, rewards_tensor
        )

    def _update_token_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        rewards_tensor: torch.Tensor,
        *,
        old_log_probs: torch.Tensor | None = None,
        old_values: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Update retained token actions; online callers supply behavior scores."""

        # Get initial values and log probs (for old policy)
        with torch.no_grad():
            if old_log_probs is None:
                old_log_probs, _ = self.compute_log_probs(
                    input_ids, attention_mask, temperature
                )
            if old_values is None:
                hidden_states = self.get_hidden_states(input_ids, attention_mask)
                old_values = self.value_head(
                    hidden_states.to(next(self.value_head.parameters()).dtype)
                ).squeeze(-1)[:, :-1]

        advantages, returns = self.compute_episode_advantages(
            rewards_tensor, response_mask[:, 1:], old_values
        )
        if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
            raise FloatingPointError("PPO advantages and returns must remain finite")

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0

        response_mask_shift = response_mask[:, 1:]

        batch_size = input_ids.shape[0]
        mini_batch_size = self.config.mini_batch_size
        if type(mini_batch_size) is not int or mini_batch_size < 1:
            raise ValueError("mini_batch_size must be a positive integer")
        num_updates = 0
        for _ in range(self.config.num_ppo_epochs):
            order = (
                torch.randperm(batch_size, device=self.device)
                if mini_batch_size < batch_size
                else torch.arange(batch_size, device=self.device)
            )
            for idx in order.split(mini_batch_size):
                mb_ids = input_ids[idx]
                mb_attention = attention_mask[idx]
                mb_mask = response_mask_shift[idx]
                # Forward pass
                log_probs, entropy = self.compute_log_probs(
                    mb_ids, mb_attention, temperature
                )
                hidden_states = self.get_hidden_states(mb_ids, mb_attention)
                values = self.value_head(
                    hidden_states.to(next(self.value_head.parameters()).dtype)
                ).squeeze(-1)[:, :-1]

                # Policy loss
                policy_loss, clip_fraction = self.ppo_loss(
                    log_probs, old_log_probs[idx], advantages[idx], mb_mask
                )

                # Value loss
                vf_loss = self.value_loss(
                    values, old_values[idx], returns[idx], mb_mask
                )

                # Entropy bonus over response tokens only
                entropy_bonus = (entropy * mb_mask).sum() / mb_mask.sum().clamp(min=1)

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_coef * vf_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                # Stage adaptive state until this inner update has committed both
                # optimizers. Rejected losses/gradients must not consume adaptation.
                proposed_kl_coef = None
                if self.config.beta > 0 and self.ref_model is not None:
                    with torch.no_grad():
                        ref_outputs = self.ref_model(
                            input_ids=mb_ids, attention_mask=mb_attention
                        )
                        ref_logits = ref_outputs.logits[:, :-1, :].float() / temperature
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        ref_token_log_probs = ref_log_probs.gather(
                            dim=-1, index=mb_ids[:, 1:].unsqueeze(-1)
                        ).squeeze(-1)

                    kl = self.compute_kl_divergence(
                        log_probs, ref_token_log_probs, mb_mask
                    )

                    kl_coef = self.config.beta
                    if self.kl_controller:
                        proposed_kl_coef = self.kl_controller.propose(kl.item(), 1)
                        kl_coef = proposed_kl_coef

                    total_loss = total_loss + kl_coef * kl

                # Reject an invalid objective before discarding existing gradients.
                if total_loss.numel() != 1 or not torch.isfinite(total_loss).all():
                    raise FloatingPointError("PPO loss must be a finite scalar")

                # Both optimizers share this backward pass. Check both gradient
                # populations before either optimizer can commit an update.
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.value_head.parameters(),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    self.value_optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "PPO gradients must remain finite"
                    ) from exc

                self._ppo_update_in_flight = True
                self.policy_optimizer.step()
                self.value_optimizer.step()
                if proposed_kl_coef is not None and self.kl_controller is not None:
                    self.kl_controller.kl_coef = proposed_kl_coef

                total_policy_loss += policy_loss.item()
                total_value_loss += vf_loss.item()
                total_entropy += entropy_bonus.item()
                total_clip_fraction += clip_fraction.item()

                num_updates += 1
        # Average the measured inner optimizer updates.
        num_epochs = num_updates
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
            "average_reward": float(rewards_tensor.double().mean().item()),
            "optimizer_updates": float(num_updates),
        }

        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        return metrics

    async def train(
        self,
        prompts: list[str],
        num_episodes: int | None = None,
    ) -> dict[str, Any]:
        """Train to a total update budget using retained on-policy token samples.

        Prompts cycle deterministically in mini_batch_size windows, with
        num_generations completions per prompt. A complete checkpoint restores
        the prompt cursor (global_step) and sampling RNG. Rewards may be sync or
        async; each receives the original prompt and decoded sampled completion.
        See ppo_online for the bounded, single-process sampling protocol.
        """
        from .ppo_online import train_online

        return await train_online(self, prompts, num_episodes)

    def save_runtime_checkpoint(self, output_dir: str) -> None:
        """Publish a PPO runtime only between complete online batches."""
        if getattr(self, "_ppo_online_active", False):
            raise RuntimeError("Cannot checkpoint an active PPO online operation")
        super().save_runtime_checkpoint(output_dir)

    def load_runtime_checkpoint(
        self, checkpoint_dir: str, *, trusted: bool = False
    ) -> bool:
        """Restore a complete PPO runtime between online batches."""
        if getattr(self, "_ppo_online_active", False):
            raise RuntimeError("Cannot reload an active PPO online operation")
        return super().load_runtime_checkpoint(checkpoint_dir, trusted=trusted)

    def save_checkpoint(self, path: str | None = None) -> str:
        """Publish policy and critic state through the shared runtime."""
        if path is None:
            path = os.path.join(
                self.config.output_dir, f"ppo-checkpoint-{self.global_step}"
            )
        self.save_runtime_checkpoint(path)
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
    Load a native causal model and run token-exact, single-process online PPO.

    Args:
        model_name: HuggingFace model name
        prompts: Training prompts
        reward_fn: Reward function
        num_episodes: Number of training episodes
        **kwargs: Additional config parameters

    Returns:
        Training results
    """
    from .ppo_online import validate_online_config, validate_prompts

    config = PPOConfig(model_name=model_name, num_episodes=num_episodes, **kwargs)
    validate_training_config(config)
    validate_online_config(config)
    validate_prompts(prompts)
    if config.use_4bit or config.use_8bit:
        raise ValueError("Online PPO convenience loading does not support quantization")
    if not _load_transformers_ppo():
        raise ImportError("Online PPO requires transformers")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if np is not None:
        np.random.seed(config.seed)
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(
        model_name, revision=config.model_revision, trust_remote_code=False
    )
    model = cast(Any, AutoModelForCausalLM).from_pretrained(
        model_name,
        revision=config.model_revision,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                task_type="CAUSAL_LM",
            ),
        )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    reference = (
        copy.deepcopy(model).eval().requires_grad_(False) if config.beta > 0 else None
    )
    trainer = PPOTrainer(config, model, tokenizer, reward_fn, reference)
    metrics = await trainer.train(prompts)
    checkpoint = trainer.save_checkpoint()
    return {**metrics, "checkpoint": checkpoint}
