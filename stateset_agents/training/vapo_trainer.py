"""
Value-Augmented Policy Optimization (VAPO) Training for StateSet Agents

VAPO is an advanced PPO variant that addresses key weaknesses in value-based RL
for long chain-of-thought reasoning. It achieves state-of-the-art results
(60.4 on AIME 2024) through seven key modifications:

1. Value network warmup (50 steps)
2. Decoupled GAE computation (separate lambda for critic/policy)
3. Length-adaptive lambda for policy
4. Asymmetric clipping (Clip-Higher)
5. Token-level loss normalization
6. Positive example LM loss addition
7. Group-sampling strategy

Reference: https://arxiv.org/abs/2504.05118
"""

import asyncio
import json
import logging
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
from .config import TrainingConfig, get_config_for_task

try:
    import numpy as np
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please install: pip install peft datasets wandb")
    raise

# Lazy import transformers to avoid torch/torchvision compatibility issues
_transformers_vapo_loaded = False
AutoModelForCausalLM = None
AutoTokenizer = None
get_cosine_schedule_with_warmup = None

def _load_transformers_vapo():
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_vapo_loaded, AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup
    if _transformers_vapo_loaded:
        return True
    # Allow pre-injected mocks without importing transformers.
    if AutoModelForCausalLM is not None and AutoTokenizer is not None:
        _transformers_vapo_loaded = True
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
        _transformers_vapo_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


@dataclass
class VAPOConfig(TrainingConfig):
    """
    Configuration for VAPO training.

    VAPO uses value-based RL with seven key modifications for stable
    long-CoT reasoning training.
    """

    # Model
    model_name: str = "gpt2"

    # Iterative training (legacy alias)
    num_iterations: int = 1

    # Generation parameters
    max_prompt_length: int = 256
    max_completion_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Model optimization
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Value network parameters
    value_hidden_size: int = 1024
    value_num_layers: int = 2
    value_warmup_steps: int = 50  # Steps to pretrain value network

    # Decoupled GAE parameters
    lambda_critic: float = 1.0  # GAE lambda for value network (unbiased)
    lambda_policy_alpha: float = 0.05  # Alpha for length-adaptive lambda

    # Asymmetric clipping (Clip-Higher)
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28

    # Token-level loss
    use_token_level_loss: bool = True

    # Positive example LM loss
    use_positive_lm_loss: bool = True
    positive_lm_weight: float = 0.1  # mu weight for NLL loss

    # Group sampling
    group_size: int = 16  # Samples per prompt
    num_prompts_per_batch: int = 512

    # Learning rates (separate for actor and critic)
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 2e-6

    # Mini-batch
    mini_batch_size: int = 512

    # Value loss coefficient
    value_loss_coef: float = 0.5

    # Entropy bonus (usually small or 0 for reasoning)
    entropy_coef: float = 0.0

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> "VAPOConfig":
        """Create VAPO config from standard training config"""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


class VAPOModelManager:
    """Manages model loading for VAPO training"""

    def __init__(self, config: VAPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load model and tokenizer with optional LoRA"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch.float16 if self.config.fp16 else (
                torch.bfloat16 if self.config.bf16 else torch.float32
            ),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **model_kwargs
        )

        # Add LoRA adapters if configured
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model

        logger.info(f"Model loaded on {self.device}")
        return self.model, self.tokenizer


class ValueHead(nn.Module):
    """
    Value head network for VAPO.

    Predicts state values for advantage estimation.
    Uses a simple MLP on top of the language model hidden states.
    """

    def __init__(
        self,
        hidden_size: int,
        value_hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_size = hidden_size

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_size, value_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_size = value_hidden_size

        # Final layer outputs scalar value
        layers.append(nn.Linear(in_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute values from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            values: [batch, seq_len, 1]
        """
        return self.network(hidden_states)


class LengthAdaptiveGAE:
    """
    Implements VAPO's Length-Adaptive GAE.

    Uses different lambda values for critic (lambda=1.0 for unbiased)
    and policy (adaptive based on sequence length).

    lambda_policy = 1 - 1/(alpha * length)

    This ensures the sum of GAE coefficients is proportional to output length,
    balancing bias-variance across variable-length sequences.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_critic: float = 1.0,
        lambda_policy_alpha: float = 0.05,
    ):
        self.gamma = gamma
        self.lambda_critic = lambda_critic
        self.lambda_policy_alpha = lambda_policy_alpha

    def compute_lambda_policy(self, sequence_length: int) -> float:
        """
        Compute length-adaptive lambda for policy.

        lambda = 1 - 1/(alpha * length + 1)

        This ensures lambda is in (0, 1) and increases with sequence length.
        """
        return 1.0 - 1.0 / (self.lambda_policy_alpha * sequence_length + 1.0)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        lambda_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: [batch, seq_len] rewards at each step
            values: [batch, seq_len] value predictions
            dones: [batch, seq_len] episode termination flags
            lambda_value: GAE lambda parameter

        Returns:
            advantages: [batch, seq_len]
            returns: [batch, seq_len] (advantages + values)
        """
        batch_size, seq_len = rewards.shape
        device = rewards.device

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=device)

        # Compute GAE backwards
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=device)
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            last_gae = delta + self.gamma * lambda_value * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values

        return advantages, returns

    def compute_decoupled_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute decoupled GAE for VAPO.

        Returns separate advantages for critic (lambda=1) and policy (length-adaptive).

        Args:
            rewards: [batch, seq_len]
            values: [batch, seq_len]
            dones: [batch, seq_len]
            sequence_lengths: [batch] length of each sequence

        Returns:
            critic_advantages: For value function training (lambda=1)
            policy_advantages: For policy training (length-adaptive lambda)
            returns: Value targets
        """
        # Critic GAE (unbiased, lambda=1)
        critic_advantages, returns = self.compute_gae(
            rewards, values, dones, self.lambda_critic
        )

        # Policy GAE (length-adaptive)
        batch_size = rewards.shape[0]
        policy_advantages = torch.zeros_like(rewards)

        for i in range(batch_size):
            seq_len = int(sequence_lengths[i].item())
            lambda_policy = self.compute_lambda_policy(seq_len)

            # Compute per-sample GAE with adaptive lambda
            single_rewards = rewards[i:i+1]
            single_values = values[i:i+1]
            single_dones = dones[i:i+1]

            adv, _ = self.compute_gae(
                single_rewards, single_values, single_dones, lambda_policy
            )
            policy_advantages[i] = adv[0]

        return critic_advantages, policy_advantages, returns


class VAPOTrainer:
    """
    Value-Augmented Policy Optimization (VAPO) Trainer

    VAPO achieves 60.4 on AIME 2024 (SOTA) through seven key modifications to PPO:

    1. Value warmup: Pretrain value network for 50 steps
    2. Decoupled GAE: Separate lambda for critic (1.0) and policy (adaptive)
    3. Length-adaptive lambda: lambda = 1 - 1/(alpha * length)
    4. Clip-Higher: Asymmetric clipping [1-0.2, 1+0.28]
    5. Token-level loss: Normalize by total tokens
    6. Positive LM loss: Add NLL on correct samples
    7. Group sampling: More samples per prompt, fewer prompts

    Reference: https://arxiv.org/abs/2504.05118
    """

    def __init__(
        self,
        config: VAPOConfig,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        verifier_fn: Optional[Callable[[str, str], bool]] = None,
    ):
        # Ensure transformers is loaded for scheduler
        _load_transformers_vapo()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.verifier_fn = verifier_fn
        self.device = next(model.parameters()).device

        # Get hidden size from model config
        if hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", 768)
        else:
            hidden_size = 768

        # Initialize value head
        self.value_head = ValueHead(
            hidden_size=hidden_size,
            value_hidden_size=config.value_hidden_size,
            num_layers=config.value_num_layers,
        ).to(self.device)

        # Initialize GAE computer
        self.gae_computer = LengthAdaptiveGAE(
            gamma=0.99,
            lambda_critic=config.lambda_critic,
            lambda_policy_alpha=config.lambda_policy_alpha,
        )

        # Separate optimizers for actor and critic
        self.actor_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.actor_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        self.critic_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=config.critic_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        # Schedulers
        total_steps = config.num_episodes * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        if get_cosine_schedule_with_warmup is not None:
            self.actor_scheduler = get_cosine_schedule_with_warmup(
                self.actor_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            self.critic_scheduler = get_cosine_schedule_with_warmup(
                self.critic_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            # Fallback to constant learning rate if scheduler unavailable
            self.actor_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.actor_optimizer, factor=1.0, total_iters=total_steps
            )
            self.critic_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.critic_optimizer, factor=1.0, total_iters=total_steps
            )

        # Metrics
        self.metrics_history = {
            "policy_loss": [],
            "value_loss": [],
            "positive_lm_loss": [],
            "average_reward": [],
            "accuracy": [],
            "explained_variance": [],
        }

        self.global_step = 0
        self.value_warmup_complete = False

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden states from model"""
        with torch.set_grad_enabled(self.model.training):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Use last hidden state
            hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute values for a sequence"""
        hidden_states = self.get_hidden_states(input_ids, attention_mask)
        values = self.value_head(hidden_states)
        return values.squeeze(-1)

    def compute_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    async def generate_group_responses(
        self,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Generate a group of responses for VAPO"""
        responses = []

        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.config.group_size):
                input_ids = prompt_tokens["input_ids"].to(self.device)
                attention_mask = prompt_tokens["attention_mask"].to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                full_ids = outputs[0]
                response_length = len(full_ids) - prompt_length

                response_mask = torch.zeros(len(full_ids), device=self.device)
                response_mask[prompt_length:] = 1.0

                response_text = self.tokenizer.decode(
                    full_ids[prompt_length:], skip_special_tokens=True
                )

                responses.append({
                    "response": response_text,
                    "input_ids": full_ids,
                    "attention_mask": torch.ones_like(full_ids),
                    "response_mask": response_mask,
                    "sequence_length": response_length,
                    "prompt_length": prompt_length,
                })

        self.model.train()
        return responses

    async def warmup_value_network(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        Pretrain value network with Monte-Carlo returns.

        This mitigates value initialization bias by training the value
        network to predict actual returns before joint training.
        """
        logger.info(f"Starting value network warmup ({self.config.value_warmup_steps} steps)")

        total_value_loss = 0.0
        explained_variances = []

        for step in range(self.config.value_warmup_steps):
            # Sample prompt
            prompt = np.random.choice(prompts)

            # Generate responses
            responses = await self.generate_group_responses(prompt)

            if len(responses) == 0:
                continue

            # Compute rewards (Monte-Carlo returns)
            rewards = []
            for resp in responses:
                reward = self.reward_fn(prompt, resp["response"])
                rewards.append(reward)

            # Prepare batch
            max_len = max(len(r["input_ids"]) for r in responses)
            batch_size = len(responses)

            batch_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            batch_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            batch_response_mask = torch.zeros(batch_size, max_len, device=self.device)

            for i, resp in enumerate(responses):
                seq_len = len(resp["input_ids"])
                batch_input_ids[i, :seq_len] = resp["input_ids"]
                batch_attention_mask[i, :seq_len] = resp["attention_mask"]
                batch_response_mask[i, :seq_len] = resp["response_mask"]

            # Monte-Carlo return targets (same reward for all tokens in response)
            rewards_tensor = torch.tensor(rewards, device=self.device)
            mc_returns = rewards_tensor.unsqueeze(1).expand(batch_size, max_len)
            mc_returns = mc_returns * batch_response_mask

            # Compute values
            with torch.no_grad():
                hidden_states = self.get_hidden_states(batch_input_ids, batch_attention_mask)

            values = self.value_head(hidden_states).squeeze(-1)

            # Value loss (MSE)
            value_loss = F.mse_loss(
                values * batch_response_mask,
                mc_returns,
                reduction="sum"
            ) / batch_response_mask.sum().clamp(min=1)

            # Update value network
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()

            total_value_loss += value_loss.item()

            # Compute explained variance
            with torch.no_grad():
                var_returns = mc_returns[batch_response_mask > 0].var()
                var_residual = (mc_returns - values)[batch_response_mask > 0].var()
                if var_returns > 1e-8:
                    explained_var = 1 - var_residual / var_returns
                    explained_variances.append(explained_var.item())

            if (step + 1) % 10 == 0:
                avg_loss = total_value_loss / (step + 1)
                avg_ev = np.mean(explained_variances[-10:]) if explained_variances else 0
                logger.info(f"Value warmup step {step + 1}/{self.config.value_warmup_steps} | "
                           f"Loss: {avg_loss:.4f} | EV: {avg_ev:.4f}")

        self.value_warmup_complete = True
        logger.info("Value network warmup complete")

        return {
            "warmup_value_loss": total_value_loss / self.config.value_warmup_steps,
            "warmup_explained_variance": np.mean(explained_variances) if explained_variances else 0,
        }

    def compute_vapo_losses(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        policy_advantages: torch.Tensor,
        critic_advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        response_mask: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAPO losses:
        1. Policy loss with Clip-Higher and token-level normalization
        2. Value loss
        3. Positive example LM loss
        """
        # Importance ratios
        log_ratio = current_log_probs - old_log_probs.detach()
        ratio = torch.exp(log_ratio)

        # Clip-Higher (asymmetric)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_eps_low,
            1.0 + self.config.clip_eps_high,
        )

        # Policy surrogate
        unclipped_obj = ratio * policy_advantages
        clipped_obj = clipped_ratio * policy_advantages
        surrogate = torch.min(unclipped_obj, clipped_obj)

        # Apply mask
        masked_surrogate = surrogate * response_mask

        # Token-level normalization
        if self.config.use_token_level_loss:
            total_tokens = response_mask.sum().clamp(min=1)
            policy_loss = -masked_surrogate.sum() / total_tokens
        else:
            policy_loss = -masked_surrogate.sum(dim=-1).mean()

        # Value loss (MSE with optional clipping)
        value_pred = values * response_mask
        value_target = returns * response_mask

        if self.config.value_clip > 0:
            # Clipped value loss
            old_values = values.detach()
            clipped_values = old_values + torch.clamp(
                values - old_values,
                -self.config.value_clip,
                self.config.value_clip,
            )
            value_loss_unclipped = (value_pred - value_target) ** 2
            value_loss_clipped = (clipped_values * response_mask - value_target) ** 2
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        else:
            value_loss = (value_pred - value_target) ** 2

        value_loss = value_loss.sum() / response_mask.sum().clamp(min=1)

        # Positive example LM loss
        if self.config.use_positive_lm_loss and positive_mask.sum() > 0:
            # NLL loss on correct samples
            positive_log_probs = current_log_probs * positive_mask
            positive_lm_loss = -positive_log_probs.sum() / positive_mask.sum().clamp(min=1)
        else:
            positive_lm_loss = torch.tensor(0.0, device=self.device)

        return policy_loss, value_loss, positive_lm_loss

    async def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        Execute one VAPO training step.

        1. Generate responses with group sampling
        2. Compute rewards and verify correctness
        3. Compute values and decoupled GAE
        4. Update policy and value networks
        """
        self.model.train()
        self.value_head.train()

        # Warmup value network if not done
        if not self.value_warmup_complete:
            warmup_metrics = await self.warmup_value_network(prompts)
            return warmup_metrics

        all_policy_losses = []
        all_value_losses = []
        all_positive_lm_losses = []
        all_rewards = []
        all_accuracies = []
        all_explained_variances = []

        for prompt in prompts[:self.config.per_device_train_batch_size]:
            # Generate group responses
            responses = await self.generate_group_responses(prompt)

            if len(responses) < 2:
                continue

            # Compute rewards and identify correct samples
            rewards = []
            is_correct = []

            for resp in responses:
                reward = self.reward_fn(prompt, resp["response"])
                rewards.append(reward)

                if self.verifier_fn:
                    correct = self.verifier_fn(prompt, resp["response"])
                else:
                    correct = reward > 0.5
                is_correct.append(correct)

            accuracy = sum(is_correct) / len(is_correct)
            all_accuracies.append(accuracy)
            all_rewards.extend(rewards)

            # Prepare batch
            max_len = max(len(r["input_ids"]) for r in responses)
            batch_size = len(responses)

            batch_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            batch_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            batch_response_mask = torch.zeros(batch_size, max_len, device=self.device)
            sequence_lengths = torch.zeros(batch_size, device=self.device)

            for i, resp in enumerate(responses):
                seq_len = len(resp["input_ids"])
                batch_input_ids[i, :seq_len] = resp["input_ids"]
                batch_attention_mask[i, :seq_len] = resp["attention_mask"]
                batch_response_mask[i, :seq_len] = resp["response_mask"]
                sequence_lengths[i] = resp["sequence_length"]

            # Positive mask for LM loss (correct samples only)
            positive_mask = torch.zeros(batch_size, max_len - 1, device=self.device)
            for i, correct in enumerate(is_correct):
                if correct:
                    # Mask response tokens for correct samples
                    resp_mask = batch_response_mask[i, 1:]  # Shift for next-token
                    positive_mask[i] = resp_mask

            # Get old log probs and values
            with torch.no_grad():
                old_log_probs = self.compute_token_log_probs(batch_input_ids, batch_attention_mask)
                old_values = self.compute_values(batch_input_ids, batch_attention_mask)

            # Create reward tensor (same reward at each response token)
            rewards_tensor = torch.tensor(rewards, device=self.device)
            reward_sequence = rewards_tensor.unsqueeze(1).expand(batch_size, max_len)
            reward_sequence = reward_sequence * batch_response_mask

            # Terminal mask (1 at last token of each response)
            dones = torch.zeros(batch_size, max_len, device=self.device)
            for i, resp in enumerate(responses):
                last_idx = resp["prompt_length"] + resp["sequence_length"] - 1
                if last_idx < max_len:
                    dones[i, last_idx] = 1.0

            # Compute decoupled GAE
            critic_advantages, policy_advantages, returns = self.gae_computer.compute_decoupled_gae(
                reward_sequence, old_values, dones, sequence_lengths
            )

            # Normalize advantages
            policy_adv_masked = policy_advantages[batch_response_mask > 0]
            if len(policy_adv_masked) > 1:
                policy_advantages = (policy_advantages - policy_adv_masked.mean()) / (policy_adv_masked.std() + 1e-8)

            # Compute current log probs and values
            current_log_probs = self.compute_token_log_probs(batch_input_ids, batch_attention_mask)
            current_values = self.compute_values(batch_input_ids, batch_attention_mask)

            # Shift masks for loss computation
            shifted_response_mask = batch_response_mask[:, 1:]
            shifted_policy_adv = policy_advantages[:, :-1]
            shifted_critic_adv = critic_advantages[:, :-1]
            shifted_returns = returns[:, :-1]
            shifted_values = current_values[:, :-1]

            # Compute VAPO losses
            policy_loss, value_loss, positive_lm_loss = self.compute_vapo_losses(
                current_log_probs,
                old_log_probs,
                shifted_policy_adv,
                shifted_critic_adv,
                shifted_values,
                shifted_returns,
                shifted_response_mask,
                positive_mask,
            )

            # Total loss
            total_loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                + self.config.positive_lm_weight * positive_lm_loss
            )

            # Backward and update
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.config.max_grad_norm)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            all_policy_losses.append(policy_loss.item())
            all_value_losses.append(value_loss.item())
            all_positive_lm_losses.append(positive_lm_loss.item())

            # Compute explained variance
            with torch.no_grad():
                returns_masked = returns[batch_response_mask > 0]
                values_masked = current_values[batch_response_mask > 0]
                if len(returns_masked) > 1:
                    var_returns = returns_masked.var()
                    var_residual = (returns_masked - values_masked).var()
                    if var_returns > 1e-8:
                        explained_var = 1 - var_residual / var_returns
                        all_explained_variances.append(explained_var.item())

        # Update schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.global_step += 1

        # Compute metrics
        metrics = {
            "policy_loss": np.mean(all_policy_losses) if all_policy_losses else 0.0,
            "value_loss": np.mean(all_value_losses) if all_value_losses else 0.0,
            "positive_lm_loss": np.mean(all_positive_lm_losses) if all_positive_lm_losses else 0.0,
            "average_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "accuracy": np.mean(all_accuracies) if all_accuracies else 0.0,
            "explained_variance": np.mean(all_explained_variances) if all_explained_variances else 0.0,
            "actor_lr": self.actor_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
            "global_step": self.global_step,
        }

        # Store metrics
        for key in ["policy_loss", "value_loss", "average_reward", "accuracy", "explained_variance"]:
            if key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])

        return metrics

    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save value head
        torch.save(
            self.value_head.state_dict(),
            os.path.join(output_dir, "value_head.pt")
        )

        # Save training state
        state = {
            "global_step": self.global_step,
            "value_warmup_complete": self.value_warmup_complete,
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
            "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
            "metrics_history": self.metrics_history,
        }
        torch.save(state, os.path.join(output_dir, "training_state.pt"))

        # Save config
        config_path = os.path.join(output_dir, "vapo_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint"""
        # Load value head
        value_head_path = os.path.join(checkpoint_dir, "value_head.pt")
        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(torch.load(value_head_path, map_location=self.device))

        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.value_warmup_complete = state.get("value_warmup_complete", True)
            self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
            self.actor_scheduler.load_state_dict(state["actor_scheduler_state_dict"])
            self.critic_scheduler.load_state_dict(state["critic_scheduler_state_dict"])
            self.metrics_history = state.get("metrics_history", self.metrics_history)

        logger.info(f"Checkpoint loaded from {checkpoint_dir}")


async def train_with_vapo(
    model_name: str,
    reward_fn: Callable[[str, str], float],
    train_prompts: List[str],
    config: Optional[VAPOConfig] = None,
    verifier_fn: Optional[Callable[[str, str], bool]] = None,
    output_dir: str = "./outputs/vapo",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, List[float]]]:
    """
    Train a model using VAPO algorithm.

    VAPO is the current SOTA for long-CoT reasoning (60.4 on AIME 2024).

    Args:
        model_name: HuggingFace model name or path
        reward_fn: Function (prompt, response) -> reward
        train_prompts: List of training prompts
        config: VAPO configuration
        verifier_fn: Optional binary verifier (prompt, response) -> correct
        output_dir: Directory to save checkpoints
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name

    Returns:
        Tuple of (model, tokenizer, metrics_history)
    """
    logger.info("=" * 60)
    logger.info("VAPO Training - Value-Augmented Policy Optimization")
    logger.info("=" * 60)
    logger.info("SOTA: 60.4 on AIME 2024")
    logger.info("Key: Value warmup + Decoupled GAE + Length-adaptive lambda")

    if config is None:
        config = VAPOConfig(
            model_name=model_name,
            output_dir=output_dir,
        )

    # Initialize W&B
    if use_wandb and wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"vapo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["vapo", "rl-training", "value-based", "reasoning"],
        )

    # Load model
    logger.info(f"Loading model: {model_name}")
    model_manager = VAPOModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    # Create trainer
    trainer = VAPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        verifier_fn=verifier_fn,
    )

    # Training loop
    logger.info(f"Starting training with {len(train_prompts)} prompts")
    logger.info(f"Value warmup steps: {config.value_warmup_steps}")
    logger.info(f"Group size: {config.group_size}")

    os.makedirs(output_dir, exist_ok=True)

    for iteration in range(config.num_episodes):
        # Sample batch
        batch_size = min(config.per_device_train_batch_size, len(train_prompts))
        batch_indices = np.random.choice(len(train_prompts), batch_size, replace=False)
        batch_prompts = [train_prompts[i] for i in batch_indices]

        # Train step
        metrics = await trainer.train_step(batch_prompts)

        # Log
        if iteration % config.logging_steps == 0:
            if "warmup_value_loss" in metrics:
                logger.info(f"Value warmup | Loss: {metrics['warmup_value_loss']:.4f}")
            else:
                logger.info(
                    f"Iter {iteration}/{config.num_episodes} | "
                    f"Policy: {metrics['policy_loss']:.4f} | "
                    f"Value: {metrics['value_loss']:.4f} | "
                    f"Reward: {metrics['average_reward']:.4f} | "
                    f"Acc: {metrics['accuracy']:.2%} | "
                    f"EV: {metrics['explained_variance']:.4f}"
                )

            if use_wandb:
                wandb.log(metrics, step=iteration)

        # Save checkpoint
        if (iteration + 1) % config.save_steps == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{iteration + 1}")
            trainer.save_checkpoint(checkpoint_dir)

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    trainer.save_checkpoint(final_dir)

    if use_wandb:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("VAPO Training Complete!")
    logger.info("=" * 60)

    return model, tokenizer, trainer.metrics_history


# Export
__all__ = [
    "VAPOConfig",
    "VAPOTrainer",
    "ValueHead",
    "LengthAdaptiveGAE",
    "train_with_vapo",
]
