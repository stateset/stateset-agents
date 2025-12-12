"""
Group Expectation Policy Optimization (GEPO) Training for StateSet Agents

GEPO is an advanced RL algorithm that uses group-level importance weights to
exponentially reduce variance under high KL divergence. This makes it particularly
robust for heterogeneous and distributed training environments.

Key innovations:
- Group Expectation Importance Weights (GEIW) instead of per-token or per-sequence
- Exponentially reduces variance when KL divergence is high
- Superior stability under network latency and heterogeneous compute

Reference: https://arxiv.org/abs/2508.17850
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# Try to import from gspo_trainer, with fallback for standalone usage
try:
    from .gspo_trainer import GSPOConfig, GSPOModelManager
except ImportError:
    # Fallback - define minimal versions if gspo_trainer unavailable
    GSPOConfig = None
    GSPOModelManager = None

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
_transformers_gepo_loaded = False
AutoModelForCausalLM = None
AutoTokenizer = None
get_cosine_schedule_with_warmup = None

def _load_transformers_gepo():
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_gepo_loaded, AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup
    if _transformers_gepo_loaded:
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
        _transformers_gepo_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


@dataclass
class GEPOConfig(TrainingConfig):
    """
    Configuration for GEPO training.

    GEPO uses group-level importance weights which provide superior stability
    compared to token-level (GRPO) or sequence-level (GSPO) weights.
    """

    # Model
    model_name: str = "gpt2"

    # GEPO specific parameters
    group_size: int = 8  # Number of responses per prompt (G)

    # Clipping (standard PPO-style, applied after GEPO coefficient computation)
    clip_eps: float = 0.2  # Clipping epsilon for policy ratio

    # KL penalty (typically set to 0 for GEPO as group weights handle divergence)
    beta: float = 0.0
    use_reference_model: bool = False

    # Training parameters from the paper
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.03  # 3% linear warmup
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 8

    # Generation parameters
    max_prompt_length: int = 256
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Model optimization
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Advantage computation
    use_group_baseline: bool = True  # Use within-group baseline normalization

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> "GEPOConfig":
        """Create GEPO config from standard training config"""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        # Override num_generations with group_size
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


class GEPOModelManager:
    """Manages model loading for GEPO training"""

    def __init__(self, config: GEPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.ref_model = None
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


class GEPOTrainer:
    """
    Group Expectation Policy Optimization (GEPO) Trainer

    GEPO improves upon GRPO and GSPO by using group-level importance weights:

    w_GEIW(y|x) = p(y|x) / E_q[q(y|x)]

    where the group expectation is computed as:
    E_q[q(y|x)] ≈ Σ q(y^i|x)² / Σ q(y^i|x)

    This exponentially reduces variance under high KL divergence, making training
    stable even with network delays or heterogeneous compute resources.

    Reference: https://arxiv.org/abs/2508.17850
    """

    def __init__(
        self,
        config: GEPOConfig,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        ref_model: Optional[Any] = None,
    ):
        # Ensure transformers is loaded for scheduler
        _load_transformers_gepo()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ref_model = ref_model
        self.device = next(model.parameters()).device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        # Scheduler
        total_steps = config.num_episodes * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        if get_cosine_schedule_with_warmup is not None:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            # Fallback to constant learning rate if scheduler unavailable
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=total_steps
            )

        # Metrics tracking
        self.metrics_history = {
            "policy_loss": [],
            "average_reward": [],
            "kl_divergence": [],
            "gepo_coefficient": [],
            "advantage_std": [],
        }

        self.global_step = 0

    def compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for response tokens.

        Returns:
            token_log_probs: Log probs for each token [batch, seq_len]
            sequence_log_probs: Sum of log probs per sequence [batch]
        """
        with torch.set_grad_enabled(self.model.training):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out prompt tokens and padding
        response_mask = torch.zeros_like(shift_mask)
        response_mask[:, response_start_idx:] = shift_mask[:, response_start_idx:]

        masked_log_probs = token_log_probs * response_mask

        # Sum over sequence
        sequence_log_probs = masked_log_probs.sum(dim=-1)

        return token_log_probs, sequence_log_probs

    def compute_gepo_coefficient(
        self,
        learner_seq_probs: torch.Tensor,
        sampler_seq_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GEPO coefficient using Group Expectation Importance Weights.

        The GEPO coefficient is:
        coef = p_learner / E_q[q_sampler]

        where E_q[q] ≈ Σ q² / Σ q (normalized within-group probabilities)

        This aggregates across the entire group using a common denominator,
        providing superior variance reduction compared to per-token or per-sequence
        importance weights.

        Args:
            learner_seq_probs: Sequence probabilities from current policy [group_size]
            sampler_seq_probs: Sequence probabilities from sampling policy [group_size]

        Returns:
            GEPO coefficients for each sequence [group_size]
        """
        # Detach sampler probs for stable computation
        sampler_detached = sampler_seq_probs.detach()

        # Compute normalized within-group probabilities: q_hat = q / sum(q)
        q_hat = sampler_detached / sampler_detached.sum()

        # Compute group expectation denominator: E_q[q] ≈ sum(q_hat * q)
        group_expectation = (q_hat * sampler_detached).sum()

        # GEPO coefficient: p / E_q[q]
        gepo_coef = learner_seq_probs / group_expectation

        return gepo_coef

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute group-relative advantages using within-group baseline normalization.

        A_i = (r_i - mean(rewards)) / std(rewards)

        Args:
            rewards: Tensor of rewards for group [group_size]

        Returns:
            advantages: Normalized advantages [group_size]
            stats: Reward statistics
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std()

        # Avoid division by zero
        if std_reward < 1e-8:
            std_reward = torch.tensor(1.0, device=rewards.device)

        advantages = (rewards - mean_reward) / std_reward

        stats = {
            "mean_reward": mean_reward.item(),
            "std_reward": std_reward.item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
        }

        return advantages, stats

    async def generate_group_responses(
        self,
        prompt: str,
        group_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate a group of responses for a single prompt.

        Returns list of dicts with:
            - response: Generated text
            - input_ids: Tokenized full sequence
            - attention_mask: Attention mask
            - response_start_idx: Index where response starts
        """
        responses = []

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            for _ in range(group_size):
                # Generate response
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

                # Decode response
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_text = self.tokenizer.decode(
                    outputs[0][prompt_length:], skip_special_tokens=True
                )

                responses.append({
                    "response": response_text,
                    "input_ids": outputs[0],
                    "attention_mask": torch.ones_like(outputs[0]),
                    "response_start_idx": prompt_length - 1,  # -1 for shift in log prob
                })

        self.model.train()
        return responses

    async def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        Execute one GEPO training step.

        For each prompt:
        1. Generate G responses (group)
        2. Compute rewards for each response
        3. Compute group advantages
        4. Compute GEPO coefficients
        5. Apply clipped policy gradient

        Args:
            prompts: List of prompts to train on

        Returns:
            Training metrics
        """
        self.model.train()

        total_loss = 0.0
        all_rewards = []
        all_advantages = []
        all_gepo_coefs = []
        num_groups = 0

        accumulated_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for prompt in prompts:
            # Generate group of responses
            group_responses = await self.generate_group_responses(
                prompt, self.config.group_size
            )

            if len(group_responses) < 2:
                logger.warning(f"Insufficient responses for prompt, skipping")
                continue

            # Compute rewards
            rewards = []
            for resp in group_responses:
                reward = self.reward_fn(prompt, resp["response"])
                rewards.append(reward)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            all_rewards.extend(rewards)

            # Compute group advantages
            advantages, reward_stats = self.compute_group_advantages(rewards_tensor)
            all_advantages.extend(advantages.tolist())

            # Stack inputs for batch processing
            max_len = max(r["input_ids"].shape[0] for r in group_responses)
            batch_input_ids = torch.zeros(
                len(group_responses), max_len, dtype=torch.long, device=self.device
            )
            batch_attention_mask = torch.zeros(
                len(group_responses), max_len, dtype=torch.long, device=self.device
            )

            for i, resp in enumerate(group_responses):
                seq_len = resp["input_ids"].shape[0]
                batch_input_ids[i, :seq_len] = resp["input_ids"]
                batch_attention_mask[i, :seq_len] = resp["attention_mask"]

            response_start_idx = group_responses[0]["response_start_idx"]

            # Compute current policy log probs (with gradients)
            _, learner_seq_log_probs = self.compute_sequence_log_probs(
                batch_input_ids, batch_attention_mask, response_start_idx
            )
            learner_seq_probs = torch.exp(learner_seq_log_probs)

            # Compute old policy log probs (detached, from generation time)
            # In practice, we use the same model but detached
            with torch.no_grad():
                _, sampler_seq_log_probs = self.compute_sequence_log_probs(
                    batch_input_ids, batch_attention_mask, response_start_idx
                )
                sampler_seq_probs = torch.exp(sampler_seq_log_probs)

            # Compute GEPO coefficients
            gepo_coefs = self.compute_gepo_coefficient(learner_seq_probs, sampler_seq_probs)
            all_gepo_coefs.extend(gepo_coefs.detach().tolist())

            # Apply PPO-style clipping
            clipped_coefs = torch.clamp(
                gepo_coefs,
                1.0 - self.config.clip_eps,
                1.0 + self.config.clip_eps,
            )

            # Compute policy loss: -min(coef * A, clipped_coef * A)
            unclipped_obj = gepo_coefs * advantages
            clipped_obj = clipped_coefs * advantages

            policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

            # Accumulate loss
            accumulated_loss = accumulated_loss + policy_loss
            num_groups += 1

        if num_groups == 0:
            logger.warning("No valid groups in batch")
            return {"policy_loss": 0.0, "average_reward": 0.0}

        # Average loss over groups
        final_loss = accumulated_loss / num_groups

        # Backward pass
        self.optimizer.zero_grad()
        final_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        # Update
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        # Compute metrics
        metrics = {
            "policy_loss": final_loss.item(),
            "average_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "advantage_std": np.std(all_advantages) if all_advantages else 0.0,
            "gepo_coefficient_mean": np.mean(all_gepo_coefs) if all_gepo_coefs else 1.0,
            "gepo_coefficient_std": np.std(all_gepo_coefs) if all_gepo_coefs else 0.0,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "global_step": self.global_step,
        }

        # Store metrics
        for key in ["policy_loss", "average_reward"]:
            if key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])

        return metrics

    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics_history": self.metrics_history,
        }
        torch.save(state, os.path.join(output_dir, "training_state.pt"))

        # Save config
        config_path = os.path.join(output_dir, "gepo_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint"""
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.metrics_history = state.get("metrics_history", self.metrics_history)
            logger.info(f"Checkpoint loaded from {checkpoint_dir}")


async def train_with_gepo(
    model_name: str,
    reward_fn: Callable[[str, str], float],
    train_prompts: List[str],
    config: Optional[GEPOConfig] = None,
    output_dir: str = "./outputs/gepo",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, List[float]]]:
    """
    Train a model using GEPO algorithm.

    Args:
        model_name: HuggingFace model name or path
        reward_fn: Function that takes (prompt, response) and returns reward
        train_prompts: List of training prompts
        config: GEPO configuration (uses defaults if None)
        output_dir: Directory to save checkpoints
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name

    Returns:
        Tuple of (model, tokenizer, metrics_history)
    """
    logger.info("=" * 60)
    logger.info("GEPO Training - Group Expectation Policy Optimization")
    logger.info("=" * 60)

    # Create config if not provided
    if config is None:
        config = GEPOConfig(
            model_name=model_name,
            output_dir=output_dir,
        )

    # Initialize W&B
    if use_wandb and wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"gepo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["gepo", "rl-training"],
        )

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model_manager = GEPOModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    # Create trainer
    trainer = GEPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        ref_model=model_manager.ref_model,
    )

    # Training loop
    logger.info(f"Starting training with {len(train_prompts)} prompts")
    logger.info(f"Group size: {config.group_size}")
    logger.info(f"Total iterations: {config.num_episodes}")

    os.makedirs(output_dir, exist_ok=True)

    for iteration in range(config.num_episodes):
        # Sample batch of prompts
        batch_size = min(config.per_device_train_batch_size, len(train_prompts))
        batch_indices = np.random.choice(len(train_prompts), batch_size, replace=False)
        batch_prompts = [train_prompts[i] for i in batch_indices]

        # Train step
        metrics = await trainer.train_step(batch_prompts)

        # Log metrics
        if iteration % config.logging_steps == 0:
            logger.info(
                f"Iteration {iteration}/{config.num_episodes} | "
                f"Loss: {metrics['policy_loss']:.4f} | "
                f"Reward: {metrics['average_reward']:.4f} | "
                f"GEPO Coef: {metrics['gepo_coefficient_mean']:.4f}"
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
    logger.info("GEPO Training Complete!")
    logger.info("=" * 60)

    return model, tokenizer, trainer.metrics_history


# Export
__all__ = [
    "GEPOConfig",
    "GEPOTrainer",
    "train_with_gepo",
]
