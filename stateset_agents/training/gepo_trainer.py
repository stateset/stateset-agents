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

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import rl_losses
from .checkpoint_io import load_checkpoint_file
from .config import TrainingConfig
from .trainer_runtime import (
    SharedModelManager,
    build_group_batch,
    save_checkpoint_artifacts,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import wandb as _wandb

    wandb: Any = _wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from peft import LoraConfig as _LoraConfig
    from peft import TaskType as _TaskType
    from peft import get_peft_model as _get_peft_model

    LoraConfig: Any = _LoraConfig
    TaskType: Any = _TaskType
    get_peft_model: Any = _get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None

# Lazy import transformers to avoid torch/torchvision compatibility issues
_transformers_gepo_loaded = False
AutoModelForCausalLM: Any | None = None
AutoTokenizer: Any | None = None
get_cosine_schedule_with_warmup: Any | None = None


def _load_transformers_gepo() -> bool:
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_gepo_loaded, AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup
    if _transformers_gepo_loaded:
        return True
    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import get_cosine_schedule_with_warmup as _get_cosine

        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        get_cosine_schedule_with_warmup = _get_cosine
        _transformers_gepo_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


def _require_peft() -> None:
    """Require PEFT only when LoRA model loading actually needs it."""
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError(
            "PEFT is required for GEPO LoRA training. "
            "Install with `pip install stateset-agents[training]` or `pip install peft`."
        )


def _require_wandb() -> None:
    """Require W&B only when experiment logging is enabled."""
    if wandb is None:
        raise ImportError(
            "wandb is required for GEPO logging. "
            "Install with `pip install stateset-agents[training]` or `pip install wandb`."
        )


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

    # Inner gradient updates per rollout (mu). The default of 1 keeps GEPO
    # strictly on-policy; values > 1 reuse each rollout, and the sampler
    # (old-policy) log probs are the snapshot taken at rollout time.
    # Cadence (shared with DAPO and VAPO): the LR scheduler advances once per
    # inner update; global_step counts train_steps, not inner updates.
    num_gradient_updates: int = 1

    @classmethod
    def from_training_config(
        cls, config: TrainingConfig, **kwargs: Any
    ) -> "GEPOConfig":
        """Create GEPO config from standard training config"""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        # Override num_generations with group_size
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


class GEPOModelManager(SharedModelManager):
    """Manages model loading for GEPO training"""

    def __init__(self, config: GEPOConfig):
        super().__init__(config)

    def _get_transformers(self) -> tuple[Any, Any]:
        if not _load_transformers_gepo():
            raise ImportError(
                "transformers is required for GEPO training. "
                "Install with `pip install stateset-agents[training]`."
            )
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers GEPO loader did not initialize correctly")
        return AutoTokenizer, AutoModelForCausalLM

    def _peft_components(self) -> tuple[Any, Any, Any]:
        _require_peft()
        return LoraConfig, TaskType, get_peft_model


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
        ref_model: Any | None = None,
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
        self.metrics_history: dict[str, list[float]] = {
            "policy_loss": [],
            "average_reward": [],
            "kl_divergence": [],
            "gepo_coefficient": [],
            "advantage_std": [],
        }

        self.global_step = 0
        self.rollout_samples_total = 0

    @staticmethod
    def build_response_mask(
        attention_mask: torch.Tensor,
        response_start_idx: int,
    ) -> torch.Tensor:
        """
        Build a mask (on the shifted next-token-prediction axis) that selects
        response tokens only, excluding prompt tokens and padding.

        `response_start_idx` is the (unshifted) index of the first response
        token; the shifted axis has length `seq_len - 1`, so the equivalent
        shifted start index is `max(response_start_idx - 1, 0)`, matching the
        convention used in gspo_trainer.py.
        """
        shift_mask = attention_mask[:, 1:].contiguous()
        shifted_start = max(response_start_idx - 1, 0)

        response_mask = torch.zeros_like(shift_mask)
        if shifted_start < response_mask.shape[1]:
            response_mask[:, shifted_start:] = shift_mask[:, shifted_start:]

        return response_mask

    def compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out prompt tokens and padding
        response_mask = self.build_response_mask(attention_mask, response_start_idx)

        masked_log_probs = token_log_probs * response_mask

        # Sum over sequence
        sequence_log_probs = masked_log_probs.sum(dim=-1)

        return token_log_probs, sequence_log_probs

    @staticmethod
    def compute_gepo_coefficient_static(
        learner_seq_log_probs: torch.Tensor,
        sampler_seq_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GEPO coefficient using Group Expectation Importance Weights,
        entirely in log space to avoid underflow for realistic (very negative)
        sequence log-probability sums.

        coef_i = p_i / E_qhat[q], where E_qhat[q] = sum(q^2) / sum(q)

        Args:
            learner_seq_log_probs: Sequence log-probs from current policy [group_size]
            sampler_seq_log_probs: Sequence log-probs from sampling policy [group_size]

        Returns:
            GEPO coefficients (linear space) for each sequence [group_size]
        """
        sampler_lp = sampler_seq_log_probs.detach()

        # log E_qhat[q] = log( sum(q^2)/sum(q) ) = logsumexp(2*lq) - logsumexp(lq)
        log_group_expectation = torch.logsumexp(
            2 * sampler_lp, dim=0
        ) - torch.logsumexp(sampler_lp, dim=0)

        log_coef = learner_seq_log_probs - log_group_expectation
        # Same overflow guard as DAPO/VAPO, at GEPO's historical +/-30 bound:
        # group coefficients are not ratios against a clip boundary, so the
        # wider window is kept.
        return rl_losses.safe_exp_ratio(log_coef, clamp=30.0)

    def compute_gepo_coefficient(
        self,
        learner_seq_log_probs: torch.Tensor,
        sampler_seq_log_probs: torch.Tensor,
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
            learner_seq_log_probs: Sequence log-probs from current policy [group_size]
            sampler_seq_log_probs: Sequence log-probs from sampling policy [group_size]

        Returns:
            GEPO coefficients for each sequence [group_size]
        """
        return self.compute_gepo_coefficient_static(
            learner_seq_log_probs, sampler_seq_log_probs
        )

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute group-relative advantages using within-group baseline normalization.

        A_i = (r_i - mean(rewards)) / std(rewards)

        Args:
            rewards: Tensor of rewards for group [group_size]

        Returns:
            advantages: Normalized advantages [group_size]
            stats: Reward statistics
        """
        advantages = rl_losses.group_advantages(rewards)

        std_reward = rewards.float().std(correction=0) if rewards.numel() > 1 else 0.0
        stats = {
            "mean_reward": rewards.mean().item(),
            "std_reward": float(std_reward),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
        }

        return advantages, stats

    async def generate_group_responses(
        self,
        prompt: str,
        group_size: int,
    ) -> list[dict[str, Any]]:
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
                response_text = self.tokenizer.decode(
                    outputs[0][prompt_length:], skip_special_tokens=True
                )

                responses.append(
                    {
                        "response": response_text,
                        "input_ids": outputs[0],
                        "attention_mask": torch.ones_like(outputs[0]),
                        "response_start_idx": prompt_length,
                    }
                )

        self.model.train()
        self.rollout_samples_total += len(responses)
        return responses

    async def train_step(
        self,
        prompts: list[str],
    ) -> dict[str, float]:
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

        all_rewards: list[float] = []
        all_advantages: list[float] = []
        all_gepo_coefs: list[float] = []

        # --- Rollout phase ------------------------------------------------
        # Generate every group and freeze the sampler (old-policy) sequence
        # log probs here, before any inner update can move the weights. When
        # they were recomputed from the just-updated model the GEPO
        # coefficient collapsed to ~1 and clipping never fired.
        rollouts: list[dict[str, Any]] = []
        for prompt in prompts:
            group_responses = await self.generate_group_responses(
                prompt, self.config.group_size
            )

            if len(group_responses) < 2:
                logger.warning("Insufficient responses for prompt, skipping")
                continue

            rewards = [
                self.reward_fn(prompt, resp["response"]) for resp in group_responses
            ]
            rewards_tensor = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            )
            all_rewards.extend(rewards)

            advantages, _reward_stats = self.compute_group_advantages(rewards_tensor)
            all_advantages.extend(advantages.tolist())

            batch_input_ids, batch_attention_mask, _ = build_group_batch(
                group_responses, self.device, include_response_mask=False
            )
            response_start_idx = group_responses[0]["response_start_idx"]

            with torch.no_grad():
                _, sampler_seq_log_probs = self.compute_sequence_log_probs(
                    batch_input_ids, batch_attention_mask, response_start_idx
                )

            rollouts.append(
                {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "response_start_idx": response_start_idx,
                    "advantages": advantages,
                    "sampler_seq_log_probs": sampler_seq_log_probs.detach(),
                }
            )

        num_groups = len(rollouts)
        if num_groups == 0:
            logger.warning("No valid groups in batch")
            return {"policy_loss": 0.0, "average_reward": 0.0}

        # --- Update phase --------------------------------------------------
        # ``num_gradient_updates`` (mu) inner updates reuse the same rollouts
        # against the frozen sampler log probs, exactly like DAPO. With the
        # default of 1 the formula is identical to the previous single-update
        # path (only dropout RNG ordering differs, since the sampler log probs
        # are now drawn before the learner ones).
        num_updates = max(1, getattr(self.config, "num_gradient_updates", 1))
        final_loss = torch.tensor(0.0, device=self.device)

        for _ in range(num_updates):
            all_gepo_coefs = []
            accumulated_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for rollout in rollouts:
                _, learner_seq_log_probs = self.compute_sequence_log_probs(
                    rollout["input_ids"],
                    rollout["attention_mask"],
                    rollout["response_start_idx"],
                )

                gepo_coefs = self.compute_gepo_coefficient(
                    learner_seq_log_probs, rollout["sampler_seq_log_probs"]
                )
                all_gepo_coefs.extend(gepo_coefs.detach().tolist())

                # Shared PPO-style clipped surrogate (already a loss).
                policy_loss = rl_losses.clipped_surrogate(
                    gepo_coefs,
                    rollout["advantages"],
                    clip_low=self.config.clip_eps,
                    clip_high=self.config.clip_eps,
                ).mean()

                accumulated_loss = accumulated_loss + policy_loss

            final_loss = accumulated_loss / num_groups

            self.optimizer.zero_grad()
            final_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.optimizer.step()
            # Cadence convention shared with DAPO/VAPO: the LR scheduler
            # advances once per inner update, global_step counts train_steps.
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

    def save_checkpoint(self, output_dir: str) -> None:
        """Save model checkpoint"""
        save_checkpoint_artifacts(
            self.model,
            self.tokenizer,
            output_dir,
            training_state={
                "global_step": self.global_step,
                "rollout_samples_total": self.rollout_samples_total,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics_history": self.metrics_history,
            },
            config_dict=self.config.to_dict(),
            config_filename="gepo_config.json",
        )

    def load_checkpoint(self, checkpoint_dir: str, *, trusted: bool = False) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_dir: Directory written by :meth:`save_checkpoint`.
            trusted: Pass ``True`` only for checkpoints from a source you
                control; the default unpickles with ``weights_only=True`` so a
                malicious checkpoint cannot execute code.
        """
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = load_checkpoint_file(
                state_path, map_location=self.device, trusted=trusted
            )
            self.global_step = state["global_step"]
            self.rollout_samples_total = int(
                state.get("rollout_samples_total", self.rollout_samples_total)
            )
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.metrics_history = state.get("metrics_history", self.metrics_history)
            logger.info(f"Checkpoint loaded from {checkpoint_dir}")


async def train_with_gepo(
    model_name: str,
    reward_fn: Callable[[str, str], float],
    train_prompts: list[str],
    config: GEPOConfig | None = None,
    output_dir: str = "./outputs/gepo",
    use_wandb: bool = False,
    wandb_project: str | None = None,
) -> tuple[Any, Any, dict[str, list[float]]]:
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
        _require_wandb()
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
    trainer.metrics_history["rollout_samples_total"] = [
        float(trainer.rollout_samples_total)
    ]

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
