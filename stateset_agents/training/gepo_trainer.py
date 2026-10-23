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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch

from ..core.errors import ConfigurationError
from . import objectives, rl_losses
from .config import TrainingConfig
from .runtime_state import (
    TrainingRuntime,
    _reject_unsupported_resume,
    validate_training_config,
)
from .spine.loop import (
    dedicated_run_config,
    run_dedicated_loop,
    sample_prompts,
    seed_dedicated_run,
)
from .spine.rollouts import PromptGroupRollouts
from .spine.scoring import CallableScorer
from .spine.update import OptimizerStep, UpdateLoop, nonfinite_grad_settings
from .trainer_runtime import (
    SharedModelManager,
    build_group_batch,
    hf_generate_group,
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
    # committed inner update (a pass skipped under nonfinite_grad_policy="skip"
    # does not advance it); global_step counts train_steps, not inner updates.
    num_gradient_updates: int = 1

    def validate_for_training(self) -> None:
        """GEPO drops groups of fewer than two responses, so a smaller
        ``group_size`` would train on nothing."""
        super().validate_for_training()
        if type(self.group_size) is not int or self.group_size < 2:
            raise ConfigurationError("group_size must be an integer of at least 2")

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


class GEPOTrainer(TrainingRuntime):
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
        reward_fn: Callable[[str, str], float | Awaitable[float]],
        ref_model: Any | None = None,
    ):
        # Ensure transformers is loaded for scheduler
        _load_transformers_gepo()

        validate_training_config(config)
        if float(getattr(config, "beta", 0.0) or 0.0) > 0:
            raise ValueError(
                "GEPOTrainer has no reference-model KL penalty; beta must be 0 "
                f"(got {config.beta}). Use the GRPO or GSPO trainer for a KL term."
            )
        self.config = config
        # GEPO objective: group-expectation importance coefficient, PPO clip.
        self._objective = objectives.resolve_objective(
            config,
            "gepo",
            max_completion_length=int(config.max_completion_length),
            supported_ratios=("group_expectation", "sequence", "sequence_token"),
            clip_low=float(config.clip_eps),
            clip_high=float(config.clip_eps),
        )
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

    def _build_optimizer_step(self) -> OptimizerStep:
        """The spine commit for one inner update (``num_gradient_updates`` pass).

        Gradients must be finite (``config.nonfinite_grad_policy`` raises or
        skips the pass otherwise), the policy is clipped to ``max_grad_norm``,
        then the optimizer and the LR scheduler step once.
        ``global_step`` counts ``train_step`` calls (``end_train_step``).
        """
        return OptimizerStep(
            self,
            torch,
            optimizers=[self.optimizer],
            schedulers=[self.scheduler],
            clip_groups=[list(self.model.parameters())],
            max_grad_norm=self.config.max_grad_norm,
            counter="train_steps",
            **nonfinite_grad_settings(self.config),
            named_modules=[self.model],
            label="GEPO",
        )

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
        # Never re-enable autograd inside a caller's no_grad (sampler pass).
        with torch.set_grad_enabled(self.model.training and torch.is_grad_enabled()):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shared shift-by-one gather in fp32 (no full-vocabulary log-softmax
        # tensor); an all-ones mask keeps every position's log-prob.
        token_log_probs, _ = rl_losses.gather_token_logprobs(
            logits, input_ids, torch.ones_like(input_ids, dtype=torch.float32)
        )
        # Sum over response tokens only (prompt and padding masked out).
        response_mask = self.build_response_mask(attention_mask, response_start_idx)
        sequence_log_probs = (token_log_probs * response_mask).sum(dim=-1)
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
        coef: torch.Tensor = rl_losses.safe_exp_ratio(log_coef, clamp=30.0)
        return coef

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

    def compute_gepo_loss(
        self,
        learner_seq_log_probs: torch.Tensor,
        sampler_seq_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        *,
        token_log_probs: torch.Tensor | None = None,
        response_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GEPO objective for one prompt group via ``objectives.policy_loss``.

        With ``token_log_probs``/``response_mask`` the per-token learner
        log-probs are passed through, so sequence ratios (``gspo``,
        ``gspo_token``) are length-normalised as GSPO defines them. Without
        them each sequence is one "token" carrying its summed log-prob, which
        is only correct for the group-expectation ratio. The whole call is one
        group, so the group-expectation denominator spans it.
        """
        if token_log_probs is not None and response_mask is not None:
            logp_cur = token_log_probs
            mask = response_mask.to(token_log_probs.dtype)
        else:
            if self._objective.ratio != "group_expectation":
                raise ValueError(
                    f"ratio={self._objective.ratio!r} needs token_log_probs and "
                    "response_mask for length normalisation"
                )
            logp_cur = learner_seq_log_probs.unsqueeze(-1)
            mask = torch.ones_like(logp_cur)
        group_ids = torch.zeros(
            logp_cur.shape[0], dtype=torch.long, device=logp_cur.device
        )
        result = objectives.policy_loss(
            logp_cur=logp_cur,
            mask=mask,
            advantages=advantages,
            objective=self._objective,
            logp_old=sampler_seq_log_probs.detach(),
            group_ids=group_ids,
        )
        return result.loss

    async def _compute_reward(self, prompt: str, response: str) -> float:
        """Score one reply with ``reward_fn``, sync or async (spine scorer)."""
        return await CallableScorer(self.reward_fn).score(prompt, response)

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
        group_ids = torch.zeros(
            rewards.numel(), dtype=torch.long, device=rewards.device
        )
        objective = getattr(self, "_objective", None) or objectives.OBJECTIVES["gepo"]
        advantages = objectives.compute_advantages(rewards, group_ids, objective)

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

        Batched HuggingFace sampling (``trainer_runtime.hf_generate_group``,
        ``generation_batch_size`` rows per ``generate`` call, trimmed after
        EOS). Returns its response dicts (``response``, ``input_ids``,
        ``attention_mask``, ``response_mask``, ``sequence_length``,
        ``prompt_length``) plus ``response_start_idx``, the index where the
        response starts.
        """
        responses: list[dict[str, Any]] = await hf_generate_group(
            self.model,
            self.tokenizer,
            self.config,
            self.device,
            prompt,
            group_size,
        )
        for response in responses:
            response["response_start_idx"] = response["prompt_length"]
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
        async def build(
            prompt: str, group_responses: list[dict[str, Any]]
        ) -> dict[str, Any]:
            rewards = [
                await self._compute_reward(prompt, resp["response"])
                for resp in group_responses
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

            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "response_start_idx": response_start_idx,
                "advantages": advantages,
                "sampler_seq_log_probs": sampler_seq_log_probs.detach(),
            }

        # One group per prompt (at least two responses for group advantages).
        rollouts = await PromptGroupRollouts(
            lambda prompt: self.generate_group_responses(
                prompt, self.config.group_size
            ),
            min_group_size=2,
            label="GEPO",
        ).collect(prompts, build)

        num_groups = len(rollouts)
        if num_groups == 0:
            logger.warning("No valid groups in batch")
            return self._step_metrics(0.0, [], [], [], updates=0)

        # --- Update phase --------------------------------------------------
        # ``num_gradient_updates`` (mu) inner updates reuse the same rollouts
        # against the frozen sampler log probs, exactly like DAPO. With the
        # default of 1 the formula is identical to the previous single-update
        # path (only dropout RNG ordering differs, since the sampler log probs
        # are now drawn before the learner ones).
        num_updates = max(1, getattr(self.config, "num_gradient_updates", 1))

        def group_loss(_pass: int, _index: int, rollout: dict[str, Any]) -> Any:
            learner_token_log_probs, learner_seq_log_probs = (
                self.compute_sequence_log_probs(
                    rollout["input_ids"],
                    rollout["attention_mask"],
                    rollout["response_start_idx"],
                )
            )
            with torch.no_grad():
                all_gepo_coefs.extend(
                    self.compute_gepo_coefficient(
                        learner_seq_log_probs.detach(),
                        rollout["sampler_seq_log_probs"],
                    ).tolist()
                )
            return self.compute_gepo_loss(
                learner_seq_log_probs,
                rollout["sampler_seq_log_probs"],
                rollout["advantages"],
                token_log_probs=learner_token_log_probs,
                response_mask=self.build_response_mask(
                    rollout["attention_mask"], rollout["response_start_idx"]
                ),
            )

        # One graph per pass: the mean of the group losses is backpropagated,
        # then one validated optimizer step. Cadence convention shared with
        # DAPO/VAPO: the LR scheduler advances once per committed inner
        # update, global_step counts train_steps.
        passes = UpdateLoop(
            self._build_optimizer_step(), passes=num_updates, mode="single_graph"
        ).run(
            rollouts,
            group_loss,
            on_pass_start=lambda _pass: all_gepo_coefs.clear(),
            initial_loss=lambda: torch.tensor(
                0.0, device=self.device, requires_grad=True
            ),
            reduce=lambda total: total / num_groups,
        )

        metrics = self._step_metrics(
            passes.losses[-1],
            all_rewards,
            all_advantages,
            all_gepo_coefs,
            updates=passes.committed,
        )

        # Store metrics
        for key in ["policy_loss", "average_reward"]:
            if key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])

        return metrics

    def _step_metrics(
        self,
        policy_loss: Any,
        rewards: list[float],
        advantages: list[float],
        coefficients: list[float],
        *,
        updates: int,
    ) -> dict[str, Any]:
        """One ``train_step``'s metrics; a step without a valid group reports
        the same keys (no updates, coefficient 1)."""
        return {
            "policy_loss": policy_loss,
            "average_reward": np.mean(rewards) if rewards else 0.0,
            "advantage_std": np.std(advantages) if advantages else 0.0,
            "gepo_coefficient_mean": np.mean(coefficients) if coefficients else 1.0,
            "gepo_coefficient_std": np.std(coefficients) if coefficients else 0.0,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "global_step": self.global_step,
            "optimizer_updates": updates,
        }

    def save_checkpoint(self, output_dir: str) -> None:
        """Publish a complete resumable checkpoint."""
        self.save_runtime_checkpoint(output_dir)

    def load_checkpoint(self, checkpoint_dir: str, *, trusted: bool = False) -> bool:
        """Restore policy weights and all training components."""
        return self.load_runtime_checkpoint(checkpoint_dir, trusted=trusted)


async def train_with_gepo(
    model_name: str,
    reward_fn: Callable[[str, str], float | Awaitable[float]],
    train_prompts: list[str],
    config: GEPOConfig | None = None,
    output_dir: str | None = None,
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
        output_dir: Directory to save checkpoints; overrides
            ``config.output_dir`` when given (``model_name`` likewise overrides
            ``config.model_name``). Without it, ``config.output_dir`` is used,
            or ``./outputs/gepo`` when no config is given.
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name

    Returns:
        Tuple of (model, tokenizer, metrics_history)
    """
    config = dedicated_run_config(
        config,
        GEPOConfig,
        model_name=model_name,
        output_dir=output_dir,
        default_output_dir="./outputs/gepo",
        entrypoint="train_with_gepo",
    )
    _reject_unsupported_resume(config, "train_with_gepo")

    logger.info("=" * 60)
    logger.info("GEPO Training - Group Expectation Policy Optimization")
    logger.info("=" * 60)

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
    seed_dedicated_run(config.seed)
    logger.info(f"Loading model: {config.model_name}")
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

    os.makedirs(config.output_dir, exist_ok=True)

    def report(iteration: int, metrics: dict[str, Any]) -> None:
        if iteration % config.logging_steps == 0:
            logger.info(
                f"Iteration {iteration}/{config.num_episodes} | "
                f"Loss: {metrics['policy_loss']:.4f} | "
                f"Reward: {metrics['average_reward']:.4f} | "
                f"GEPO Coef: {metrics['gepo_coefficient_mean']:.4f}"
            )

            if use_wandb:
                wandb.log(metrics, step=iteration)

    # The spine's dedicated outer loop (periodic and final saves included).
    await run_dedicated_loop(
        iterations=config.num_episodes,
        batch=sample_prompts(train_prompts, config.per_device_train_batch_size),
        step=trainer.train_step,
        save=trainer.save_checkpoint,
        output_dir=config.output_dir,
        save_steps=config.save_steps,
        final_name="final",
        report=report,
        seed=config.seed,
    )

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
