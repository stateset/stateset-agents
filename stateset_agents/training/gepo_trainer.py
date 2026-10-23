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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import objectives, rl_losses
from .config import (
    TrainingConfig,
    config_value,
    require_finite,
    validate_group_size,
)
from .dedicated_reporting import WandbRun
from .runtime_state import (
    TrainingRuntime,
    _reject_unsupported_resume,
    validate_training_config,
)
from .spine.algorithm import (
    AlgorithmSpec,
    AlgorithmTrainer,
    StepRollouts,
    run_algorithm_step,
)
from .spine.loop import (
    dedicated_run_config,
    run_dedicated_loop,
    sample_prompts,
    seed_dedicated_run,
)
from .spine.rollouts import PromptGroupRollouts
from .spine.scoring import CallableScorer
from .spine.update import UpdatePasses
from .trainer_runtime import (
    SharedModelManager,
    build_group_batch,
    hf_generate_group,
)

logger = logging.getLogger(__name__)


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
        """Also require two responses per group (:func:`validate_group_size`)
        and a PPO clip in [0, 1)."""
        super().validate_for_training()
        validate_group_size(self.group_size)
        require_finite(self, "clip_eps", minimum=0, maximum=1, exclusive_maximum=True)

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
    """Manages model loading for GEPO training (transformers and PEFT through
    :mod:`~stateset_agents.training.dedicated_deps`)."""

    algorithm = "GEPO"

    def __init__(self, config: GEPOConfig):
        super().__init__(config)


class GEPOTrainer(AlgorithmTrainer, TrainingRuntime):
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
        validate_training_config(config, GEPOConfig)
        if float(config_value(config, "beta") or 0.0) > 0:
            raise ValueError(
                "GEPOTrainer has no reference-model KL penalty; beta must be 0 "
                f"(got {config.beta}). Use the GRPO or GSPO trainer for a KL term."
            )
        self.config = config
        # GEPO objective: group-expectation importance coefficient, PPO clip
        # (GEPO_SPEC).
        self._objective = self._resolve_algorithm_objective()
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

        # ``lr_scheduler_type`` over every committed update pass of the run
        # (num_episodes x num_gradient_updates, the spine budget), clamped at
        # the end. Sizing it over num_episodes alone made the cosine reach 0
        # halfway through a two-pass run and then climb back to the full LR.
        self.scheduler = self._build_lr_scheduler(self.optimizer)

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
        return await self._score_reply(prompt, response)

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
        Execute one GEPO training step (:data:`GEPO_SPEC` on the spine step).

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
        return await run_algorithm_step(self, self.algorithm, prompts)

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


@dataclass
class _GEPOStep:
    """One GEPO step's rollout and pass data (for the metrics)."""

    rewards: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)


async def _gepo_score(
    trainer: GEPOTrainer, prompt: str, response: str, context: Mapping[str, Any]
) -> float:
    return await CallableScorer(trainer.reward_fn).score(prompt, response)


async def _gepo_rollouts(trainer: GEPOTrainer, prompts: list[Any]) -> StepRollouts:
    """One group per prompt (at least two responses for group advantages),
    with the sampler (old-policy) sequence log probs frozen here, before any
    inner update can move the weights. When they were recomputed from the
    just-updated model the GEPO coefficient collapsed to ~1 and clipping
    never fired."""
    step = _GEPOStep()

    async def build(
        prompt: str, group_responses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        rewards = [
            await trainer._compute_reward(prompt, resp["response"])
            for resp in group_responses
        ]
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=trainer.device
        )
        step.rewards.extend(rewards)

        advantages, _reward_stats = trainer.compute_group_advantages(rewards_tensor)
        step.advantages.extend(advantages.tolist())

        batch_input_ids, batch_attention_mask, _ = build_group_batch(
            group_responses, trainer.device, include_response_mask=False
        )
        response_start_idx = group_responses[0]["response_start_idx"]

        with torch.no_grad():
            _, sampler_seq_log_probs = trainer.compute_sequence_log_probs(
                batch_input_ids, batch_attention_mask, response_start_idx
            )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "response_start_idx": response_start_idx,
            "advantages": advantages,
            "sampler_seq_log_probs": sampler_seq_log_probs.detach(),
        }

    items = await PromptGroupRollouts(
        lambda prompt: trainer.generate_group_responses(
            prompt, trainer.config.group_size
        ),
        min_group_size=2,
        label="GEPO",
    ).collect(prompts, build)
    if not items:
        logger.warning("No valid groups in batch")
    return StepRollouts(items=items, info=step)


def _gepo_group_loss(
    trainer: GEPOTrainer,
    rollouts: StepRollouts,
    _pass: int,
    _index: int,
    rollout: dict[str, Any],
) -> Any:
    learner_token_log_probs, learner_seq_log_probs = trainer.compute_sequence_log_probs(
        rollout["input_ids"],
        rollout["attention_mask"],
        rollout["response_start_idx"],
    )
    with torch.no_grad():
        rollouts.info.coefficients.extend(
            trainer.compute_gepo_coefficient(
                learner_seq_log_probs.detach(),
                rollout["sampler_seq_log_probs"],
            ).tolist()
        )
    return trainer.compute_gepo_loss(
        learner_seq_log_probs,
        rollout["sampler_seq_log_probs"],
        rollout["advantages"],
        token_log_probs=learner_token_log_probs,
        response_mask=trainer.build_response_mask(
            rollout["attention_mask"], rollout["response_start_idx"]
        ),
    )


def _gepo_update_options(
    trainer: GEPOTrainer, rollouts: StepRollouts
) -> dict[str, Any]:
    """One graph per pass: the mean of the group losses is backpropagated,
    then one validated optimizer step; each pass reports its own
    coefficients."""
    num_groups = len(rollouts.items)
    return {
        "on_pass_start": lambda _pass: rollouts.info.coefficients.clear(),
        "initial_loss": lambda: torch.tensor(
            0.0, device=trainer.device, requires_grad=True
        ),
        "reduce": lambda total: total / num_groups,
    }


def _gepo_metrics(
    trainer: GEPOTrainer, rollouts: StepRollouts, passes: UpdatePasses | None
) -> dict[str, Any]:
    if passes is None:
        return trainer._step_metrics(0.0, [], [], [], updates=0)
    step: _GEPOStep = rollouts.info
    metrics = trainer._step_metrics(
        passes.losses[-1],
        step.rewards,
        step.advantages,
        step.coefficients,
        updates=passes.committed,
    )

    # Store metrics
    for key in ["policy_loss", "average_reward"]:
        if key in trainer.metrics_history:
            trainer.metrics_history[key].append(metrics[key])
    return metrics


#: GEPO: the group-expectation importance coefficient with a PPO clip over
#: prompt groups of at least two replies; one graph per pass, ``mu``
#: (``num_gradient_updates``) passes against the frozen sampler log probs
#: (cadence shared with DAPO/VAPO: the LR scheduler advances once per
#: committed pass, ``global_step`` counts train steps).
GEPO_SPEC = AlgorithmSpec(
    name="gepo",
    objective="gepo",
    objective_options=lambda config: {
        "supported_ratios": ("group_expectation", "sequence", "sequence_token"),
        "clip_low": float(config.clip_eps),
        "clip_high": float(config.clip_eps),
    },
    counter="train_steps",
    update_mode="single_graph",
    passes=lambda config: max(1, config_value(config, "num_gradient_updates")),
    score=_gepo_score,
    rollouts=_gepo_rollouts,
    group_loss=_gepo_group_loss,
    metrics=_gepo_metrics,
    update_options=_gepo_update_options,
)
GEPOTrainer.algorithm = GEPO_SPEC


async def train_with_gepo(
    model_name: str,
    reward_fn: Callable[[str, str], float | Awaitable[float]],
    train_prompts: list[str],
    config: GEPOConfig | None = None,
    output_dir: str | None = None,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    callbacks: list[Any] | None = None,
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
        callbacks: Get ``on_step_end`` after each iteration (after its
            checkpoint); one setting ``should_abort`` stops training after the
            final save

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

    with WandbRun(
        "gepo",
        enabled=use_wandb,
        project=wandb_project,
        config=config,
        tags=["gepo", "rl-training"],
    ) as reporter:
        return await _run_gepo(
            config, reward_fn, train_prompts, callbacks or [], reporter
        )


async def _run_gepo(
    config: GEPOConfig,
    reward_fn: Callable[[str, str], float | Awaitable[float]],
    train_prompts: list[str],
    callbacks: list[Any],
    reporter: WandbRun,
) -> tuple[Any, Any, dict[str, list[float]]]:
    """:func:`train_with_gepo` after its config and W&B run are set up."""
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

            reporter.log(metrics, step=iteration)

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
        callbacks=callbacks,
        seed=config.seed,
    )

    trainer.metrics_history["rollout_samples_total"] = [
        float(trainer.rollout_samples_total)
    ]

    logger.info("=" * 60)
    logger.info("GEPO Training Complete!")
    logger.info("=" * 60)

    return model, tokenizer, trainer.metrics_history


# Export
__all__ = [
    "GEPOConfig",
    "GEPO_SPEC",
    "GEPOTrainer",
    "train_with_gepo",
]
