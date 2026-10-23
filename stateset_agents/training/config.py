"""
Training configuration and profiles for GRPO Agent Framework

This module provides training configurations that implement best practices
and integrate seamlessly with HuggingFace and Weights & Biases.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.errors import ConfigurationError


def estimate_grpo_optimizer_steps(
    rollout_batches: int,
    *,
    gradient_accumulation_steps: int = 1,
    num_gradient_updates: int = 1,
    task_ids: Sequence[str | None] | None = None,
) -> int:
    """Estimate native GRPO's optimizer-step budget for a learning-rate schedule.

    Token-path inner updates each commit a full step, bypassing accumulation.
    Their budget is an upper bound when batches instead use sequence fallback,
    emit no gradients, or skip updates. A single update per batch uses ordinary
    accumulation. Supplying task IDs closes each task segment's partial window,
    matching the trainer's flush before task-memory consolidation.
    """
    batches = max(0, int(rollout_batches))
    inner = max(1, int(num_gradient_updates or 1))
    accumulation = max(1, int(gradient_accumulation_steps or 1))
    if inner > 1:
        return batches * inner
    if task_ids is not None:
        if len(task_ids) != batches:
            raise ValueError("Task schedule must describe every rollout batch")
        total, size, current = 0, 0, None
        for task in task_ids:
            if task is not None and not isinstance(task, str):
                raise ValueError("Task identifiers must be strings or None")
            if task is not None and current is not None and task != current:
                total += (size + accumulation - 1) // accumulation
                size = 0
            if task is not None:
                current = task
            size += 1
        return total + (size + accumulation - 1) // accumulation
    return (batches + accumulation - 1) // accumulation


class TrainingProfile(Enum):
    """Pre-defined training profiles based on stability/performance trade-offs"""

    CONSERVATIVE = "conservative"  # Maximum stability
    BALANCED = "balanced"  # Default, good balance
    AGGRESSIVE = "aggressive"  # Maximum performance
    EXPERIMENTAL = "experimental"  # Cutting edge


_NATIVE_LR_SCHEDULERS = frozenset({"cosine", "linear", "constant"})


def validate_group_size(group_size: object) -> None:
    """Group-relative algorithms (DAPO, GEPO, VAPO) need two responses per
    prompt: a lone response has no group baseline (GEPO and VAPO drop such a
    group, DAPO gives it a zero advantage), so a smaller ``group_size``
    trains on nothing."""
    if type(group_size) is not int or group_size < 2:
        raise ConfigurationError("group_size must be an integer of at least 2")


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration compatible with HuggingFace patterns
    """

    # Basic training parameters
    model_name: str = "gpt2"
    # Immutable Hugging Face commit/tag used for tokenizer, policy, reference
    # model, and rollout backend. Leave unset for normal exploratory use;
    # publication-grade benchmarks should always pin a full commit SHA.
    model_revision: str | None = None
    num_episodes: int = 1000
    num_epochs: int = 1
    learning_rate: float = 5e-6

    # Optimization parameters (HuggingFace compatible)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"

    # Batch and generation settings
    per_device_train_batch_size: int = 1
    # Aliases for compatibility with some tests/configs
    batch_size: int | None = None
    # Note for manual generate_trajectories/training_step loops: only every
    # 4th training_step takes an optimizer step by default.
    gradient_accumulation_steps: int = 4
    num_generations: int = 16  # GRPO: trajectories per scenario
    # Roll every group out in lockstep, one batched generate per conversation
    # turn, when the agent has generate_turns_batch and the environment
    # supports concurrent episodes. False forces one episode at a time.
    batched_rollouts: bool = True
    # What MultiTurnGRPOTrainer does when a step's gradients are non-finite:
    # "raise" (fail fast) or "skip" (log which parameters and loss terms were
    # non-finite, discard that update, and abort after
    # max_consecutive_nonfinite_skips consecutive skips).
    nonfinite_grad_policy: str = "raise"
    max_consecutive_nonfinite_skips: int = 3
    generation_batch_size: int = 4
    # Rollout/evaluation parallelism (requires environments that support clone()).
    rollout_concurrency: int = 1
    # Evaluation settings
    eval_num_generations: int = 4

    # Length constraints
    max_prompt_length: int = 512
    max_completion_length: int = 512
    # Alias used by some tests; not directly consumed by trainer but accepted
    max_steps_per_episode: int | None = None

    # Hardware optimization
    bf16: bool = True
    fp16: bool = False
    use_cpu: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    gradient_checkpointing: bool = False

    # Evaluation and checkpointing
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3

    # Output configuration
    output_dir: str = "./outputs"
    run_name: str | None = None
    # Backwards-compat alias (some tests/configs use this name).
    wandb_run_name: str | None = None
    overwrite_output_dir: bool = True

    # Weights & Biases integration
    report_to: str = "wandb"  # "wandb", "tensorboard", "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None

    # Training stability and monitoring
    auto_adjust: bool = True
    early_stopping: bool = False
    patience: int = 50

    # GRPO specific parameters
    gamma: float = 0.99  # Discount factor for cumulative returns
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    reward_clip: float = 10.0
    advantage_normalization: bool = True
    baseline_type: str = "group_mean"  # "group_mean", "global_mean"

    # Enhanced GRPO parameters from real-world implementation
    beta: float = 0.0  # KL penalty coefficient
    use_reference_model: bool = False  # Whether to use reference model for KL
    clip_ratio: float = (
        0.2  # PPO ratio clipping when old log probs are available; else advantage clamp
    )
    # The GRPO importance ratio is a per-token-MEAN (length-normalised) sequence
    # ratio, so it sits within ~1e-3 of 1.0 and the classic +/-0.2 token-level
    # PPO clip (`clip_ratio`) is meaningless for it — it never triggers. GSPO
    # clips this quantity at ~3e-4/4e-4 instead; see `gspo_config.py`.
    seq_clip_ratio: float = 3e-4  # Sequence-mean ratio clip (GSPO scale)
    value_clip: float = 0.2  # Value function clipping
    # Entropy bonus coefficient. On by default (0.01); set 0.0 for the plain
    # objective (the token path then skips the full-vocabulary softmax).
    entropy_coef: float = 0.01

    # Continual learning (replay + LwF + EWC)
    continual_strategy: str = "none"  # "none", "replay", "lwf", "ewc", combos
    replay_buffer_size: int = 2000
    replay_ratio: float = 0.5
    replay_min_size: int = 100
    replay_sampling: str = "uniform"  # "uniform", "recent", "reward", "balanced"
    replay_storage: str = "reservoir"  # "reservoir", "fifo"
    continual_kl_beta: float = 0.0  # KL penalty for LwF reference model
    ewc_lambda: float = 0.0
    ewc_num_samples: int = 64
    ewc_decay: float = 0.9
    task_id_key: str = "task_id"

    # Data processing
    max_examples: int | None = None
    eval_split_size: float = 0.1
    stratify_by_task: bool = True
    data_format: str = "jsonl"  # "jsonl", "json", "csv"

    # Post-training evaluation
    run_post_eval: bool = True
    post_eval_samples: int = 5
    post_eval_detailed: bool = True

    # Model optimization
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    # Generation settings for evaluation
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # vLLM configuration (5-20x faster generation)
    use_vllm: bool = False  # Enable vLLM for generation
    vllm_gpu_memory_utilization: float = 0.85  # Fraction of GPU memory for vLLM
    vllm_tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    vllm_enable_prefix_caching: bool = True  # Cache KV for repeated prefixes
    vllm_max_model_len: int | None = None  # Max sequence length (auto if None)
    vllm_quantization: str | None = None  # "awq", "gptq", or None
    vllm_enable_chunked_prefill: bool = True  # Better memory efficiency

    # Advanced options
    seed: int = 42
    remove_unused_columns: bool = False
    disable_tqdm: bool = False

    # Continual learning task schedule (optional)
    task_schedule: list[str] | None = None
    task_switch_steps: int = 0
    resume_from_checkpoint: str | None = None

    # Policy objective selection. ``None`` keeps each trainer's native
    # objective (GRPO -> sequence-clipped GRPO, DAPO -> dapo, ...). Set to a
    # preset name from ``stateset_agents.training.objectives.OBJECTIVES``
    # (grpo, dr_grpo, bnpo, dapo, gspo, gspo_token, gepo, rloo,
    # reinforce_pp_baseline, cispo, ppo) to train with that objective instead;
    # ``objective_overrides`` tweaks individual PolicyObjective fields, e.g.
    # ``{"clip_high": 0.3, "is_cap": 3.0}``. See docs/OBJECTIVES.md.
    objective: str | None = None
    objective_overrides: dict[str, Any] | None = None
    # Inner optimizer updates per rollout batch on the GRPO trainers' token
    # path (PPO/DAPO-style mini-epochs). 1 keeps the on-policy single update;
    # above 1 the old policy's per-token log-probs are frozen once and each
    # update re-evaluates the clipped objective against them.
    num_gradient_updates: int = 1
    # Push the policy's weights into the agent's attached rollout backend
    # (``MultiTurnAgent.set_rollout_backend``) after every optimizer step so
    # engine rollouts stay on-policy. Disable for a deliberately asynchronous
    # engine and pair it with ``old_logprobs_source="sampler"``.
    rollout_sync: bool = True
    # Old-policy log-probs on the GRPO token path: ``"recompute"`` treats each
    # rollout batch as exactly on-policy (ratio 1 at the first update);
    # ``"sampler"`` uses the log-probs recorded at sampling time
    # (``sampler_log_probs``), so the importance ratio corrects for a stale or
    # numerically different rollout engine (TRL's vLLM importance-sampling
    # correction). Falls back to ``"recompute"`` when rollouts carry none.
    old_logprobs_source: str = "recompute"
    # Compatibility paths change the estimator and must be requested explicitly.
    allow_sequence_fallback: bool = False
    allow_missing_sampler_logprobs: bool = False
    # Optional measured provider rate. Runtime metrics cover rollout + update
    # time only; provisioning/storage charges remain in the provider ledger.
    compute_cost_per_hour: float | None = None
    # Keyword-only so existing positional calls to derived configs also retain
    # their field order. Zero disables the consecutive-zero-advantage guard.
    zero_signal_patience: int = field(default=5, kw_only=True)
    # Zero every dropout module (including LoRA dropout) in the policy before
    # RL training, so separately scored old/reference log-probs and the
    # learner's forward pass agree (TRL's ``disable_dropout``). Keyword-only
    # so subclasses keep their positional arguments.
    disable_dropout: bool = field(default=False, kw_only=True)
    # Run Python shipped inside the model's Hugging Face repo when loading it.
    # Off by default: only enable for a repo you trust whose architecture is
    # not native to transformers (curated presets opt in explicitly).
    trust_remote_code: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        if self.nonfinite_grad_policy not in ("raise", "skip"):
            raise ConfigurationError(
                "nonfinite_grad_policy must be 'raise' or 'skip', got "
                f"{self.nonfinite_grad_policy!r}"
            )
        if self.max_consecutive_nonfinite_skips < 0:
            raise ConfigurationError("max_consecutive_nonfinite_skips must be >= 0")
        # Map compatibility aliases
        if self.batch_size is not None and self.batch_size > 0:
            self.per_device_train_batch_size = int(self.batch_size)
        # The alias is applied once; keeping it would re-impose this value on
        # every replace()/from_dict() copy and undo later batch-size changes.
        self.batch_size = None
        if self.wandb_run_name is not None and self.run_name is None:
            self.run_name = self.wandb_run_name

    @classmethod
    def from_profile(cls, profile: TrainingProfile, **overrides) -> "TrainingConfig":
        """Create configuration from predefined profile"""
        base_config: dict[str, Any] = {}

        if profile == TrainingProfile.CONSERVATIVE:
            base_config = {
                "learning_rate": 1e-6,
                "num_generations": 32,
                "gradient_accumulation_steps": 8,
                "max_grad_norm": 0.5,
                "warmup_ratio": 0.2,
                "eval_steps": 25,
                "auto_adjust": True,
                "early_stopping": True,
                "patience": 30,
            }
        elif profile == TrainingProfile.BALANCED:
            base_config = {
                "learning_rate": 5e-6,
                "num_generations": 16,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.1,
                "eval_steps": 50,
                "auto_adjust": True,
                "early_stopping": False,
                "patience": 50,
            }
        elif profile == TrainingProfile.AGGRESSIVE:
            base_config = {
                "learning_rate": 1e-5,
                "num_generations": 8,
                "gradient_accumulation_steps": 2,
                "max_grad_norm": 2.0,
                "warmup_ratio": 0.05,
                "eval_steps": 100,
                "auto_adjust": False,
                "early_stopping": False,
                "patience": 100,
            }
        elif profile == TrainingProfile.EXPERIMENTAL:
            base_config = {
                "learning_rate": 2e-5,
                "num_generations": 12,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.5,
                "warmup_ratio": 0.0,
                "eval_steps": 20,
                "auto_adjust": True,
                "early_stopping": True,
                "patience": 20,
                "lr_scheduler_type": "linear",
            }

        # Apply overrides
        base_config.update(overrides)

        return cls(**base_config)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary"""
        return cls(**config_dict)

    def save(self, path: str):
        """Save configuration to file"""
        config_dict = self.to_dict()

        # Convert enum values to strings
        if isinstance(config_dict.get("wandb_tags"), list):
            # Ensure all tags are strings
            config_dict["wandb_tags"] = [str(tag) for tag in config_dict["wandb_tags"]]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from file"""
        with open(path) as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def validate(self) -> list[str]:
        """Validate configuration and return warnings"""
        warnings = []

        # Learning rate checks
        if self.learning_rate > 1e-4:
            warnings.append(
                f"Learning rate {self.learning_rate} is very high, consider reducing"
            )

        if self.learning_rate < 1e-7:
            warnings.append(
                f"Learning rate {self.learning_rate} is very low, training may be slow"
            )

        # Batch size checks
        effective_batch_size = (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )
        if effective_batch_size < 4:
            warnings.append(
                f"Effective batch size {effective_batch_size} is small, may cause instability"
            )

        # Generation checks
        if self.num_generations < 4:
            warnings.append(
                f"num_generations {self.num_generations} is low for GRPO, consider increasing"
            )

        # Hardware checks
        if self.bf16 and self.fp16:
            warnings.append("Both bf16 and fp16 enabled, bf16 will take precedence")

        # W&B checks
        if self.report_to == "wandb" and not self.wandb_project:
            warnings.append("W&B reporting enabled but no project specified")

        # Objective selection checks (torch-free: only the preset names).
        if self.objective is not None:
            from .objectives import OBJECTIVES

            if self.objective not in OBJECTIVES:
                warnings.append(
                    f"objective '{self.objective}' is not a known preset; "
                    f"expected one of: {', '.join(sorted(OBJECTIVES))}"
                )

        if int(self.num_gradient_updates) < 1:
            warnings.append("num_gradient_updates must be >= 1")
        if self.old_logprobs_source not in ("recompute", "sampler"):
            warnings.append(
                "old_logprobs_source must be 'recompute' or 'sampler', got "
                f"{self.old_logprobs_source!r}"
            )

        # Continual learning checks
        valid_strategies = {
            "none",
            "replay",
            "lwf",
            "ewc",
            "replay_lwf",
            "replay_ewc",
            "replay_lwf_ewc",
        }
        if self.continual_strategy not in valid_strategies:
            warnings.append(
                f"continual_strategy '{self.continual_strategy}' not recognized"
            )
        if self.replay_sampling not in {"uniform", "recent", "reward", "balanced"}:
            warnings.append(f"replay_sampling '{self.replay_sampling}' not recognized")
        if self.replay_ratio < 0:
            warnings.append("replay_ratio must be non-negative")
        if self.replay_ratio > 1.0:
            warnings.append("replay_ratio > 1.0 may overweight replay data")
        if self.replay_buffer_size < 0:
            warnings.append("replay_buffer_size must be non-negative")
        if self.replay_min_size > self.replay_buffer_size:
            warnings.append("replay_min_size exceeds replay_buffer_size")
        if self.ewc_lambda > 0 and self.ewc_num_samples <= 0:
            warnings.append("ewc_num_samples must be > 0 when ewc_lambda > 0")
        if self.ewc_decay < 0.0 or self.ewc_decay > 1.0:
            warnings.append("ewc_decay should be between 0 and 1 for stable updates")
        if self.beta > 0 and not self.use_reference_model:
            warnings.append(
                f"beta={self.beta} has no effect without use_reference_model=True; "
                "the KL penalty needs a frozen reference policy"
            )
        if self.continual_kl_beta > 0 and "lwf" not in self.continual_strategy:
            warnings.append(
                "continual_kl_beta set but continual_strategy does not include lwf"
            )
        if self.task_schedule and self.task_switch_steps <= 0:
            warnings.append(
                "task_schedule set but task_switch_steps is <= 0; tasks will not advance"
            )
        if self.eval_split_size < 0.0 or self.eval_split_size > 1.0:
            warnings.append("eval_split_size should be between 0 and 1")

        if self.resume_from_checkpoint:
            try:
                if not Path(self.resume_from_checkpoint).exists():
                    warnings.append(
                        f"resume_from_checkpoint '{self.resume_from_checkpoint}' does not exist"
                    )
            except (OSError, ValueError):
                warnings.append("resume_from_checkpoint path could not be validated")

        return warnings

    def _supported_lr_schedulers(self) -> frozenset[str] | None:
        """Schedulers the native trainers implement; ``None`` accepts any name.

        A config that hands ``lr_scheduler_type`` to another library (TRL)
        overrides this to defer to that library's own set.
        """
        return _NATIVE_LR_SCHEDULERS

    def _validate_lr_scheduler_type(self) -> None:
        schedulers = self._supported_lr_schedulers()
        name = self.lr_scheduler_type
        if (
            isinstance(name, str)
            and name
            and (schedulers is None or name in schedulers)
        ):
            return
        expected = (
            "one of " + ", ".join(sorted(schedulers))
            if schedulers is not None
            else "a non-empty scheduler name"
        )
        raise ConfigurationError(f"lr_scheduler_type must be {expected} (got {name!r})")

    def validate_for_training(self) -> None:
        """Reject invalid run semantics before allocating a model or optimizer."""
        if type(self.zero_signal_patience) is not int or self.zero_signal_patience < 0:
            raise ConfigurationError(
                "zero_signal_patience must be a nonnegative integer"
            )
        for name in (
            "num_episodes",
            "num_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "num_generations",
            "generation_batch_size",
            "rollout_concurrency",
            "eval_num_generations",
            "max_prompt_length",
            "max_completion_length",
            "num_gradient_updates",
            "save_total_limit",
            "logging_steps",
            "save_steps",
            "eval_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        self._validate_lr_scheduler_type()
        for name in ("learning_rate", "temperature", "max_grad_norm"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigurationError(f"{name} must be finite and positive")
        for name in ("warmup_ratio", "eval_split_size", "top_p", "lora_dropout"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ConfigurationError(f"{name} must be between zero and one")
        if self.top_p == 0:
            raise ConfigurationError("top_p must be positive")
        if self.bf16 and self.fp16:
            raise ConfigurationError("bf16 and fp16 are mutually exclusive")
        if getattr(self, "use_4bit", False) and getattr(self, "use_8bit", False):
            raise ConfigurationError("use_4bit and use_8bit are mutually exclusive")
        if type(self.seed) is not int or self.seed < 0:
            raise ConfigurationError("seed must be a nonnegative integer")
        if self.compute_cost_per_hour is not None and (
            type(self.compute_cost_per_hour) not in (int, float)
            or not math.isfinite(self.compute_cost_per_hour)
            or self.compute_cost_per_hour < 0
        ):
            raise ConfigurationError(
                "compute_cost_per_hour must be finite and nonnegative"
            )
        for name in (
            "beta",
            "entropy_coef",
            "weight_decay",
            "clip_ratio",
            "seq_clip_ratio",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"{name} must be finite and nonnegative")
        if self.old_logprobs_source not in {"recompute", "sampler"}:
            raise ConfigurationError(
                "old_logprobs_source must be 'recompute' or 'sampler'"
            )
        if not self.rollout_sync and self.old_logprobs_source != "sampler":
            raise ConfigurationError(
                "Asynchronous rollouts require old_logprobs_source='sampler'"
            )
        if self.objective is not None:
            from .objectives import OBJECTIVES

            if self.objective not in OBJECTIVES:
                raise ConfigurationError(f"Unknown objective: {self.objective!r}")
        if self.objective_overrides:
            from .objectives import OBJECTIVES

            OBJECTIVES[self.objective or "grpo"].with_(**self.objective_overrides)
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ConfigurationError("model_name must be nonempty")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ConfigurationError("model_revision must be a nonempty string or None")
        if (
            self.resume_from_checkpoint
            and not Path(self.resume_from_checkpoint).is_dir()
        ):
            raise ConfigurationError(
                "resume_from_checkpoint must be an existing directory"
            )

    def get_effective_batch_size(self) -> int:
        """Get effective batch size"""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def get_total_steps(self, num_episodes: int | None = None) -> int:
        """Estimate native GRPO optimizer steps, assuming one batch per episode."""
        episodes = self.num_episodes if num_episodes is None else num_episodes
        return estimate_grpo_optimizer_steps(
            episodes,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_gradient_updates=self.num_gradient_updates,
        )

    def get_warmup_steps(self, total_steps: int | None = None) -> int:
        """Calculate warmup steps"""
        if total_steps is None:
            total_steps = self.get_total_steps()
        return int(total_steps * self.warmup_ratio)


# Pre-defined configurations for common use cases
CUSTOMER_SERVICE_CONFIG = TrainingConfig(
    num_episodes=800,
    learning_rate=3e-6,
    num_generations=12,
    max_completion_length=256,
    eval_steps=40,
    wandb_tags=["customer-service", "conversation"],
    wandb_notes="Customer service agent training with politeness and helpfulness focus",
)

TUTORING_CONFIG = TrainingConfig(
    num_episodes=1200,
    learning_rate=5e-6,
    num_generations=16,
    max_completion_length=512,
    eval_steps=60,
    patience=60,
    wandb_tags=["tutoring", "education"],
    wandb_notes="Educational tutoring agent with step-by-step explanation focus",
)

CREATIVE_WRITING_CONFIG = TrainingConfig(
    num_episodes=600,
    learning_rate=8e-6,
    num_generations=8,
    max_completion_length=1024,
    eval_steps=30,
    lr_scheduler_type="linear",
    wandb_tags=["creative-writing", "generation"],
    wandb_notes="Creative writing assistant with imagination and storytelling focus",
)

RESEARCH_ASSISTANT_CONFIG = TrainingConfig(
    num_episodes=1000,
    learning_rate=4e-6,
    num_generations=20,
    max_completion_length=768,
    eval_steps=50,
    early_stopping=True,
    patience=40,
    wandb_tags=["research", "analysis"],
    wandb_notes="Research assistant with factual accuracy and citation focus",
)


def get_config_for_task(task_type: str, **overrides) -> TrainingConfig:
    """Get pre-configured setup for common tasks"""

    task_configs = {
        "customer_service": CUSTOMER_SERVICE_CONFIG,
        "tutoring": TUTORING_CONFIG,
        "creative_writing": CREATIVE_WRITING_CONFIG,
        "research_assistant": RESEARCH_ASSISTANT_CONFIG,
    }

    if task_type in task_configs:
        base_config = task_configs[task_type]

        # Create new config with overrides
        config_dict = base_config.to_dict()
        config_dict.update(overrides)

        return TrainingConfig.from_dict(config_dict)
    else:
        # Return default balanced config with overrides
        return TrainingConfig.from_profile(TrainingProfile.BALANCED, **overrides)


def optimize_config_for_hardware(
    config: TrainingConfig, gpu_memory_gb: float | None = None, num_gpus: int = 1
) -> TrainingConfig:
    """Optimize configuration based on available hardware"""

    optimized_dict = config.to_dict()

    # GPU memory optimizations
    if gpu_memory_gb is not None:
        if gpu_memory_gb < 8:
            # Small GPU optimizations
            optimized_dict.update(
                {
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "num_generations": 8,
                    "max_completion_length": 256,
                    "bf16": True,
                    "dataloader_num_workers": 0,
                }
            )
        elif gpu_memory_gb < 16:
            # Medium GPU optimizations
            optimized_dict.update(
                {
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "num_generations": 12,
                    "max_completion_length": 512,
                    "bf16": True,
                    "dataloader_num_workers": 2,
                }
            )
        else:
            # Large GPU optimizations
            optimized_dict.update(
                {
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "num_generations": 16,
                    "max_completion_length": 768,
                    "bf16": True,
                    "dataloader_num_workers": 4,
                }
            )

    # Multi-GPU optimizations
    if num_gpus > 1:
        optimized_dict["per_device_train_batch_size"] = max(
            1, optimized_dict["per_device_train_batch_size"] // num_gpus
        )
        optimized_dict["dataloader_num_workers"] = min(
            8, optimized_dict["dataloader_num_workers"] * num_gpus
        )

    return TrainingConfig.from_dict(optimized_dict)


def create_config_from_args(
    model_name: str, task_type: str = "general", profile: str = "balanced", **kwargs
) -> TrainingConfig:
    """Create configuration from command line arguments"""

    # Start with task-specific config or profile. A non-balanced profile is
    # layered over a task config (its tuning keys win); "balanced" is the
    # default and leaves the task's own tuning in place.
    if task_type != "general":
        config = get_config_for_task(task_type)
        if TrainingProfile(profile) != TrainingProfile.BALANCED:
            neutral = TrainingConfig().to_dict()
            # Every profile sets the same tuning keys; find them by diffing
            # each profile against the plain defaults.
            profile_keys = {
                name
                for member in TrainingProfile
                for name, value in TrainingConfig.from_profile(member).to_dict().items()
                if value != neutral[name]
            }
            tuned = TrainingConfig.from_profile(TrainingProfile(profile)).to_dict()
            config_dict = config.to_dict()
            config_dict.update({name: tuned[name] for name in profile_keys})
            config = TrainingConfig.from_dict(config_dict)
    else:
        config = TrainingConfig.from_profile(TrainingProfile(profile))

    # Apply model-specific optimizations
    if "large" in model_name.lower() or "13b" in model_name.lower():
        # Large model optimizations
        config_dict = config.to_dict()
        config_dict.update(
            {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "bf16": True,
                "max_completion_length": min(512, config_dict["max_completion_length"]),
            }
        )
        config = TrainingConfig.from_dict(config_dict)

    # Apply user overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = TrainingConfig.from_dict(config_dict)

    return config


# Example usage and testing
def example_config_usage():
    """Example of how to use training configurations"""

    # Create from profile
    conservative_config = TrainingConfig.from_profile(TrainingProfile.CONSERVATIVE)
    print(f"Conservative LR: {conservative_config.learning_rate}")

    # Create for specific task
    customer_config = get_config_for_task("customer_service", num_episodes=1200)
    print(f"Customer service episodes: {customer_config.num_episodes}")

    # Optimize for hardware
    optimized_config = optimize_config_for_hardware(
        customer_config, gpu_memory_gb=12.0, num_gpus=2
    )
    print(f"Optimized batch size: {optimized_config.per_device_train_batch_size}")

    # Validate configuration
    warnings = optimized_config.validate()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    # Save and load
    optimized_config.save("config.json")
    loaded_config = TrainingConfig.load("config.json")
    print(f"Loaded config learning rate: {loaded_config.learning_rate}")


if __name__ == "__main__":
    example_config_usage()


def cuda_is_available() -> bool:
    """Return whether a CUDA device is usable, without importing torch eagerly.

    ``torch`` is an optional heavy dependency and configuration modules must
    stay importable without it (see ``tests/unit/test_torch_import_policy.py``),
    so the import happens inside this helper rather than at module level.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is installed in CI
        return False
    return bool(torch.cuda.is_available())
