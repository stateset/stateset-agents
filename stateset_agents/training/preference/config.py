"""Torch-free configuration for preference optimization and reward models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

from .encoding import PROMPT_FORMATS

__all__ = [
    "ONLINE_METHODS",
    "PREFERENCE_METHODS",
    "REFERENCE_FREE_METHODS",
    "REFERENCE_UPDATES",
    "OnlineDPOConfig",
    "PreferenceConfig",
    "RewardModelConfig",
    "validate_lora",
]

#: Policy optimization methods trained by :class:`PreferenceTrainer`.
PREFERENCE_METHODS = ("dpo", "ipo", "simpo", "kto", "orpo")
#: Methods without a reference model.
REFERENCE_FREE_METHODS = frozenset({"simpo", "orpo"})

_SCHEDULERS = ("linear", "cosine", "constant")
_NONFINITE = ("raise", "skip")
_PRECISION = ("no", "bf16")

_ConfigT = TypeVar("_ConfigT", bound="_LoopConfig")


def _int(value: Any, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _float(
    value: Any,
    name: str,
    *,
    low: float | None = None,
    positive: bool = False,
    below: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    if low is not None and number < low:
        raise ValueError(f"{name} must be >= {low}, got {value!r}")
    if below is not None and number >= below:
        raise ValueError(f"{name} must be < {below}, got {value!r}")


def _choice(value: Any, name: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def validate_lora(config: Any) -> None:
    """Validate the ``use_lora`` / ``lora_*`` fields a config carries."""
    if not isinstance(config.use_lora, bool):
        raise ValueError("use_lora must be a boolean")
    _int(config.lora_r, "lora_r", minimum=1)
    _int(config.lora_alpha, "lora_alpha", minimum=1)
    _float(config.lora_dropout, "lora_dropout", low=0.0, below=1.0)
    targets = config.lora_target_modules
    if targets is not None and (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(name, str) and name for name in targets)
    ):
        raise ValueError("lora_target_modules must be a nonempty list of names")


@dataclass
class _LoopConfig:
    """Optimization and data settings shared by both trainers."""

    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr_scheduler: str = "linear"
    warmup_steps: int = 0
    num_epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_length: int = 1024
    max_prompt_length: int | None = 512
    prompt_format: str = "auto"
    append_eos: bool = True
    disable_dropout: bool = True
    nonfinite_policy: str = "raise"
    mixed_precision: str = "no"
    seed: int = 42
    device: str = "auto"
    logging_steps: int = 10
    eval_steps: int | None = None
    save_steps: int | None = None
    output_dir: str | None = None
    validation_fraction: float = 0.0

    def validate(self) -> None:
        """Raise ``ValueError`` for any invalid setting."""
        _float(self.learning_rate, "learning_rate", positive=True)
        _float(self.weight_decay, "weight_decay", low=0.0)
        _float(self.adam_beta1, "adam_beta1", low=0.0, below=1.0)
        _float(self.adam_beta2, "adam_beta2", low=0.0, below=1.0)
        _float(self.adam_epsilon, "adam_epsilon", positive=True)
        _choice(self.lr_scheduler, "lr_scheduler", _SCHEDULERS)
        _int(self.warmup_steps, "warmup_steps", minimum=0)
        _int(self.num_epochs, "num_epochs", minimum=1)
        if self.max_steps is not None:
            _int(self.max_steps, "max_steps", minimum=1)
        _int(self.batch_size, "batch_size", minimum=1)
        _int(self.gradient_accumulation_steps, "gradient_accumulation_steps", minimum=1)
        _float(self.max_grad_norm, "max_grad_norm", positive=True)
        _int(self.max_length, "max_length", minimum=2)
        if self.max_prompt_length is not None:
            _int(self.max_prompt_length, "max_prompt_length", minimum=1)
            if self.max_prompt_length >= self.max_length:
                raise ValueError("max_prompt_length must be smaller than max_length")
        _choice(self.prompt_format, "prompt_format", PROMPT_FORMATS)
        for name in ("append_eos", "disable_dropout"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        _choice(self.nonfinite_policy, "nonfinite_policy", _NONFINITE)
        _choice(self.mixed_precision, "mixed_precision", _PRECISION)
        _int(self.seed, "seed", minimum=0)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")
        _int(self.logging_steps, "logging_steps", minimum=1)
        for name in ("eval_steps", "save_steps"):
            value = getattr(self, name)
            if value is not None:
                _int(value, name, minimum=1)
        if self.save_steps is not None and not self.output_dir:
            raise ValueError("save_steps needs output_dir")
        _float(self.validation_fraction, "validation_fraction", low=0.0, below=1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[_ConfigT], values: dict[str, Any]) -> _ConfigT:
        """Build a validated config, rejecting unknown keys."""
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown {cls.__name__} settings: {unknown}")
        config = cls(**values)
        config.validate()
        return config


@dataclass
class PreferenceConfig(_LoopConfig):
    """Settings for :class:`~stateset_agents.training.preference.PreferenceTrainer`.

    ``method``: ``dpo`` (``label_smoothing > 0`` gives conservative DPO),
    ``ipo``, ``simpo`` (reference-free, ``simpo_gamma`` target margin),
    ``kto`` (unpaired, ``kto_*_weight`` class weights) or ``orpo``
    (reference-free, SFT + odds ratio). ``beta`` is DPO's/KTO's KL strength,
    IPO's regularizer τ, SimPO's reward scale and ORPO's λ.
    """

    method: str = "dpo"
    beta: float = 0.1
    label_smoothing: float = 0.0
    simpo_gamma: float = 0.5
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0
    precompute_reference_logps: bool = False
    logprob_dtype: str | None = None
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = field(default=None)

    @property
    def reference_free(self) -> bool:
        return self.method in REFERENCE_FREE_METHODS

    def validate(self) -> None:
        super().validate()
        _choice(self.method, "method", PREFERENCE_METHODS)
        _float(self.beta, "beta", positive=True)
        _float(self.label_smoothing, "label_smoothing", low=0.0, below=0.5)
        if self.label_smoothing and self.method not in {"dpo", "simpo"}:
            raise ValueError("label_smoothing applies to dpo and simpo only")
        _float(self.simpo_gamma, "simpo_gamma", low=0.0)
        _float(self.kto_desirable_weight, "kto_desirable_weight", positive=True)
        _float(self.kto_undesirable_weight, "kto_undesirable_weight", positive=True)
        if not isinstance(self.precompute_reference_logps, bool):
            raise ValueError("precompute_reference_logps must be a boolean")
        if self.precompute_reference_logps and self.reference_free:
            raise ValueError(f"{self.method} is reference-free; nothing to precompute")
        if self.logprob_dtype is not None:
            _choice(
                str(self.logprob_dtype).strip().lower(),
                "logprob_dtype",
                ("fp32", "float32", "bf16", "bfloat16", "fp16", "float16", "half"),
            )
        validate_lora(self)


@dataclass
class RewardModelConfig(_LoopConfig):
    """Settings for :class:`~stateset_agents.training.preference.RewardModelTrainer`.

    ``use_margin`` subtracts each pair's ``margin`` inside the Bradley-Terry
    sigmoid; ``center_rewards_coefficient`` adds ``c (r_w + r_l)²``.
    ``use_lora`` trains LoRA adapters on the backbone (needs ``peft``) plus
    the full reward head instead of the whole backbone.
    """

    learning_rate: float = 1e-5
    use_margin: bool = True
    center_rewards_coefficient: float | None = None
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = field(default=None)

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.use_margin, bool):
            raise ValueError("use_margin must be a boolean")
        validate_lora(self)
        if self.center_rewards_coefficient is not None:
            _float(
                self.center_rewards_coefficient,
                "center_rewards_coefficient",
                low=0.0,
            )


#: Paired methods :class:`OnlineDPOTrainer` trains each round with.
ONLINE_METHODS = ("dpo", "ipo")
#: ``fixed``: every round uses the initial policy as its reference;
#: ``previous_round``: round ``r`` uses the policy as it ended round ``r - 1``.
REFERENCE_UPDATES = ("fixed", "previous_round")


@dataclass
class OnlineDPOConfig(PreferenceConfig):
    """Settings for :class:`~stateset_agents.training.preference.OnlineDPOTrainer`.

    Each of ``rounds`` rounds samples ``samples_per_prompt`` completions per
    prompt from the current policy (``max_new_tokens``,
    ``sampling_temperature``, ``top_p``; ``generation_batch_size`` rows at a
    time), scores them with the judge, pairs the best against the worst when
    their scores differ by more than ``min_score_margin``, and trains
    ``method`` (``dpo`` or ``ipo``) on that round's pairs for ``num_epochs``
    (or ``max_steps`` updates) with a fresh optimizer. ``reference_update``
    keeps the initial policy as the reference or refreshes it each round.
    """

    rounds: int = 3
    samples_per_prompt: int = 4
    max_new_tokens: int = 128
    sampling_temperature: float = 1.0
    top_p: float = 1.0
    min_score_margin: float = 0.0
    reference_update: str = "fixed"
    generation_batch_size: int = 16

    def validate(self) -> None:
        super().validate()
        _choice(self.method, "method", ONLINE_METHODS)
        _int(self.rounds, "rounds", minimum=1)
        _int(self.samples_per_prompt, "samples_per_prompt", minimum=2)
        _int(self.max_new_tokens, "max_new_tokens", minimum=1)
        _float(self.sampling_temperature, "sampling_temperature", positive=True)
        _float(self.top_p, "top_p", positive=True)
        if float(self.top_p) > 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p!r}")
        _float(self.min_score_margin, "min_score_margin", low=0.0)
        _choice(self.reference_update, "reference_update", REFERENCE_UPDATES)
        _int(self.generation_batch_size, "generation_batch_size", minimum=1)
        if self.validation_fraction:
            raise ValueError(
                "online DPO builds its pairs each round; pass eval_prompts "
                "instead of validation_fraction"
            )
