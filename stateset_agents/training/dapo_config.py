"""
Configuration surface for DAPO training.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ConfigurationError
from .config import (
    TrainingConfig,
    reject_dead_fields,
    require_finite,
    require_positive_ints,
    validate_group_size,
)


@dataclass
class DAPOConfig(TrainingConfig):
    """
    Configuration for DAPO training.

    DAPO uses four key techniques for stable long-CoT training:
    1. Clip-Higher (asymmetric clipping)
    2. Dynamic Sampling (filter trivial accuracy)
    3. Token-level loss normalization
    4. Overlong reward shaping
    """

    model_name: str = "gpt2"

    max_prompt_length: int = 256
    max_completion_length: int = 512

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28

    group_size: int = 16
    # Deprecated (removed in v0.71.0): never read. Prompts per iteration are
    # per_device_train_batch_size; a non-default value is rejected.
    prompt_batch_size: int = 512
    mini_batch_size: int = 512
    # Inner gradient updates per rollout (mu). Cadence (shared with GEPO and
    # VAPO): the LR scheduler advances once per inner update; global_step
    # counts train_steps, not inner updates.
    num_gradient_updates: int = 16

    use_dynamic_sampling: bool = True
    min_accuracy_threshold: float = 0.0
    max_accuracy_threshold: float = 1.0
    dynamic_sampling_buffer_size: int = 1024

    use_overlong_shaping: bool = True
    max_generation_length: int = 20480
    overlong_cache_length: int = 4096
    overlong_penalty: float = -1.0

    use_token_level_loss: bool = True

    # Precision of the per-token log-softmax. None (default) means fp32, the
    # most accurate; "bf16"/"fp16" trade numerics for peak memory on
    # large-vocab models.
    logprob_dtype: str | None = None

    beta: float = 0.0
    use_reference_model: bool = False

    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant"
    temperature: float = 1.0
    top_p: float = 0.7

    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.85
    vllm_tensor_parallel_size: int = 1
    vllm_enable_prefix_caching: bool = True

    # Deprecated (removed in v0.71.0): DAPO has no evaluation loop to repeat;
    # a non-default value is rejected.
    eval_repeats: int = 32

    def validate_for_training(self) -> None:
        """Also require two responses per group (:func:`validate_group_size`)
        and DAPO's own numeric fields in range."""
        super().validate_for_training()
        validate_group_size(self.group_size)
        reject_dead_fields(
            self,
            {"prompt_batch_size": "per_device_train_batch_size", "eval_repeats": None},
        )
        require_positive_ints(
            self,
            "dynamic_sampling_buffer_size",
            "max_generation_length",
            "mini_batch_size",
        )
        if (
            type(self.overlong_cache_length) is not int
            or self.overlong_cache_length < 0
        ):
            raise ConfigurationError(
                "overlong_cache_length must be a nonnegative integer "
                f"(got {self.overlong_cache_length!r})"
            )
        require_finite(
            self, "clip_eps_low", minimum=0, maximum=1, exclusive_maximum=True
        )
        require_finite(self, "clip_eps_high", minimum=0)
        require_finite(
            self,
            "min_accuracy_threshold",
            "max_accuracy_threshold",
            minimum=0,
            maximum=1,
        )
        if self.min_accuracy_threshold > self.max_accuracy_threshold:
            raise ConfigurationError(
                "min_accuracy_threshold must not exceed max_accuracy_threshold"
            )
        require_finite(self, "overlong_penalty")

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> DAPOConfig:
        """Create a DAPO config from a standard training config."""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


__all__ = ["DAPOConfig"]
