"""
Configuration surface for DAPO training.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TrainingConfig


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
    prompt_batch_size: int = 512
    mini_batch_size: int = 512
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

    eval_repeats: int = 32

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> DAPOConfig:
        """Create a DAPO config from a standard training config."""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


__all__ = ["DAPOConfig"]
