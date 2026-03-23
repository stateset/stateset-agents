"""
Configuration surface for VAPO training.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TrainingConfig


@dataclass
class VAPOConfig(TrainingConfig):
    """
    Configuration for VAPO training.

    VAPO uses value-based RL with seven key modifications for stable
    long-CoT reasoning training.
    """

    model_name: str = "gpt2"
    num_iterations: int = 1

    max_prompt_length: int = 256
    max_completion_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    value_hidden_size: int = 1024
    value_num_layers: int = 2
    value_warmup_steps: int = 50

    lambda_critic: float = 1.0
    lambda_policy_alpha: float = 0.05

    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28

    use_token_level_loss: bool = True

    use_positive_lm_loss: bool = True
    positive_lm_weight: float = 0.1

    group_size: int = 16
    num_prompts_per_batch: int = 512

    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 2e-6

    mini_batch_size: int = 512

    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> VAPOConfig:
        """Create VAPO config from standard training config."""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


__all__ = ["VAPOConfig"]
