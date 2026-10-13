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

    # VAPO/VC-PPO use an undiscounted return: with one reward on the final
    # token, gamma < 1 shrinks critic targets geometrically toward the start
    # of the response, and the length-adaptive lambda assumes gamma = 1.
    # Redeclared here only to change the inherited default (field order and
    # positional arguments are unchanged).
    gamma: float = 1.0
    lambda_critic: float = 1.0
    # Length-adaptive policy lambda = 1 - 1/(alpha * L) (VAPO eq. 7). The
    # paper's 0.05 targets long reasoning outputs: completions with
    # alpha * L <= 1 (under 20 tokens at 0.05) get lambda = 0, i.e. one-step
    # TD advantages that rely entirely on the critic (a warning is logged).
    # Raise alpha above 1 / typical_length for short completions.
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

    # Inner gradient updates per rollout (mu). The default of 1 keeps VAPO
    # strictly on-policy; values > 1 reuse each rollout, and the importance
    # ratio is then computed against log probs frozen at rollout time.
    # Cadence (shared with DAPO and GEPO): the LR schedulers advance once per
    # inner update; global_step counts train_steps, not inner updates.
    num_gradient_updates: int = 1

    # Precision of the per-token log-softmax. None (default) means fp32, the
    # most accurate; "bf16"/"fp16" trade numerics for peak memory on
    # large-vocab models.
    logprob_dtype: str | None = None

    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> VAPOConfig:
        """Create VAPO config from standard training config."""
        config_dict = config.to_dict()
        # A base config still on TrainingConfig's discount default did not
        # choose it; VAPO's own default (undiscounted) applies instead.
        if config_dict.get("gamma") == TrainingConfig().gamma:
            config_dict.pop("gamma")
        config_dict.update(kwargs)
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


__all__ = ["VAPOConfig"]
