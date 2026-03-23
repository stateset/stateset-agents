"""
Configuration surface for GSPO training.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch

from stateset_agents.exceptions import (
    IMPORT_EXCEPTIONS as BITSANDBYTES_IMPORT_EXCEPTIONS,
)

from .config import TrainingConfig


@dataclass
class GSPOConfig(TrainingConfig):
    """Configuration for GSPO training."""

    num_generations: int = 4
    beta: float = 0.0

    clip_range_left: float = 3e-4
    clip_range_right: float = 4e-4

    num_iterations: int = 1
    mini_batch_size: int = 1

    num_outer_iterations: int = 1
    generations_per_iteration: int = 100

    max_prompt_length: int = 256
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    gradient_checkpointing: bool = True
    use_8bit: bool = False
    use_4bit: bool = False

    use_vllm: bool = False
    use_gspo_token: bool = False

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> GSPOConfig:
        """Create GSPO config from standard training config."""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        return cls(**config_dict)

    def validate(self) -> list[str]:
        """Validate configuration and return warnings."""
        warnings = super().validate()

        if self.gradient_checkpointing and self.use_lora:
            warnings.append(
                "gradient_checkpointing with LoRA can require input gradients; "
                "StateSet Agents disables `use_cache` and enables input grads automatically."
            )

        model_name_lower = self.model_name.lower()
        if (
            "qwen" in model_name_lower
            and any(size in model_name_lower for size in ["3b", "7b"])
            and not (self.use_4bit or self.use_8bit)
        ):
            warnings.append(
                "Qwen 3B/7B models often require quantization on consumer GPUs; "
                "consider setting `use_4bit=True` to reduce memory usage."
            )

        if self.use_4bit or self.use_8bit:
            if not torch.cuda.is_available():
                warnings.append(
                    "4-bit/8-bit quantization requires CUDA; disable `use_4bit/use_8bit` "
                    "or run on a CUDA-enabled machine."
                )
            elif importlib.util.find_spec("bitsandbytes") is None:
                warnings.append(
                    "bitsandbytes is required for 4-bit/8-bit quantization. "
                    "Install with `pip install stateset-agents[trl]` or `pip install bitsandbytes`."
                )
            else:
                try:
                    import bitsandbytes  # noqa: F401  # type: ignore[import-not-found]
                except BITSANDBYTES_IMPORT_EXCEPTIONS:
                    warnings.append(
                        "bitsandbytes is installed but failed to import. "
                        "If you just installed it in a notebook, restart the runtime/kernel."
                    )

        return warnings


__all__ = ["GSPOConfig"]
