"""
Type stubs for transformers library.
"""

from typing import Any

import torch

class PreTrainedTokenizer:
    """Stub for PreTrainedTokenizer."""

    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, **kwargs) -> list[int]: ...
    def decode(self, tokens: list[int], **kwargs) -> str: ...
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs) -> str: ...
    def __call__(self, text: str, **kwargs) -> dict[str, Any]: ...

class PreTrainedTokenizerFast(PreTrainedTokenizer):
    """Stub for PreTrainedTokenizerFast."""

    pass

class PreTrainedModel:
    """Stub for PreTrainedModel."""

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor: ...
    def __call__(self, *args, **kwargs) -> Any: ...

class AutoTokenizer:
    """Stub for AutoTokenizer."""

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> PreTrainedTokenizer: ...

class AutoModelForCausalLM:
    """Stub for AutoModelForCausalLM."""

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> PreTrainedModel: ...

class GenerationConfig:
    """Stub for GenerationConfig."""

    def __init__(self, **kwargs): ...
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    repetition_penalty: float
    pad_token_id: int | None
    eos_token_id: int | None

class StoppingCriteria:
    """Stub for StoppingCriteria."""

    pass

class StoppingCriteriaList:
    """Stub for StoppingCriteriaList."""

    def __init__(self, criteria: list[StoppingCriteria]): ...

class get_cosine_schedule_with_warmup:
    """Stub for get_cosine_schedule_with_warmup."""

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int): ...

class get_linear_schedule_with_warmup:
    """Stub for get_linear_schedule_with_warmup."""

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int): ...
