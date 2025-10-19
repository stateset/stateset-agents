"""
Type stubs for transformers library.
"""

from typing import Any, Dict, List, Optional, Protocol, Union

import torch

class PreTrainedTokenizer:
    """Stub for PreTrainedTokenizer."""

    pad_token_id: Optional[int]
    eos_token_id: Optional[int]

    def encode(self, text: str, **kwargs) -> List[int]: ...
    def decode(self, tokens: List[int], **kwargs) -> str: ...
    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
    def __call__(self, text: str, **kwargs) -> Dict[str, Any]: ...

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
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]

class StoppingCriteria:
    """Stub for StoppingCriteria."""

    pass

class StoppingCriteriaList:
    """Stub for StoppingCriteriaList."""

    def __init__(self, criteria: List[StoppingCriteria]): ...

class get_cosine_schedule_with_warmup:
    """Stub for get_cosine_schedule_with_warmup."""

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int): ...

class get_linear_schedule_with_warmup:
    """Stub for get_linear_schedule_with_warmup."""

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int): ...
