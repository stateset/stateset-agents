"""
Support utilities for agent backends.

This module centralises the lightweight stub backend used throughout the
framework so `core.agent` can stay focused on orchestration logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StubGenerationConfig:
    """Minimal generation configuration used by the stub backend."""

    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool = True
    repetition_penalty: float = 1.0
    pad_token_id: int = 0
    eos_token_id: int = 0


class StubTokenizer:
    """Lightweight tokenizer for stubbed agent backends."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.model_max_length = 4096
        self.chat_template: Optional[str] = None

    def __call__(
        self,
        prompt: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        return {"prompt": prompt, "input_ids": [[0]]}

    def decode(self, text: Any, skip_special_tokens: bool = True) -> str:
        return text if isinstance(text, str) else str(text)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **_: Any,
    ) -> str:
        parts = []
        for message in messages:
            parts.append(
                f"{message.get('role', 'user').capitalize()}: {message.get('content', '')}"
            )
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n".join(parts)


class StubModelConfig:
    """Configuration for StubModel to match PyTorch model interface."""
    def __init__(self):
        self.hidden_size = 768
        self.vocab_size = 50257


class StubModel:
    """Minimal text responder for offline/dev usage."""

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        if responses:
            self._responses = responses
        else:
            self._responses = [
                "I'm operating in stub mode. Let's keep iterating.",
                "This is a lightweight response from the stub backend.",
            ]
        self._index = 0
        self.config = StubModelConfig()
        self._training = False

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_response = self._responses[self._index % len(self._responses)]
        self._index += 1
        if context and isinstance(context, dict) and context.get("user_hint"):
            return f"{base_response} Hint noted: {context['user_hint']}"
        return base_response

    @property
    def device(self) -> str:
        return "cpu"

    def named_parameters(self):
        """Return empty iterator to match PyTorch interface."""
        return iter([])

    def parameters(self):
        """Return empty iterator to match PyTorch interface."""
        return iter([])

    def train(self, mode: bool = True):
        """Set training mode to match PyTorch interface."""
        self._training = mode
        return self

    def eval(self):
        """Set evaluation mode to match PyTorch interface."""
        self._training = False
        return self

    def to(self, device):
        """No-op to match PyTorch interface."""
        return self

    def save_pretrained(self, path: str):
        """No-op save to match HuggingFace interface."""
        pass

    def __call__(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Make model callable to match PyTorch interface."""
        # Create mock output that looks like transformer model output
        class MockOutput:
            def __init__(self):
                # For stub mode, return dummy loss and logits
                batch_size = 1
                seq_len = 10
                vocab_size = 50257

                # Import torch if available, otherwise create simple objects
                try:
                    import torch
                    self.loss = torch.tensor(0.5, requires_grad=True)
                    self.logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
                except ImportError:
                    self.loss = 0.5
                    self.logits = [[0.0] * vocab_size for _ in range(seq_len)]

        return MockOutput()


@dataclass
class StubBackend:
    """Container for stub backend components."""

    tokenizer: StubTokenizer
    model: StubModel
    generation_config: StubGenerationConfig


def create_stub_backend(
    *,
    stub_responses: Optional[List[str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
    pad_token_id: Optional[int],
    eos_token_id: Optional[int],
) -> StubBackend:
    """Build a stub backend tailored to the provided generation settings."""
    tokenizer = StubTokenizer()
    model = StubModel(stub_responses)
    generation_config = StubGenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_token_id or 0,
        eos_token_id=eos_token_id or 0,
    )
    return StubBackend(tokenizer=tokenizer, model=model, generation_config=generation_config)


__all__ = [
    "StubBackend",
    "StubGenerationConfig",
    "StubModel",
    "StubTokenizer",
    "create_stub_backend",
]
