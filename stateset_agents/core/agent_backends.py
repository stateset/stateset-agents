"""
Support utilities for agent backends.

This module centralises the lightweight stub backend used throughout the
framework so `core.agent` can stay focused on orchestration logic.

It also defines the ModelBackend protocol for dependency injection,
allowing tests to inject mock implementations without patching globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol defining the tokenizer interface for dependency injection."""

    pad_token: str | None
    eos_token: str | None
    pad_token_id: int | None
    eos_token_id: int | None
    model_max_length: int
    chat_template: str | None

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Tokenize input text."""
        ...

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        ...

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""
        ...

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        """Apply chat template to messages."""
        ...


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining the model interface for dependency injection."""

    @property
    def device(self) -> Any:
        """Return the device the model is on."""
        ...

    def parameters(self) -> Any:
        """Return model parameters."""
        ...

    def train(self, mode: bool = True) -> ModelProtocol:
        """Set training mode."""
        ...

    def eval(self) -> ModelProtocol:
        """Set evaluation mode."""
        ...

    def to(self, device: Any) -> ModelProtocol:
        """Move model to device."""
        ...


@runtime_checkable
class GenerationConfigProtocol(Protocol):
    """Protocol defining the generation config interface."""

    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    repetition_penalty: float
    pad_token_id: int
    eos_token_id: int


class ModelBackend(Protocol):
    """Protocol for model backends supporting dependency injection.

    This protocol allows different model implementations (HuggingFace,
    vLLM, stub, mock) to be used interchangeably with the Agent class.

    Example:
        >>> class MyMockBackend:
        ...     @property
        ...     def tokenizer(self) -> TokenizerProtocol:
        ...         return MockTokenizer()
        ...     @property
        ...     def model(self) -> ModelProtocol:
        ...         return MockModel()
        ...     @property
        ...     def generation_config(self) -> GenerationConfigProtocol:
        ...         return MockConfig()
        >>>
        >>> # Use in tests without patching globals
        >>> agent = Agent(config, backend=MyMockBackend())
    """

    @property
    def tokenizer(self) -> TokenizerProtocol:
        """Return the tokenizer."""
        ...

    @property
    def model(self) -> ModelProtocol:
        """Return the model."""
        ...

    @property
    def generation_config(self) -> GenerationConfigProtocol:
        """Return the generation configuration."""
        ...


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
        self.chat_template: str | None = None

    def __call__(
        self,
        prompt: str | list[str],
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        **_: Any,
    ) -> dict[str, Any]:
        texts = [prompt] if isinstance(prompt, str) else list(prompt)

        batch_ids: list[list[int]] = []
        for text in texts:
            tokens = [ord(ch) % 256 for ch in (text or "")]
            if not tokens:
                tokens = [0]
            if max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            batch_ids.append(tokens)

        if padding:
            max_len = max(len(t) for t in batch_ids)
            batch_masks = []
            for i, tokens in enumerate(batch_ids):
                pad_len = max_len - len(tokens)
                batch_masks.append([1] * len(tokens) + [0] * pad_len)
                batch_ids[i] = tokens + [self.pad_token_id] * pad_len
        else:
            batch_masks = [[1] * len(t) for t in batch_ids]

        if return_tensors == "pt":
            try:
                import torch
                return {
                    "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
                }
            except ImportError:
                pass

        return {"input_ids": batch_ids, "attention_mask": batch_masks}

    def decode(self, text: Any, skip_special_tokens: bool = True) -> str:
        return text if isinstance(text, str) else str(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Naive encode for stub usage."""
        return [ord(ch) for ch in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
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

    def save_pretrained(self, path: str) -> None:
        """No-op save to match HuggingFace interface."""
        pass


class StubModelConfig:
    """Configuration for StubModel to match PyTorch model interface."""

    def __init__(self) -> None:
        self.hidden_size = 768
        self.vocab_size = 50257


class StubModel:
    """Minimal text responder for offline/dev usage."""

    def __init__(self, responses: list[str] | None = None) -> None:
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

    @property
    def training(self) -> bool:
        return self._training

    def generate(
        self,
        prompt: Any = None,
        context: dict[str, Any] | None = None,
        *,
        input_ids: Any = None,
        attention_mask: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Tensor-mode: return stub token ids matching shape expectations
        if input_ids is not None:
            try:
                import torch
                if hasattr(input_ids, "shape"):
                    bs = input_ids.shape[0]
                    prompt_len = input_ids.shape[1]
                else:
                    bs = 1
                    prompt_len = 10
                max_new = kwargs.get("max_new_tokens", 5)
                total_len = prompt_len + max_new
                return torch.ones(bs, total_len, dtype=torch.long)
            except ImportError:
                return [[1] * 15]

        # String-mode: return a stub text response
        base_response = self._responses[self._index % len(self._responses)]
        self._index += 1
        if context and isinstance(context, dict) and context.get("user_hint"):
            return f"{base_response} Hint noted: {context['user_hint']}"
        return base_response

    @property
    def device(self) -> str:
        return "cpu"

    def named_parameters(self) -> Any:
        """Return empty iterator to match PyTorch interface."""
        return iter([])

    def parameters(self) -> Any:
        """Return empty iterator to match PyTorch interface."""
        return iter([])

    def train(self, mode: bool = True) -> "StubModel":
        """Set training mode to match PyTorch interface."""
        self._training = mode
        return self

    def eval(self) -> "StubModel":
        """Set evaluation mode to match PyTorch interface."""
        self._training = False
        return self

    def to(self, device: Any) -> "StubModel":
        """No-op to match PyTorch interface."""
        return self

    def save_pretrained(self, path: str) -> None:
        """No-op save to match HuggingFace interface."""
        pass

    def __call__(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        labels: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Make model callable to match PyTorch interface."""
        batch_size = 1
        seq_len = 10
        vocab_size = self.config.vocab_size
        hidden_size = self.config.hidden_size
        output_hidden = kwargs.get("output_hidden_states", False)

        try:
            if input_ids is not None and hasattr(input_ids, "shape"):
                batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
            elif input_ids is not None and isinstance(input_ids, (list, tuple)):
                batch_size = len(input_ids)
                seq_len = len(input_ids[0]) if input_ids else 10
        except Exception:
            pass

        class MockOutput:
            def __init__(
                self, bs: int, sl: int, vs: int, hs: int, include_hidden: bool
            ) -> None:
                self.loss: Any
                self.logits: Any
                self.hidden_states: Any
                try:
                    import torch
                    self.loss = torch.tensor(0.5, requires_grad=True)
                    self.logits = torch.randn(bs, sl, vs, requires_grad=True)
                    if include_hidden:
                        self.hidden_states = (torch.randn(bs, sl, hs),)
                    else:
                        self.hidden_states = None
                except ImportError:
                    self.loss = 0.5
                    self.logits = [[[0.0] * vs for _ in range(sl)] for _ in range(bs)]
                    self.hidden_states = None

        return MockOutput(batch_size, seq_len, vocab_size, hidden_size, output_hidden)


@dataclass
class StubBackend:
    """Container for stub backend components."""

    tokenizer: StubTokenizer
    model: StubModel
    generation_config: StubGenerationConfig


def create_stub_backend(
    *,
    stub_responses: list[str] | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
    pad_token_id: int | None,
    eos_token_id: int | None,
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
    return StubBackend(
        tokenizer=tokenizer, model=model, generation_config=generation_config
    )


__all__ = [
    # Protocols for dependency injection
    "GenerationConfigProtocol",
    "ModelBackend",
    "ModelProtocol",
    "TokenizerProtocol",
    # Stub implementations
    "StubBackend",
    "StubGenerationConfig",
    "StubModel",
    "StubTokenizer",
    "create_stub_backend",
]
