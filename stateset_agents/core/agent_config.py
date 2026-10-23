"""Agent configuration models and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ConfigurationError


class ConfigValidationError(ConfigurationError):
    """Raised when agent configuration validation fails.

    A :class:`~stateset_agents.core.errors.ConfigurationError`, so still a
    ``ValueError``; ``str(exc)`` is unchanged.
    """

    def __init__(
        self,
        field: str,
        value: Any,
        message: str,
        suggestions: list[str] | None = None,
    ):
        self.field = field
        self.value = value
        self.suggestions = suggestions or []
        suggestion_text = (
            f" Suggestions: {', '.join(self.suggestions)}" if self.suggestions else ""
        )
        super().__init__(f"Invalid {field}={value!r}: {message}{suggestion_text}")


#: Stop strings ``MultiTurnAgent`` used unconditionally before text fidelity
#: became the default: speaker labels and the first blank line. Opt back in
#: with ``AgentConfig(stop_sequences=list(LEGACY_STOP_SEQUENCES))``.
LEGACY_STOP_SEQUENCES: tuple[str, ...] = ("User:", "System:", "\n\n")


@dataclass
class AgentConfig:
    """Configuration for agent behavior.

    Compatible with HuggingFace-style generation and loading options while also
    supporting StateSet-specific stub, planning, and PEFT controls.

    Text fidelity: an assistant turn's text is exactly what the policy
    generated (``tokenizer.decode(token_ids, skip_special_tokens=True)``).
    Generation ends at EOS or ``max_new_tokens`` unless ``stop_sequences`` are
    configured; then the reply is cut before the first stop string, in the
    text *and* in the recorded ``token_ids`` alike (a token straddling the
    stop string is dropped whole). Pass :data:`LEGACY_STOP_SEQUENCES` to get
    the pre-0.65 behaviour of stopping at speaker labels and blank lines.

    ``top_k=0`` or ``top_k=None`` disables top-k filtering (plain sampling
    over the full vocabulary, filtered only by ``top_p``).

    ``enable_persona_hints`` appends canned sentences (e.g. a billing
    acknowledgement) to :meth:`MultiTurnAgent.generate_response` replies that
    miss an obvious keyword. It is off by default and is never applied to the
    rollout APIs (``generate_turn`` / ``generate_turns``) used for training.
    """

    model_name: str
    model_revision: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int | None = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    system_prompt: str | None = None
    use_chat_template: bool = True

    # HuggingFace model configuration
    torch_dtype: str = "bfloat16"
    # SDPA ships with PyTorch. FlashAttention remains an explicit opt-in
    # because setting it without the optional flash-attn wheel makes model
    # loading fail before training starts.
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    use_peft: bool = False
    peft_config: dict[str, Any] | None = None
    peft_path: str | None = (
        None  # Load a pre-trained LoRA adapter from disk (e.g. outputs/run_v1)
    )
    use_stub_model: bool = False
    stub_responses: list[str] | None = None

    # Reasoning Capabilities (DeepSeek-R1 style)
    enable_reasoning: bool = False
    reasoning_tag: str = "think"

    # Long-term planning
    enable_planning: bool = False
    planning_config: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    # Text fidelity (see class docstring)
    stop_sequences: list[str] = field(default_factory=list)
    enable_persona_hints: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all configuration parameters."""
        if self.chat_template_kwargs is not None:
            reserved = {
                "conversation",
                "messages",
                "tokenize",
                "add_generation_prompt",
                "return_tensors",
                "return_dict",
                "return_assistant_tokens_mask",
                "chat_template",
            }
            if (
                not isinstance(self.chat_template_kwargs, dict)
                or any(not isinstance(key, str) for key in self.chat_template_kwargs)
                or reserved.intersection(self.chat_template_kwargs)
            ):
                raise ConfigValidationError(
                    "chat_template_kwargs",
                    self.chat_template_kwargs,
                    "must be a dict of template options without tokenizer control arguments",
                )
        if not self.model_name or not isinstance(self.model_name, str):
            raise ConfigValidationError(
                "model_name",
                self.model_name,
                "must be a non-empty string",
                ["meta-llama/Llama-2-7b-chat-hf", "gpt2", "stub://test"],
            )
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ConfigValidationError(
                "model_revision",
                self.model_revision,
                "must be a non-empty tag or immutable commit when provided",
                ["Pin the full Hugging Face commit SHA for benchmark runs"],
            )

        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ConfigValidationError(
                "max_new_tokens",
                self.max_new_tokens,
                "must be a positive integer >= 1",
                ["256", "512", "1024", "2048"],
            )
        if self.max_new_tokens > 8192:
            raise ConfigValidationError(
                "max_new_tokens",
                self.max_new_tokens,
                "exceeds maximum of 8192 tokens",
                ["1024", "2048", "4096"],
            )

        if not isinstance(self.temperature, (int, float)) or self.temperature < 0.0:
            raise ConfigValidationError(
                "temperature",
                self.temperature,
                "must be a non-negative number",
                ["0.0 (deterministic)", "0.7 (balanced)", "1.0 (creative)"],
            )
        if self.temperature > 2.0:
            raise ConfigValidationError(
                "temperature",
                self.temperature,
                "exceeds maximum of 2.0 (produces incoherent output)",
                ["0.7", "0.9", "1.0"],
            )

        if not isinstance(self.top_p, (int, float)) or not 0.0 <= self.top_p <= 1.0:
            raise ConfigValidationError(
                "top_p",
                self.top_p,
                "must be between 0.0 and 1.0",
                ["0.9 (recommended)", "0.95", "1.0 (disabled)"],
            )

        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 0
        ):
            raise ConfigValidationError(
                "top_k",
                self.top_k,
                "must be a non-negative integer (0 or None disables top-k)",
                ["50", "0 (disabled)", "None (disabled)"],
            )
        if self.top_k is not None and self.top_k > 1000:
            raise ConfigValidationError(
                "top_k",
                self.top_k,
                "exceeds reasonable maximum of 1000",
                ["50", "100", "200"],
            )

        if (
            not isinstance(self.repetition_penalty, (int, float))
            or self.repetition_penalty < 1.0
        ):
            raise ConfigValidationError(
                "repetition_penalty",
                self.repetition_penalty,
                "must be >= 1.0 (1.0 = no penalty)",
                ["1.0 (disabled)", "1.1 (light)", "1.2 (moderate)"],
            )
        if self.repetition_penalty > 2.0:
            raise ConfigValidationError(
                "repetition_penalty",
                self.repetition_penalty,
                "exceeds reasonable maximum of 2.0",
                ["1.1", "1.2", "1.5"],
            )

        valid_dtypes = ["float16", "bfloat16", "float32"]
        if self.torch_dtype not in valid_dtypes:
            raise ConfigValidationError(
                "torch_dtype",
                self.torch_dtype,
                f"must be one of {valid_dtypes}",
                valid_dtypes,
            )

        valid_attn = ["flash_attention_2", "sdpa", "eager", None]
        if self.attn_implementation not in valid_attn:
            raise ConfigValidationError(
                "attn_implementation",
                self.attn_implementation,
                f"must be one of {valid_attn}",
                ["flash_attention_2 (fastest)", "sdpa (good)", "eager (compatible)"],
            )

        if self.peft_config and not self.use_peft:
            raise ConfigValidationError(
                "peft_config",
                self.peft_config,
                "peft_config provided but use_peft=False; set use_peft=True to enable LoRA",
                ["Set use_peft=True"],
            )

        if self.system_prompt and len(self.system_prompt) > 10000:
            raise ConfigValidationError(
                "system_prompt",
                f"<{len(self.system_prompt)} chars>",
                "system prompt exceeds 10000 character limit",
                ["Shorten the system prompt", "Move context to user messages"],
            )

        stop_sequences: Any = self.stop_sequences
        if isinstance(stop_sequences, str) or not isinstance(
            stop_sequences, (list, tuple)
        ):
            raise ConfigValidationError(
                "stop_sequences",
                self.stop_sequences,
                "must be a list of non-empty strings",
                ["[] (EOS only)", "list(LEGACY_STOP_SEQUENCES)", '["User:"]'],
            )
        if any(not isinstance(s, str) or not s for s in stop_sequences):
            raise ConfigValidationError(
                "stop_sequences",
                self.stop_sequences,
                "must contain only non-empty strings",
                ['["User:"]'],
            )
        self.stop_sequences = list(stop_sequences)

        if self.planning_config is not None and not isinstance(
            self.planning_config, dict
        ):
            raise ConfigValidationError(
                "planning_config",
                self.planning_config,
                "must be a dict of planning configuration overrides",
                ["Use a dict or set enable_planning=False"],
            )
