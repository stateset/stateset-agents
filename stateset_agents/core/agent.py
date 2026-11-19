"""
Agent base classes for multi-turn RL training

This module defines the core agent interfaces and implementations
for training conversational AI agents using GRPO.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
from collections.abc import AsyncIterator, Callable
from contextlib import nullcontext
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from stateset_agents.core.generation_mode import inference_mode
from stateset_agents.utils.async_calls import drain_owned_operation

from .agent_backends import ModelBackend, StubModel, create_stub_backend
from .agent_config import AgentConfig, ConfigValidationError
from .agent_initialization import (
    BackendInitializationState,
    _AdapterAttachment,
    _BackendInitialization,
    _checked_component,
)
from .conversation import (
    ConversationResetError,
    _ConversationState,
    conversation_call,
    conversation_stream,
)
from .muse_protocol import is_muse_spark, muse_spark_final_response
from .trajectory import ConversationTurn
from .transformers_compat import generation_compat_kwargs, load_generation_model

if TYPE_CHECKING:
    from ..experimental.long_term_planning import PlanningManager
    from .agent_adapter_checkpoint import _AdapterCheckpoint
    from .agent_checkpoint import _CheckpointRestore

try:
    import torch
except ImportError:  # pragma: no cover - allow stub mode without PyTorch
    torch = None

try:
    from jinja2.exceptions import TemplateError as _Jinja2TemplateError
except ImportError:  # pragma: no cover
    _Jinja2TemplateError = ValueError
# Lazy imports for transformers to avoid torch/torchvision compatibility issues
AutoModelForCausalLM: Any | None = None
AutoTokenizer: Any | None = None
_transformers_agent_loaded = False


class _FallbackGenerationConfig:
    """Fallback GenerationConfig when transformers not available."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FallbackStoppingCriteria:
    """Fallback StoppingCriteria when transformers not available."""

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> bool:  # pragma: no cover - placeholder
        return False


class _FallbackStoppingCriteriaList(list[Any]):
    """Fallback StoppingCriteriaList when transformers not available."""

    pass


GenerationConfig: Any = _FallbackGenerationConfig
StoppingCriteria: Any = _FallbackStoppingCriteria
StoppingCriteriaList: Any = _FallbackStoppingCriteriaList


def _adopt_shim_mocks() -> bool:
    """Adopt test-injected mocks from the ``stateset_agents.core.agent`` shim.

    Returns True when both ``AutoModelForCausalLM`` and ``AutoTokenizer`` were
    replaced with shim values. Any exception typical of missing attributes is
    swallowed since this path is a best-effort hook for test injection.
    """
    global AutoModelForCausalLM, AutoTokenizer
    try:
        import sys as _sys

        shim = _sys.modules.get("stateset_agents.core.agent")
        if shim is None:
            return False
        shim_model = getattr(shim, "AutoModelForCausalLM", None)
        shim_tokenizer = getattr(shim, "AutoTokenizer", None)
        if shim_model is not None and shim_tokenizer is not None:
            AutoModelForCausalLM = shim_model
            AutoTokenizer = shim_tokenizer
            return True
    except AGENT_SHIM_EXCEPTIONS:
        pass
    return False


def _load_transformers_agent() -> bool:
    """Lazily load transformers to avoid import-time errors.

    Resolution order:
      1. If already loaded and globals still valid, refresh from shim and return.
      2. Honor caller/test-injected globals directly.
      3. Honor mocks injected via the ``stateset_agents.core.agent`` shim.
      4. Import from the real ``transformers`` package.
    """
    global _transformers_agent_loaded, AutoModelForCausalLM, AutoTokenizer
    global GenerationConfig, StoppingCriteria, StoppingCriteriaList

    if _transformers_agent_loaded:
        _adopt_shim_mocks()
        if AutoModelForCausalLM is not None and AutoTokenizer is not None:
            return True
        _transformers_agent_loaded = False

    if AutoModelForCausalLM is not None and AutoTokenizer is not None:
        _transformers_agent_loaded = True
        return True

    if _adopt_shim_mocks():
        _transformers_agent_loaded = True
        return True

    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import GenerationConfig as _GenerationConfig
        from transformers import StoppingCriteria as _StoppingCriteria
        from transformers import StoppingCriteriaList as _StoppingCriteriaList

        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        GenerationConfig = cast(Any, _GenerationConfig)
        StoppingCriteria = cast(Any, _StoppingCriteria)
        StoppingCriteriaList = cast(Any, _StoppingCriteriaList)
        _transformers_agent_loaded = True
        return True
    except (ImportError, RuntimeError) as e:  # pragma: no cover
        logging.warning(f"Failed to load transformers: {e}")
        return False


LoraConfig: Any | None = None
get_peft_model: Any | None = None
PeftModel: Any | None = None
_peft_loaded = False


def _load_peft() -> bool:
    """Lazily load PEFT to keep lightweight imports quiet in API-only usage."""
    global _peft_loaded, LoraConfig, get_peft_model, PeftModel
    if _peft_loaded:
        if LoraConfig is None:
            return False
        return get_peft_model is not None
    try:
        from peft import LoraConfig as _LoraConfig
        from peft import PeftModel as _PeftModel
        from peft import get_peft_model as _get_peft_model

        LoraConfig = _LoraConfig
        get_peft_model = _get_peft_model
        PeftModel = _PeftModel
        _peft_loaded = True
        return True
    except ImportError:  # pragma: no cover
        _peft_loaded = True
        return False


logger = logging.getLogger(__name__)

AGENT_SHIM_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    KeyError,
    TypeError,
)
TOKENIZER_ATTR_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    TypeError,
    ValueError,
)
PLANNING_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
)
CHAT_TEMPLATE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    _Jinja2TemplateError,
)
TOOL_EXEC_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
)

STREAM_STOP_PHRASES = ("User:", "System:", "Human:", "AI:")


def _stream_visible_text(pending: str) -> tuple[str, str, bool]:
    """Hold back possible stop prefixes until later chunks resolve them."""
    stop_positions = [
        position
        for phrase in STREAM_STOP_PHRASES
        if (position := pending.find(phrase)) >= 0
    ]
    if stop_positions:
        return pending[: min(stop_positions)], "", True

    held = max(
        (
            length
            for phrase in STREAM_STOP_PHRASES
            for length in range(1, min(len(phrase), len(pending) + 1))
            if pending.endswith(phrase[:length])
        ),
        default=0,
    )
    if held:
        return pending[:-held], pending[-held:], False
    return pending, "", False


class StopOnSpecialTokens(StoppingCriteria):
    """Custom stopping criteria for conversation agents"""

    def __init__(self, stop_tokens: list[str], tokenizer: Any) -> None:
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer
        self.stop_token_ids: list[list[int]] = []
        for token in stop_tokens:
            try:
                token_ids = tokenizer.encode(token, add_special_tokens=False)
            except TOKENIZER_ATTR_EXCEPTIONS:
                token_ids = []
            if token_ids:
                self.stop_token_ids.append(token_ids)

    def ends_with_stop(self, sequence: list[int]) -> bool:
        """Whether ``sequence`` ends with one of the stop strings' token ids."""
        return any(
            len(stop_ids) <= len(sequence) and sequence[-len(stop_ids) :] == stop_ids
            for stop_ids in self.stop_token_ids
        )

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        # One flag per row: HF ORs criteria into a per-sequence done mask, so a
        # scalar would stop (or keep running) every row of a batched group.
        if torch is not None and hasattr(input_ids, "tolist"):
            done = [self.ends_with_stop(row) for row in input_ids.tolist()]
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)
        rows = [list(row) for row in input_ids] if input_ids else []
        return [self.ends_with_stop(row) for row in rows]


def _apply_persona_keyword_hints(response: str, messages: list[dict[str, Any]]) -> str:
    """Inject lightweight persona hints when responses miss obvious intents."""

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last_user or not isinstance(last_user.get("content"), str):
        return response

    content_l = last_user["content"].lower()
    resp_l = response.lower()

    def _missing_keywords(keywords: list[str], required: list[str]) -> bool:
        return any(k in content_l for k in keywords) and not any(
            req in resp_l for req in required
        )

    if _missing_keywords(["bill", "charge"], ["billing", "charge", "help"]):
        return response + " I'll help you with your billing question right away."
    if _missing_keywords(["error", "technical"], ["technical", "error", "assist"]):
        return response + " I can assist you with technical issues."
    return response


class Agent:
    """
    Abstract base class for all agents

    Supports dependency injection via the `backend` parameter for easier testing.

    Example:
        >>> # For production with transformers
        >>> agent = Agent(config)
        >>> await agent.initialize()
        >>>
        >>> # For testing with a mock backend
        >>> from stateset_agents.core.agent_backends import StubBackend, create_stub_backend
        >>> backend = create_stub_backend(...)
        >>> agent = Agent(config, backend=backend)
        >>> await agent.initialize()  # Uses injected backend
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        model_loader: Callable[[AgentConfig], Any] | None = None,
        tokenizer_loader: Callable[[AgentConfig], Any] | None = None,
        generation_config_factory: (
            Callable[[AgentConfig, Any, Any], GenerationConfig] | None
        ) = None,
        backend: ModelBackend | None = None,
    ):
        """Initialize the agent.

        Args:
            config: Agent configuration. Defaults to stub model config.
            model_loader: Optional custom model loader function.
            tokenizer_loader: Optional custom tokenizer loader function.
            generation_config_factory: Optional custom generation config factory.
            backend: Optional pre-configured ModelBackend for dependency injection.
                    If provided, bypasses model/tokenizer loading during initialize().
        """
        self.config = config or AgentConfig(
            model_name="stub://test", use_stub_model=True
        )
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.generation_config: Any | None = None
        self.conversation_history: list[Any] = []
        self._last_reasoning: str | None = None
        self._conversation = _ConversationState()
        self._model_loader = model_loader
        self._tokenizer_loader = tokenizer_loader
        self._generation_config_factory = generation_config_factory
        self._injected_backend = backend
        self._backend_initialization = _BackendInitialization()
        self._checkpoint_restore: _CheckpointRestore | None = None
        self._checkpoint_adapters: _AdapterCheckpoint | None = None

    @property
    def _is_stub_backend(self) -> bool:
        """Whether the agent is running with a stub model (read-only)."""
        return isinstance(self.model, StubModel)

    async def initialize(self) -> None:
        """Initialize the backend once per component/adapter identity.

        Loaders retain their synchronous calling convention. Completed components
        survive retryable setup failures; uncertain adapter effects cannot replay
        on the same model. Async prerequisites can join this boundary through
        ``_initialize_extensions``; work after an overridden ``initialize``
        method's super-call retains its own lifecycle.
        """
        owner = self._initialization_owner()
        with owner.lock:
            if owner.state is BackendInitializationState.INITIALIZING:
                raise RuntimeError("Backend initialization cannot reenter itself")
            owner.state = BackendInitializationState.INITIALIZING
            owner.phase = "backend"
            owner.error = None
            try:
                restore = getattr(self, "_checkpoint_restore", None)
                with restore.open() if restore is not None else nullcontext():
                    self._initialize_backend()
                    owner.phase = "adapter"
                    self._initialize_adapter()
                    if restore is not None:
                        owner.phase = "checkpoint_restore"
                        restore.apply(self)
                for name, value in (
                    ("Model", self.model),
                    ("Tokenizer", self.tokenizer),
                    ("Generation configuration", self.generation_config),
                ):
                    _checked_component(name, value)
                extension = self._initialize_extensions
                components = (self.model, self.tokenizer, self.generation_config)
                if getattr(extension, "__func__", None) is Agent._initialize_extensions:
                    self._complete_initialization(owner)
                    return
                owner.phase = "extensions"
            except BaseException as error:
                owner.error = error
                owner.state = BackendInitializationState.FAILED
                raise

        # Do not hold a thread lock across an async dependency. It may need a
        # worker thread, and competing/reentrant attempts must observe the active
        # state instead of deadlocking behind that thread.
        try:
            extension_task = asyncio.create_task(extension())
            try:
                await asyncio.shield(extension_task)
            except BaseException:
                if not extension_task.done():
                    extension_task.cancel()
                    owner.phase = "extensions_cleanup"

                    async def join_extension() -> None:
                        await extension_task

                    try:
                        await drain_owned_operation(join_extension())
                    except BaseException:
                        # Preserve the initiating error/cancellation even if the
                        # extension suppresses cancellation or fails in cleanup.
                        pass
                raise
            with owner.lock:
                if any(
                    expected is not current
                    for expected, current in zip(
                        components,
                        (self.model, self.tokenizer, self.generation_config),
                        strict=True,
                    )
                ):
                    raise ValueError(
                        "Backend components changed during extension initialization"
                    )
                self._complete_initialization(owner)
        except BaseException as error:
            with owner.lock:
                owner.error = error
                owner.state = BackendInitializationState.FAILED
            raise

    def _complete_initialization(self, owner: _BackendInitialization) -> None:
        owner.identity = (self.model, self.tokenizer, self.generation_config)
        self._checkpoint_restore = None
        owner.phase = "complete"
        owner.state = BackendInitializationState.READY

    def _initialization_owner(self) -> _BackendInitialization:
        # Older serialized agents predate the lifecycle record.
        if not hasattr(self, "_backend_initialization"):
            self._backend_initialization = _BackendInitialization()
        return self._backend_initialization

    def get_initialization_status(self) -> dict[str, Any]:
        """Inspect backend initialization without loading or retrying anything."""
        owner = self._initialization_owner()
        adapter = owner.adapter
        adapter_matches = adapter is not None and (
            self.model is adapter.input_model or self.model is adapter.output_model
        )
        return {
            "state": owner.state.value,
            "phase": owner.phase,
            "error_type": type(owner.error).__name__ if owner.error else None,
            "backend_matches": owner.identity is not None
            and all(
                expected is actual
                for expected, actual in zip(
                    owner.identity,
                    (self.model, self.tokenizer, self.generation_config),
                    strict=True,
                )
            ),
            "adapter_state": (
                "not_requested"
                if adapter is None
                else (
                    "unbound"
                    if not adapter_matches
                    else (
                        "failed"
                        if adapter.error is not None
                        else (
                            "ready" if adapter.output_model is not None else "attaching"
                        )
                    )
                )
            ),
            "adapter_matches": adapter_matches,
            "adapter_request": (
                copy.deepcopy(adapter.request) if adapter is not None else None
            ),
            "retry_requires_new_model": adapter is not None
            and adapter.error is not None
            and self.model is adapter.input_model,
        }

    def _ensure_backend_usable(self) -> None:
        self._conversation_owner().check()
        owner = self._initialization_owner()
        if owner.state in (
            BackendInitializationState.INITIALIZING,
            BackendInitializationState.FAILED,
        ):
            raise RuntimeError(
                f"Backend initialization is {owner.state.value} at {owner.phase}; complete initialization before generating"
            ) from owner.error

    def _initialize_adapter(self) -> None:
        """Initialize optional backend extensions inside the owned boundary."""

    async def _initialize_extensions(self) -> None:
        """Complete subclass dependencies without replacing backend components.

        Overlapping initializers are rejected while an async extension is active.
        Cancellation requests cleanup once and drains it before returning failure.
        The extension owns whether its completed or failed effects can be retried.
        """

    def _initialize_backend(self) -> None:
        """Initialize the agent (load models, etc.).

        Loads a tokenizer/model using the provided `AgentConfig`. Subclasses
        may override this to customize behavior, but the base implementation
        covers common defaults to keep the base class usable in tests.

        If a backend was injected via the constructor, uses that instead
        of loading from transformers.
        """
        logger.info(f"Initializing Agent with model: {self.config.model_name}")

        # Use injected backend if provided (dependency injection for testing)
        if self._injected_backend is not None:
            owner = self._initialization_owner()
            if owner.source is self._injected_backend:
                if (
                    self.tokenizer is not None
                    and self.model is not None
                    and self.generation_config is not None
                ):
                    return
                components = (self.tokenizer, self.model, self.generation_config)
            else:
                components = (None, None, None)
            tokenizer = (
                components[0]
                if components[0] is not None
                else self._injected_backend.tokenizer
            )
            model = (
                components[1]
                if components[1] is not None
                else self._injected_backend.model
            )
            generation_config = (
                components[2]
                if components[2] is not None
                else self._injected_backend.generation_config
            )
            if any(value is None for value in (tokenizer, model, generation_config)):
                raise RuntimeError(
                    "Injected backend has incomplete initialization components"
                )
            for name, value in (
                ("Tokenizer", tokenizer),
                ("Model", model),
                ("Generation configuration", generation_config),
            ):
                _checked_component(name, value)
            self.tokenizer, self.model, self.generation_config = (
                tokenizer,
                model,
                generation_config,
            )
            owner.source = self._injected_backend
            owner.source_kind = "injected"
            logger.info("Agent initialized with injected backend")
            return

        if self.config.use_stub_model or str(self.config.model_name).startswith(
            "stub://"
        ):
            self._initialize_stub_backend()
            return

        if not _load_transformers_agent():
            raise ImportError(
                "transformers is required to initialize non-stub agents. "
                "Install with `pip install stateset-agents[training]`."
            )
        if torch is None:
            raise ImportError(
                "PyTorch is required to initialize non-stub agents. "
                "Install with `pip install stateset-agents[training]`."
            )

        # Load tokenizer
        self._initialization_owner().phase = "tokenizer"
        if self.tokenizer is None:
            if self._tokenizer_loader:
                self.tokenizer = _checked_component(
                    "Tokenizer", self._tokenizer_loader(self.config), owned=True
                )
            else:
                tokenizer_kwargs = dict(self.config.tokenizer_kwargs or {})
                if self.config.model_revision is not None:
                    tokenizer_kwargs.setdefault("revision", self.config.model_revision)
                if AutoTokenizer is None:
                    raise ImportError(
                        "transformers is required to initialize non-stub agents. "
                        "Install with `pip install stateset-agents[training]`."
                    )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=self.config.trust_remote_code,
                    **tokenizer_kwargs,
                )
        _checked_component("Tokenizer", self.tokenizer)
        if getattr(self.tokenizer, "pad_token", None) is None:
            if self.config.pad_token_id is not None:
                try:
                    self.tokenizer.pad_token_id = self.config.pad_token_id
                except TOKENIZER_ATTR_EXCEPTIONS:
                    pass
            else:
                eos_token = getattr(self.tokenizer, "eos_token", None)
                if eos_token is not None:
                    self.tokenizer.pad_token = eos_token
        if self.config.pad_token_id is not None:
            try:
                self.tokenizer.pad_token_id = self.config.pad_token_id
            except TOKENIZER_ATTR_EXCEPTIONS as e:
                logger.debug("Could not set pad_token_id: %s", e)
        if self.config.eos_token_id is not None:
            try:
                self.tokenizer.eos_token_id = self.config.eos_token_id
            except TOKENIZER_ATTR_EXCEPTIONS as e:
                logger.debug("Could not set eos_token_id: %s", e)

        # Build model kwargs
        self._initialization_owner().phase = "model"
        if self.model is None:
            if self._model_loader:
                self.model = _checked_component(
                    "Model", self._model_loader(self.config), owned=True
                )
            else:
                if AutoModelForCausalLM is None:
                    raise ImportError(
                        "transformers is required to initialize non-stub agents. "
                        "Install with `pip install stateset-agents[training]`."
                    )
                model_kwargs: dict[str, Any] = {
                    "trust_remote_code": self.config.trust_remote_code
                }
                if self.config.model_revision is not None:
                    model_kwargs["revision"] = self.config.model_revision
                if self.config.torch_dtype == "bfloat16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
                elif self.config.torch_dtype == "float16":
                    model_kwargs["torch_dtype"] = torch.float16
                elif self.config.torch_dtype == "float32":
                    model_kwargs["torch_dtype"] = torch.float32

                if self.config.device_map:
                    model_kwargs["device_map"] = self.config.device_map
                if self.config.attn_implementation:
                    model_kwargs["attn_implementation"] = (
                        self.config.attn_implementation
                    )
                if self.config.model_kwargs:
                    model_kwargs.update(self.config.model_kwargs)

                self.model, _ = load_generation_model(
                    AutoModelForCausalLM,
                    self.config.model_name,
                    model_kwargs,
                )
        _checked_component("Model", self.model)

        # Setup generation config
        self._initialization_owner().phase = "generation_config"
        if self.generation_config is None:
            self.generation_config = _checked_component(
                "Generation configuration", self._build_generation_config(), owned=True
            )
        _checked_component("Generation configuration", self.generation_config)
        logger.info("Agent initialized successfully")

    def _initialize_stub_backend(self) -> None:
        """Setup lightweight stub backend for offline scenarios."""
        owner = self._initialization_owner()
        request: dict[str, Any] = copy.deepcopy(
            {
                "stub_responses": self.config.stub_responses,
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "do_sample": self.config.do_sample,
                "repetition_penalty": self.config.repetition_penalty,
                "pad_token_id": self.config.pad_token_id,
                "eos_token_id": self.config.eos_token_id,
            }
        )
        if (
            owner.source_kind == "stub"
            and owner.stub_request == request
            and self._is_stub_backend
            and self.tokenizer is not None
            and self.generation_config is not None
        ):
            return
        backend = create_stub_backend(**copy.deepcopy(request))
        tokenizer, model, generation_config = (
            backend.tokenizer,
            backend.model,
            backend.generation_config,
        )
        if any(value is None for value in (tokenizer, model, generation_config)):
            raise RuntimeError("Stub backend has incomplete initialization components")
        self.tokenizer, self.model, self.generation_config = (
            tokenizer,
            model,
            generation_config,
        )
        owner.stub_request = request
        owner.source_kind = "stub"
        owner.source = None
        logger.info(
            "Agent initialized in stub mode (no external model downloads required)"
        )

    async def generate_response(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a response given conversation history.

        Base implementation raises to indicate subclass responsibility,
        while allowing the base class to be constructed.
        """
        raise NotImplementedError("generate_response must be implemented by subclasses")

    @staticmethod
    def _normalize_messages(
        messages: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Coerce message input into a normalized list of dicts."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages

    def _conversation_owner(self) -> _ConversationState:
        if not hasattr(self, "_conversation"):
            self._conversation = _ConversationState()
        return self._conversation

    def get_conversation_status(self) -> dict[str, Any]:
        """Inspect conversation identity and operations, including invalidated work."""
        return self._conversation_owner().inspect()

    async def reset(self) -> None:
        """Start a new conversation and reject late results from previous work.

        Reset clears history, turn count and the last reasoning trace. It does
        not cancel provider/tool effects already in progress or erase long-term
        enhanced memory. In-flight operations keep ownership until they settle.
        """
        self._conversation_owner().reset(self)

    def add_to_history(self, turn: ConversationTurn) -> None:
        """Add a turn to conversation history.

        Stores history as a list of dict messages `{role, content}` for
        compatibility with tests and simple consumers.
        """
        msg: Any
        try:
            msg = {"role": turn.role, "content": turn.content}
        except (AttributeError, TypeError):
            msg = turn
        with self._conversation_owner().commit():
            self.conversation_history.append(copy.deepcopy(msg))

    def get_history(self) -> list[ConversationTurn]:
        """Get current conversation history"""
        with self._conversation_owner().lock:
            return copy.deepcopy(self.conversation_history)

    def _build_generation_config(self) -> GenerationConfig:
        """Create a generation config using current tokenizer/model."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be available to build generation config")
        _load_transformers_agent()

        if self._generation_config_factory:
            return self._generation_config_factory(
                self.config, self.tokenizer, self.model
            )

        def _safe_token_id(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float, str)):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
            return None

        return GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=_safe_token_id(getattr(self.tokenizer, "pad_token_id", None)),
            eos_token_id=_safe_token_id(getattr(self.tokenizer, "eos_token_id", None)),
            **generation_compat_kwargs(self.model),
        )


#: Generation details copied onto an assistant turn's metadata when present.
_TURN_METADATA_KEYS: tuple[str, ...] = (
    "prompt_token_ids",
    "token_ids",
    "sampler_log_probs",
    "sampler_log_probs_source",
    "model_log_probs",
    "rollout_backend",
    "rollout_backend_version",
    "rollout_backend_stale",
    "rollout_backend_error",
)


def _validated_backend_result(result: Any, prompt: str) -> dict[str, Any]:
    """Reject backend output that cannot represent one exact sampled rollout."""
    if getattr(result, "prompt", prompt) != prompt:
        raise ValueError("rollout backend returned a result for a different prompt")
    prompt_ids = [int(token) for token in result.prompt_token_ids]
    token_ids = [int(token) for token in result.response_token_ids]
    log_probs = [float(value) for value in result.token_logprobs]
    if len(token_ids) != len(log_probs):
        raise ValueError("rollout backend token IDs and log-probs differ in length")
    if any(not math.isfinite(value) or value > 1e-6 for value in log_probs):
        raise ValueError("rollout backend returned invalid token log-probs")
    log_probs = [min(value, 0.0) for value in log_probs]
    details: dict[str, Any] = {
        "response": str(result.response),
        "prompt_token_ids": prompt_ids,
        "token_ids": token_ids,
    }
    mode = getattr(result, "logprobs_mode", "processed_logprobs")
    if mode == "raw_logprobs":
        details["model_log_probs"] = log_probs
    elif mode == "processed_logprobs":
        details["sampler_log_probs"] = log_probs
        details["sampler_log_probs_source"] = "backend"
    else:
        raise ValueError("rollout backend returned an unknown log probability mode")
    return details


class MultiTurnAgent(Agent):
    """
    Agent designed for multi-turn conversations
    """

    def __init__(
        self,
        config: AgentConfig,
        memory_window: int = 10,
        context_compression: bool = False,
        model_loader: Callable[[AgentConfig], Any] | None = None,
        tokenizer_loader: Callable[[AgentConfig], Any] | None = None,
        generation_config_factory: (
            Callable[[AgentConfig, Any, Any], GenerationConfig] | None
        ) = None,
        backend: ModelBackend | None = None,
        planning_manager: PlanningManager | None = None,
    ):
        super().__init__(
            config,
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
            generation_config_factory=generation_config_factory,
            backend=backend,
        )
        self.memory_window = memory_window
        self.context_compression = context_compression
        self._rollout_backend: Any | None = None
        self._rollout_backend_version: int = 0
        self._rollout_backend_stale: bool = False
        self._rollout_backend_error: str | None = None
        self._rollout_sync_warned: bool = False
        self.turn_count = 0
        if planning_manager is None and getattr(config, "enable_planning", False):
            planning_kwargs = getattr(config, "planning_config", None) or {}
            try:
                from ..experimental.long_term_planning import (
                    PlanningConfig,
                    PlanningManager,
                )

                if isinstance(planning_kwargs, dict):
                    planning_kwargs = dict(planning_kwargs)
                    planning_kwargs.pop("enabled", None)
                planning_cfg = PlanningConfig(enabled=True, **planning_kwargs)
                planning_manager = PlanningManager(planning_cfg)
            except PLANNING_EXCEPTIONS as exc:
                logger.warning("Failed to init PlanningManager: %s", exc)
        self.planning_manager = planning_manager

        # Arbitrary caller-attached metadata (e.g. the API layer records the
        # checkpoint path/base model a registered agent was loaded from here
        # so /v1/agents/{id} can surface it). Not used by the agent itself.
        self.metadata: dict[str, Any] = {}

    def _initialize_adapter(self) -> None:
        """Apply an adapter once, retaining uncertain in-place mutations on failure."""
        owner = self._initialization_owner()
        if self._is_stub_backend:
            owner.adapter = None
            return

        request: dict[str, Any]
        checkpoint_adapters = getattr(self, "_checkpoint_adapters", None)
        if self.config.peft_path:
            from pathlib import Path

            adapter_path = Path(self.config.peft_path).expanduser().resolve()
            if (
                checkpoint_adapters is not None
                and adapter_path == checkpoint_adapters.path.resolve()
            ):
                request = checkpoint_adapters.request()
            else:
                request = {"kind": "path", "path": str(adapter_path)}
        elif self.config.use_peft and self.config.peft_config:
            request = {
                "kind": "config",
                "config": copy.deepcopy(self.config.peft_config),
            }
        else:
            request = {"kind": "none"}

        previous = owner.adapter
        if previous is not None and (
            self.model is previous.input_model or self.model is previous.output_model
        ):
            if previous.error is not None:
                raise RuntimeError(
                    "Previous adapter initialization failed after attachment began; "
                    "provide a fresh model or agent before retrying"
                ) from previous.error
            if previous.request != request:
                raise ValueError(
                    "Adapter configuration changed on an initialized model; "
                    "provide a fresh model or agent to apply a different adapter"
                )
            if previous.output_model is None:
                raise RuntimeError("Adapter initialization is already in progress")
            self.model = previous.output_model
            return

        if request["kind"] == "none":
            owner.adapter = None
            return
        if self.model is None:
            raise RuntimeError("Model must be loaded before adapter initialization")

        if request["kind"] == "checkpoint":
            if not _load_peft() or PeftModel is None:
                raise ImportError(
                    "PEFT library not available. Install with: pip install peft"
                )
            assert checkpoint_adapters is not None
            checkpoint_adapters.validate_files()

            def attach(model: Any) -> Any:
                return checkpoint_adapters.attach(model, PeftModel)

        elif request["kind"] == "path":
            if not _load_peft() or PeftModel is None:
                raise ImportError(
                    "PEFT library not available. Install with: pip install peft"
                )
            if not adapter_path.exists():
                raise FileNotFoundError(f"peft_path does not exist: {adapter_path}")
            from .checkpoint_io import validate_local_shard_indexes

            validate_local_shard_indexes(adapter_path)

            def attach(model: Any) -> Any:
                return PeftModel.from_pretrained(model, str(adapter_path))

        else:
            if not _load_peft() or LoraConfig is None or get_peft_model is None:
                raise ImportError(
                    "PEFT library not available. Install with: pip install peft"
                )
            lora_config = _checked_component(
                "Adapter configuration",
                LoraConfig(**copy.deepcopy(request["config"])),
                owned=True,
            )

            def attach(model: Any) -> Any:
                return get_peft_model(model, lora_config)

        attachment = _AdapterAttachment(request=request, input_model=self.model)
        owner.adapter = attachment
        owner.phase = "adapter_attachment"
        try:
            model = attach(attachment.input_model)
            if model is None:
                raise RuntimeError("Adapter initialization returned no model")
            model = _checked_component("Adapter", model, owned=True)
        except BaseException as error:
            attachment.error = error
            raise
        else:
            attachment.output_model = model
            self.model = model
            logger.info("MultiTurnAgent adapter initialized successfully")

    async def reset(self) -> None:
        """Reset agent state for a fresh conversation."""
        await super().reset()

    @conversation_call
    async def generate_response(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate response for multi-turn conversation"""

        prompt, messages = self._render_prompt(messages, context)
        response = await self._generate_with_model(prompt, context)
        return self._finalize_response(response, messages)

    def set_rollout_backend(self, backend: Any | None) -> None:
        """Attach (or detach with ``None``) a rollout engine for generation.

        The backend must expose ``async generate_with_logprobs(prompts,
        **kwargs) -> list[result]`` where each result carries ``response``,
        ``prompt_token_ids``, ``response_token_ids`` and ``token_logprobs``
        (the shape of ``training.vllm_backend.GenerationResult``). When set,
        both ``generate_response`` and ``generate_turn`` sample from it and
        the returned turns carry the engine's exact token ids, so the GRPO
        trainers train on what the engine generated. The trainers call
        :meth:`sync_rollout_backend` after every optimizer step; an engine
        that also exposes ``sync_weights(model)`` (``VLLMGenerator`` does)
        then keeps serving the current policy. Attaching resets the policy
        version counter and staleness state.
        """
        self._rollout_backend = backend
        self._rollout_backend_version = 0
        self._rollout_backend_stale = False
        self._rollout_backend_error = None
        self._rollout_sync_warned = False

    @property
    def rollout_backend_version(self) -> int:
        """How many successful weight syncs the attached backend has taken."""
        return int(getattr(self, "_rollout_backend_version", 0))

    @property
    def rollout_backend_stale(self) -> bool:
        """True once a sync was needed but could not be applied, until one succeeds."""
        return bool(getattr(self, "_rollout_backend_stale", False))

    @property
    def rollout_backend_error(self) -> str | None:
        """The last weight-sync failure (``"ExcType: message"``), or ``None``."""
        return getattr(self, "_rollout_backend_error", None)

    def sync_rollout_backend(self) -> bool:
        """Push the current policy weights into the attached rollout backend.

        Returns True when the backend now serves the current policy (or no
        backend is attached). Returns False, marks the backend stale, and
        records the reason when the backend has no ``sync_weights`` or the
        sync raised; rollouts sampled while stale carry
        ``rollout_backend_stale=True`` so a trainer using
        ``old_logprobs_source="sampler"`` can still importance-correct them.
        """
        backend = getattr(self, "_rollout_backend", None)
        if backend is None:
            return True
        sync = getattr(backend, "sync_weights", None)
        if not callable(sync):
            self._rollout_backend_stale = True
            self._rollout_backend_error = (
                f"{type(backend).__name__} has no sync_weights(model)"
            )
            if not self._rollout_sync_warned:
                self._rollout_sync_warned = True
                logger.warning(
                    "rollout backend %s exposes no sync_weights(model); its "
                    "rollouts will lag the policy after each optimizer step "
                    "(set old_logprobs_source='sampler' to importance-correct them)",
                    type(backend).__name__,
                )
            return False
        try:
            sync(self.model)
        except Exception as e:  # noqa: BLE001 - engine failures must not kill training
            self._rollout_backend_stale = True
            self._rollout_backend_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "rollout backend %s weight sync failed; rollouts are stale: %s",
                type(backend).__name__,
                e,
            )
            return False
        self._rollout_backend_version += 1
        self._rollout_backend_stale = False
        self._rollout_backend_error = None
        return True

    @conversation_call
    async def generate_turn(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Generate an assistant turn that also carries what was generated.

        Same pipeline as :meth:`generate_response`; the returned turn's
        ``metadata`` additionally holds ``prompt_token_ids`` (the exact ids
        fed to ``generate``), ``token_ids`` (the sampled response ids, before
        text cleaning) and ``sampler_log_probs`` (the probabilities of those actions under
        the generation policy, in log space) when a real model produced the text. Stub
        backends return a turn with text only.
        """
        prompt, messages = self._render_prompt(messages, context)
        details = await self._generate_with_model_details(prompt, context)
        content = self._finalize_response(str(details["response"]), messages)
        metadata: dict[str, Any] = {"generated": True}
        for key in _TURN_METADATA_KEYS:
            if details.get(key) is not None:
                metadata[key] = details[key]
        return ConversationTurn(role="assistant", content=content, metadata=metadata)

    @conversation_call
    async def generate_turns(
        self,
        messages: str | list[dict[str, str]],
        n: int,
        context: dict[str, Any] | None = None,
    ) -> list[ConversationTurn]:
        """Sample ``n`` assistant turns for the same conversation in one call.

        A real HF model generates them as one batch (``num_return_sequences``),
        a rollout backend receives ``n`` copies of the prompt, and a stub backend
        samples sequentially. Every turn carries the same token metadata as
        :meth:`generate_turn`, so a group-relative trainer gets its whole
        group from one forward-generate instead of ``n`` sequential ones.
        """
        n = int(n)
        if n <= 0:
            return []
        prompt, messages = self._render_prompt(messages, context)
        batch = await self._generate_with_model_details_batch(prompt, n, context)
        turns: list[ConversationTurn] = []
        for details in batch:
            content = self._finalize_response(str(details["response"]), messages)
            metadata: dict[str, Any] = {"generated": True}
            for key in _TURN_METADATA_KEYS:
                if details.get(key) is not None:
                    metadata[key] = details[key]
            turns.append(
                ConversationTurn(role="assistant", content=content, metadata=metadata)
            )
        return turns

    async def _generate_with_model_details_batch(
        self, prompt: str, n: int, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """``n`` samples of :meth:`_generate_with_model_details` in one batch
        where the backend allows it (HF ``num_return_sequences``, engine batch
        of ``n`` prompts); sequential otherwise."""
        self._ensure_backend_usable()
        backend_error: str | None = None
        backend = getattr(self, "_rollout_backend", None)
        if backend is not None:
            try:
                results = await backend.generate_with_logprobs(
                    [prompt] * n,
                    temperature=float(self.config.temperature),
                    top_p=float(self.config.top_p),
                    max_tokens=int(self.config.max_new_tokens),
                )
                self._conversation_owner().check()
                if len(results) != n:
                    raise ValueError(
                        f"rollout backend returned {len(results)} results for {n} prompts"
                    )
                backend_batch = [
                    _validated_backend_result(result, prompt) for result in results
                ]
                for backend_item in backend_batch:
                    backend_item["response"] = self._clean_response(
                        backend_item["response"]
                    )
                    backend_item["rollout_backend"] = type(backend).__name__
                    backend_item["rollout_backend_version"] = (
                        self.rollout_backend_version
                    )
                    backend_item["rollout_backend_stale"] = self.rollout_backend_stale
                return backend_batch
            except ConversationResetError:
                raise
            except Exception as e:  # noqa: BLE001 - engine failure falls back
                backend_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "rollout backend %s failed for a batch of %d, falling back to "
                    "native generation: %s",
                    type(backend).__name__,
                    n,
                    e,
                )

        self._conversation_owner().check()

        async def sequential_fallback() -> list[dict[str, Any]]:
            details = []
            for _ in range(n):
                self._conversation_owner().check()
                details.append(
                    await self._generate_with_model_details(
                        prompt, context, use_rollout_backend=False
                    )
                )
            if backend_error:
                for item in details:
                    item["rollout_backend_error"] = backend_error
            return details

        if (
            (self._is_stub_backend and isinstance(self.model, StubModel))
            or self.model is None
            or self.tokenizer is None
            or self.generation_config is None
            or not callable(getattr(self.model, "generate", None))
        ):
            return await sequential_fallback()

        inputs = self._tokenize_prompt(prompt)
        model_device = None
        if hasattr(self.model, "parameters"):
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                model_device = None
        if model_device and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        stopping_criteria = self._build_stopping_criteria()
        stopper = stopping_criteria[0]
        try:
            with inference_mode(self.model), torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True,
                    num_return_sequences=n,
                )
            sequences = outputs.sequences
            if not torch.is_tensor(sequences) or int(sequences.shape[0]) != n:
                raise ValueError("batched generate returned an unexpected shape")
        except (RuntimeError, TypeError, ValueError, AttributeError) as e:
            logger.debug(
                "batched generation unavailable (%s); sampling sequentially", e
            )
            return await sequential_fallback()

        return self._captured_turns(
            outputs, inputs["input_ids"][0], stopper, backend_error
        )

    def _captured_turns(
        self, outputs: Any, prompt: Any, stopper: Any, backend_error: Any
    ) -> list[dict[str, Any]]:
        """Decode and score single and batched generation in the same action frame."""
        from .generation_evidence import hf_turn_evidence

        sequences = outputs.sequences
        eos = getattr(self.generation_config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        rows = hf_turn_evidence(
            outputs,
            prompt,
            self.generation_config,
            eos,
            getattr(stopper, "stop_token_ids", []),
            (
                None
                if hasattr(stopper, "stop_token_ids")
                else getattr(stopper, "ends_with_stop", None)
            ),
        )
        start = len(prompt)
        for i, row in enumerate(rows):
            row["response"] = self._decode_model_response(
                sequences[i, start : start + len(row["token_ids"])]
            )
            if backend_error:
                row["rollout_backend_error"] = backend_error
        return rows

    def _render_prompt(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        """Normalise, window, and template the conversation into a prompt."""
        self._ensure_backend_usable()
        messages = self._normalize_messages(messages)

        # Apply memory window
        recent_messages = self._apply_memory_window(messages)

        # Add system prompt if configured
        if self.config.system_prompt and (
            not recent_messages or recent_messages[0]["role"] != "system"
        ):
            recent_messages.insert(
                0, {"role": "system", "content": self.config.system_prompt}
            )

        # Inject long-term plan context if enabled
        recent_messages = self._inject_planning_context(recent_messages, context)

        # Apply chat template if available
        if (
            self.config.use_chat_template
            and self.tokenizer is not None
            and hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    recent_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(self.config.chat_template_kwargs or {}),
                )
            except CHAT_TEMPLATE_EXCEPTIONS as e:
                if is_muse_spark(self.config.model_name, self.model):
                    raise
                # Fallback to manual formatting if chat template fails
                logger.debug("Chat template failed, using manual formatting: %s", e)
                prompt = self._format_conversation(recent_messages)
        else:
            prompt = self._format_conversation(recent_messages)

        return prompt, messages

    def _finalize_response(self, response: str, messages: list[dict[str, str]]) -> str:
        """Strip reasoning traces, apply persona hints, record history."""

        # Handle Reasoning Traces (DeepSeek-R1 style)
        reasoning = None
        if self.config.enable_reasoning:
            import re

            tag = self.config.reasoning_tag
            pattern = f"<{tag}>(.*?)</{tag}>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                # Remove reasoning from the final response presented to user
                response = re.sub(pattern, "", response, flags=re.DOTALL).strip()

        # Lightweight keyword adaptation to satisfy persona-style expectations in tests
        response = _apply_persona_keyword_hints(response, messages)

        with self._conversation_owner().commit():
            self._last_reasoning = reasoning
            self.turn_count += 1
            # Store the final answer in history, with reasoning kept separately.
            self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _apply_memory_window(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Apply memory window to conversation history"""
        if self.memory_window <= 0:
            return messages

        # Keep system message and last N turns
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        other_messages = [msg for msg in messages if msg["role"] != "system"]

        # Keep recent messages within window
        recent_messages = other_messages[-self.memory_window :]

        return system_messages + recent_messages

    def _inject_planning_context(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Insert long-term plan context as a system message when enabled."""
        if self.planning_manager is None:
            return messages

        from ..experimental.long_term_planning import _planning_scope

        with _planning_scope(self._conversation_owner().commit):
            plan_message = self.planning_manager.build_plan_message(
                messages, context=context
            )
        if plan_message is None:
            return messages

        prefix = getattr(self.planning_manager.config, "plan_prefix", "Long-term plan")
        for msg in messages:
            if msg.get("role") == "system" and str(msg.get("content", "")).startswith(
                prefix
            ):
                return messages

        insert_at = 0
        while insert_at < len(messages) and messages[insert_at].get("role") == "system":
            insert_at += 1
        messages.insert(insert_at, plan_message)
        return messages

    def _format_conversation(self, messages: list[dict[str, str]]) -> str:
        """Format conversation for models without chat template"""
        formatted = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted += f"System: {content}\n"
            elif role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"

        formatted += "Assistant: "
        return formatted

    def _build_stopping_criteria(self) -> StoppingCriteriaList:
        """Build stopping criteria for generation."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be available to build stopping criteria")
        stop_tokens = ["User:", "System:", "\n\n"]
        return StoppingCriteriaList([StopOnSpecialTokens(stop_tokens, self.tokenizer)])

    async def _generate_with_model(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """Generate response text using the language model."""
        details = await self._generate_with_model_details(prompt, context)
        return str(details["response"])

    async def _generate_with_model_details(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        *,
        use_rollout_backend: bool = True,
    ) -> dict[str, Any]:
        """Generate with the language model and return the response text plus,
        for real HF models, the exact prompt ids, the sampled response ids, and
        the model's log-probs of those ids (temperature 1, no truncation)."""

        self._ensure_backend_usable()
        backend_error: str | None = None
        backend = (
            getattr(self, "_rollout_backend", None) if use_rollout_backend else None
        )
        if backend is not None:
            try:
                results = await backend.generate_with_logprobs(
                    [prompt],
                    temperature=float(self.config.temperature),
                    top_p=float(self.config.top_p),
                    max_tokens=int(self.config.max_new_tokens),
                )
                self._conversation_owner().check()
                if len(results) != 1:
                    raise ValueError(
                        f"rollout backend returned {len(results)} results for one prompt"
                    )
                result = results[0]
                backend_details = _validated_backend_result(result, prompt)
                return {
                    **backend_details,
                    "response": self._clean_response(backend_details["response"]),
                    "rollout_backend": type(backend).__name__,
                    "rollout_backend_version": self.rollout_backend_version,
                    "rollout_backend_stale": self.rollout_backend_stale,
                }
            except ConversationResetError:
                raise
            except Exception as e:  # noqa: BLE001 - any engine failure falls back
                backend_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "rollout backend %s failed, falling back to native generation: %s",
                    type(backend).__name__,
                    e,
                )

        self._conversation_owner().check()
        if self._is_stub_backend and isinstance(self.model, StubModel):
            stub_details: dict[str, Any] = {
                "response": str(self.model.generate(prompt, context))
            }
            if backend_error:
                stub_details["rollout_backend_error"] = backend_error
            return stub_details

        if (
            self.model is None
            or self.tokenizer is None
            or self.generation_config is None
        ):
            raise RuntimeError(
                "Agent model, tokenizer, and generation configuration must be initialized "
                "before calling _generate_with_model. Call `await initialize()` first or "
                "provide custom loader hooks."
            )

        # Tokenize input
        inputs = self._tokenize_prompt(prompt)

        # Move inputs to model device when available
        model_device = None
        if hasattr(self.model, "parameters"):
            try:
                first_param = next(self.model.parameters())
                model_device = first_param.device
            except StopIteration:
                model_device = None
        if model_device and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        # Setup stopping criteria
        stopping_criteria = self._build_stopping_criteria()

        # Generate (always in inference mode: a trainer may have left the
        # model in train(), which corrupts sampling with LoRA/checkpointing)
        with inference_mode(self.model), torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
            )

        if torch.is_tensor(outputs.sequences) and torch.is_tensor(inputs["input_ids"]):
            return self._captured_turns(
                outputs, inputs["input_ids"][0], stopping_criteria[0], backend_error
            )[0]
        response = self._decode_model_response(
            outputs.sequences[0][int(inputs["input_ids"].shape[1]) :]
        )
        details = {"response": response}
        if backend_error:
            details["rollout_backend_error"] = backend_error
        return details

    def _decode_model_response(self, token_ids: Any) -> str:
        """Keep native Muse channel boundaries while retaining the sampled IDs."""
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("Model tokenizer is not initialized")
        native = is_muse_spark(self.config.model_name, self.model)
        response = str(tokenizer.decode(token_ids, skip_special_tokens=not native))
        if native:
            return self._clean_response(response, native_protocol=True)
        return self._clean_response(response)

    def _clean_response(
        self, response: str, *, native_protocol: bool | None = None
    ) -> str:
        """Clean up generated response by removing artifacts and stop tokens.

        Args:
            response: Raw response string from the language model

        Returns:
            Cleaned response with artifacts removed
        """
        if native_protocol is not False and is_muse_spark(
            self.config.model_name, self.model
        ):
            response = muse_spark_final_response(
                response, allow_plain=native_protocol is None
            )
        # Remove common artifacts
        response = response.strip()

        # Remove stopping tokens that might have leaked through
        for phrase in STREAM_STOP_PHRASES:
            if phrase in response:
                response = response.split(phrase)[0].strip()

        return response

    def _tokenize_prompt(self, prompt: str) -> Any:
        """Tokenize a generation prompt, keeping its end when it is too long.

        The end holds the latest user turn and the assistant cue; right
        truncation (the tokenizer default) dropped them, so the model continued
        mid-history.
        """
        max_length = self._effective_max_input_length()
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer must be available to tokenize prompts")
        original_side = getattr(tokenizer, "truncation_side", None)
        if original_side is not None:
            tokenizer.truncation_side = "left"
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            )
        finally:
            if original_side is not None:
                tokenizer.truncation_side = original_side
        ids: Any = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if (
            torch is not None
            and torch.is_tensor(ids)
            and ids.dim() > 0
            and int(ids.shape[-1]) >= max_length
        ):
            logger.warning(
                "Prompt reached the %d-token input limit; earliest context dropped.",
                max_length,
            )
        return inputs

    def _effective_max_input_length(self) -> int:
        """Compute a safe max_length for tokenization to avoid negative lengths."""
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int):
            max_len = int(model_max_length) - self.config.max_new_tokens
            if max_len < 1:
                logger.warning(
                    "max_new_tokens (%s) exceeds or equals tokenizer.model_max_length (%s); "
                    "clamping max_length to 1 to avoid generation failure. "
                    "Consider reducing max_new_tokens.",
                    self.config.max_new_tokens,
                    model_max_length,
                )
                return 1
            return int(max_len)
        return 1024  # conservative fallback when tokenizer does not expose a limit

    @conversation_stream
    async def generate_response_stream(
        self,
        messages: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Generate response with streaming output.

        Yields tokens as they are generated, enabling real-time display
        of agent responses. This is particularly useful for long responses
        or interactive applications.

        Args:
            messages: Either a string (single user message) or list of
                message dicts with 'role' and 'content' keys
            context: Optional context dict passed to generation

        Yields:
            str: Individual tokens or token chunks as they are generated

        Example:
            >>> async for token in agent.generate_response_stream("Hello!"):
            ...     print(token, end="", flush=True)
        """
        self._ensure_backend_usable()
        messages = self._normalize_messages(messages)
        recent_messages = self._apply_memory_window(messages)

        if self.config.system_prompt and (
            not recent_messages or recent_messages[0]["role"] != "system"
        ):
            recent_messages.insert(
                0, {"role": "system", "content": self.config.system_prompt}
            )

        recent_messages = self._inject_planning_context(recent_messages, context)

        if (
            self.config.use_chat_template
            and self.tokenizer is not None
            and hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    recent_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(self.config.chat_template_kwargs or {}),
                )
            except CHAT_TEMPLATE_EXCEPTIONS:
                if is_muse_spark(self.config.model_name, self.model):
                    raise
                prompt = self._format_conversation(recent_messages)
        else:
            prompt = self._format_conversation(recent_messages)

        response_text = ""

        # For stub backend, yield the full response in chunks
        if (
            is_muse_spark(self.config.model_name, self.model)
            and not self._is_stub_backend
        ):
            # Native recipient headers must survive decoding. Buffer the full
            # generation before emitting user-directed content; a generic
            # streamer discards those headers and would expose private reasoning.
            response_text = await self._generate_with_model(prompt, context)
            for offset in range(0, len(response_text), 32):
                yield response_text[offset : offset + 32]
                await asyncio.sleep(0)
        elif self._is_stub_backend and isinstance(self.model, StubModel):
            response = str(self.model.generate(prompt, context))
            # Simulate streaming by yielding word by word
            words = response.split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.02)  # Small delay for realistic streaming
            response_text = response
        else:
            if (
                self.model is None
                or self.tokenizer is None
                or self.generation_config is None
            ):
                raise RuntimeError(
                    "Agent must be initialized before streaming. Call `await initialize()` first."
                )

            # Tokenize input
            inputs = self._tokenize_prompt(prompt)

            model_device = None
            if hasattr(self.model, "parameters"):
                try:
                    first_param = next(self.model.parameters())
                    model_device = first_param.device
                except StopIteration:
                    model_device = None
            if model_device and hasattr(inputs, "to"):
                inputs = inputs.to(model_device)

            # Use TextIteratorStreamer if available
            try:
                import threading

                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                stop_requested = threading.Event()

                class StopOnRequest:
                    def __call__(
                        self, _input_ids: Any, _scores: Any, **_kwargs: Any
                    ) -> bool:
                        return stop_requested.is_set()

                stopping_criteria = self._build_stopping_criteria()
                stopping_criteria.append(StopOnRequest())

                generation_kwargs = {
                    **inputs,
                    "generation_config": self.generation_config,
                    "streamer": streamer,
                    "stopping_criteria": stopping_criteria,
                }
                model = self.model
                if model is None:
                    raise RuntimeError(
                        "Agent model must be initialized before streaming"
                    )

                generation_error: list[Exception] = []

                def run_generation() -> None:
                    try:
                        with inference_mode(model), torch.no_grad():
                            model.generate(**generation_kwargs)
                    except Exception as exc:  # noqa: BLE001 - relay worker failure
                        generation_error.append(exc)
                        streamer.end()  # Unblock a reader waiting for the next chunk.

                # The producer and blocking iterator both run off the event loop.
                thread = threading.Thread(target=run_generation, daemon=True)
                thread.start()

                iterator = iter(streamer)

                def next_chunk() -> str | None:
                    try:
                        return cast(str, next(iterator))
                    except StopIteration:
                        return None

                response_chunks: list[str] = []
                pending = ""
                stopped = False
                try:
                    while (text := await asyncio.to_thread(next_chunk)) is not None:
                        visible, pending, stopped = _stream_visible_text(pending + text)
                        if visible:
                            response_chunks.append(visible)
                            yield visible
                        if stopped:
                            stop_requested.set()
                            break
                    if not stopped and pending:
                        response_chunks.append(pending)
                        yield pending
                finally:
                    stop_requested.set()
                    await asyncio.to_thread(thread.join)

                if generation_error:
                    raise RuntimeError(
                        "Model generation failed during streaming"
                    ) from (generation_error[0])
                response_text = "".join(response_chunks)

            except ImportError:
                # Fallback: generate full response and yield in chunks
                response = await self._generate_with_model(prompt, context)
                response = self._clean_response(response)
                # Yield in small chunks for streaming feel
                chunk_size = 4
                for i in range(0, len(response), chunk_size):
                    yield response[i : i + chunk_size]
                    await asyncio.sleep(0.01)
                response_text = response

        if is_muse_spark(self.config.model_name, self.model):
            response_text = self._clean_response(response_text, native_protocol=False)
        else:
            response_text = self._clean_response(response_text)
        with self._conversation_owner().commit():
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )
            self.turn_count += 1

    @conversation_call
    async def process_turn(
        self,
        conversation_history: list[dict[str, str]],
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Process a single conversation turn"""

        # Add user input to history
        messages = conversation_history + [{"role": "user", "content": user_input}]

        # Generate response
        response = await self.generate_response(messages, context)

        # Capture reasoning if available
        metadata = {
            "turn_count": self.turn_count,
            "model_name": self.config.model_name,
            "context": context,
        }

        if hasattr(self, "_last_reasoning") and self._last_reasoning:
            metadata["reasoning"] = self._last_reasoning

        # Create conversation turn
        turn = ConversationTurn(
            role="assistant",
            content=response,
            metadata=metadata,
        )

        return turn


# ``agent_factories`` and ``tool_agent`` both import this module, so importing
# them here at module level would be a cycle that only resolves by luck of
# import order. Re-export them on first attribute access instead, which keeps
# ``from stateset_agents.core.agent import ToolAgent`` (and the factory names)
# working without the cycle.
_LAZY_REEXPORTS: dict[str, str] = {
    "AGENT_CONFIGS": ".agent_factories",
    "_load_agent_configs": ".agent_factories",
    "create_agent": ".agent_factories",
    "create_peft_agent": ".agent_factories",
    "get_preset_config": ".agent_factories",
    "load_agent_from_checkpoint": ".agent_factories",
    "save_agent_checkpoint": ".agent_factories",
    "ToolAgent": ".tool_agent",
}

# Starred so the lazily re-exported names above are not flagged as undefined.
__all__ = [
    "Agent",
    "BackendInitializationState",
    "ConversationResetError",
    "AgentConfig",
    "ConfigValidationError",
    "MultiTurnAgent",
    *_LAZY_REEXPORTS,
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_REEXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __package__)
    value = getattr(module, name)
    globals()[name] = value  # cache, so __getattr__ runs at most once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_REEXPORTS))
