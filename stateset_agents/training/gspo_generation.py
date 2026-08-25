"""
Trajectory generation helpers for GSPO training.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from stateset_agents.core.agent import Agent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.exceptions import INFERENCE_EXCEPTIONS as VLLM_EXCEPTIONS
from stateset_agents.exceptions import MODEL_DEVICE_EXCEPTIONS

from . import rl_losses
from .gspo_config import GSPOConfig

logger = logging.getLogger(__name__)

VLLMConfig = None
VLLMGenerator = None
HuggingFaceGeneratorFallback = None
GenerationResult = None
create_generator = None
VLLM_BACKEND_AVAILABLE = False
VLLM_AVAILABLE = False
_vllm_backend_loaded = False


def _load_vllm_backend() -> bool:
    """Lazily load the vLLM backend to avoid import-time side effects."""
    global _vllm_backend_loaded, VLLMConfig, VLLMGenerator, HuggingFaceGeneratorFallback
    global GenerationResult, create_generator, VLLM_BACKEND_AVAILABLE, VLLM_AVAILABLE
    if _vllm_backend_loaded:
        return VLLM_BACKEND_AVAILABLE
    try:
        from .vllm_backend import GenerationResult as _GenerationResult
        from .vllm_backend import HuggingFaceGeneratorFallback as _HFGen
        from .vllm_backend import VLLMConfig as _VLLMConfig
        from .vllm_backend import VLLMGenerator as _VLLMGenerator
        from .vllm_backend import create_generator as _create_generator

        VLLMConfig = _VLLMConfig
        VLLMGenerator = _VLLMGenerator
        HuggingFaceGeneratorFallback = _HFGen
        GenerationResult = _GenerationResult
        create_generator = _create_generator
        VLLM_BACKEND_AVAILABLE = True
        VLLM_AVAILABLE = True
        _vllm_backend_loaded = True
        return True
    except (ImportError, RuntimeError):
        _vllm_backend_loaded = True
        VLLM_BACKEND_AVAILABLE = False
        VLLM_AVAILABLE = False
        return False


def render_prompt_for_scoring(
    tokenizer: Any, prompt: str, system_prompt: str | None = None
) -> str:
    """Render a user prompt into the exact text a chat-template model conditions on.

    Both generation (`_generate_with_hf` / vLLM rescoring) and scoring
    (current-log-prob and reference-log-prob computation in the trainer) MUST
    call this helper so the importance-ratio numerator and denominator share
    one tokenization convention. Falls back to the raw prompt when the
    tokenizer has no chat template.

    When ``system_prompt`` is truthy it is included as a leading system
    message, matching ``MultiTurnAgent``'s own message construction
    (``core/agent.py`` inserts the system prompt at index 0 before calling
    the model). This is a best-effort parity measure only: it does not
    capture memory-window truncation or any other agent-side post-processing
    of the conversation history, so the rendered text scored here can still
    diverge from what the agent actually conditioned on when generating the
    response. See `_generate_with_hf`'s docstring for the residual
    limitation.
    """
    chat_template = getattr(tokenizer, "chat_template", None) if tokenizer else None
    if tokenizer is None or not chat_template:
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def build_scoring_text(prompt_text: str, response: str) -> str:
    """Concatenate rendered prompt text and response with no injected separator.

    Generation continues directly from the rendered prompt, so scoring must
    concatenate the exact same way — inserting a space would score a token
    that was never actually sampled, biasing the importance ratio.
    """
    return prompt_text + response


def _get_model_device(model: Any) -> torch.device | None:
    """Best-effort helper to locate a model's device without assuming attributes."""
    if model is None:
        return None
    try:
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        return getattr(model, "device", None)
    except MODEL_DEVICE_EXCEPTIONS:
        return getattr(model, "device", None)


class GSPOTrajectoryGenerator:
    """
    Handles efficient trajectory generation for GSPO training.

    Supports two generation backends:
    1. vLLM (preferred): 5-20x faster with automatic log prob extraction
    2. HuggingFace (fallback): Standard generation when vLLM unavailable
    """

    def __init__(
        self, config: GSPOConfig, agent: Agent, environment: ConversationEnvironment
    ):
        self.config = config
        self.agent = agent
        self.environment = environment
        self.vllm_generator: Any | None = None
        self.sampling_params: Any | None = None
        self._vllm_initialized = False
        self._temperature_bias_warned = False
        self._post_processing_warned = False

        if self.config.use_vllm and _load_vllm_backend():
            self._setup_vllm_generator()

    def _setup_vllm_generator(self) -> None:
        """Setup vLLM generator with config parameters."""
        logger.info("Setting up vLLM generator for fast generation...")

        if not _load_vllm_backend():
            logger.warning("vLLM backend failed to load, will use HuggingFace fallback")
            return
        vllm_config_cls = VLLMConfig
        vllm_generator_cls = VLLMGenerator
        if vllm_config_cls is None or vllm_generator_cls is None:
            logger.warning(
                "vLLM backend symbols unavailable, using HuggingFace fallback"
            )
            return

        vllm_config = vllm_config_cls(
            model_name=self.config.model_name,
            gpu_memory_utilization=getattr(
                self.config, "vllm_gpu_memory_utilization", 0.85
            ),
            tensor_parallel_size=getattr(self.config, "vllm_tensor_parallel_size", 1),
            enable_prefix_caching=getattr(
                self.config, "vllm_enable_prefix_caching", True
            ),
            max_model_len=getattr(self.config, "vllm_max_model_len", None),
            quantization=getattr(self.config, "vllm_quantization", None),
            enable_chunked_prefill=getattr(
                self.config, "vllm_enable_chunked_prefill", True
            ),
            max_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            dtype=(
                "float16"
                if self.config.fp16
                else ("bfloat16" if self.config.bf16 else "auto")
            ),
        )

        self.vllm_generator = vllm_generator_cls(vllm_config)

    async def initialize_vllm(self) -> bool:
        """Initialize the vLLM engine before generation."""
        if self.vllm_generator is None:
            return False

        if self._vllm_initialized:
            return True

        try:
            generator = self.vllm_generator
            if generator is None:
                return False
            success = await generator.initialize()
            self._vllm_initialized = success
            if success:
                logger.info(
                    "vLLM generator initialized - 5-20x faster generation enabled!"
                )
            return bool(success)
        except VLLM_EXCEPTIONS as e:
            logger.warning(
                f"Failed to initialize vLLM: {e}. Using HuggingFace fallback."
            )
            self._vllm_initialized = False
            return False

    @property
    def using_vllm(self) -> bool:
        """Check if vLLM is being used for generation."""
        return bool(self._vllm_initialized and self.vllm_generator is not None)

    async def generate_group_responses(
        self, prompt: str, num_responses: int
    ) -> list[tuple[str, float]]:
        """Generate a group of responses for a single prompt."""
        if self.using_vllm:
            return await self._generate_with_vllm(prompt, num_responses)
        return await self._generate_with_hf(prompt, num_responses)

    def _warn_vllm_temperature_bias_once(self) -> None:
        """Emit a one-time warning that vLLM's cumulative_logprob is biased as an
        old-policy log prob when sampling temperature != 1.0 and rescoring is
        disabled.
        """
        if self._temperature_bias_warned:
            return
        self._temperature_bias_warned = True
        logger.warning(
            "GSPO: rescore_old_log_probs is disabled and sampling temperature "
            "(%.3f) != 1.0. vLLM's cumulative_logprob is a temperature-scaled "
            "log prob and is biased as an old-policy log prob for the "
            "importance ratio; residual bias will not be corrected. Set "
            "config.rescore_old_log_probs=True to rescore rollouts with an "
            "HF forward pass at the true policy temperature.",
            self.config.temperature,
        )

    async def _generate_with_vllm(
        self, prompt: str, num_responses: int
    ) -> list[tuple[str, float]]:
        """Generate responses using vLLM."""
        try:
            generator = self.vllm_generator
            if generator is None:
                return await self._generate_with_hf(prompt, num_responses)
            grouped_results = await generator.generate_groups(
                prompts=[prompt],
                num_generations_per_prompt=num_responses,
            )

            results = grouped_results[prompt]

            if getattr(self.config, "rescore_old_log_probs", True):
                tokenizer = getattr(self.agent, "tokenizer", None)
                agent_config = getattr(self.agent, "config", None)
                system_prompt = getattr(agent_config, "system_prompt", None)
                rendered_prompt = render_prompt_for_scoring(
                    tokenizer, prompt, system_prompt
                )
                responses = []
                for result in results:
                    log_prob = await self._compute_sequence_log_prob(
                        rendered_prompt, result.response
                    )
                    responses.append((result.response, log_prob))
            else:
                if self.config.temperature != 1.0:
                    self._warn_vllm_temperature_bias_once()
                responses = [
                    (result.response, result.cumulative_logprob) for result in results
                ]

            logger.debug("vLLM generated %s responses for prompt", len(responses))
            return responses
        except VLLM_EXCEPTIONS as e:
            logger.warning(f"vLLM generation failed: {e}. Falling back to HuggingFace.")
            return await self._generate_with_hf(prompt, num_responses)

    def _warn_post_processing_divergence_once(self) -> None:
        """One-time warning that the rendered scoring prompt may not exactly
        match what the agent conditioned on when generating the response.
        """
        if self._post_processing_warned:
            return
        self._post_processing_warned = True
        logger.warning(
            "GSPO: agent exposes a conversation memory window "
            "(memory_window=%s); scoring renders only the current user turn "
            "(plus system prompt when set) and cannot reconstruct any "
            "prior-turn context or other agent-side post-processing the live "
            "agent may have used when generating this response. This is a "
            "residual scoring/generation mismatch — see `_generate_with_hf`'s "
            "docstring.",
            getattr(self.agent, "memory_window", None),
        )

    async def _generate_with_hf(
        self, prompt: str, num_responses: int
    ) -> list[tuple[str, float]]:
        """Generate responses using HuggingFace sequentially.

        Limitation: the text scored by `_compute_sequence_log_prob` is the
        prompt rendered via `render_prompt_for_scoring` — a single user turn
        (plus the agent's system prompt, when set) run through the
        tokenizer's chat template. This is NOT guaranteed to be byte-identical
        to what `MultiTurnAgent.generate_response` actually conditioned on:
        the live agent may fold in conversation memory/history, truncate to a
        context window, or otherwise post-process messages before generation.
        Full parity would require threading the agent's exact rendered input
        back out of `generate_response`, which is out of scope here. Treat
        the resulting old-policy log prob as an approximation that is exact
        only for stateless, system-prompt-only agents.
        """
        responses = []

        tokenizer = getattr(self.agent, "tokenizer", None)
        agent_config = getattr(self.agent, "config", None)
        system_prompt = getattr(agent_config, "system_prompt", None)
        rendered_prompt = render_prompt_for_scoring(tokenizer, prompt, system_prompt)

        if getattr(self.agent, "memory_window", 0):
            self._warn_post_processing_divergence_once()

        for _ in range(num_responses):
            messages = [{"role": "user", "content": prompt}]
            response = await self.agent.generate_response(messages)
            log_prob = await self._compute_sequence_log_prob(rendered_prompt, response)
            responses.append((response, log_prob))

        return responses

    async def generate_batch_groups(
        self, prompts: list[str], num_responses_per_prompt: int
    ) -> dict[str, list[tuple[str, float]]]:
        """Generate response groups for multiple prompts efficiently."""
        if self.using_vllm:
            try:
                generator = self.vllm_generator
                if generator is None:
                    return {
                        prompt: await self._generate_with_hf(
                            prompt, num_responses_per_prompt
                        )
                        for prompt in prompts
                    }
                grouped_results = await generator.generate_groups(
                    prompts=prompts,
                    num_generations_per_prompt=num_responses_per_prompt,
                )

                if getattr(self.config, "rescore_old_log_probs", True):
                    tokenizer = getattr(self.agent, "tokenizer", None)
                    agent_config = getattr(self.agent, "config", None)
                    system_prompt = getattr(agent_config, "system_prompt", None)
                    batch_results: dict[str, list[tuple[str, float]]] = {}
                    for prompt, results in grouped_results.items():
                        rendered_prompt = render_prompt_for_scoring(
                            tokenizer, prompt, system_prompt
                        )
                        batch_results[prompt] = [
                            (
                                r.response,
                                await self._compute_sequence_log_prob(
                                    rendered_prompt, r.response
                                ),
                            )
                            for r in results
                        ]
                    return batch_results

                if self.config.temperature != 1.0:
                    self._warn_vllm_temperature_bias_once()
                return {
                    prompt: [(r.response, r.cumulative_logprob) for r in results]
                    for prompt, results in grouped_results.items()
                }
            except VLLM_EXCEPTIONS as e:
                logger.warning(
                    f"Batch vLLM generation failed: {e}. Falling back to sequential."
                )

        results = {}
        for prompt in prompts:
            results[prompt] = await self._generate_with_hf(
                prompt, num_responses_per_prompt
            )
        return results

    async def _compute_sequence_log_prob(
        self, prompt_text: str, response: str
    ) -> float:
        """Compute the log probability of a sequence.

        `prompt_text` must be the exact text the response continues from
        (e.g. the chat-template-rendered prompt used at generation time), so
        that scoring tokenizes the same text the model actually sampled.
        """
        tokenizer = getattr(self.agent, "tokenizer", None)
        model = getattr(self.agent, "model", None)
        if tokenizer is None or model is None:
            raise RuntimeError(
                "Agent tokenizer and model are required for GSPO scoring"
            )

        full_text = build_scoring_text(prompt_text, response)
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length
            + self.config.max_completion_length,
            add_special_tokens=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
            add_special_tokens=False,
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        model_device = _get_model_device(model)
        if model_device and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        input_ids = inputs["input_ids"]
        # Shared gather (fp32 log-softmax), masked to response positions:
        # unshifted position p is a response token when p >= prompt_length,
        # i.e. shifted index max(prompt_length - 1, 0) onwards.
        response_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if prompt_length < response_mask.shape[-1]:
            response_mask[..., prompt_length:] = 1.0

        response_start = max(prompt_length - 1, 0)
        if response_start >= input_ids.shape[-1] - 1:
            # Degenerate case (prompt fills the window): fall back to the
            # whole sequence rather than returning 0.
            all_ones = torch.ones_like(input_ids, dtype=torch.float32)
            token_log_probs, _ = rl_losses.gather_token_logprobs(
                logits, input_ids, all_ones
            )
            return float(token_log_probs.sum().item())

        token_log_probs, _ = rl_losses.gather_token_logprobs(
            logits, input_ids, response_mask
        )
        return float(token_log_probs.sum().item())


__all__ = [
    "GSPOTrajectoryGenerator",
    "build_scoring_text",
    "render_prompt_for_scoring",
    "GenerationResult",
    "HuggingFaceGeneratorFallback",
    "VLLM_AVAILABLE",
    "VLLM_BACKEND_AVAILABLE",
    "VLLMConfig",
    "VLLMGenerator",
    "_load_vllm_backend",
]
