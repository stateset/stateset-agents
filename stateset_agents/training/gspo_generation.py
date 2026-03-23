"""
Trajectory generation helpers for GSPO training.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from stateset_agents.core.agent import Agent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.exceptions import INFERENCE_EXCEPTIONS as VLLM_EXCEPTIONS
from stateset_agents.exceptions import MODEL_DEVICE_EXCEPTIONS

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
        self.vllm_generator = None
        self._vllm_initialized = False

        if self.config.use_vllm and _load_vllm_backend():
            self._setup_vllm_generator()

    def _setup_vllm_generator(self) -> None:
        """Setup vLLM generator with config parameters."""
        logger.info("Setting up vLLM generator for fast generation...")

        if not _load_vllm_backend():
            logger.warning("vLLM backend failed to load, will use HuggingFace fallback")
            return

        vllm_config = VLLMConfig(
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
            dtype="float16"
            if self.config.fp16
            else ("bfloat16" if self.config.bf16 else "auto"),
        )

        self.vllm_generator = VLLMGenerator(vllm_config)

    async def initialize_vllm(self) -> bool:
        """Initialize the vLLM engine before generation."""
        if self.vllm_generator is None:
            return False

        if self._vllm_initialized:
            return True

        try:
            success = await self.vllm_generator.initialize()
            self._vllm_initialized = success
            if success:
                logger.info(
                    "vLLM generator initialized - 5-20x faster generation enabled!"
                )
            return success
        except VLLM_EXCEPTIONS as e:
            logger.warning(f"Failed to initialize vLLM: {e}. Using HuggingFace fallback.")
            self._vllm_initialized = False
            return False

    @property
    def using_vllm(self) -> bool:
        """Check if vLLM is being used for generation."""
        return self._vllm_initialized and self.vllm_generator is not None

    async def generate_group_responses(
        self, prompt: str, num_responses: int
    ) -> list[tuple[str, float]]:
        """Generate a group of responses for a single prompt."""
        if self.using_vllm:
            return await self._generate_with_vllm(prompt, num_responses)
        return await self._generate_with_hf(prompt, num_responses)

    async def _generate_with_vllm(
        self, prompt: str, num_responses: int
    ) -> list[tuple[str, float]]:
        """Generate responses using vLLM."""
        try:
            grouped_results = await self.vllm_generator.generate_groups(
                prompts=[prompt],
                num_generations_per_prompt=num_responses,
            )

            results = grouped_results[prompt]
            responses = [
                (result.response, result.cumulative_logprob) for result in results
            ]

            logger.debug("vLLM generated %s responses for prompt", len(responses))
            return responses
        except VLLM_EXCEPTIONS as e:
            logger.warning(f"vLLM generation failed: {e}. Falling back to HuggingFace.")
            return await self._generate_with_hf(prompt, num_responses)

    async def _generate_with_hf(
        self, prompt: str, num_responses: int
    ) -> list[tuple[str, float]]:
        """Generate responses using HuggingFace sequentially."""
        responses = []

        for _ in range(num_responses):
            messages = [{"role": "user", "content": prompt}]
            response = await self.agent.generate_response(messages)
            log_prob = await self._compute_sequence_log_prob(prompt, response)
            responses.append((response, log_prob))

        return responses

    async def generate_batch_groups(
        self, prompts: list[str], num_responses_per_prompt: int
    ) -> dict[str, list[tuple[str, float]]]:
        """Generate response groups for multiple prompts efficiently."""
        if self.using_vllm:
            try:
                grouped_results = await self.vllm_generator.generate_groups(
                    prompts=prompts,
                    num_generations_per_prompt=num_responses_per_prompt,
                )

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

    async def _compute_sequence_log_prob(self, prompt: str, response: str) -> float:
        """Compute the log probability of a sequence."""
        full_text = prompt + " " + response
        inputs = self.agent.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length
            + self.config.max_completion_length,
            add_special_tokens=False,
        )

        prompt_tokens = self.agent.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
            add_special_tokens=False,
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        model_device = _get_model_device(self.agent.model)
        if model_device and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        with torch.no_grad():
            outputs = self.agent.model(**inputs)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        response_start = max(prompt_length - 1, 0)
        if response_start >= token_log_probs.shape[-1]:
            return token_log_probs.sum().item()

        response_log_probs = token_log_probs[..., response_start:]
        sequence_log_prob = response_log_probs.sum().item()
        return sequence_log_prob


__all__ = [
    "GSPOTrajectoryGenerator",
    "GenerationResult",
    "HuggingFaceGeneratorFallback",
    "VLLM_AVAILABLE",
    "VLLM_BACKEND_AVAILABLE",
    "VLLMConfig",
    "VLLMGenerator",
    "_load_vllm_backend",
]
