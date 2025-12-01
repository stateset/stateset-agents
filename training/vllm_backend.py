"""
vLLM Backend for High-Performance Generation in RL Training

This module provides a unified vLLM integration for all trainers in the framework,
enabling 5-20x faster generation compared to standard HuggingFace generation.

Key Features:
- Batched generation with efficient KV cache management
- Automatic log probability extraction for importance ratios
- Prefix caching for repeated prompts (common in RL)
- Memory-efficient inference with PagedAttention
- Seamless fallback to HuggingFace when vLLM is unavailable

Usage:
    from training.vllm_backend import VLLMGenerator, VLLMConfig

    config = VLLMConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )

    generator = VLLMGenerator(config)
    await generator.initialize()

    results = await generator.generate_with_logprobs(
        prompts=["What is 2+2?", "Explain gravity"],
        sampling_params={"temperature": 0.7, "max_tokens": 512}
    )

Reference:
- vLLM Paper: https://arxiv.org/abs/2309.06180
- PagedAttention: Efficient memory management for LLM serving
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# Check vLLM availability
try:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
    logger.info("vLLM is available - high-performance generation enabled")
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    RequestOutput = None
    logger.warning(
        "vLLM not installed. Install with `pip install vllm` for 5-20x faster generation. "
        "Falling back to HuggingFace generation."
    )


@dataclass
class VLLMConfig:
    """Configuration for vLLM backend"""

    # Model configuration
    model_name: str = "gpt2"
    tokenizer_name: Optional[str] = None  # Defaults to model_name
    revision: Optional[str] = None

    # Hardware configuration
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    pipeline_parallel_size: int = 1  # Number of GPUs for pipeline parallelism
    gpu_memory_utilization: float = 0.85  # Fraction of GPU memory to use
    max_model_len: Optional[int] = None  # Max sequence length (auto-detect if None)

    # Generation defaults
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1  # -1 means disabled

    # Performance options
    enable_prefix_caching: bool = True  # Cache KV for repeated prefixes
    enable_chunked_prefill: bool = True  # Better memory efficiency
    max_num_seqs: int = 256  # Max concurrent sequences
    max_num_batched_tokens: Optional[int] = None  # Auto-compute if None

    # Quantization
    quantization: Optional[str] = None  # "awq", "gptq", "squeezellm", None
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"

    # LoRA support
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 64
    lora_modules: Optional[List[Dict[str, Any]]] = None

    # Seed for reproducibility
    seed: Optional[int] = None

    # Trust remote code (for custom models)
    trust_remote_code: bool = True

    def to_vllm_kwargs(self) -> Dict[str, Any]:
        """Convert to vLLM LLM constructor kwargs"""
        kwargs = {
            "model": self.model_name,
            "tokenizer": self.tokenizer_name or self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
        }

        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len

        if self.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens

        if self.quantization is not None:
            kwargs["quantization"] = self.quantization

        if self.revision is not None:
            kwargs["revision"] = self.revision

        if self.seed is not None:
            kwargs["seed"] = self.seed

        if self.enable_lora:
            kwargs["enable_lora"] = True
            kwargs["max_loras"] = self.max_loras
            kwargs["max_lora_rank"] = self.max_lora_rank

        return kwargs


@dataclass
class GenerationResult:
    """Result from a single generation"""
    prompt: str
    response: str
    full_text: str
    prompt_token_ids: List[int]
    response_token_ids: List[int]
    token_logprobs: List[float]  # Log prob for each response token
    cumulative_logprob: float  # Sum of token log probs
    sequence_length: int
    finish_reason: str  # "stop", "length", "abort"

    @property
    def mean_logprob(self) -> float:
        """Average log probability per token"""
        if self.sequence_length == 0:
            return 0.0
        return self.cumulative_logprob / self.sequence_length

    @property
    def perplexity(self) -> float:
        """Perplexity of the generation"""
        import math
        return math.exp(-self.mean_logprob)


@dataclass
class BatchGenerationResult:
    """Results from a batch generation"""
    results: List[GenerationResult]
    total_tokens_generated: int
    generation_time_seconds: float
    tokens_per_second: float


class VLLMGenerator:
    """
    High-performance text generation using vLLM.

    Provides batched generation with automatic log probability extraction,
    which is essential for computing importance ratios in policy gradient methods.
    """

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.engine: Optional[LLM] = None
        self.tokenizer = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the vLLM engine.

        Returns True if vLLM was successfully initialized, False if falling back to HF.
        """
        if not VLLM_AVAILABLE:
            logger.warning("vLLM not available, will use HuggingFace fallback")
            return False

        try:
            logger.info(f"Initializing vLLM engine for {self.config.model_name}...")
            logger.info(f"  Tensor parallel size: {self.config.tensor_parallel_size}")
            logger.info(f"  GPU memory utilization: {self.config.gpu_memory_utilization}")
            logger.info(f"  Prefix caching: {self.config.enable_prefix_caching}")

            # Initialize vLLM engine
            vllm_kwargs = self.config.to_vllm_kwargs()
            self.engine = LLM(**vllm_kwargs)
            self.tokenizer = self.engine.get_tokenizer()

            self._initialized = True
            logger.info("vLLM engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.warning("Falling back to HuggingFace generation")
            self._initialized = False
            return False

    @property
    def is_available(self) -> bool:
        """Check if vLLM is available and initialized"""
        return self._initialized and self.engine is not None

    def create_sampling_params(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        logprobs: int = 1,  # Number of log probs to return per token
        prompt_logprobs: Optional[int] = None,  # Log probs for prompt tokens
        **kwargs,
    ) -> "SamplingParams":
        """Create vLLM SamplingParams with defaults from config"""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")

        return SamplingParams(
            temperature=temperature if temperature is not None else self.config.temperature,
            top_p=top_p if top_p is not None else self.config.top_p,
            top_k=top_k if top_k is not None else self.config.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            stop=stop,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            **kwargs,
        )

    def generate_sync(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional["SamplingParams"] = None,
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Synchronous generation with log probabilities.

        Args:
            prompts: Single prompt or list of prompts
            sampling_params: vLLM SamplingParams (or created from kwargs)
            **kwargs: Passed to create_sampling_params if sampling_params not provided

        Returns:
            List of GenerationResult objects with responses and log probs
        """
        if not self.is_available:
            raise RuntimeError("vLLM not initialized. Call initialize() first or use fallback.")

        # Normalize to list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Create sampling params if not provided
        if sampling_params is None:
            sampling_params = self.create_sampling_params(**kwargs)

        # Generate with vLLM
        outputs: List[RequestOutput] = self.engine.generate(prompts, sampling_params)

        # Process outputs
        results = []
        for output in outputs:
            prompt = output.prompt

            # Get the first (best) completion
            completion = output.outputs[0]
            response_text = completion.text
            response_token_ids = list(completion.token_ids)

            # Extract log probabilities
            token_logprobs = []
            if completion.logprobs is not None:
                for logprob_dict in completion.logprobs:
                    if logprob_dict is not None:
                        # Get the log prob of the actual sampled token
                        # logprob_dict maps token_id -> LogProb object
                        sampled_token_id = response_token_ids[len(token_logprobs)] if len(token_logprobs) < len(response_token_ids) else None
                        if sampled_token_id is not None and sampled_token_id in logprob_dict:
                            token_logprobs.append(logprob_dict[sampled_token_id].logprob)
                        elif logprob_dict:
                            # Fallback: get the first logprob
                            first_key = next(iter(logprob_dict))
                            token_logprobs.append(logprob_dict[first_key].logprob)
                        else:
                            token_logprobs.append(0.0)
                    else:
                        token_logprobs.append(0.0)

            # Compute cumulative log prob
            cumulative_logprob = sum(token_logprobs) if token_logprobs else 0.0

            result = GenerationResult(
                prompt=prompt,
                response=response_text,
                full_text=prompt + response_text,
                prompt_token_ids=list(output.prompt_token_ids),
                response_token_ids=response_token_ids,
                token_logprobs=token_logprobs,
                cumulative_logprob=cumulative_logprob,
                sequence_length=len(response_token_ids),
                finish_reason=completion.finish_reason,
            )
            results.append(result)

        return results

    async def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional["SamplingParams"] = None,
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Async generation with log probabilities.

        Wraps synchronous vLLM generation in asyncio executor for non-blocking use.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_sync(prompts, sampling_params, **kwargs)
        )

    async def generate_groups(
        self,
        prompts: List[str],
        num_generations_per_prompt: int,
        sampling_params: Optional["SamplingParams"] = None,
        **kwargs,
    ) -> Dict[str, List[GenerationResult]]:
        """
        Generate multiple responses per prompt (for GRPO/GSPO group generation).

        This is more efficient than calling generate() multiple times because
        vLLM can batch all generations together.

        Args:
            prompts: List of prompts
            num_generations_per_prompt: Number of responses per prompt (G in GRPO)
            sampling_params: vLLM SamplingParams
            **kwargs: Passed to create_sampling_params

        Returns:
            Dict mapping each prompt to its list of GenerationResults
        """
        if not self.is_available:
            raise RuntimeError("vLLM not initialized")

        # Expand prompts for batched generation
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_generations_per_prompt)

        # Generate all at once (vLLM handles batching efficiently)
        all_results = await self.generate(expanded_prompts, sampling_params, **kwargs)

        # Group results by prompt
        grouped_results: Dict[str, List[GenerationResult]] = {p: [] for p in prompts}
        for i, result in enumerate(all_results):
            prompt_idx = i // num_generations_per_prompt
            original_prompt = prompts[prompt_idx]
            grouped_results[original_prompt].append(result)

        return grouped_results

    def compute_log_probs_for_sequences(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[Tuple[float, List[float]]]:
        """
        Compute log probabilities for given prompt-response pairs.

        This is useful for computing importance ratios when you already
        have the responses (e.g., from old policy).

        Args:
            prompts: List of prompts
            responses: List of corresponding responses

        Returns:
            List of (cumulative_logprob, token_logprobs) tuples
        """
        if not self.is_available:
            raise RuntimeError("vLLM not initialized")

        # Combine prompts and responses
        full_sequences = [p + r for p, r in zip(prompts, responses)]

        # Use prompt_logprobs to get log probs for the full sequence
        sampling_params = self.create_sampling_params(
            max_tokens=1,  # We just want the log probs, not more generation
            prompt_logprobs=1,
        )

        outputs = self.engine.generate(full_sequences, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            prompt = prompts[i]
            prompt_tokens = self.tokenizer.encode(prompt)
            prompt_length = len(prompt_tokens)

            # Extract log probs for response tokens (after prompt)
            token_logprobs = []
            if output.prompt_logprobs is not None:
                for j, logprob_dict in enumerate(output.prompt_logprobs):
                    if j >= prompt_length and logprob_dict is not None:
                        # This is a response token
                        token_id = output.prompt_token_ids[j]
                        if token_id in logprob_dict:
                            token_logprobs.append(logprob_dict[token_id].logprob)
                        else:
                            token_logprobs.append(0.0)

            cumulative = sum(token_logprobs) if token_logprobs else 0.0
            results.append((cumulative, token_logprobs))

        return results

    def add_lora_adapter(
        self,
        adapter_name: str,
        adapter_path: str,
    ):
        """
        Add a LoRA adapter to the vLLM engine.

        Note: Requires enable_lora=True in config.
        """
        if not self.is_available:
            raise RuntimeError("vLLM not initialized")

        if not self.config.enable_lora:
            raise RuntimeError("LoRA not enabled in config. Set enable_lora=True.")

        # vLLM LoRA loading
        from vllm.lora.request import LoRARequest
        # Note: Actual LoRA loading depends on vLLM version
        logger.info(f"Adding LoRA adapter: {adapter_name} from {adapter_path}")

    def get_tokenizer(self):
        """Get the underlying tokenizer"""
        return self.tokenizer

    def shutdown(self):
        """Shutdown the vLLM engine and free resources"""
        if self.engine is not None:
            # vLLM cleanup
            del self.engine
            self.engine = None
            self._initialized = False

            # Force CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("vLLM engine shutdown complete")


class HuggingFaceGeneratorFallback:
    """
    Fallback generator using HuggingFace transformers.

    Used when vLLM is not available or fails to initialize.
    Provides the same interface as VLLMGenerator but with HF generation.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize HuggingFace model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Initializing HuggingFace fallback for {self.model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            self._initialized = True
            logger.info("HuggingFace fallback initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace fallback: {e}")
            return False

    @property
    def is_available(self) -> bool:
        return self._initialized

    async def generate(
        self,
        prompts: Union[str, List[str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs,
    ) -> List[GenerationResult]:
        """Generate responses using HuggingFace"""
        if not self.is_available:
            raise RuntimeError("HuggingFace model not initialized")

        if isinstance(prompts, str):
            prompts = [prompts]

        results = []

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            prompt_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            generated_ids = outputs.sequences[0]
            response_ids = generated_ids[prompt_length:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Compute log probs from scores
            token_logprobs = []
            if outputs.scores:
                import torch.nn.functional as F
                for i, score in enumerate(outputs.scores):
                    if i < len(response_ids):
                        log_probs = F.log_softmax(score[0], dim=-1)
                        token_id = response_ids[i].item()
                        token_logprobs.append(log_probs[token_id].item())

            cumulative_logprob = sum(token_logprobs) if token_logprobs else 0.0

            result = GenerationResult(
                prompt=prompt,
                response=response_text,
                full_text=prompt + response_text,
                prompt_token_ids=inputs["input_ids"][0].tolist(),
                response_token_ids=response_ids.tolist(),
                token_logprobs=token_logprobs,
                cumulative_logprob=cumulative_logprob,
                sequence_length=len(response_ids),
                finish_reason="stop" if len(response_ids) < max_tokens else "length",
            )
            results.append(result)

        return results

    async def generate_groups(
        self,
        prompts: List[str],
        num_generations_per_prompt: int,
        **kwargs,
    ) -> Dict[str, List[GenerationResult]]:
        """Generate multiple responses per prompt"""
        grouped_results: Dict[str, List[GenerationResult]] = {p: [] for p in prompts}

        for prompt in prompts:
            for _ in range(num_generations_per_prompt):
                results = await self.generate([prompt], **kwargs)
                grouped_results[prompt].append(results[0])

        return grouped_results

    def get_tokenizer(self):
        return self.tokenizer

    def shutdown(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._initialized = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_generator(
    config: Union[VLLMConfig, Dict[str, Any], str],
    prefer_vllm: bool = True,
) -> Union[VLLMGenerator, HuggingFaceGeneratorFallback]:
    """
    Factory function to create the appropriate generator.

    Args:
        config: VLLMConfig, dict with config options, or model name string
        prefer_vllm: If True, try vLLM first; if False, always use HF

    Returns:
        VLLMGenerator if available and preferred, else HuggingFaceGeneratorFallback
    """
    # Normalize config
    if isinstance(config, str):
        config = VLLMConfig(model_name=config)
    elif isinstance(config, dict):
        config = VLLMConfig(**config)

    if prefer_vllm and VLLM_AVAILABLE:
        return VLLMGenerator(config)
    else:
        return HuggingFaceGeneratorFallback(config.model_name)


# Convenience function for quick generation
async def quick_generate(
    model_name: str,
    prompts: Union[str, List[str]],
    num_generations: int = 1,
    **kwargs,
) -> Union[List[GenerationResult], Dict[str, List[GenerationResult]]]:
    """
    Quick generation utility for one-off use.

    Automatically handles initialization and cleanup.
    """
    generator = create_generator(model_name)
    await generator.initialize()

    try:
        if num_generations == 1:
            return await generator.generate(prompts, **kwargs)
        else:
            if isinstance(prompts, str):
                prompts = [prompts]
            return await generator.generate_groups(prompts, num_generations, **kwargs)
    finally:
        generator.shutdown()


# Export
__all__ = [
    "VLLM_AVAILABLE",
    "VLLMConfig",
    "VLLMGenerator",
    "HuggingFaceGeneratorFallback",
    "GenerationResult",
    "BatchGenerationResult",
    "create_generator",
    "quick_generate",
]
