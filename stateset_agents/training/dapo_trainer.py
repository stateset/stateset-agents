"""
Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) Training

DAPO is an advanced RL algorithm designed for long chain-of-thought (CoT) reasoning.
It achieves state-of-the-art results (50 points on AIME 2024) through four key techniques:

1. Clip-Higher: Asymmetric clipping to prevent entropy collapse
2. Dynamic Sampling: Filter out prompts with 0% or 100% accuracy
3. Token-Level Policy Gradient: Normalize by total tokens, not samples
4. Overlong Reward Shaping: Soft penalty for sequences approaching max length

Reference: https://arxiv.org/abs/2503.14476
GitHub: https://github.com/BytedTsinghua-SIA/DAPO
"""

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from .dapo_config import DAPOConfig

logger = logging.getLogger(__name__)

DAPO_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    OSError,
    asyncio.TimeoutError,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = cast(Any, None)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = cast(Any, None)
    TaskType = cast(Any, None)
    get_peft_model = cast(Any, None)

# Lazy import transformers to avoid torch/torchvision compatibility issues
_transformers_dapo_loaded = False
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None
get_cosine_schedule_with_warmup: Any = None
get_constant_schedule: Any = None


def _load_transformers_dapo() -> bool:
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_dapo_loaded, AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup, get_constant_schedule
    if _transformers_dapo_loaded:
        return True
    if AutoModelForCausalLM is not None and AutoTokenizer is not None:
        _transformers_dapo_loaded = True
        return True
    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import get_constant_schedule as _get_constant
        from transformers import get_cosine_schedule_with_warmup as _get_cosine

        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        get_cosine_schedule_with_warmup = _get_cosine
        get_constant_schedule = _get_constant
        _transformers_dapo_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


def _require_transformers_dapo() -> None:
    """Ensure transformers components are available before model loading."""
    if not _load_transformers_dapo():
        raise ImportError(
            "transformers is required for DAPO training. "
            "Install with `pip install stateset-agents[training]` or `pip install transformers`."
        )


def _require_peft() -> None:
    """Ensure PEFT is available before using LoRA features."""
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError(
            "PEFT is required for DAPO LoRA training. "
            "Install with `pip install stateset-agents[training]` or `pip install peft`."
        )


def _require_wandb() -> None:
    """Ensure Weights & Biases is available before logging."""
    if wandb is None:
        raise ImportError(
            "wandb is required for DAPO logging. "
            "Install with `pip install stateset-agents[training]` or `pip install wandb`."
        )


# Lazy import vLLM backend to avoid torch/torchvision compatibility issues
# vllm imports transformers which imports torchvision at module level
VLLMConfig: Any = None
VLLMGenerator: Any = None
GenerationResult: Any = None
VLLM_BACKEND_AVAILABLE = False
_vllm_backend_loaded = False


def _load_vllm_backend() -> bool:
    """Lazily load vLLM backend to avoid import-time errors."""
    global _vllm_backend_loaded, VLLMConfig, VLLMGenerator, GenerationResult, VLLM_BACKEND_AVAILABLE
    if _vllm_backend_loaded:
        return VLLM_BACKEND_AVAILABLE
    try:
        from .vllm_backend import VLLM_AVAILABLE as _VLLM_BACKEND_AVAILABLE
        from .vllm_backend import GenerationResult as _GenerationResult
        from .vllm_backend import VLLMConfig as _VLLMConfig
        from .vllm_backend import VLLMGenerator as _VLLMGenerator

        VLLMConfig = _VLLMConfig
        VLLMGenerator = _VLLMGenerator
        GenerationResult = _GenerationResult
        VLLM_BACKEND_AVAILABLE = _VLLM_BACKEND_AVAILABLE
        _vllm_backend_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load vLLM backend: {e}")
        _vllm_backend_loaded = True
        return False


class DAPOModelManager:
    """Manages model loading for DAPO training"""

    def __init__(self, config: DAPOConfig):
        self.config = config
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load model and tokenizer with optional LoRA"""
        logger.info(f"Loading model: {self.config.model_name}")
        _require_transformers_dapo()
        auto_tokenizer = AutoTokenizer
        auto_model = AutoModelForCausalLM
        if auto_tokenizer is None or auto_model is None:
            raise ImportError("transformers exports are unavailable for DAPO")

        # Load tokenizer
        self.tokenizer = auto_tokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch.float16
            if self.config.fp16
            else (torch.bfloat16 if self.config.bf16 else torch.float32),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }

        # Load base model
        base_model = auto_model.from_pretrained(self.config.model_name, **model_kwargs)

        # Add LoRA adapters if configured
        if self.config.use_lora:
            _require_peft()
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model_with_lora = get_peft_model(base_model, lora_config)
            model_with_lora.print_trainable_parameters()
            self.model = model_with_lora
        else:
            self.model = base_model

        logger.info(f"Model loaded on {self.device}")
        return self.model, self.tokenizer


class DAPORewardShaper:
    """
    Implements DAPO's Overlong Reward Shaping.

    Applies graduated length penalty:
    - 0 if |y| <= L_max - L_cache
    - Linear penalty from 0 to -1 if L_max - L_cache < |y| <= L_max
    - -1 if |y| > L_max (truncated)
    """

    def __init__(
        self,
        max_length: int = 20480,
        cache_length: int = 4096,
        penalty: float = -1.0,
    ):
        self.max_length = max_length
        self.cache_length = cache_length
        self.penalty = penalty
        self.soft_start = max_length - cache_length

    def compute_length_reward(self, sequence_length: int) -> float:
        """
        Compute length-based reward adjustment.

        R_length(y) = {
            0,                                    if |y| <= L_max - L_cache
            [(L_max - L_cache) - |y|]/L_cache,   if L_max - L_cache < |y| <= L_max
            -1,                                   if L_max < |y|
        }
        """
        if sequence_length <= self.soft_start:
            return 0.0
        elif sequence_length <= self.max_length:
            # Linear interpolation from 0 to penalty
            progress = (sequence_length - self.soft_start) / self.cache_length
            return self.penalty * progress
        else:
            # Truncated - full penalty
            return self.penalty

    def shape_reward(
        self,
        base_reward: float,
        sequence_length: int,
        weight: float = 1.0,
    ) -> float:
        """Apply length shaping to base reward"""
        length_adjustment = self.compute_length_reward(sequence_length)
        return base_reward + weight * length_adjustment


class DynamicSamplingBuffer:
    """
    Implements DAPO's Dynamic Sampling.

    Filters out prompts where all responses are correct (accuracy=1)
    or all responses are wrong (accuracy=0), as these provide no gradient signal.
    """

    def __init__(
        self,
        buffer_size: int = 1024,
        min_accuracy: float = 0.0,
        max_accuracy: float = 1.0,
    ):
        self.buffer_size = buffer_size
        self.min_accuracy = min_accuracy
        self.max_accuracy = max_accuracy
        self.buffer: list[dict[str, Any]] = []

    def should_include(self, accuracy: float) -> bool:
        """Check if sample should be included based on accuracy"""
        return self.min_accuracy < accuracy < self.max_accuracy

    def add_sample(self, sample: dict[str, Any], accuracy: float) -> bool:
        """
        Add sample to buffer if it passes the accuracy filter.

        Returns True if sample was added.
        """
        if self.should_include(accuracy):
            self.buffer.append(sample)
            return True
        return False

    def get_batch(self, batch_size: int) -> list[dict[str, Any]]:
        """Get a batch of samples from buffer"""
        if len(self.buffer) < batch_size:
            return []

        batch = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]
        return batch

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch"""
        return len(self.buffer) >= batch_size

    def clear(self):
        """Clear the buffer"""
        self.buffer = []

    @property
    def size(self) -> int:
        return len(self.buffer)


class DAPOTrainer:
    """
    Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) Trainer

    DAPO achieves 50 points on AIME 2024 through:

    1. Clip-Higher: Uses asymmetric clipping (eps_low=0.2, eps_high=0.28)
       to allow more exploration while maintaining stability.

    2. Dynamic Sampling: Filters prompts with trivial accuracy (0% or 100%)
       to ensure all gradients are meaningful.

    3. Token-Level Loss: Normalizes by total token count instead of sample count
       to prevent length-based learning bias.

    4. Overlong Reward Shaping: Applies soft then hard penalties as sequences
       approach maximum length.

    Now with optional vLLM support for 5-20x faster generation!

    Reference: https://arxiv.org/abs/2503.14476
    """

    def __init__(
        self,
        config: DAPOConfig,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float | Awaitable[float]],
        verifier_fn: Callable[[str, str], bool] | None = None,
    ):
        # Ensure transformers is loaded for scheduler
        _load_transformers_dapo()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.verifier_fn = verifier_fn  # For binary correctness (e.g., math)
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # vLLM generator for fast generation
        self.vllm_generator: Any | None = None
        self._vllm_initialized = False

        # Setup vLLM if configured
        if config.use_vllm and VLLM_BACKEND_AVAILABLE:
            self._setup_vllm()

        # Reward shaper for length penalty
        self.reward_shaper = DAPORewardShaper(
            max_length=config.max_generation_length,
            cache_length=config.overlong_cache_length,
            penalty=config.overlong_penalty,
        )

        # Dynamic sampling buffer
        self.sampling_buffer = DynamicSamplingBuffer(
            buffer_size=config.dynamic_sampling_buffer_size,
            min_accuracy=config.min_accuracy_threshold,
            max_accuracy=config.max_accuracy_threshold,
        )

        # Optimizer (constant learning rate as per paper)
        params = list(self.model.parameters())
        if not params:
            self._stub_param = torch.nn.Parameter(torch.zeros(1))
            params = [self._stub_param]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )

        # Constant scheduler (DAPO uses constant LR)
        if get_constant_schedule is not None:
            self.scheduler = get_constant_schedule(self.optimizer)
        else:
            # Fallback to constant learning rate if scheduler unavailable
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=config.num_episodes * config.num_epochs,
            )

        # Metrics
        self.metrics_history: dict[str, list[float]] = {
            "policy_loss": [],
            "average_reward": [],
            "accuracy": [],
            "filtered_ratio": [],
            "avg_sequence_length": [],
        }

        self.global_step = 0

    async def _compute_reward(self, prompt: str, response: str) -> float:
        """Support sync or async reward callables."""
        reward = self.reward_fn(prompt, response)
        if asyncio.iscoroutine(reward):
            reward_value = await cast(Awaitable[float], reward)
        else:
            reward_value = cast(float, reward)
        return float(reward_value)

    def _setup_vllm(self):
        """Setup vLLM generator"""
        logger.info("Setting up vLLM for fast DAPO generation...")

        # Load vLLM backend lazily
        if not _load_vllm_backend():
            logger.warning("vLLM backend failed to load, will use HuggingFace fallback")
            return

        vllm_config_cls = VLLMConfig
        vllm_generator_cls = VLLMGenerator
        if vllm_config_cls is None or vllm_generator_cls is None:
            logger.warning("vLLM backend exports unavailable, using HuggingFace fallback")
            return

        vllm_config = vllm_config_cls(
            model_name=self.config.model_name,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            enable_prefix_caching=self.config.vllm_enable_prefix_caching,
            max_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            dtype="bfloat16" if self.config.bf16 else "float16",
        )

        self.vllm_generator = vllm_generator_cls(vllm_config)

    async def initialize_vllm(self) -> bool:
        """Initialize vLLM engine"""
        if self.vllm_generator is None:
            return False

        if self._vllm_initialized:
            return True

        try:
            success = bool(await self.vllm_generator.initialize())
            self._vllm_initialized = success
            if success:
                logger.info(
                    "vLLM initialized for DAPO - generation will be 5-20x faster!"
                )
            return success
        except DAPO_EXCEPTIONS as e:
            logger.warning(f"Failed to initialize vLLM: {e}")
            return False

    @property
    def using_vllm(self) -> bool:
        """Check if vLLM is being used"""
        return self._vllm_initialized and self.vllm_generator is not None

    def compute_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities.

        Returns:
            token_log_probs: Log prob for each token [batch, seq_len]
            token_counts: Number of response tokens per sequence [batch]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_response_mask = response_mask[:, 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask non-response tokens
        masked_log_probs = token_log_probs * shift_response_mask

        # Count response tokens
        token_counts = shift_response_mask.sum(dim=-1)

        return masked_log_probs, token_counts

    def compute_importance_ratio(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token importance ratio.

        r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
            = exp(log_pi_theta - log_pi_theta_old)
        """
        log_ratio = current_log_probs - old_log_probs
        return torch.exp(log_ratio)

    def compute_dapo_loss(
        self,
        importance_ratios: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DAPO loss with Clip-Higher and token-level normalization.

        L = -(1/sum(|o_i|)) * sum_i sum_t min(r_t * A, clip(r_t, 1-eps_low, 1+eps_high) * A)

        Key differences from PPO:
        - Asymmetric clipping (eps_low != eps_high)
        - Normalize by total token count, not sample count
        """
        # Apply asymmetric clipping (Clip-Higher)
        clipped_ratios = torch.clamp(
            importance_ratios,
            1.0 - self.config.clip_eps_low,
            1.0 + self.config.clip_eps_high,
        )

        # Compute surrogate objectives
        unclipped_obj = importance_ratios * advantages
        clipped_obj = clipped_ratios * advantages

        # Take minimum (pessimistic bound)
        surrogate = torch.min(unclipped_obj, clipped_obj)

        # Apply response mask
        masked_surrogate = surrogate * response_mask

        if self.config.use_token_level_loss:
            # Token-level normalization: divide by total response tokens
            total_tokens = response_mask.sum()
            if total_tokens > 0:
                loss = -masked_surrogate.sum() / total_tokens
            else:
                loss = torch.tensor(0.0, device=self.device)
        else:
            # Sample-level normalization (standard)
            loss = -masked_surrogate.sum(dim=-1).mean()

        return loss

    async def generate_group_responses(
        self,
        prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Generate a group of responses for a prompt.

        Uses vLLM for fast batched generation if available.

        Returns list of dicts containing:
            - response: Generated text
            - input_ids: Full tokenized sequence
            - response_mask: Mask for response tokens
            - sequence_length: Length of response
        """
        # Try vLLM first (much faster for batched generation)
        if self.using_vllm:
            return await self._generate_with_vllm(prompt)

        # Fallback to HuggingFace generation
        return await self._generate_with_hf(prompt)

    async def _generate_with_vllm(self, prompt: str) -> list[dict[str, Any]]:
        """Generate responses using vLLM (5-20x faster)"""
        try:
            generator = self.vllm_generator
            if generator is None:
                return await self._generate_with_hf(prompt)
            # Generate all responses in a single batched call
            grouped_results = await generator.generate_groups(
                prompts=[prompt],
                num_generations_per_prompt=self.config.group_size,
            )

            results = grouped_results[prompt]
            responses = []

            for result in results:
                prompt_length = len(result.prompt_token_ids)
                full_ids = torch.tensor(
                    result.prompt_token_ids + result.response_token_ids,
                    device=self.device,
                )

                # Create response mask
                response_mask = torch.zeros(len(full_ids), device=self.device)
                response_mask[prompt_length:] = 1.0

                responses.append(
                    {
                        "response": result.response,
                        "input_ids": full_ids,
                        "attention_mask": torch.ones_like(full_ids),
                        "response_mask": response_mask,
                        "sequence_length": result.sequence_length,
                        "prompt_length": prompt_length,
                        "token_logprobs": result.token_logprobs,  # Already computed!
                        "cumulative_logprob": result.cumulative_logprob,
                    }
                )

            logger.debug(f"vLLM generated {len(responses)} responses for DAPO")
            return responses

        except DAPO_EXCEPTIONS as e:
            logger.warning(f"vLLM generation failed: {e}. Falling back to HuggingFace.")
            return await self._generate_with_hf(prompt)

    async def _generate_with_hf(self, prompt: str) -> list[dict[str, Any]]:
        """Generate responses using HuggingFace (sequential fallback)"""
        responses = []

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.config.group_size):
                input_ids = prompt_tokens["input_ids"].to(self.device)
                attention_mask = prompt_tokens["attention_mask"].to(self.device)

                # Generate with DAPO parameters
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                full_ids = outputs[0]
                response_length = len(full_ids) - prompt_length

                # Create response mask (1 for response tokens, 0 for prompt)
                response_mask = torch.zeros(len(full_ids), device=self.device)
                response_mask[prompt_length:] = 1.0

                response_text = self.tokenizer.decode(
                    full_ids[prompt_length:], skip_special_tokens=True
                )

                responses.append(
                    {
                        "response": response_text,
                        "input_ids": full_ids,
                        "attention_mask": torch.ones_like(full_ids),
                        "response_mask": response_mask,
                        "sequence_length": response_length,
                        "prompt_length": prompt_length,
                    }
                )

        self.model.train()
        return responses

    def compute_group_accuracy(
        self,
        prompt: str,
        responses: list[dict[str, Any]],
    ) -> float:
        """Compute accuracy for a group of responses"""
        if self.verifier_fn is None:
            # If no verifier, use reward threshold
            correct = sum(1 for r in responses if float(r.get("reward", 0.0)) > 0.5)
        else:
            correct = sum(
                1 for r in responses if self.verifier_fn(prompt, r["response"])
            )
        return correct / len(responses)

    async def collect_samples_with_dynamic_sampling(
        self,
        prompts: list[str],
        target_batch_size: int,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Collect samples using dynamic sampling.

        Continues generating until we have enough non-trivial samples
        (accuracy not 0 or 1).
        """
        collected_samples: list[dict[str, Any]] = []
        prompts_processed = 0
        prompts_filtered = 0

        prompt_idx = 0
        while len(collected_samples) < target_batch_size and prompt_idx < len(prompts):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            prompts_processed += 1

            # Generate group
            group_responses = await self.generate_group_responses(prompt)

            # Compute accuracy
            accuracy = self.compute_group_accuracy(prompt, group_responses)

            # Check if should include (dynamic sampling filter)
            if self.config.use_dynamic_sampling:
                if not self.sampling_buffer.should_include(accuracy):
                    prompts_filtered += 1
                    continue

            # Compute rewards and advantages
            rewards = []
            for resp in group_responses:
                base_reward = await self._compute_reward(prompt, resp["response"])
                resp["reward"] = base_reward

                # Apply overlong reward shaping
                if self.config.use_overlong_shaping:
                    shaped_reward = self.reward_shaper.shape_reward(
                        base_reward, resp["sequence_length"]
                    )
                else:
                    shaped_reward = base_reward

                rewards.append(shaped_reward)

            rewards_tensor = torch.tensor(rewards, device=self.device)

            # Group-relative advantages
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std()
            if std_reward < 1e-8:
                std_reward = torch.tensor(1.0, device=self.device)
            advantages = (rewards_tensor - mean_reward) / std_reward

            # Store sample
            sample = {
                "prompt": prompt,
                "responses": group_responses,
                "rewards": rewards,
                "advantages": advantages,
                "accuracy": accuracy,
            }
            collected_samples.append(sample)

        filter_ratio = prompts_filtered / max(prompts_processed, 1)
        logger.debug(
            f"Dynamic sampling: {prompts_filtered}/{prompts_processed} filtered ({filter_ratio:.1%})"
        )

        return collected_samples, filter_ratio

    async def train_step(
        self,
        prompts: list[str],
    ) -> dict[str, float]:
        """
        Execute one DAPO training step.

        1. Collect samples with dynamic sampling
        2. For each mini-batch:
           a. Compute current policy log probs
           b. Compute importance ratios
           c. Apply Clip-Higher
           d. Compute token-level loss
           e. Update model
        """
        self.model.train()

        # Collect samples with dynamic sampling
        samples, filter_ratio = await self.collect_samples_with_dynamic_sampling(
            prompts, self.config.mini_batch_size
        )

        if len(samples) == 0:
            logger.warning("No valid samples after dynamic sampling")
            return {"policy_loss": 0.0, "filtered_ratio": 1.0}

        total_loss = 0.0
        all_rewards: list[float] = []
        all_accuracies: list[float] = []
        all_seq_lengths: list[int] = []
        num_updates = 0

        # Process samples
        for sample in samples:
            responses = sample["responses"]
            advantages = sample["advantages"]

            all_rewards.extend(sample["rewards"])
            all_accuracies.append(sample["accuracy"])
            all_seq_lengths.extend([r["sequence_length"] for r in responses])

            # Prepare batch
            max_len = max(len(r["input_ids"]) for r in responses)
            batch_size = len(responses)

            batch_input_ids = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=self.device
            )
            batch_attention_mask = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=self.device
            )
            batch_response_mask = torch.zeros(batch_size, max_len, device=self.device)

            for i, resp in enumerate(responses):
                seq_len = len(resp["input_ids"])
                batch_input_ids[i, :seq_len] = resp["input_ids"]
                batch_attention_mask[i, :seq_len] = resp["attention_mask"]
                batch_response_mask[i, :seq_len] = resp["response_mask"]

            # Get old log probs (from generation, detached)
            with torch.no_grad():
                old_token_log_probs, _ = self.compute_token_log_probs(
                    batch_input_ids, batch_attention_mask, batch_response_mask
                )

            # Multiple gradient updates per rollout (mu updates)
            for _ in range(min(self.config.num_gradient_updates, 1)):
                # Compute current log probs
                current_token_log_probs, token_counts = self.compute_token_log_probs(
                    batch_input_ids, batch_attention_mask, batch_response_mask
                )

                # Compute importance ratios
                importance_ratios = self.compute_importance_ratio(
                    current_token_log_probs, old_token_log_probs.detach()
                )

                # Expand advantages to token level
                # advantages: [batch_size] -> [batch_size, seq_len]
                token_advantages = advantages.unsqueeze(1).expand_as(importance_ratios)

                # Compute DAPO loss
                loss = self.compute_dapo_loss(
                    importance_ratios,
                    token_advantages,
                    batch_response_mask[:, 1:],  # Shift for next-token prediction
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                _params = list(self.model.parameters())
                if _params:
                    torch.nn.utils.clip_grad_norm_(
                        _params, self.config.max_grad_norm
                    )

                # Update
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                num_updates += 1

        self.global_step += 1

        # Compute metrics
        metrics = {
            "policy_loss": float(total_loss / max(num_updates, 1)),
            "average_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "accuracy": float(np.mean(all_accuracies)) if all_accuracies else 0.0,
            "filtered_ratio": filter_ratio,
            "avg_sequence_length": float(np.mean(all_seq_lengths))
            if all_seq_lengths
            else 0.0,
            "learning_rate": float(self.config.learning_rate),
            "global_step": float(self.global_step),
        }

        # Store metrics
        for key in ["policy_loss", "average_reward", "accuracy", "filtered_ratio"]:
            if key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])

        return metrics

    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
        }
        torch.save(state, os.path.join(output_dir, "training_state.pt"))

        # Save config
        config_path = os.path.join(output_dir, "dapo_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Checkpoint saved to {output_dir}")

from .dapo_entrypoints import train_reasoning_with_dapo, train_with_dapo


# Export
__all__ = [
    "DAPOConfig",
    "DAPOTrainer",
    "DAPORewardShaper",
    "DynamicSamplingBuffer",
    "train_with_dapo",
    "train_reasoning_with_dapo",
]
