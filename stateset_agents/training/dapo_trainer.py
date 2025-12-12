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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import framework components
from .config import TrainingConfig, get_config_for_task

try:
    import numpy as np
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please install: pip install peft datasets wandb")
    raise

# Lazy import transformers to avoid torch/torchvision compatibility issues
_transformers_dapo_loaded = False
AutoModelForCausalLM = None
AutoTokenizer = None
get_cosine_schedule_with_warmup = None
get_constant_schedule = None

def _load_transformers_dapo():
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_dapo_loaded, AutoModelForCausalLM, AutoTokenizer
    global get_cosine_schedule_with_warmup, get_constant_schedule
    if _transformers_dapo_loaded:
        return True
    try:
        from transformers import (
            AutoModelForCausalLM as _AutoModelForCausalLM,
            AutoTokenizer as _AutoTokenizer,
            get_cosine_schedule_with_warmup as _get_cosine,
            get_constant_schedule as _get_constant,
        )
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        get_cosine_schedule_with_warmup = _get_cosine
        get_constant_schedule = _get_constant
        _transformers_dapo_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False

# Lazy import vLLM backend to avoid torch/torchvision compatibility issues
# vllm imports transformers which imports torchvision at module level
VLLMConfig = None
VLLMGenerator = None
GenerationResult = None
VLLM_BACKEND_AVAILABLE = False
_vllm_backend_loaded = False

def _load_vllm_backend():
    """Lazily load vLLM backend to avoid import-time errors."""
    global _vllm_backend_loaded, VLLMConfig, VLLMGenerator, GenerationResult, VLLM_BACKEND_AVAILABLE
    if _vllm_backend_loaded:
        return VLLM_BACKEND_AVAILABLE
    try:
        from .vllm_backend import (
            VLLMConfig as _VLLMConfig,
            VLLMGenerator as _VLLMGenerator,
            GenerationResult as _GenerationResult,
            VLLM_AVAILABLE as _VLLM_BACKEND_AVAILABLE,
        )
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


@dataclass
class DAPOConfig(TrainingConfig):
    """
    Configuration for DAPO training.

    DAPO uses four key techniques for stable long-CoT training:
    1. Clip-Higher (asymmetric clipping)
    2. Dynamic Sampling (filter trivial accuracy)
    3. Token-level loss normalization
    4. Overlong reward shaping
    """

    # Model
    model_name: str = "gpt2"

    # Generation parameters
    max_prompt_length: int = 256
    max_completion_length: int = 512

    # Model optimization
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # DAPO-specific clipping (Clip-Higher)
    clip_eps_low: float = 0.2  # Lower clipping bound (conservative)
    clip_eps_high: float = 0.28  # Upper clipping bound (allows exploration)

    # Group/sampling parameters
    group_size: int = 16  # Responses per prompt (G)
    prompt_batch_size: int = 512  # Number of prompts per batch
    mini_batch_size: int = 512  # Mini-batch size for gradient updates
    num_gradient_updates: int = 16  # Gradient updates per rollout (mu)

    # Dynamic sampling
    use_dynamic_sampling: bool = True  # Enable dynamic sampling filter
    min_accuracy_threshold: float = 0.0  # Filter if accuracy <= this
    max_accuracy_threshold: float = 1.0  # Filter if accuracy >= this
    dynamic_sampling_buffer_size: int = 1024  # Buffer for filtered samples

    # Overlong reward shaping
    use_overlong_shaping: bool = True
    max_generation_length: int = 20480  # L_max (tokens)
    overlong_cache_length: int = 4096  # L_cache (soft penalty interval)
    overlong_penalty: float = -1.0  # Penalty for truncated sequences

    # Token-level loss normalization
    use_token_level_loss: bool = True

    # No KL penalty in DAPO (model can diverge for reasoning)
    beta: float = 0.0
    use_reference_model: bool = False

    # Training parameters from paper
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant"  # DAPO uses constant LR
    temperature: float = 1.0
    top_p: float = 0.7

    # vLLM configuration (5-20x faster generation)
    use_vllm: bool = False  # Enable vLLM for generation
    vllm_gpu_memory_utilization: float = 0.85
    vllm_tensor_parallel_size: int = 1
    vllm_enable_prefix_caching: bool = True

    # Evaluation
    eval_repeats: int = 32

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> "DAPOConfig":
        """Create DAPO config from standard training config"""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        if "group_size" in kwargs:
            config_dict["num_generations"] = kwargs["group_size"]
        return cls(**config_dict)


class DAPOModelManager:
    """Manages model loading for DAPO training"""

    def __init__(self, config: DAPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load model and tokenizer with optional LoRA"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch.float16 if self.config.fp16 else (
                torch.bfloat16 if self.config.bf16 else torch.float32
            ),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **model_kwargs
        )

        # Add LoRA adapters if configured
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
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
        self.buffer: List[Dict[str, Any]] = []

    def should_include(self, accuracy: float) -> bool:
        """Check if sample should be included based on accuracy"""
        return self.min_accuracy < accuracy < self.max_accuracy

    def add_sample(self, sample: Dict[str, Any], accuracy: float) -> bool:
        """
        Add sample to buffer if it passes the accuracy filter.

        Returns True if sample was added.
        """
        if self.should_include(accuracy):
            self.buffer.append(sample)
            return True
        return False

    def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
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
        reward_fn: Callable[[str, str], float],
        verifier_fn: Optional[Callable[[str, str], bool]] = None,
    ):
        # Ensure transformers is loaded for scheduler
        _load_transformers_dapo()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.verifier_fn = verifier_fn  # For binary correctness (e.g., math)
        self.device = next(model.parameters()).device

        # vLLM generator for fast generation
        self.vllm_generator: Optional[VLLMGenerator] = None
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
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
                self.optimizer, factor=1.0, total_iters=config.num_episodes * config.num_epochs
            )

        # Metrics
        self.metrics_history = {
            "policy_loss": [],
            "average_reward": [],
            "accuracy": [],
            "filtered_ratio": [],
            "avg_sequence_length": [],
        }

        self.global_step = 0

    def _setup_vllm(self):
        """Setup vLLM generator"""
        logger.info("Setting up vLLM for fast DAPO generation...")

        # Load vLLM backend lazily
        if not _load_vllm_backend():
            logger.warning("vLLM backend failed to load, will use HuggingFace fallback")
            return

        vllm_config = VLLMConfig(
            model_name=self.config.model_name,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            enable_prefix_caching=self.config.vllm_enable_prefix_caching,
            max_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            dtype="bfloat16" if self.config.bf16 else "float16",
        )

        self.vllm_generator = VLLMGenerator(vllm_config)

    async def initialize_vllm(self) -> bool:
        """Initialize vLLM engine"""
        if self.vllm_generator is None:
            return False

        if self._vllm_initialized:
            return True

        try:
            success = await self.vllm_generator.initialize()
            self._vllm_initialized = success
            if success:
                logger.info("vLLM initialized for DAPO - generation will be 5-20x faster!")
            return success
        except Exception as e:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    ) -> List[Dict[str, Any]]:
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

    async def _generate_with_vllm(self, prompt: str) -> List[Dict[str, Any]]:
        """Generate responses using vLLM (5-20x faster)"""
        try:
            # Generate all responses in a single batched call
            grouped_results = await self.vllm_generator.generate_groups(
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

                responses.append({
                    "response": result.response,
                    "input_ids": full_ids,
                    "attention_mask": torch.ones_like(full_ids),
                    "response_mask": response_mask,
                    "sequence_length": result.sequence_length,
                    "prompt_length": prompt_length,
                    "token_logprobs": result.token_logprobs,  # Already computed!
                    "cumulative_logprob": result.cumulative_logprob,
                })

            logger.debug(f"vLLM generated {len(responses)} responses for DAPO")
            return responses

        except Exception as e:
            logger.warning(f"vLLM generation failed: {e}. Falling back to HuggingFace.")
            return await self._generate_with_hf(prompt)

    async def _generate_with_hf(self, prompt: str) -> List[Dict[str, Any]]:
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

                responses.append({
                    "response": response_text,
                    "input_ids": full_ids,
                    "attention_mask": torch.ones_like(full_ids),
                    "response_mask": response_mask,
                    "sequence_length": response_length,
                    "prompt_length": prompt_length,
                })

        self.model.train()
        return responses

    def compute_group_accuracy(
        self,
        prompt: str,
        responses: List[Dict[str, Any]],
    ) -> float:
        """Compute accuracy for a group of responses"""
        if self.verifier_fn is None:
            # If no verifier, use reward threshold
            correct = sum(
                1 for r in responses
                if self.reward_fn(prompt, r["response"]) > 0.5
            )
        else:
            correct = sum(
                1 for r in responses
                if self.verifier_fn(prompt, r["response"])
            )
        return correct / len(responses)

    async def collect_samples_with_dynamic_sampling(
        self,
        prompts: List[str],
        target_batch_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Collect samples using dynamic sampling.

        Continues generating until we have enough non-trivial samples
        (accuracy not 0 or 1).
        """
        collected_samples = []
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
                base_reward = self.reward_fn(prompt, resp["response"])

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
        logger.debug(f"Dynamic sampling: {prompts_filtered}/{prompts_processed} filtered ({filter_ratio:.1%})")

        return collected_samples, filter_ratio

    async def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
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
        all_rewards = []
        all_accuracies = []
        all_seq_lengths = []
        num_updates = 0

        # Process samples
        for sample in samples:
            prompt = sample["prompt"]
            responses = sample["responses"]
            advantages = sample["advantages"]

            all_rewards.extend(sample["rewards"])
            all_accuracies.append(sample["accuracy"])
            all_seq_lengths.extend([r["sequence_length"] for r in responses])

            # Prepare batch
            max_len = max(len(r["input_ids"]) for r in responses)
            batch_size = len(responses)

            batch_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            batch_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
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
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # Update
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                num_updates += 1

        self.global_step += 1

        # Compute metrics
        metrics = {
            "policy_loss": total_loss / max(num_updates, 1),
            "average_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "accuracy": np.mean(all_accuracies) if all_accuracies else 0.0,
            "filtered_ratio": filter_ratio,
            "avg_sequence_length": np.mean(all_seq_lengths) if all_seq_lengths else 0.0,
            "learning_rate": self.config.learning_rate,
            "global_step": self.global_step,
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


async def train_with_dapo(
    model_name: str,
    reward_fn: Callable[[str, str], float],
    train_prompts: List[str],
    config: Optional[DAPOConfig] = None,
    verifier_fn: Optional[Callable[[str, str], bool]] = None,
    output_dir: str = "./outputs/dapo",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, List[float]]]:
    """
    Train a model using DAPO algorithm.

    DAPO is optimized for long chain-of-thought reasoning tasks like math.

    Args:
        model_name: HuggingFace model name or path
        reward_fn: Function (prompt, response) -> reward
        train_prompts: List of training prompts
        config: DAPO configuration
        verifier_fn: Optional binary verifier (prompt, response) -> correct
        output_dir: Directory to save checkpoints
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name

    Returns:
        Tuple of (model, tokenizer, metrics_history)
    """
    logger.info("=" * 60)
    logger.info("DAPO Training - Decoupled Clip and Dynamic Sampling")
    logger.info("=" * 60)
    logger.info("Key techniques: Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Shaping")

    # Create config if not provided
    if config is None:
        config = DAPOConfig(
            model_name=model_name,
            output_dir=output_dir,
        )

    # Initialize W&B
    if use_wandb and wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"dapo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["dapo", "rl-training", "reasoning"],
        )

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model_manager = DAPOModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    # Create trainer
    trainer = DAPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        verifier_fn=verifier_fn,
    )

    # Initialize vLLM if configured
    if config.use_vllm:
        logger.info("Initializing vLLM for fast DAPO generation...")
        vllm_success = await trainer.initialize_vllm()
        if vllm_success:
            logger.info("vLLM initialized - generation will be 5-20x faster!")
        else:
            logger.warning("vLLM initialization failed - using HuggingFace generation")

    # Training loop
    logger.info(f"Starting training with {len(train_prompts)} prompts")
    logger.info(f"Group size: {config.group_size}")
    logger.info(f"Clip range: [{1-config.clip_eps_low:.2f}, {1+config.clip_eps_high:.2f}]")
    logger.info(f"Dynamic sampling: {config.use_dynamic_sampling}")

    os.makedirs(output_dir, exist_ok=True)

    for iteration in range(config.num_episodes):
        # Sample batch
        batch_size = min(config.per_device_train_batch_size, len(train_prompts))
        batch_indices = np.random.choice(len(train_prompts), batch_size, replace=False)
        batch_prompts = [train_prompts[i] for i in batch_indices]

        # Train step
        metrics = await trainer.train_step(batch_prompts)

        # Log
        if iteration % config.logging_steps == 0:
            logger.info(
                f"Iter {iteration}/{config.num_episodes} | "
                f"Loss: {metrics['policy_loss']:.4f} | "
                f"Reward: {metrics['average_reward']:.4f} | "
                f"Acc: {metrics['accuracy']:.2%} | "
                f"Filtered: {metrics['filtered_ratio']:.1%}"
            )

            if use_wandb:
                wandb.log(metrics, step=iteration)

        # Save checkpoint
        if (iteration + 1) % config.save_steps == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{iteration + 1}")
            trainer.save_checkpoint(checkpoint_dir)

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    trainer.save_checkpoint(final_dir)

    if use_wandb:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("DAPO Training Complete!")
    logger.info("=" * 60)

    return model, tokenizer, trainer.metrics_history


# Convenience function for math/reasoning tasks
async def train_reasoning_with_dapo(
    model_name: str,
    math_problems: List[Dict[str, str]],  # [{"problem": ..., "answer": ...}]
    output_dir: str = "./outputs/dapo-math",
    **kwargs,
) -> Tuple[Any, Any, Dict]:
    """
    Train a reasoning model using DAPO.

    Designed for math problems with verifiable answers.
    """
    # Extract prompts
    prompts = [p["problem"] for p in math_problems]
    answers = {p["problem"]: p["answer"] for p in math_problems}

    # Create verifier that checks final answer
    def verifier(prompt: str, response: str) -> bool:
        expected = answers.get(prompt, "")
        # Simple check - in practice use more sophisticated extraction
        return expected.lower() in response.lower()

    # Create reward function
    def reward_fn(prompt: str, response: str) -> float:
        if verifier(prompt, response):
            return 1.0
        return 0.0

    config = DAPOConfig(
        model_name=model_name,
        output_dir=output_dir,
        use_dynamic_sampling=True,
        use_overlong_shaping=True,
        use_token_level_loss=True,
        **kwargs,
    )

    return await train_with_dapo(
        model_name=model_name,
        reward_fn=reward_fn,
        train_prompts=prompts,
        config=config,
        verifier_fn=verifier,
        output_dir=output_dir,
    )


# Export
__all__ = [
    "DAPOConfig",
    "DAPOTrainer",
    "DAPORewardShaper",
    "DynamicSamplingBuffer",
    "train_with_dapo",
    "train_reasoning_with_dapo",
]
