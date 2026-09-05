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
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

import numpy as np
import torch

from . import objectives, rl_losses
from .dapo_config import DAPOConfig
from .trainer_runtime import (
    SharedModelManager,
    build_group_batch,
    hf_generate_group,
    save_checkpoint_artifacts,
)

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


class DAPOModelManager(SharedModelManager):
    """Manages model loading for DAPO training"""

    def __init__(self, config: DAPOConfig):
        super().__init__(config)

    def _get_transformers(self) -> tuple[Any, Any]:
        _require_transformers_dapo()
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers exports are unavailable for DAPO")
        return AutoTokenizer, AutoModelForCausalLM

    def _peft_components(self) -> tuple[Any, Any, Any]:
        _require_peft()
        return LoraConfig, TaskType, get_peft_model


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

    def clear(self) -> None:
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
        # Parsed once here rather than per forward pass.
        self._logprob_dtype = rl_losses.resolve_logprob_dtype(
            getattr(config, "logprob_dtype", None)
        )
        # The DAPO objective (Clip-Higher, token- or sequence-level
        # aggregation) as one declarative PolicyObjective.
        self._objective = objectives.resolve_objective(
            config,
            "dapo",
            max_completion_length=int(config.max_completion_length),
            supported_ratios=("token", "sequence", "sequence_token"),
            clip_low=float(config.clip_eps_low),
            clip_high=float(config.clip_eps_high),
            aggregate="token_mean" if config.use_token_level_loss else "seq_mean",
        )
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
        # Exact completion count used by measured benchmark evidence. This
        # includes groups later rejected by dynamic sampling.
        self.rollout_samples_total = 0

    async def _compute_reward(self, prompt: str, response: str) -> float:
        """Support sync or async reward callables."""
        reward = self.reward_fn(prompt, response)
        if asyncio.iscoroutine(reward):
            reward_value = await cast(Awaitable[float], reward)
        else:
            reward_value = cast(float, reward)
        return float(reward_value)

    def _setup_vllm(self) -> None:
        """Setup vLLM generator"""
        logger.info("Setting up vLLM for fast DAPO generation...")

        # Load vLLM backend lazily
        if not _load_vllm_backend():
            logger.warning("vLLM backend failed to load, will use HuggingFace fallback")
            return

        vllm_config_cls = VLLMConfig
        vllm_generator_cls = VLLMGenerator
        if vllm_config_cls is None or vllm_generator_cls is None:
            logger.warning(
                "vLLM backend exports unavailable, using HuggingFace fallback"
            )
            return

        vllm_config = vllm_config_cls(
            model_name=self.config.model_name,
            revision=getattr(self.config, "model_revision", None),
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
        masked_log_probs, shift_response_mask = rl_losses.gather_token_logprobs(
            outputs.logits, input_ids, response_mask, dtype=self._logprob_dtype
        )
        return masked_log_probs, shift_response_mask.sum(dim=-1)

    def compute_importance_ratio(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token importance ratio.

        r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
            = exp(log_pi_theta - log_pi_theta_old)

        The log-ratio is clamped before exponentiating (see
        ``rl_losses.safe_exp_ratio``) so a single wildly off-policy token
        cannot overflow to inf and make the whole batch's loss non-finite.
        """
        return rl_losses.safe_exp_ratio(current_log_probs - old_log_probs)

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Advantages for one group of rewards [group_size] using the
        configured objective's estimator (group-normalised by default).

        Groups of size 1 (or constant rewards) yield zeros rather than NaN.
        """
        group_ids = torch.zeros(
            rewards.numel(), dtype=torch.long, device=rewards.device
        )
        objective = getattr(self, "_objective", None) or objectives.OBJECTIVES["dapo"]
        return objectives.compute_advantages(rewards, group_ids, objective)

    def compute_dapo_loss_from_log_probs(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """DAPO objective from per-token log-probs via ``objectives.policy_loss``.

        ``advantages`` is one value per sequence ``[G]`` (broadcast over
        tokens) or per token ``[G, T]``.
        """
        result = objectives.policy_loss(
            logp_cur=current_log_probs,
            mask=response_mask,
            advantages=advantages,
            objective=self._objective,
            logp_old=old_log_probs,
        )
        return result.loss, result.metrics

    def compute_dapo_loss(
        self,
        importance_ratios: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DAPO loss with Clip-Higher and token-level normalization.

        L = -(1/sum(|o_i|)) * sum_i sum_t min(r_t * A, clip(r_t, 1-eps_low, 1+eps_high) * A)

        Ratio-based entry point kept for callers holding ratios; ``train_step``
        uses :meth:`compute_dapo_loss_from_log_probs`. Both evaluate the same
        :class:`~stateset_agents.training.objectives.PolicyObjective`.
        """
        per_token = objectives.surrogate(
            self._objective, importance_ratios, advantages, importance_ratios
        )
        return objectives.aggregate(self._objective, per_token, response_mask)

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
            responses = await self._generate_with_vllm(prompt)
        else:
            # Fallback to HuggingFace generation
            responses = await self._generate_with_hf(prompt)
        self.rollout_samples_total += len(responses)
        return responses

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
        responses: list[dict[str, Any]] = await hf_generate_group(
            self.model,
            self.tokenizer,
            self.config,
            self.device,
            prompt,
            self.config.group_size,
        )
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

    def _build_batch_tensors(
        self, responses: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad a group of response dicts into batch tensors."""
        batch_input_ids, batch_attention_mask, batch_response_mask = build_group_batch(
            responses, self.device
        )
        assert batch_response_mask is not None
        return batch_input_ids, batch_attention_mask, batch_response_mask

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
            advantages = self.compute_group_advantages(rewards_tensor)

            # Freeze old-policy log probs at rollout time (before any inner updates
            # mutate the model), so importance ratios are computed against the
            # policy that actually generated these responses.
            (
                batch_input_ids,
                batch_attention_mask,
                batch_response_mask,
            ) = self._build_batch_tensors(group_responses)
            with torch.no_grad():
                old_token_log_probs, _ = self.compute_token_log_probs(
                    batch_input_ids, batch_attention_mask, batch_response_mask
                )

            # Store sample
            sample = {
                "prompt": prompt,
                "responses": group_responses,
                "rewards": rewards,
                "advantages": advantages,
                "accuracy": accuracy,
                "old_token_log_probs": old_token_log_probs,
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
            (
                batch_input_ids,
                batch_attention_mask,
                batch_response_mask,
            ) = self._build_batch_tensors(responses)

            # Old log probs: prefer the ones frozen at rollout time (before any
            # inner updates could have moved the policy). Fall back to computing
            # them here once, before the inner loop, for backward compatibility
            # with callers that don't populate this key.
            old_token_log_probs = sample.get("old_token_log_probs")
            if old_token_log_probs is None:
                with torch.no_grad():
                    old_token_log_probs, _ = self.compute_token_log_probs(
                        batch_input_ids, batch_attention_mask, batch_response_mask
                    )
            old_token_log_probs = old_token_log_probs.detach()

            # Multiple gradient updates per rollout (mu updates)
            for _ in range(max(1, self.config.num_gradient_updates)):
                # Compute current log probs
                current_token_log_probs, token_counts = self.compute_token_log_probs(
                    batch_input_ids, batch_attention_mask, batch_response_mask
                )

                # Clip-Higher surrogate + aggregation through the objective.
                loss, _objective_metrics = self.compute_dapo_loss_from_log_probs(
                    current_token_log_probs,
                    old_token_log_probs,
                    advantages,
                    batch_response_mask[:, 1:],  # Shift for next-token prediction
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                _params = list(self.model.parameters())
                if _params:
                    torch.nn.utils.clip_grad_norm_(_params, self.config.max_grad_norm)

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
            "avg_sequence_length": (
                float(np.mean(all_seq_lengths)) if all_seq_lengths else 0.0
            ),
            "learning_rate": float(self.config.learning_rate),
            "global_step": float(self.global_step),
        }

        # Store metrics
        for key in ["policy_loss", "average_reward", "accuracy", "filtered_ratio"]:
            if key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])

        return metrics

    def save_checkpoint(self, output_dir: str) -> None:
        """Save model checkpoint"""
        save_checkpoint_artifacts(
            self.model,
            self.tokenizer,
            output_dir,
            training_state={
                "global_step": self.global_step,
                "rollout_samples_total": self.rollout_samples_total,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics_history": self.metrics_history,
            },
            config_dict=self.config.to_dict(),
            config_filename="dapo_config.json",
        )


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
