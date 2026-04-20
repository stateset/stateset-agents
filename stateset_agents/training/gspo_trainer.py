"""
Group Sequence Policy Optimization (GSPO) Training for StateSet Agents

This module implements GSPO, a stable and efficient RL algorithm for training
large language models. GSPO uses sequence-level importance ratios and performs
sequence-level clipping, rewarding, and optimization.

Reference: https://arxiv.org/abs/2507.18071v2
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Import framework components (absolute imports so legacy `training.*` works)
from stateset_agents.core.agent import Agent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.exceptions import (
    ATTRIBUTE_VALUE_EXCEPTIONS as INPUT_GRADS_EXCEPTIONS,
)
from stateset_agents.exceptions import GPU_EXCEPTIONS as GPU_PROPERTIES_EXCEPTIONS
from stateset_agents.exceptions import (
    IMPORT_EXCEPTIONS as BITSANDBYTES_IMPORT_EXCEPTIONS,
)
from stateset_agents.exceptions import MODEL_IO_EXCEPTIONS as MODEL_LOAD_EXCEPTIONS
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction as MultiObjectiveReward,
)

from .gspo_config import GSPOConfig
from .gspo_generation import GSPOTrajectoryGenerator, VLLM_AVAILABLE, _get_model_device

logger = logging.getLogger(__name__)

# Optional training dependencies
try:
    import wandb as _wandb
    wandb: Any | None = _wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def _require_peft() -> None:
    """Ensure PEFT is available before using LoRA/quantization features."""
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError(
            "PEFT is required for GSPO LoRA/quantized training. "
            "Install with `pip install stateset-agents[training]` or `pip install peft`."
        )


def _require_bitsandbytes() -> None:
    """Ensure bitsandbytes is importable when k-bit loading is requested."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit/8-bit quantization requires CUDA. "
            "Disable `use_4bit/use_8bit` or run on a CUDA-enabled machine."
        )

    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "bitsandbytes is required for 4-bit/8-bit quantization. "
            "Install with `pip install stateset-agents[trl]` or `pip install bitsandbytes`."
        )

    try:
        import bitsandbytes  # noqa: F401  # type: ignore[import-not-found]
    except BITSANDBYTES_IMPORT_EXCEPTIONS as exc:  # pragma: no cover
        raise ImportError(
            "bitsandbytes is installed but failed to import. "
            "If you just installed it in a notebook, restart the runtime/kernel. "
            "Otherwise, verify your CUDA/PyTorch/bitsandbytes compatibility."
        ) from exc


def _require_wandb() -> None:
    """Ensure Weights & Biases is available before logging."""
    if wandb is None:
        raise ImportError(
            "wandb is required for GSPO logging. "
            "Install with `pip install stateset-agents[training]` or `pip install wandb`."
        )


# Lazy import transformers to avoid torch/torchvision compatibility issues
_transformers_loaded = False
AutoModelForCausalLM: Any | None = None
AutoTokenizer: Any | None = None
TrainingArguments: Any | None = None
get_cosine_schedule_with_warmup: Any | None = None


def _load_transformers() -> bool:
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_loaded, AutoModelForCausalLM, AutoTokenizer
    global TrainingArguments, get_cosine_schedule_with_warmup
    if _transformers_loaded:
        return True
    # Allow tests/consumers to pre-inject mocks without importing transformers.
    if AutoModelForCausalLM is not None and AutoTokenizer is not None:
        _transformers_loaded = True
        return True
    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import TrainingArguments as _TrainingArguments
        from transformers import get_cosine_schedule_with_warmup as _get_cosine

        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        TrainingArguments = _TrainingArguments
        get_cosine_schedule_with_warmup = _get_cosine
        _transformers_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


def _enable_input_require_grads(model: Any) -> None:
    """
    Ensure at least one checkpoint input requires gradients.

    Some Transformer implementations use `torch.utils.checkpoint` internally for
    `gradient_checkpointing`. With PEFT/LoRA, the base model weights are often
    frozen, so hidden states may not require grad unless explicitly enabled.
    """
    if getattr(model, "_stateset_input_grads_enabled", False):
        return

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
            model._stateset_input_grads_enabled = True
            return
        except INPUT_GRADS_EXCEPTIONS as e:  # pragma: no cover
            logger.debug("enable_input_require_grads failed: %s", e)

    try:
        if not hasattr(model, "get_input_embeddings"):
            return
        embeddings = model.get_input_embeddings()
        if embeddings is None:
            return

        def _require_grads_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, torch.Tensor):
                return output.requires_grad_(True)
            return output

        embeddings.register_forward_hook(_require_grads_hook)
        model._stateset_input_grads_enabled = True
    except INPUT_GRADS_EXCEPTIONS as e:  # pragma: no cover
        logger.debug("Failed to register input grad hook: %s", e)


class GSPOModelManager:
    """Manages model loading and LoRA configuration for GSPO training"""

    def __init__(self, config: GSPOConfig):
        self.config = config
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.ref_model: Any | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load model and tokenizer with LoRA if specified"""
        logger.info(f"Loading model: {self.config.model_name}")

        model_name_lower = self.config.model_name.lower()
        if (
            torch.cuda.is_available()
            and "qwen" in model_name_lower
            and any(size in model_name_lower for size in ["3b", "7b"])
            and not (self.config.use_4bit or self.config.use_8bit)
        ):
            try:
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                if total_mem_gb < 24:
                    logger.warning(
                        "Model %s on %.1fGB GPU: consider `use_4bit=True` to reduce memory usage",
                        self.config.model_name,
                        total_mem_gb,
                    )
            except GPU_PROPERTIES_EXCEPTIONS:  # pragma: no cover
                logger.warning(
                    "Model %s: consider `use_4bit=True` to reduce memory usage",
                    self.config.model_name,
                )

        # Load transformers lazily
        if not _load_transformers():
            raise ImportError("transformers is required but failed to load")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers GSPO loader did not initialize correctly")

        try:
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
                "torch_dtype": torch.float16
                if self.config.fp16
                else (torch.bfloat16 if self.config.bf16 else torch.float32),
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True,
            }

            # PEFT is required for LoRA or k-bit training features
            if self.config.use_lora or self.config.use_8bit or self.config.use_4bit:
                _require_peft()

            # Add quantization if specified
            if self.config.use_8bit:
                _require_bitsandbytes()
                model_kwargs["load_in_8bit"] = True
            elif self.config.use_4bit:
                _require_bitsandbytes()
                model_kwargs["load_in_4bit"] = True

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )

            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                if hasattr(base_model, "config") and hasattr(
                    base_model.config, "use_cache"
                ):
                    base_model.config.use_cache = False
                if hasattr(base_model, "gradient_checkpointing_enable"):
                    base_model.gradient_checkpointing_enable()
                else:  # pragma: no cover
                    logger.warning(
                        "gradient_checkpointing requested but model does not support it"
                    )
                if self.config.use_lora and not (
                    self.config.use_8bit or self.config.use_4bit
                ):
                    _enable_input_require_grads(base_model)

            # Prepare model for training if using quantization
            if self.config.use_8bit or self.config.use_4bit:
                base_model = prepare_model_for_kbit_training(base_model)

            # Add LoRA adapters
            if self.config.use_lora:
                # Determine target modules based on model architecture
                if self.config.lora_target_modules:
                    target_modules = self.config.lora_target_modules
                elif "gpt2" in self.config.model_name.lower():
                    target_modules = ["c_attn", "c_proj"]
                elif (
                    "llama" in self.config.model_name.lower()
                    or "qwen" in self.config.model_name.lower()
                ):
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                else:
                    target_modules = ["q_proj", "v_proj"]

                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )

                self.model = get_peft_model(base_model, lora_config)
                if self.model is not None:
                    self.model.print_trainable_parameters()

                logger.info("LoRA adapters added to model")
            else:
                self.model = base_model

            # Load reference model if using KL penalty
            if self.config.beta > 0 and self.config.use_reference_model:
                logger.info("Loading reference model for KL penalty...")
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=model_kwargs["torch_dtype"],
                    device_map="auto" if torch.cuda.device_count() > 1 else None,
                )
                if self.ref_model is not None:
                    self.ref_model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.tokenizer

        except MODEL_LOAD_EXCEPTIONS as e:
            logger.error(f"Failed to load model: {e}")
            raise

class GSPOTrainer:
    """
    Group Sequence Policy Optimization trainer

    Implements the GSPO algorithm from: https://arxiv.org/abs/2507.18071v2
    """

    def __init__(
        self,
        config: GSPOConfig,
        model: Any,
        tokenizer: Any,
        agent: Agent,
        environment: ConversationEnvironment,
        reward_model: MultiObjectiveReward,
        ref_model: Any | None = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.agent = agent
        self.environment = environment
        self.reward_model = reward_model
        self.ref_model = ref_model

        # Optimizer — stub models have no parameters; use a dummy param.
        params = list(self.model.parameters())
        if not params:
            self._stub_param = torch.nn.Parameter(torch.zeros(1))
            params = [self._stub_param]
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate)

        # Scheduler — fallback to constant LR when transformers unavailable.
        total_steps = max(
            1, int(config.num_outer_iterations) * int(config.num_iterations)
        )
        warmup_steps = int(total_steps * float(config.warmup_ratio))
        _load_transformers()
        if get_cosine_schedule_with_warmup is not None:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda step: 1.0
            )

        # Trajectory generator
        self.generator = GSPOTrajectoryGenerator(config, agent, environment)

        # Metrics
        self.training_metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "clipping_fraction": [],
            "average_reward": [],
            "sequence_importance_ratio": [],
        }

    def compute_sequence_importance_ratio(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sequence-level importance ratio with length normalization.

        s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
               = exp(1/|y_i| * Σ log(π_θ(y_i,t|x,y_<t) / π_θ_old(y_i,t|x,y_<t)))

        Args:
            current_log_probs: Log probabilities under current policy (sum over sequence)
            old_log_probs: Log probabilities under old policy (sum over sequence)
            sequence_lengths: Length of each sequence

        Returns:
            Sequence importance ratios
        """
        # Compute log ratio sum
        log_ratio_sum = current_log_probs - old_log_probs

        # Apply length normalization
        normalized_log_ratio = log_ratio_sum / sequence_lengths

        # Convert to ratio via exp
        importance_ratio = torch.exp(normalized_log_ratio)

        return importance_ratio

    def compute_group_advantages(
        self, rewards: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute group-relative advantages (normalized rewards).

        Â_i = (r(x, y_i) - mean(rewards)) / std(rewards)

        Args:
            rewards: Rewards for a group of responses [group_size]

        Returns:
            advantages: Normalized advantages
            stats: Statistics about rewards
        """
        rewards_tensor = (
            rewards
            if isinstance(rewards, torch.Tensor)
            else torch.as_tensor(rewards, dtype=torch.float32)
        )
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std(unbiased=False)

        # Avoid division by zero / NaNs (e.g., group_size == 1)
        if torch.isnan(std_reward) or std_reward < 1e-8:
            std_reward_safe = std_reward.new_tensor(1.0)
        else:
            std_reward_safe = std_reward

        advantages = (rewards_tensor - mean_reward) / std_reward_safe

        stats = {
            "mean_reward": float(mean_reward.item()),
            "std_reward": float(std_reward.item()),
            "max_reward": float(rewards_tensor.max().item()),
            "min_reward": float(rewards_tensor.min().item()),
        }

        return advantages, stats

    def _compute_group_sequence_log_probs(
        self, prompt: str, responses: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-sequence log probabilities for response tokens (with gradients).

        Returns:
            sequence_log_probs: Sum of response token log probs [group_size]
            response_lengths: Number of response tokens used for normalization [group_size]
        """
        full_texts = [f"{prompt} {response}" for response in responses]

        # Use right-padding for stable response_start_idx across the batch.
        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"

        try:
            inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length
                + self.config.max_completion_length,
                add_special_tokens=False,
            )
        finally:
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = original_padding_side

        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
            add_special_tokens=False,
        )
        prompt_length = int(prompt_tokens["input_ids"].shape[1])
        response_start_idx = max(prompt_length - 1, 0)

        model_device = _get_model_device(self.model)
        if model_device is not None:
            if hasattr(inputs, "to"):
                inputs = inputs.to(model_device)
            else:
                inputs = {
                    k: v.to(model_device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

        outputs = self.model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()
        shift_mask = inputs["attention_mask"][:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        response_mask = torch.zeros_like(shift_mask)
        if response_start_idx < response_mask.shape[1]:
            response_mask[:, response_start_idx:] = shift_mask[:, response_start_idx:]

        sequence_log_probs = (token_log_probs * response_mask).sum(dim=-1)
        response_lengths = (
            response_mask.sum(dim=-1).to(dtype=torch.float32).clamp(min=1.0)
        )

        return sequence_log_probs, response_lengths

    async def train_step(
        self, queries: list[str | dict[str, Any]], num_groups: int = 1
    ) -> dict[str, float]:
        """
        Execute one GSPO training step.

        Args:
            queries: List of prompts/queries (strings or dicts with prompt/context)
            num_groups: Number of query groups to process

        Returns:
            Training metrics
        """
        self.model.train()
        model_device = _get_model_device(self.model)

        total_loss = torch.tensor(0.0, device=model_device)
        total_clipped = 0
        total_samples = 0
        all_rewards = []
        all_importance_ratios = []

        for query in queries[:num_groups]:
            if isinstance(query, dict):
                prompt = str(query.get("prompt", ""))
                query_context = query.get("context")
                if not isinstance(query_context, dict):
                    query_context = {}
            else:
                prompt = str(query)
                query_context = {}
            prompt_value = prompt
            query_context_value = dict(query_context)

            # Generate group of responses for this query
            group_responses = await self.generator.generate_group_responses(
                prompt, self.config.num_generations
            )

            # Extract responses and old log probs
            responses = [resp for resp, _ in group_responses]
            old_log_probs = torch.tensor(
                [log_prob for _, log_prob in group_responses],
                dtype=torch.float32,
                device=model_device,
            )

            # Compute rewards for each response in parallel
            async def _compute_reward_for_response(
                resp: str,
                *,
                _prompt: str = prompt_value,
                _query_context: dict[str, Any] = query_context_value,
            ) -> float:
                turn = ConversationTurn(
                    role="assistant", content=resp, metadata={"generated": True}
                )
                reward_context = {"user_query": _prompt}
                reward_context.update(_query_context)
                reward_info = await self.reward_model.compute_reward(
                    trajectory=None,
                    turn=turn,
                    context=reward_context,
                )
                return float(reward_info.total_reward)

            rewards = list(
                await asyncio.gather(
                    *[_compute_reward_for_response(r) for r in responses]
                )
            )

            rewards_tensor = torch.tensor(
                rewards, dtype=torch.float32, device=model_device
            )
            all_rewards.extend(rewards)

            # Compute group advantages
            advantages, reward_stats = self.compute_group_advantages(rewards_tensor)

            # Compute current log probs (with gradients) for each response
            (
                current_log_probs,
                sequence_lengths,
            ) = self._compute_group_sequence_log_probs(prompt, responses)
            old_log_probs = old_log_probs.to(current_log_probs.device)

            # Compute sequence importance ratios
            importance_ratios = self.compute_sequence_importance_ratio(
                current_log_probs, old_log_probs, sequence_lengths
            )
            all_importance_ratios.extend(importance_ratios.detach().cpu().tolist())

            # Apply clipping to importance ratios
            clipped_ratios = torch.clamp(
                importance_ratios,
                1 - self.config.clip_range_left,
                1 + self.config.clip_range_right,
            )

            # Count clipped sequences
            num_clipped = (importance_ratios != clipped_ratios).sum().item()
            total_clipped += num_clipped
            total_samples += len(responses)

            # Compute policy loss using GSPO objective
            # J_GSPO = E[1/G * Σ min(s_i(θ) * Â_i, clip(s_i(θ)) * Â_i)]

            # Unclipped objective
            unclipped_obj = importance_ratios * advantages

            # Clipped objective
            clipped_obj = clipped_ratios * advantages

            # Take minimum (conservative policy update)
            policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

            # Add KL penalty if specified
            if self.config.beta > 0 and self.ref_model is not None:
                # Compute KL divergence with reference model (batched for efficiency)
                ref_log_probs = self._compute_batch_ref_log_probs(prompt, responses)
                if model_device is not None:
                    ref_log_probs = ref_log_probs.to(model_device)

                # KL = log(π_θ/π_ref) = log π_θ - log π_ref
                kl_div = (current_log_probs - ref_log_probs) / sequence_lengths
                kl_penalty = self.config.beta * kl_div.mean()

                total_loss_item = policy_loss + kl_penalty
            else:
                total_loss_item = policy_loss

            # Accumulate loss
            total_loss += total_loss_item

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (skip for stub models with no real parameters)
        model_params = list(self.model.parameters())
        if model_params:
            torch.nn.utils.clip_grad_norm_(model_params, self.config.max_grad_norm)

        # Update parameters
        self.optimizer.step()
        self.scheduler.step()

        # Compute metrics
        clipping_fraction = total_clipped / max(total_samples, 1)
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        avg_importance_ratio = (
            np.mean(all_importance_ratios) if all_importance_ratios else 1.0
        )

        metrics = {
            "policy_loss": total_loss.item(),
            "clipping_fraction": clipping_fraction,
            "average_reward": avg_reward,
            "sequence_importance_ratio": avg_importance_ratio,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        # Store metrics
        for key, value in metrics.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)

        return metrics

    async def _compute_ref_log_prob(self, prompt: str, response: str) -> float:
        """Compute log probability under reference model"""
        full_text = prompt + " " + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length
            + self.config.max_completion_length,
            add_special_tokens=False,
        )

        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
            add_special_tokens=False,
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        model_device = _get_model_device(self.ref_model)
        if model_device and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        ref_model = self.ref_model
        if ref_model is None:
            raise RuntimeError("Reference model must be initialized before use")
        with torch.no_grad():
            outputs = ref_model(**inputs)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        response_start = max(prompt_length - 1, 0)
        if response_start >= token_log_probs.shape[-1]:
            return float(token_log_probs.sum().item())

        response_log_probs = token_log_probs[..., response_start:]
        sequence_log_prob = float(response_log_probs.sum().item())

        return sequence_log_prob

    def _compute_batch_ref_log_probs(
        self, prompt: str, responses: list[str]
    ) -> torch.Tensor:
        """Compute log probabilities under reference model for a batch of responses."""
        full_texts = [f"{prompt} {response}" for response in responses]

        # Batch tokenize with padding
        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"

        try:
            inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length
                + self.config.max_completion_length,
                add_special_tokens=False,
            )
        finally:
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = original_padding_side

        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
            add_special_tokens=False,
        )
        prompt_length = int(prompt_tokens["input_ids"].shape[1])
        response_start_idx = max(prompt_length - 1, 0)

        model_device = _get_model_device(self.ref_model)
        if model_device is not None:
            inputs = {
                k: v.to(model_device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        ref_model = self.ref_model
        if ref_model is None:
            raise RuntimeError("Reference model must be initialized before use")
        with torch.no_grad():
            outputs = ref_model(**inputs)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()
        shift_mask = inputs["attention_mask"][:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask for response tokens only (exclude prompt and padding)
        response_mask = torch.zeros_like(shift_mask)
        if response_start_idx < response_mask.shape[1]:
            response_mask[:, response_start_idx:] = shift_mask[:, response_start_idx:]

        sequence_log_probs = (token_log_probs * response_mask).sum(dim=-1)
        return sequence_log_probs

    def save_model(self, output_dir: str) -> None:
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training metrics
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)

        logger.info(f"Model saved to {output_dir}")


from .gspo_entrypoints import train_customer_service_with_gspo, train_with_gspo


# Backwards-compatible alias (some users expect snake_case-like class names).
GSPO_Trainer = GSPOTrainer


# Export main components
__all__ = [
    "GSPOConfig",
    "GSPOModelManager",
    "GSPOTrajectoryGenerator",
    "GSPOTrainer",
    "GSPO_Trainer",
    "VLLM_AVAILABLE",
    "train_with_gspo",
    "train_customer_service_with_gspo",
]
