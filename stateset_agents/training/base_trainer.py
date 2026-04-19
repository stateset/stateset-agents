"""
Base Trainer Infrastructure for StateSet Agents

This module provides shared infrastructure for all RL trainers (GSPO, VAPO, DAPO, etc.),
eliminating code duplication and ensuring consistent behavior across algorithms.

Key abstractions:
- BaseConfig: Common configuration fields
- BaseModelManager: Unified model/tokenizer loading with LoRA/quantization
- BaseTrainer: Shared training loop infrastructure
- BaseTrajectoryGenerator: Common generation patterns
"""

import asyncio
import importlib.util
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast
from collections.abc import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

BASE_TRAINER_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    OSError,
    asyncio.TimeoutError,
)

# Type variable for config classes
ConfigT = TypeVar("ConfigT", bound="BaseTrainerConfig")

# Lazy imports to avoid torch/torchvision compatibility issues
_transformers_loaded = False
_AutoModelForCausalLM: Any = None
_AutoTokenizer: Any = None
_get_cosine_schedule_with_warmup: Any = None
_get_constant_schedule: Any = None


def _load_transformers() -> bool:
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_loaded, _AutoModelForCausalLM, _AutoTokenizer
    global _get_cosine_schedule_with_warmup, _get_constant_schedule

    if _transformers_loaded:
        return True

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_constant_schedule,
            get_cosine_schedule_with_warmup,
        )

        _AutoModelForCausalLM = AutoModelForCausalLM
        _AutoTokenizer = AutoTokenizer
        _get_cosine_schedule_with_warmup = get_cosine_schedule_with_warmup
        _get_constant_schedule = get_constant_schedule
        _transformers_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


def _load_peft():
    """Lazily load PEFT for LoRA support."""
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        return LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        logger.warning("PEFT not available. LoRA will be disabled.")
        return None, None, None, None


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
        except BASE_TRAINER_EXCEPTIONS as e:  # pragma: no cover
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
    except BASE_TRAINER_EXCEPTIONS as e:  # pragma: no cover
        logger.debug("Failed to register input grad hook: %s", e)


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
    except (ImportError, RuntimeError, OSError) as exc:  # pragma: no cover
        raise ImportError(
            "bitsandbytes is installed but failed to import. "
            "If you just installed it in a notebook, restart the runtime/kernel. "
            "Otherwise, verify your CUDA/PyTorch/bitsandbytes compatibility."
        ) from exc


# vLLM backend lazy loading
_vllm_backend_loaded = False
_VLLMConfig: Any = None
_VLLMGenerator: Any = None
_VLLM_AVAILABLE = False


def _load_vllm_backend() -> bool:
    """Lazily load vLLM backend."""
    global _vllm_backend_loaded, _VLLMConfig, _VLLMGenerator, _VLLM_AVAILABLE

    if _vllm_backend_loaded:
        return _VLLM_AVAILABLE

    try:
        from .vllm_backend import VLLM_AVAILABLE, VLLMConfig, VLLMGenerator

        _VLLMConfig = VLLMConfig
        _VLLMGenerator = VLLMGenerator
        _VLLM_AVAILABLE = VLLM_AVAILABLE
        _vllm_backend_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load vLLM backend: {e}")
        _vllm_backend_loaded = True
        return False


@dataclass
class BaseTrainerConfig:
    """
    Base configuration shared by all RL trainers.

    This consolidates common parameters to eliminate duplication across
    GSPO, VAPO, DAPO, and other trainer configurations.
    """

    # Model identification
    model_name: str = "gpt2"

    # Generation parameters
    max_prompt_length: int = 256
    max_completion_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Training parameters
    learning_rate: float = 1e-5
    num_epochs: int = 3
    num_episodes: int = 100
    warmup_ratio: float = 0.1

    # Optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Precision
    fp16: bool = False
    bf16: bool = True

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit: bool = False
    use_4bit: bool = False

    # vLLM acceleration
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.85
    vllm_tensor_parallel_size: int = 1
    vllm_enable_prefix_caching: bool = True

    # Reference model (for KL penalty)
    beta: float = 0.0
    use_reference_model: bool = False

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./outputs"

    # W&B integration
    report_to: str = "wandb"
    wandb_project: str = "stateset-agents"
    wandb_run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BaseTrainerConfig":
        """Create config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)


class BaseModelManager:
    """
    Unified model and tokenizer management for all trainers.

    Handles:
    - Model loading with proper dtype and device mapping
    - Tokenizer setup with padding configuration
    - LoRA adapter application
    - Quantization (4-bit/8-bit)
    - Gradient checkpointing
    - Reference model loading for KL penalty
    """

    def __init__(self, config: BaseTrainerConfig):
        self.config = config
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.ref_model: Any | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load model and tokenizer with all configured optimizations."""
        logger.info(f"Loading model: {self.config.model_name}")

        if not _load_transformers():
            raise ImportError("transformers is required but failed to load")

        auto_tokenizer = _AutoTokenizer
        auto_model = _AutoModelForCausalLM
        if auto_tokenizer is None or auto_model is None:
            raise ImportError("transformers exports are unavailable")

        # Load tokenizer
        self.tokenizer = auto_tokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build model loading kwargs
        model_kwargs = self._build_model_kwargs()

        # Load base model
        base_model = auto_model.from_pretrained(self.config.model_name, **model_kwargs)

        # Enable gradient checkpointing
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

        # Prepare for quantized training
        if self.config.use_8bit or self.config.use_4bit:
            (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            ) = _load_peft()
            if prepare_model_for_kbit_training:
                base_model = prepare_model_for_kbit_training(base_model)

        # Apply LoRA
        if self.config.use_lora:
            self.model = self._apply_lora(base_model)
        else:
            self.model = base_model

        # Load reference model if needed
        if self.config.beta > 0 and self.config.use_reference_model:
            self._load_reference_model(model_kwargs)

        logger.info(f"Model loaded successfully on {self.device}")
        return self.model, self.tokenizer

    def _build_model_kwargs(self) -> dict[str, Any]:
        """Build kwargs for model loading."""
        kwargs = {
            "torch_dtype": self._get_dtype(),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }

        if self.config.use_8bit:
            _require_bitsandbytes()
            kwargs["load_in_8bit"] = True
        elif self.config.use_4bit:
            _require_bitsandbytes()
            kwargs["load_in_4bit"] = True

        return kwargs

    def _get_dtype(self) -> torch.dtype:
        """Get the appropriate dtype based on config."""
        if self.config.fp16:
            return torch.float16
        elif self.config.bf16:
            return torch.bfloat16
        return torch.float32

    def _apply_lora(self, base_model: Any) -> Any:
        """Apply LoRA adapters to the model."""
        LoraConfig, TaskType, get_peft_model, _ = _load_peft()

        if LoraConfig is None:
            logger.warning("PEFT not available, skipping LoRA")
            return base_model

        target_modules = self._get_lora_target_modules()

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA adapters applied to model")
        return model

    def _get_lora_target_modules(self) -> list[str]:
        """Determine LoRA target modules based on model architecture."""
        if self.config.lora_target_modules:
            return self.config.lora_target_modules

        model_name_lower = self.config.model_name.lower()

        if "gpt2" in model_name_lower:
            return ["c_attn", "c_proj"]
        elif "llama" in model_name_lower or "qwen" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gemma" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            return ["q_proj", "v_proj"]

    def _load_reference_model(self, model_kwargs: dict[str, Any]) -> None:
        """Load frozen reference model for KL penalty."""
        logger.info("Loading reference model for KL penalty...")
        auto_model = _AutoModelForCausalLM
        if auto_model is None:
            raise ImportError("transformers exports are unavailable")
        self.ref_model = auto_model.from_pretrained(
            self.config.model_name,
            torch_dtype=model_kwargs["torch_dtype"],
            device_map="auto" if torch.cuda.device_count() > 1 else None,
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False


class BaseTrajectoryGenerator:
    """
    Base class for trajectory generation with vLLM acceleration.

    Provides common generation patterns used by GSPO, VAPO, DAPO, etc.
    """

    def __init__(self, config: BaseTrainerConfig):
        self.config = config
        self.vllm_generator: Any | None = None
        self._vllm_initialized = False

        if config.use_vllm:
            self._setup_vllm()

    def _setup_vllm(self) -> None:
        """Setup vLLM generator if available."""
        if not _load_vllm_backend():
            logger.warning("vLLM not available, will use HuggingFace fallback")
            return

        vllm_config_cls = _VLLMConfig
        vllm_generator_cls = _VLLMGenerator
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
            dtype="float16"
            if self.config.fp16
            else ("bfloat16" if self.config.bf16 else "auto"),
        )

        self.vllm_generator = vllm_generator_cls(vllm_config)

    async def initialize_vllm(self) -> bool:
        """Initialize vLLM engine."""
        if self.vllm_generator is None:
            return False

        if self._vllm_initialized:
            return True

        try:
            success = bool(await self.vllm_generator.initialize())
            self._vllm_initialized = success
            if success:
                logger.info("vLLM initialized - 5-20x faster generation enabled!")
            return success
        except BASE_TRAINER_EXCEPTIONS as e:
            logger.warning(f"Failed to initialize vLLM: {e}")
            return False

    @property
    def using_vllm(self) -> bool:
        """Check if vLLM is active."""
        return self._vllm_initialized and self.vllm_generator is not None


class BaseTrainer(ABC, Generic[ConfigT]):
    """
    Abstract base class for all RL trainers.

    Provides common functionality:
    - Optimizer and scheduler setup
    - Metrics tracking
    - Checkpoint management
    - W&B integration
    - Training loop infrastructure
    """

    def __init__(
        self,
        config: ConfigT,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.device = next(model.parameters()).device

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Metrics tracking
        self.metrics_history: dict[str, list[float]] = {
            "policy_loss": [],
            "average_reward": [],
        }
        self.global_step = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with config parameters."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        _load_transformers()
        scheduler_factory = _get_cosine_schedule_with_warmup
        if scheduler_factory is None:
            raise ImportError("transformers scheduler exports are unavailable")

        total_steps = self.config.num_episodes * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        return scheduler_factory(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def compute_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities.

        This is a common operation across all RL trainers.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            response_mask: Optional mask for response tokens only

        Returns:
            token_log_probs: Per-token log probs [batch, seq_len-1]
            sequence_log_probs: Sum of log probs per sequence [batch]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Apply response mask if provided
        if response_mask is not None:
            shift_mask = response_mask[:, 1:].contiguous()
            token_log_probs = token_log_probs * shift_mask

        # Compute sequence log probs
        sequence_log_probs = token_log_probs.sum(dim=-1)

        return token_log_probs, sequence_log_probs

    def compute_kl_divergence(
        self,
        current_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policies.

        Args:
            current_logits: Logits from current policy
            reference_logits: Logits from reference policy
            mask: Optional mask for valid positions

        Returns:
            KL divergence (scalar)
        """
        # KL(π_θ || π_ref) = Σ π_θ(x) * [log π_θ(x) - log π_ref(x)]
        # Previous code used F.kl_div(current_log, ref_probs) which computes
        # KL(ref || current) — the WRONG direction for RL policy gradient.
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)

        # Manual forward-KL: π_θ * (log π_θ - log π_ref)
        current_probs = current_log_probs.exp()
        kl = (current_probs * (current_log_probs - reference_log_probs)).sum(dim=-1)

        if mask is not None:
            kl = kl * mask
            return kl.sum() / mask.sum().clamp(min=1)

        return kl.mean()

    def clip_gradients(self) -> float:
        """Clip gradients and return the norm."""
        params = list(self.model.parameters())
        if not params:
            return 0.0
        return float(
            torch.nn.utils.clip_grad_norm_(
            params,
            self.config.max_grad_norm,
            ).item()
        )

    def get_gradient_stats(self) -> dict[str, float]:
        """Compute gradient statistics for monitoring.

        Returns dict with grad_norm, grad_max, grad_mean, and num_zero_grads.
        Useful for detecting vanishing/exploding gradients and dead parameters.
        """
        total_norm = 0.0
        max_grad = 0.0
        grad_sum = 0.0
        param_count = 0
        zero_count = 0

        for p in self.model.parameters():
            if p.grad is not None:
                grad = p.grad.detach()
                param_norm = grad.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, grad.abs().max().item())
                grad_sum += grad.abs().mean().item()
                param_count += 1
                if param_norm < 1e-10:
                    zero_count += 1

        return {
            "grad_norm": total_norm ** 0.5,
            "grad_max": max_grad,
            "grad_mean": grad_sum / max(param_count, 1),
            "num_zero_grads": zero_count,
        }

    def get_learning_rate(self) -> float:
        """Get current learning rate from optimizer/scheduler."""
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups:
                return float(pg.get("lr", 0.0))
        return float(getattr(self.config, "learning_rate", 0.0))

    def get_training_metrics(self) -> dict[str, float]:
        """Gather comprehensive training state metrics.

        Combines gradient stats, learning rate, and step info into a
        single dict suitable for logging.
        """
        metrics: dict[str, float] = {
            "global_step": float(self.global_step),
            "learning_rate": self.get_learning_rate(),
        }

        # Gradient stats (only if model has parameters)
        try:
            grad_stats = self.get_gradient_stats()
            metrics.update(grad_stats)
        except (RuntimeError, AttributeError):
            pass

        return metrics

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log metrics to W&B and internal history."""
        step = step or self.global_step

        # Enrich with training state if not already present
        if "learning_rate" not in metrics:
            metrics["learning_rate"] = self.get_learning_rate()

        # Update history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        # Log to W&B if available
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass

        # Console logging
        if step % self.config.logging_steps == 0:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Step {step} | {metrics_str}")

    def save_checkpoint(self, path: str | None = None) -> str:
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.output_dir, f"checkpoint-{self.global_step}"
            )

        os.makedirs(path, exist_ok=True)

        # Save model
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save optimizer and scheduler state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "metrics_history": self.metrics_history,
            },
            os.path.join(path, "training_state.pt"),
        )

        logger.info(f"Checkpoint saved to {path}")
        return path

    @abstractmethod
    async def train_step(self, batch: Any) -> dict[str, float]:
        """
        Execute a single training step.

        Must be implemented by subclasses (GSPO, VAPO, DAPO, etc.)
        """
        pass

    @abstractmethod
    async def train(self) -> dict[str, Any]:
        """
        Execute the full training loop.

        Must be implemented by subclasses.
        """
        pass


# Utility functions for common operations


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance.

    Handles edge cases:
    - Single element: returns zero tensor
    - Zero std: returns zero-mean tensor without dividing by std
    """
    if advantages.numel() <= 1:
        return torch.zeros_like(advantages)

    mean = advantages.mean()
    std = advantages.std()

    if std < eps:
        # All values are the same, return zero-centered
        return advantages - mean

    return (advantages - mean) / (std + eps)


def compute_group_advantages(
    rewards: list[float],
    baseline_type: str = "mean",
) -> list[float]:
    """
    Compute group-relative advantages.

    Args:
        rewards: List of rewards for the group
        baseline_type: "mean" or "median"

    Returns:
        List of advantages
    """
    import numpy as np

    if not rewards:
        return []

    rewards_array = np.array(rewards)

    if baseline_type == "mean":
        baseline = rewards_array.mean()
    elif baseline_type == "median":
        baseline = np.median(rewards_array)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    advantages = rewards_array - baseline
    return cast(list[float], advantages.tolist())


def create_response_mask(
    input_ids: torch.Tensor,
    prompt_lengths: list[int],
) -> torch.Tensor:
    """
    Create a mask that is 1 for response tokens and 0 for prompt tokens.

    Args:
        input_ids: Token IDs [batch, seq_len]
        prompt_lengths: Length of prompt for each sequence in batch

    Returns:
        mask: [batch, seq_len] with 1s for response tokens
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.float)

    for i, prompt_len in enumerate(prompt_lengths):
        mask[i, prompt_len:] = 1.0

    return mask
