"""
Kimi-K2.5 configuration utilities for StateSet Agents

This module provides helper functions for creating optimized configurations
for Kimi-K2.5 models based on their specific architecture (1T MoE, 32B activated).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from stateset_agents.training.config import (
    TrainingConfig,
    TrainingProfile,
    get_config_for_task,
)

try:
    from stateset_agents import MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.core.reward import create_domain_reward
    from stateset_agents.training import train_with_gspo
    from stateset_agents.training.gspo_trainer import GSPOConfig
except Exception:  # pragma: no cover - optional heavy deps
    MultiTurnAgent = None  # type: ignore
    ConversationEnvironment = None  # type: ignore
    create_domain_reward = None  # type: ignore
    train_with_gspo = None  # type: ignore
    GSPOConfig = None  # type: ignore

logger = logging.getLogger(__name__)


# Kimi-K2.5 model identifiers
KIMI_MODELS = {
    "kimi-k2.5": "moonshotai/Kimi-K2.5",
    "kimi-k2.5-base": "moonshotai/Kimi-K2.5",
    "kimi-k2.5-instruct": "moonshotai/Kimi-K2.5-Instruct",
}

# LoRA target modules for Kimi-K2.5 MoE model
KIMI_K25_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

KIMI_K25_SUPPORTED_VARIANTS = [
    "moonshotai/Kimi-K2.5",
    "moonshotai/Kimi-K2.5-Base",
    "moonshotai/Kimi-K2.5-Instruct",
]

KIMI_K25_MODEL_SPECS = {
    "moonshotai/Kimi-K2.5": {
        "total_params": "1T",
        "activated_params": "32B",
        "architecture": "MoE",
        "context_length": "256K",
    },
}

KIMI_K25_INFO = {
    "model_name": "moonshotai/Kimi-K2.5",
    "total_params": "1T",
    "activated_params": "32B",
    "context_length": "256K",
    "multimodal": True,
    "supported_engines": ["vLLM", "SGLang", "KTransformers"],
}


def get_kimi_config(
    model_name: str = "moonshotai/Kimi-K2.5",
    task: str = "conversational",
    profile: str = "balanced",
    use_lora: bool = True,
    use_vllm: bool = False,
    **kwargs,
) -> TrainingConfig:
    """
    Get optimized training configuration for Kimi-K2.5 models.

    Args:
        model_name: Kimi model identifier
        task: Task type (conversational, customer_service, technical_support, coding)
        profile: Training profile (balanced, conservative, aggressive)
        use_lora: Use LoRA for efficient training (recommended for MoE)
        use_vllm: Use vLLM for faster generation
        **kwargs: Additional configuration overrides

    Returns:
        Optimized TrainingConfig for Kimi-K2.5

    Examples:
        >>> config = get_kimi_config(task="customer_service")
        >>> config = get_kimi_config(use_lora=True, use_vllm=True)
    """

    # Start with base profile
    base_profile = TrainingProfile(profile)
    config_dict = TrainingConfig.from_profile(base_profile).to_dict()

    # Update base training parameters for Kimi-K2.5
    base_updates = {
        "model_name": model_name,
        "learning_rate": 3e-6,  # Lower LR for stability with MoE
        "per_device_train_batch_size": 1,  # Small batch due to MoE memory
        "gradient_accumulation_steps": 16,  # Large accumulation for effective batch
        "num_generations": 8,  # Moderate group size
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "top_p": 0.95,
    }
    config_dict.update(base_updates)

    # Task-specific optimizations
    task_configs = {
        "conversational": {
            "max_prompt_length": 2048,
            "max_completion_length": 2048,
            "temperature": 1.0,
        },
        "customer_service": {
            "max_prompt_length": 1536,
            "max_completion_length": 1024,
            "temperature": 0.7,
            "eval_steps": 50,
        },
        "technical_support": {
            "max_prompt_length": 2048,
            "max_completion_length": 1536,
            "temperature": 0.7,
            "eval_steps": 75,
        },
        "coding": {
            "max_prompt_length": 4096,
            "max_completion_length": 2048,
            "temperature": 0.6,
            "eval_steps": 100,
        },
    }

    if task in task_configs:
        config_dict.update(task_configs[task])

    # Profile-specific adjustments
    if profile == "conservative":
        config_dict.update(
            {
                "learning_rate": 1e-6,
                "num_generations": 12,
                "gradient_accumulation_steps": 20,
                "max_grad_norm": 0.5,
            }
        )
    elif profile == "aggressive":
        config_dict.update(
            {
                "learning_rate": 5e-6,
                "num_generations": 6,
                "gradient_accumulation_steps": 12,
                "max_grad_norm": 1.5,
            }
        )

    # LoRA configuration for MoE
    if use_lora:
        config_dict.update(
            {
                "use_lora": True,
                "lora_r": 64,  # Higher rank for MoE models
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "lora_target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "gradient_checkpointing": True,
            }
        )

    # vLLM configuration
    if use_vllm:
        config_dict.update(
            {
                "use_vllm": True,
                "vllm_gpu_memory_utilization": 0.85,
                "vllm_tensor_parallel_size": 1,
                "vllm_enable_prefix_caching": True,
                "vllm_max_model_len": 256000,  # Kimi-K2.5 supports 256K context
                "vllm_enable_chunked_prefill": True,
            }
        )

    # Apply user overrides
    config_dict.update(kwargs)

    # Create final config
    config = TrainingConfig.from_dict(config_dict)

    logger.info(f"Created Kimi-K2.5 config: {model_name}")
    logger.info(f"  Task: {task}, Profile: {profile}")
    logger.info(f"  LoRA: {use_lora}, vLLM: {use_vllm}")
    logger.info(
        f"  Batch size: {config.per_device_train_batch_size}, "
        f"Grad accumulation: {config.gradient_accumulation_steps}"
    )

    return config


def get_kimi_system_prompt(task: str = "conversational") -> str:
    """
    Get optimized system prompts for Kimi-K2.5 models.

    Args:
        task: Task type

    Returns:
        System prompt string

    Examples:
        >>> prompt = get_kimi_system_prompt("customer_service")
    """

    base_intro = "You are Kimi, an AI assistant created by Moonshot AI."

    prompts = {
        "conversational": (
            f"{base_intro} You are helpful, accurate, and engaging. "
            "You think carefully before responding and provide thorough, "
            "well-reasoned answers."
        ),
        "customer_service": (
            f"{base_intro} You are a professional customer service representative. "
            "You are empathetic, patient, and solutions-oriented. You help customers "
            "with their inquiries professionally and efficiently."
        ),
        "technical_support": (
            f"{base_intro} You are a knowledgeable technical support specialist. "
            "You provide clear, detailed explanations and step-by-step guidance. "
            "You troubleshoot issues systematically and verify solutions work."
        ),
        "coding": (
            f"{base_intro} You are an expert programmer and coding assistant. "
            "You write clean, efficient, well-documented code. You explain "
            "your reasoning when debugging and follow best practices."
        ),
    }

    return prompts.get(task, prompts["conversational"])


def validate_kimi_config(config: TrainingConfig) -> list[str]:
    """
    Validate Kimi-K2.5 configuration and return warnings/recommendations.

    Args:
        config: TrainingConfig to validate

    Returns:
        List of warning messages

    Examples:
        >>> warnings = validate_kimi_config(config)
        >>> if warnings:
        ...     print("Warnings:")
        ...     for w in warnings:
        ...         print(f"  - {w}")
    """

    warnings = []

    # Check model name
    if "kimi" not in config.model_name.lower():
        warnings.append(
            "Model name doesn't contain 'kimi'. Verify this is a Kimi-K2.5 model."
        )

    # Check learning rate for MoE
    if config.learning_rate > 1e-5:
        warnings.append(
            f"Learning rate {config.learning_rate} may be too high for MoE model. "
            "Consider reducing to 3e-6 or lower."
        )

    # Check batch size
    if config.per_device_train_batch_size > 2:
        warnings.append(
            f"Batch size {config.per_device_train_batch_size} may cause OOM "
            "with 1T MoE model. Recommend 1-2 per device."
        )

    # Check effective batch size
    effective_batch = (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    if effective_batch < 8:
        warnings.append(
            f"Effective batch size {effective_batch} is small. "
            "Consider increasing gradient_accumulation_steps."
        )

    # Check LoRA rank for MoE
    if config.use_lora and config.lora_r < 32:
        warnings.append(
            f"LoRA rank {config.lora_r} may be too low for MoE model. "
            "Recommend 64 or higher."
        )

    # Check context length
    if config.max_prompt_length > 256000:
        warnings.append(
            f"Max prompt length {config.max_prompt_length} exceeds Kimi-K2.5 limit (256K)."
        )

    return warnings


def get_kimi_k25_system_prompt(task: str = "conversational") -> str:
    """Return a task-specific system prompt for Kimi-K2.5."""
    if task == "technical_support":
        return (
            "You are Kimi, a helpful and empathetic technical support specialist "
            "created by Moonshot AI. You help users troubleshoot technical issues "
            "with clear, detailed explanations using reasoning when helpful."
        )
    base_intro = "You are Kimi, an AI assistant created by Moonshot AI."
    if task == "customer_service":
        return (
            f"{base_intro} You are a helpful, empathetic, and professional customer service "
            "assistant who resolves issues efficiently."
        )
    if task in {"sales", "sales_assistant"}:
        return (
            f"{base_intro} You are a persuasive and helpful sales assistant who "
            "matches customers with the right products."
        )
    if task in {"coding", "coding_assistant"}:
        return (
            f"{base_intro} You are an expert coding and programming assistant who writes clean, "
            "well-documented code and explains your reasoning."
        )
    if task == "vision":
        return (
            f"{base_intro} You can analyze images and text together and explain "
            "your reasoning clearly."
        )
    return base_intro


def _default_kimi_tags(task: str) -> list[str]:
    tags = ["kimi-k25", "gspo"]
    if task:
        tags.append(task)
    return tags


def _resolve_grad_steps(gpu_memory_gb: int | None, num_gpus: int) -> int:
    grad_steps = 16
    if gpu_memory_gb is not None:
        if gpu_memory_gb >= 80:
            grad_steps = 8
        elif gpu_memory_gb >= 40:
            grad_steps = 16
        else:
            grad_steps = 24
    if num_gpus > 1 and grad_steps < 16:
        grad_steps = 16
    return grad_steps


@dataclass
class KimiK25Config:
    """Lightweight configuration container for Kimi-K2.5 training."""

    model_name: str = "moonshotai/Kimi-K2.5"
    task: str = "conversational"
    profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 64
    lora_alpha: int | None = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(KIMI_K25_LORA_TARGET_MODULES)
    )

    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.85
    vllm_enable_prefix_caching: bool = True
    vllm_max_model_len: int = 256000
    vllm_enable_chunked_prefill: bool = True

    use_4bit: bool = False
    use_8bit: bool = False
    bf16: bool = True

    temperature: float = 0.7
    top_p: float = 0.95
    temperature_thinking: float = 1.0
    temperature_instant: float = 0.6

    max_prompt_length: int = 8192
    max_completion_length: int = 4096

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_generations: int = 8
    learning_rate: float = 3e-6

    num_iterations: int = 1
    num_outer_iterations: int = 100
    generations_per_iteration: int = 100

    clip_range_left: float = 3e-4
    clip_range_right: float = 4e-4

    save_steps_every: int = 50
    output_dir: str = "./outputs/kimi_k25_finetuned"

    report_to: str = "wandb"
    wandb_project: str | None = "kimi-k25"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    use_wandb: bool = True

    device_map: str = "auto"
    trust_remote_code: bool = True
    gpu_memory_gb: int | None = None
    num_gpus: int = 1

    def __post_init__(self) -> None:
        if self.system_prompt is None:
            self.system_prompt = get_kimi_k25_system_prompt(self.task)
        if not self.wandb_tags:
            self.wandb_tags = _default_kimi_tags(self.task)
        if not self.use_wandb:
            self.report_to = "none"
        if not self.use_lora:
            self.lora_r = None
            self.lora_alpha = None
        if self.use_4bit:
            self.use_8bit = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "KimiK25Config":
        return cls(**config_dict)

    def get_effective_batch_size(self) -> int:
        return int(self.per_device_train_batch_size * self.gradient_accumulation_steps)

    def validate(self) -> list[str]:
        warnings: list[str] = []
        if self.num_outer_iterations <= 0 or self.num_iterations <= 0:
            warnings.append("num_outer_iterations must be > 0")
        if self.learning_rate < 0:
            warnings.append("learning rate must be non-negative")
        if self.learning_rate >= 1e-4:
            warnings.append("learning rate is very high for Kimi-K2.5")
        if self.learning_rate < 1e-7:
            warnings.append("learning rate is very low for Kimi-K2.5")
        if self.per_device_train_batch_size >= 8:
            warnings.append("large batch size may cause OOM on Kimi-K2.5")
        if (
            self.gpu_memory_gb is not None
            and self.gpu_memory_gb < 40
            and not (self.use_4bit or self.use_8bit)
        ):
            warnings.append("GPU memory may be insufficient without quantization")
        if not self.use_lora and not (self.use_4bit or self.use_8bit):
            warnings.append("LoRA is recommended for Kimi-K2.5 without quantization")
        return warnings


def get_kimi_k25_config(
    model_name: str = "moonshotai/Kimi-K2.5",
    task: str = "conversational",
    profile: str = "balanced",
    use_lora: bool = True,
    use_vllm: bool = True,
    use_4bit: bool = False,
    use_8bit: bool = False,
    use_wandb: bool = True,
    wandb_enabled: bool | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    gpu_memory_gb: int | None = None,
    num_gpus: int = 1,
    num_iterations: int | None = None,
    num_outer_iterations: int | None = None,
    generations_per_iteration: int | None = None,
    learning_rate: float | None = None,
    num_generations: int | None = None,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    vllm_gpu_memory_utilization: float | None = None,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
    temperature: float | None = None,
    save_steps_every: int | None = None,
    output_dir: str | None = None,
    system_prompt: str | None = None,
    wandb_tags: list[str] | None = None,
    **kwargs: Any,
) -> KimiK25Config:
    if model_name not in KIMI_K25_SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported Kimi-K2.5 model: {model_name}")

    if wandb_enabled is not None:
        use_wandb = wandb_enabled

    if lora_r is None:
        lora_r = 128 if use_4bit else 64
    if lora_alpha is None:
        lora_alpha = 128
    if not use_lora:
        lora_r = None
        lora_alpha = None

    if learning_rate is None:
        learning_rate = 3e-6
    if num_generations is None:
        num_generations = 8
    if per_device_train_batch_size is None:
        per_device_train_batch_size = 1
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = _resolve_grad_steps(gpu_memory_gb, num_gpus)

    if max_prompt_length is None:
        max_prompt_length = 8192
    if max_completion_length is None:
        max_completion_length = 4096
    if temperature is None:
        temperature = 0.7

    if save_steps_every is None:
        save_steps_every = 50
    if output_dir is None:
        output_dir = "./outputs/kimi_k25_finetuned"

    if wandb_project is None and use_wandb:
        wandb_project = "kimi-k25"

    if wandb_tags is None:
        wandb_tags = _default_kimi_tags(task)

    config = KimiK25Config(
        model_name=model_name,
        task=task,
        profile=profile,
        system_prompt=system_prompt,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_modules=list(KIMI_K25_LORA_TARGET_MODULES),
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization
        if vllm_gpu_memory_utilization is not None
        else 0.85,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        temperature=temperature,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        learning_rate=learning_rate,
        num_iterations=num_iterations if num_iterations is not None else 1,
        num_outer_iterations=num_outer_iterations
        if num_outer_iterations is not None
        else 100,
        generations_per_iteration=generations_per_iteration
        if generations_per_iteration is not None
        else 100,
        save_steps_every=save_steps_every,
        output_dir=output_dir,
        report_to="wandb" if use_wandb else "none",
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_tags=wandb_tags,
        use_wandb=use_wandb,
        gpu_memory_gb=gpu_memory_gb,
        num_gpus=num_gpus,
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def get_kimi_k25_conversational_config(**overrides: Any) -> KimiK25Config:
    tags = overrides.pop("wandb_tags", None) or _default_kimi_tags("conversational")
    return get_kimi_k25_config(
        task="conversational",
        max_prompt_length=4096,
        max_completion_length=2048,
        temperature=0.7,
        wandb_tags=tags,
        **overrides,
    )


def get_kimi_k25_vision_config(**overrides: Any) -> KimiK25Config:
    tags = overrides.pop("wandb_tags", None) or _default_kimi_tags("vision")
    return get_kimi_k25_config(
        task="vision",
        max_prompt_length=8192,
        max_completion_length=4096,
        temperature=1.0,
        wandb_tags=tags,
        **overrides,
    )


def get_kimi_k25_config_gspo(
    kimi_config: KimiK25Config, base_config: TrainingConfig | None = None
) -> Any | None:
    if GSPOConfig is None:
        return None
    if base_config is None:
        base_config = get_config_for_task(
            kimi_config.task, model_name=kimi_config.model_name
        )
    gspo_config = GSPOConfig.from_training_config(base_config)
    gspo_config.model_name = kimi_config.model_name
    gspo_config.use_lora = kimi_config.use_lora
    if kimi_config.lora_r is not None:
        gspo_config.lora_r = kimi_config.lora_r
    if kimi_config.lora_alpha is not None:
        gspo_config.lora_alpha = kimi_config.lora_alpha
    gspo_config.lora_target_modules = list(KIMI_K25_LORA_TARGET_MODULES)
    gspo_config.num_generations = kimi_config.num_generations
    gspo_config.learning_rate = kimi_config.learning_rate
    gspo_config.clip_range_left = kimi_config.clip_range_left
    gspo_config.clip_range_right = kimi_config.clip_range_right
    gspo_config.use_vllm = kimi_config.use_vllm
    return gspo_config


def validate_kimi_k25_config(config: KimiK25Config) -> list[str]:
    return config.validate()


async def setup_kimi_k25_training(
    task: str = "customer_service",
    model_name: str = "moonshotai/Kimi-K2.5",
    use_lora: bool = True,
    use_vllm: bool = True,
    **kwargs: Any,
):
    if MultiTurnAgent is None or ConversationEnvironment is None:
        raise RuntimeError("Training dependencies are unavailable")

    kimi_config = get_kimi_k25_config(
        task=task,
        model_name=model_name,
        use_lora=use_lora,
        use_vllm=use_vllm,
        **kwargs,
    )

    from stateset_agents.core.agent import AgentConfig

    agent = MultiTurnAgent(  # type: ignore[call-arg]
        AgentConfig(model_name=model_name, system_prompt=kimi_config.system_prompt)
    )
    init_result = agent.initialize()
    if asyncio.iscoroutine(init_result):
        await init_result

    env = ConversationEnvironment(
        scenarios=[{"id": "kimi_k25", "topic": task, "context": task}]
    )

    if create_domain_reward is None:
        raise RuntimeError("Reward utilities are unavailable")
    reward_model = create_domain_reward(task)

    gspo_config = get_kimi_k25_config_gspo(kimi_config, get_config_for_task(task))

    if train_with_gspo is None:
        raise RuntimeError("GSPO trainer is unavailable")
    train_result = train_with_gspo(  # type: ignore[misc]
        agent=agent,
        environment=env,
        reward_fn=reward_model,
        config=gspo_config,
    )
    if asyncio.iscoroutine(train_result):
        await train_result

    return agent, env, reward_model, gspo_config


def get_kimi_hardware_requirements(model_name: str) -> dict[str, Any]:
    """
    Get estimated hardware requirements for Kimi-K2.5 training.

    Args:
        model_name: Kimi model identifier

    Returns:
        Dictionary with hardware requirements

    Examples:
        >>> reqs = get_kimi_hardware_requirements("moonshotai/Kimi-K2.5")
        >>> print(f"GPU memory: {reqs['gpu_memory_gb']}GB+")
    """

    # Base requirements for 1T MoE model (32B activated)
    base_requirements = {
        "model_name": model_name,
        "gpu_memory_gb": 80,  # Minimum for 32B activated with LoRA
        "gpu_count": 1,
        "system_memory_gb": 32,
        "disk_space_gb": 100,  # Model weights + checkpoints
        "recommended_gpu": "A100 80GB, H100 80GB, or equivalent",
    }

    # Check if model might be smaller (quantized, etc.)
    if "int4" in model_name.lower() or "4bit" in model_name.lower():
        base_requirements.update(
            {
                "gpu_memory_gb": 40,
                "recommended_gpu": "A100 40GB, A6000, or equivalent",
            }
        )

    return base_requirements


def get_kimi_k25_lora_target_modules() -> list[str]:
    """
    Get the LoRA target modules for Kimi-K2.5 models.

    Returns:
        List of module names for LoRA finetuning

    Examples:
        >>> modules = get_kimi_k25_lora_target_modules()
        >>> print(modules)
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    """
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def get_vllm_launch_command(
    model_name: str = "moonshotai/Kimi-K2.5",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int | None = None,
) -> str:
    """
    Generate vLLM launch command for Kimi-K2.5 models.

    Args:
        model_name: HuggingFace model name
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        host: Host to bind the server
        port: Port for the API server
        max_model_len: Maximum sequence length (default: 256000 for Kimi-K2.5)

    Returns:
        vLLM launch command string

    Examples:
        >>> cmd = get_vllm_launch_command(tensor_parallel_size=2)
        >>> print(cmd)
    """
    if max_model_len is None:
        max_model_len = 256000  # Kimi-K2.5 supports 256K context

    cmd_parts = [
        "vllm serve",
        model_name,
        f"--tensor-parallel-size {tensor_parallel_size}",
        "--mm-encoder-tp-mode data",
        "--tool-call-parser kimi_k2",
        "--reasoning-parser kimi_k2",
        f"--gpu-memory-utilization {gpu_memory_utilization}",
        f"--max-model-len {max_model_len}",
        f"--host {host}",
        f"--port {port}",
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
    ]

    return " ".join(cmd_parts)


def print_kimi_summary():
    """Print summary of Kimi-K2.5 integration capabilities."""

    print("\n" + "=" * 80)
    print("Kimi-K2.5 Integration for StateSet Agents")
    print("=" * 80)

    print("\n📋 Model Specifications:")
    specs = [
        ("Architecture", "Mixture-of-Experts (MoE)"),
        ("Total Parameters", "1T"),
        ("Activated Parameters", "32B"),
        ("Number of Experts", "384"),
        ("Selected Experts", "8 per token"),
        ("Context Length", "256K tokens"),
        ("Vision Encoder", "MoonViT (400M)"),
        ("Attention", "MLA (Multi-head Latent Attention)"),
        ("Activation", "SwiGLU"),
    ]
    for name, value in specs:
        print(f"  • {name:.<30} {value}")

    print("\n🚀 Supported Training Algorithms:")
    algorithms = [
        ("GSPO", "✅ Recommended - Stable sequence-level optimization"),
        ("GRPO", "✅ Supported - Group-based policy optimization"),
        ("PPO", "✅ Supported - Standard PPO baseline"),
        ("GAPO", "✅ Supported - Group-augmented policy optimization"),
    ]
    for name, status in algorithms:
        print(f"  • {name:.<20} {status}")

    print("\n⚙️ Key Features:")
    features = [
        "Native multimodality (vision + text)",
        "Thinking mode for complex reasoning",
        "Agent swarm capabilities",
        "Multi-step tool calling",
        "Long context (256K tokens)",
        "LoRA training support",
        "vLLM acceleration",
        "INT4 quantization",
    ]
    for feature in features:
        print(f"  ✓ {feature}")

    print("\n📁 Quick Start:")
    print("  1. Training:")
    print("     python examples/finetune_kimi_k25_gspo.py \\")
    print("       --task customer_service --use-lora --use-vllm")
    print("\n  2. Configuration:")
    print("     from examples.kimi_k25_config import get_kimi_config")
    print("     config = get_kimi_config(task='customer_service')")
    print("\n  3. System prompts:")
    print("     from examples.kimi_k25_config import get_kimi_system_prompt")
    print("     prompt = get_kimi_system_prompt('coding')")

    print("\n📚 Documentation:")
    print("  • examples/KIMI_K25_INTEGRATION.md - Complete integration guide")
    print("  • examples/kimi_k25_config.py - Configuration utilities")
    print("  • examples/finetune_kimi_k25_gspo.py - Training examples")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_kimi_summary()
