"""
Kimi-K2.5 configuration utilities for StateSet Agents

This module provides helper functions for creating optimized configurations
for Kimi-K2.5 models based on their specific architecture (1T MoE, 32B activated).
"""

import logging
from typing import Dict, List, Optional
from stateset_agents.training.config import TrainingConfig, TrainingProfile

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
    }
    config_dict.update(base_updates)

    # Task-specific optimizations
    task_configs = {
        "conversational": {
            "max_prompt_length": 2048,
            "max_completion_length": 2048,
            "temperature": 1.0 if "thinking" in model_name.lower() else 0.7,
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


def validate_kimi_config(config: TrainingConfig) -> List[str]:
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


def get_kimi_hardware_requirements(model_name: str) -> Dict[str, any]:
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


def get_kimi_k25_lora_target_modules() -> List[str]:
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
    max_model_len: Optional[int] = None,
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
        f"--gpu-memory-utilization {gpu_memory_utilization}",
        f"--max-model-len {max_model_len}",
        f"--host {host}",
        f"--port {port}",
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
