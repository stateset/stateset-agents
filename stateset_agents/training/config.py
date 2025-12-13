"""
Training configuration and profiles for GRPO Agent Framework

This module provides training configurations that implement best practices
and integrate seamlessly with HuggingFace and Weights & Biases.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingProfile(Enum):
    """Pre-defined training profiles based on stability/performance trade-offs"""

    CONSERVATIVE = "conservative"  # Maximum stability
    BALANCED = "balanced"  # Default, good balance
    AGGRESSIVE = "aggressive"  # Maximum performance
    EXPERIMENTAL = "experimental"  # Cutting edge


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration compatible with HuggingFace patterns
    """

    # Basic training parameters
    model_name: str = "gpt2"
    num_episodes: int = 1000
    num_epochs: int = 1
    learning_rate: float = 5e-6

    # Optimization parameters (HuggingFace compatible)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"

    # Batch and generation settings
    per_device_train_batch_size: int = 1
    # Aliases for compatibility with some tests/configs
    batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 4
    num_generations: int = 16  # GRPO: trajectories per scenario
    generation_batch_size: int = 4

    # Length constraints
    max_prompt_length: int = 512
    max_completion_length: int = 512
    # Alias used by some tests; not directly consumed by trainer but accepted
    max_steps_per_episode: Optional[int] = None

    # Hardware optimization
    bf16: bool = True
    fp16: bool = False
    use_cpu: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Evaluation and checkpointing
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3

    # Output configuration
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    # Backwards-compat alias (some tests/configs use this name).
    wandb_run_name: Optional[str] = None
    overwrite_output_dir: bool = True

    # Weights & Biases integration
    report_to: str = "wandb"  # "wandb", "tensorboard", "none"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None

    # Training stability and monitoring
    auto_adjust: bool = True
    early_stopping: bool = False
    patience: int = 50

    # GRPO specific parameters
    gamma: float = 0.99  # Discount factor for cumulative returns
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    reward_clip: float = 10.0
    advantage_normalization: bool = True
    baseline_type: str = "group_mean"  # "group_mean", "global_mean"

    # Enhanced GRPO parameters from real-world implementation
    beta: float = 0.0  # KL penalty coefficient
    use_reference_model: bool = False  # Whether to use reference model for KL
    clip_ratio: float = 0.2  # PPO-style clipping ratio
    value_clip: float = 0.2  # Value function clipping
    entropy_coef: float = 0.01  # Entropy bonus coefficient

    # Data processing
    max_examples: Optional[int] = None
    eval_split_size: float = 0.1
    stratify_by_task: bool = True
    data_format: str = "jsonl"  # "jsonl", "json", "csv"

    # Post-training evaluation
    run_post_eval: bool = True
    post_eval_samples: int = 5
    post_eval_detailed: bool = True

    # Model optimization
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Generation settings for evaluation
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # vLLM configuration (5-20x faster generation)
    use_vllm: bool = False  # Enable vLLM for generation
    vllm_gpu_memory_utilization: float = 0.85  # Fraction of GPU memory for vLLM
    vllm_tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    vllm_enable_prefix_caching: bool = True  # Cache KV for repeated prefixes
    vllm_max_model_len: Optional[int] = None  # Max sequence length (auto if None)
    vllm_quantization: Optional[str] = None  # "awq", "gptq", or None
    vllm_enable_chunked_prefill: bool = True  # Better memory efficiency

    # Advanced options
    seed: int = 42
    remove_unused_columns: bool = False
    disable_tqdm: bool = False

    def __post_init__(self):
        # Map compatibility aliases
        if self.batch_size is not None and self.batch_size > 0:
            self.per_device_train_batch_size = int(self.batch_size)
        if self.wandb_run_name is not None and self.run_name is None:
            self.run_name = self.wandb_run_name

    @classmethod
    def from_profile(cls, profile: TrainingProfile, **overrides) -> "TrainingConfig":
        """Create configuration from predefined profile"""

        if profile == TrainingProfile.CONSERVATIVE:
            base_config = {
                "learning_rate": 1e-6,
                "num_generations": 32,
                "gradient_accumulation_steps": 8,
                "max_grad_norm": 0.5,
                "warmup_ratio": 0.2,
                "eval_steps": 25,
                "auto_adjust": True,
                "early_stopping": True,
                "patience": 30,
            }
        elif profile == TrainingProfile.BALANCED:
            base_config = {
                "learning_rate": 5e-6,
                "num_generations": 16,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.1,
                "eval_steps": 50,
                "auto_adjust": True,
                "early_stopping": False,
                "patience": 50,
            }
        elif profile == TrainingProfile.AGGRESSIVE:
            base_config = {
                "learning_rate": 1e-5,
                "num_generations": 8,
                "gradient_accumulation_steps": 2,
                "max_grad_norm": 2.0,
                "warmup_ratio": 0.05,
                "eval_steps": 100,
                "auto_adjust": False,
                "early_stopping": False,
                "patience": 100,
            }
        elif profile == TrainingProfile.EXPERIMENTAL:
            base_config = {
                "learning_rate": 2e-5,
                "num_generations": 12,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.5,
                "warmup_ratio": 0.0,
                "eval_steps": 20,
                "auto_adjust": True,
                "early_stopping": True,
                "patience": 20,
                "lr_scheduler_type": "linear",
            }
        else:
            base_config = {}

        # Apply overrides
        base_config.update(overrides)

        return cls(**base_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary"""
        return cls(**config_dict)

    def save(self, path: str):
        """Save configuration to file"""
        config_dict = self.to_dict()

        # Convert enum values to strings
        if isinstance(config_dict.get("wandb_tags"), list):
            # Ensure all tags are strings
            config_dict["wandb_tags"] = [str(tag) for tag in config_dict["wandb_tags"]]

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from file"""
        with open(path, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def validate(self) -> List[str]:
        """Validate configuration and return warnings"""
        warnings = []

        # Learning rate checks
        if self.learning_rate > 1e-4:
            warnings.append(
                f"Learning rate {self.learning_rate} is very high, consider reducing"
            )

        if self.learning_rate < 1e-7:
            warnings.append(
                f"Learning rate {self.learning_rate} is very low, training may be slow"
            )

        # Batch size checks
        effective_batch_size = (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )
        if effective_batch_size < 4:
            warnings.append(
                f"Effective batch size {effective_batch_size} is small, may cause instability"
            )

        # Generation checks
        if self.num_generations < 4:
            warnings.append(
                f"num_generations {self.num_generations} is low for GRPO, consider increasing"
            )

        # Hardware checks
        if self.bf16 and self.fp16:
            warnings.append("Both bf16 and fp16 enabled, bf16 will take precedence")

        # W&B checks
        if self.report_to == "wandb" and not self.wandb_project:
            warnings.append("W&B reporting enabled but no project specified")

        return warnings

    def get_effective_batch_size(self) -> int:
        """Get effective batch size"""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def get_total_steps(self, num_episodes: Optional[int] = None) -> int:
        """Calculate total training steps"""
        episodes = num_episodes or self.num_episodes
        return episodes // self.gradient_accumulation_steps

    def get_warmup_steps(self, total_steps: Optional[int] = None) -> int:
        """Calculate warmup steps"""
        if total_steps is None:
            total_steps = self.get_total_steps()
        return int(total_steps * self.warmup_ratio)


# Pre-defined configurations for common use cases
CUSTOMER_SERVICE_CONFIG = TrainingConfig(
    num_episodes=800,
    learning_rate=3e-6,
    num_generations=12,
    max_completion_length=256,
    eval_steps=40,
    wandb_tags=["customer-service", "conversation"],
    wandb_notes="Customer service agent training with politeness and helpfulness focus",
)

TUTORING_CONFIG = TrainingConfig(
    num_episodes=1200,
    learning_rate=5e-6,
    num_generations=16,
    max_completion_length=512,
    eval_steps=60,
    patience=60,
    wandb_tags=["tutoring", "education"],
    wandb_notes="Educational tutoring agent with step-by-step explanation focus",
)

CREATIVE_WRITING_CONFIG = TrainingConfig(
    num_episodes=600,
    learning_rate=8e-6,
    num_generations=8,
    max_completion_length=1024,
    eval_steps=30,
    lr_scheduler_type="linear",
    wandb_tags=["creative-writing", "generation"],
    wandb_notes="Creative writing assistant with imagination and storytelling focus",
)

RESEARCH_ASSISTANT_CONFIG = TrainingConfig(
    num_episodes=1000,
    learning_rate=4e-6,
    num_generations=20,
    max_completion_length=768,
    eval_steps=50,
    early_stopping=True,
    patience=40,
    wandb_tags=["research", "analysis"],
    wandb_notes="Research assistant with factual accuracy and citation focus",
)


def get_config_for_task(task_type: str, **overrides) -> TrainingConfig:
    """Get pre-configured setup for common tasks"""

    task_configs = {
        "customer_service": CUSTOMER_SERVICE_CONFIG,
        "tutoring": TUTORING_CONFIG,
        "creative_writing": CREATIVE_WRITING_CONFIG,
        "research_assistant": RESEARCH_ASSISTANT_CONFIG,
    }

    if task_type in task_configs:
        base_config = task_configs[task_type]

        # Create new config with overrides
        config_dict = base_config.to_dict()
        config_dict.update(overrides)

        return TrainingConfig.from_dict(config_dict)
    else:
        # Return default balanced config with overrides
        return TrainingConfig.from_profile(TrainingProfile.BALANCED, **overrides)


def optimize_config_for_hardware(
    config: TrainingConfig, gpu_memory_gb: Optional[float] = None, num_gpus: int = 1
) -> TrainingConfig:
    """Optimize configuration based on available hardware"""

    optimized_dict = config.to_dict()

    # GPU memory optimizations
    if gpu_memory_gb is not None:
        if gpu_memory_gb < 8:
            # Small GPU optimizations
            optimized_dict.update(
                {
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "num_generations": 8,
                    "max_completion_length": 256,
                    "bf16": True,
                    "dataloader_num_workers": 0,
                }
            )
        elif gpu_memory_gb < 16:
            # Medium GPU optimizations
            optimized_dict.update(
                {
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "num_generations": 12,
                    "max_completion_length": 512,
                    "bf16": True,
                    "dataloader_num_workers": 2,
                }
            )
        else:
            # Large GPU optimizations
            optimized_dict.update(
                {
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "num_generations": 16,
                    "max_completion_length": 768,
                    "bf16": True,
                    "dataloader_num_workers": 4,
                }
            )

    # Multi-GPU optimizations
    if num_gpus > 1:
        optimized_dict["per_device_train_batch_size"] = max(
            1, optimized_dict["per_device_train_batch_size"] // num_gpus
        )
        optimized_dict["dataloader_num_workers"] = min(
            8, optimized_dict["dataloader_num_workers"] * num_gpus
        )

    return TrainingConfig.from_dict(optimized_dict)


def create_config_from_args(
    model_name: str, task_type: str = "general", profile: str = "balanced", **kwargs
) -> TrainingConfig:
    """Create configuration from command line arguments"""

    # Start with task-specific config or profile
    if task_type != "general":
        config = get_config_for_task(task_type)
    else:
        config = TrainingConfig.from_profile(TrainingProfile(profile))

    # Apply model-specific optimizations
    if "large" in model_name.lower() or "13b" in model_name.lower():
        # Large model optimizations
        config_dict = config.to_dict()
        config_dict.update(
            {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "bf16": True,
                "max_completion_length": min(512, config_dict["max_completion_length"]),
            }
        )
        config = TrainingConfig.from_dict(config_dict)

    # Apply user overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = TrainingConfig.from_dict(config_dict)

    return config


# Example usage and testing
def example_config_usage():
    """Example of how to use training configurations"""

    # Create from profile
    conservative_config = TrainingConfig.from_profile(TrainingProfile.CONSERVATIVE)
    print(f"Conservative LR: {conservative_config.learning_rate}")

    # Create for specific task
    customer_config = get_config_for_task("customer_service", num_episodes=1200)
    print(f"Customer service episodes: {customer_config.num_episodes}")

    # Optimize for hardware
    optimized_config = optimize_config_for_hardware(
        customer_config, gpu_memory_gb=12.0, num_gpus=2
    )
    print(f"Optimized batch size: {optimized_config.per_device_train_batch_size}")

    # Validate configuration
    warnings = optimized_config.validate()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    # Save and load
    optimized_config.save("config.json")
    loaded_config = TrainingConfig.load("config.json")
    print(f"Loaded config learning rate: {loaded_config.learning_rate}")


if __name__ == "__main__":
    example_config_usage()
