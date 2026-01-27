"""
Configuration validation for StateSet Agents.

Provides JSON Schema validation and pre-flight checks.
"""

from typing import Any, Dict, List, Optional, Type, Union

import json
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Validation error information."""

    field: str
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Validation result for configuration."""

    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]

    def __bool__(self) -> bool:
        return self.is_valid

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [{"field": e.field, "message": e.message, "severity": e.severity}
                      for e in self.errors],
            "warnings": [{"field": w.field, "message": w.message, "severity": w.severity}
                        for w in self.warnings],
        }


class ConfigSchema:
    """JSON Schema definitions for configurations."""

    TRAINING_CONFIG = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "minLength": 1},
            "learning_rate": {"type": "number", "minimum": 1e-7, "maximum": 1.0},
            "num_episodes": {"type": "integer", "minimum": 1, "maximum": 100000},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024},
            "gamma": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "lambda": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "use_lora": {"type": "boolean"},
            "lora_r": {"type": "integer", "minimum": 1, "maximum": 1024},
            "lora_alpha": {"type": "integer", "minimum": 1, "maximum": 2048},
            "gradient_accumulation_steps": {"type": "integer", "minimum": 1, "maximum": 1000},
            "max_grad_norm": {"type": "number", "minimum": 0.0, "maximum": 100.0},
            "num_generations": {"type": "integer", "minimum": 2, "maximum": 100},
            "clip_range": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "required": ["model_name"],
        "additionalProperties": True,
    }

    AGENT_CONFIG = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "minLength": 1},
            "system_prompt": {"type": "string"},
            "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 8192},
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 1000},
            "repetition_penalty": {"type": "number", "minimum": 1.0, "maximum": 2.0},
        },
        "required": ["model_name"],
        "additionalProperties": True,
    }

    ENVIRONMENT_CONFIG = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "max_turns": {"type": "integer", "minimum": 1, "maximum": 1000},
            "scenarios": {
                "type": "array",
                "items": {"type": "object"},
                "minItems": 1,
            },
            "reward_on_completion": {"type": "boolean"},
            "timeout": {"type": "number", "minimum": 0.0},
        },
        "required": ["scenarios"],
        "additionalProperties": True,
    }


class ConfigValidator:
    """Configuration validator with pre-flight checks."""

    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def validate_training_config(
        self,
        config: Dict[str, Any],
        strict: bool = False,
    ) -> ValidationResult:
        """Validate training configuration."""
        self.errors = []
        self.warnings = []

        # Check required fields
        required_fields = ["model_name"]
        for field in required_fields:
            if field not in config:
                self.errors.append(
                    ValidationError(field, f"Missing required field: {field}")
                )

        # Check learning rate
        if "learning_rate" in config:
            lr = config["learning_rate"]
            if lr > 1e-3:
                self.warnings.append(
                    ValidationError(
                        "learning_rate",
                        f"Learning rate {lr} is high. Consider using 1e-4 or lower.",
                        "warning",
                    )
                )
            elif lr < 1e-7:
                self.warnings.append(
                    ValidationError(
                        "learning_rate",
                        f"Learning rate {lr} is very low. Training may be slow.",
                        "warning",
                    )
                )

        # Check batch size
        if "batch_size" in config and "per_device_train_batch_size" not in config:
            bs = config["batch_size"]
            if bs > 64:
                self.warnings.append(
                    ValidationError(
                        "batch_size",
                        f"Large batch size {bs} may cause OOM. Consider using gradient accumulation.",
                        "warning",
                    )
                )

        # Check LoRA configuration
        if config.get("use_lora", False):
            if "lora_r" in config and config["lora_r"] > 256:
                self.warnings.append(
                    ValidationError(
                        "lora_r",
                        f"LoRA rank {config['lora_r']} is high. May increase training time.",
                        "warning",
                    )
                )

        # Check effective batch size
        if "per_device_train_batch_size" in config and "gradient_accumulation_steps" in config:
            effective_batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
            if effective_batch < 8:
                self.warnings.append(
                    ValidationError(
                        "effective_batch",
                        f"Effective batch size {effective_batch} is small. Consider increasing.",
                        "warning",
                    )
                )

        # Check number of generations
        if "num_generations" in config:
            ng = config["num_generations"]
            if ng < 4:
                self.warnings.append(
                    ValidationError(
                        "num_generations",
                        f"Low number of generations ({ng}). Recommend 4+ for GRPO.",
                        "warning",
                    )
                )

        return ValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
        )

    def validate_agent_config(
        self,
        config: Dict[str, Any],
    ) -> ValidationResult:
        """Validate agent configuration."""
        self.errors = []
        self.warnings = []

        # Check model name
        if "model_name" not in config or not config["model_name"]:
            self.errors.append(
                ValidationError("model_name", "Model name is required")
            )

        # Check generation parameters
        if "temperature" in config:
            temp = config["temperature"]
            if temp > 1.5:
                self.warnings.append(
                    ValidationError(
                        "temperature",
                        f"Temperature {temp} is high. May produce random outputs.",
                        "warning",
                    )
                )

        if "top_p" in config:
            top_p = config["top_p"]
            if top_p > 0.99 and "temperature" in config and config["temperature"] > 0.8:
                self.warnings.append(
                    ValidationError(
                        "top_p",
                        "High top_p with high temperature may cause instability.",
                        "warning",
                    )
                )

        # Check max new tokens
        if "max_new_tokens" in config:
            mnt = config["max_new_tokens"]
            if mnt > 4096:
                self.warnings.append(
                    ValidationError(
                        "max_new_tokens",
                        f"High max_new_tokens ({mnt}). May slow inference.",
                        "warning",
                    )
                )

        return ValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
        )

    def validate_environment_config(
        self,
        config: Dict[str, Any],
    ) -> ValidationResult:
        """Validate environment configuration."""
        self.errors = []
        self.warnings = []

        # Check scenarios
        if "scenarios" not in config or not config["scenarios"]:
            self.errors.append(
                ValidationError("scenarios", "At least one scenario is required")
            )

        # Check max turns
        if "max_turns" in config:
            max_turns = config["max_turns"]
            if max_turns > 50:
                self.warnings.append(
                    ValidationError(
                        "max_turns",
                        f"High max_turns ({max_turns}). May slow training.",
                        "warning",
                    )
                )

        return ValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
        )


class PreFlightChecker:
    """Pre-flight checks before training/inference."""

    @staticmethod
    def check_system_requirements() -> ValidationResult:
        """Check if system meets minimum requirements."""
        errors = []
        warnings = []

        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            errors.append(
                ValidationError(
                    "python_version",
                    f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported. Need 3.8+.",
                )
            )

        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count == 0:
                    warnings.append(
                        ValidationError(
                            "cuda",
                            "CUDA available but no GPUs found",
                            "warning",
                        )
                    )
                else:
                    for i in range(gpu_count):
                        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                        if mem < 4:  # Less than 4GB
                            warnings.append(
                                ValidationError(
                                    "gpu_memory",
                                    f"GPU {i} has only {mem:.1f}GB memory",
                                    "warning",
                                )
                            )
            else:
                warnings.append(
                    ValidationError(
                        "cuda",
                        "CUDA not available. Training will be slow on CPU.",
                        "warning",
                    )
                )
        except ImportError:
            errors.append(
                ValidationError(
                    "torch",
                    "PyTorch not installed. Install with 'pip install torch'",
                )
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def check_training_readiness(
        config: Dict[str, Any],
        data_path: Optional[str] = None,
    ) -> ValidationResult:
        """Check if training can start."""
        errors = []
        warnings = []

        # Check if output directory exists
        if "output_dir" in config:
            import os
            output_dir = config["output_dir"]
            if os.path.exists(output_dir):
                warnings.append(
                    ValidationError(
                        "output_dir",
                        f"Output directory '{output_dir}' already exists. Files may be overwritten.",
                        "warning",
                    )
                )
            else:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except PermissionError:
                    errors.append(
                        ValidationError(
                            "output_dir",
                            f"Cannot create output directory '{output_dir}'. Check permissions.",
                        )
                    )

        # Check data path if provided
        if data_path and not os.path.exists(data_path):
            errors.append(
                ValidationError(
                    "data_path",
                    f"Data path '{data_path}' does not exist.",
                )
            )

        # Estimate memory requirements
        if config.get("use_lora", False):
            # LoRA is more memory efficient
            pass
        elif "model_name" in config:
            # Check model size
            model_name = config["model_name"].lower()
            if any(x in model_name for x in ["1t", "1-t", "moellama-1t"]):
                warnings.append(
                    ValidationError(
                        "model_size",
                        "Model appears to be 1T+ parameters. Ensure you have >80GB GPU memory or use quantization.",
                        "warning",
                    )
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
