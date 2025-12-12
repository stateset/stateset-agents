"""
Type System and Validation for GRPO Agent Framework

This module provides comprehensive type definitions, validation, and 
type-safe interfaces for all framework components.
"""

import asyncio
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import NotRequired, TypedDict

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
AgentType = TypeVar("AgentType", bound="Agent")
EnvironmentType = TypeVar("EnvironmentType", bound="Environment")
RewardType = TypeVar("RewardType", bound="RewardFunction")

# Basic types
UserId = str
ConversationId = str
ModelName = str
TokenCount = int
Score = float
Timestamp = float


# Configuration types
class DeviceType(Enum):
    """Supported device types"""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class ModelSize(Enum):
    """Model size categories"""

    SMALL = "small"  # < 1B parameters
    MEDIUM = "medium"  # 1B - 7B parameters
    LARGE = "large"  # 7B - 30B parameters
    XLARGE = "xlarge"  # > 30B parameters


class TrainingStage(Enum):
    """Training stages"""

    INITIALIZATION = "initialization"
    WARMUP = "warmup"
    TRAINING = "training"
    EVALUATION = "evaluation"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


# Configuration TypedDicts
class ModelConfig(TypedDict, total=False):
    """Type-safe model configuration"""

    model_name: str
    device: DeviceType
    torch_dtype: Literal["float16", "bfloat16", "float32"]
    max_length: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    pad_token_id: NotRequired[int]
    eos_token_id: NotRequired[int]
    use_cache: NotRequired[bool]
    trust_remote_code: NotRequired[bool]


class TrainingConfig(TypedDict, total=False):
    """Type-safe training configuration"""

    num_epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    save_steps: int
    eval_steps: int
    max_grad_norm: float
    weight_decay: float
    adam_epsilon: float
    lr_scheduler_type: Literal["linear", "cosine", "polynomial", "constant"]
    output_dir: str
    seed: NotRequired[int]
    dataloader_num_workers: NotRequired[int]
    fp16: NotRequired[bool]
    bf16: NotRequired[bool]


class ConversationTurn(TypedDict):
    """Type-safe conversation turn"""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: NotRequired[Timestamp]
    metadata: NotRequired[Dict[str, Any]]


class TrajectoryData(TypedDict):
    """Type-safe trajectory data"""

    conversation_id: ConversationId
    turns: List[ConversationTurn]
    total_tokens: TokenCount
    duration_seconds: float
    context_metadata: NotRequired[Dict[str, Any]]


class RewardMetrics(TypedDict):
    """Type-safe reward metrics"""

    total_score: Score
    component_scores: Dict[str, Score]
    confidence: float
    explanation: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]


# Protocol definitions for type-safe interfaces
class Configurable(Protocol):
    """Protocol for configurable components"""

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component"""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        ...


class Trainable(Protocol):
    """Protocol for trainable components"""

    async def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute a single training step"""
        ...

    def get_trainable_parameters(self) -> List[Any]:
        """Get trainable parameters"""
        ...


class Evaluable(Protocol):
    """Protocol for evaluable components"""

    async def evaluate(self, test_data: Any) -> Dict[str, float]:
        """Evaluate performance"""
        ...


class Serializable(Protocol):
    """Protocol for serializable components"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Create from dictionary"""
        ...


# Generic base classes
class TypedComponent(Generic[T], ABC):
    """Base class for type-safe components"""

    def __init__(self, config: T):
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration"""
        pass

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with validation"""
        if isinstance(self.config, dict):
            self.config.update(updates)
        else:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        self._validate_config()


class Agent(Protocol):
    """Type-safe agent interface"""

    async def generate_response(
        self,
        messages: Union[str, List[ConversationTurn]],
        **kwargs: Any,
    ) -> str:
        """Generate response to conversation"""
        ...

    async def start_conversation(
        self, user_id: UserId, initial_context: Optional[Dict[str, Any]] = None
    ) -> ConversationId:
        """Start new conversation"""
        ...


class Environment(Protocol):
    """Type-safe environment interface"""

    async def reset(self) -> Dict[str, Any]:
        """Reset environment state"""
        ...

    async def step(
        self, action: str, state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Execute environment step"""
        ...


class RewardFunction(Protocol):
    """Type-safe reward function interface"""

    async def compute_reward(
        self, trajectory: TrajectoryData, context: Optional[Dict[str, Any]] = None
    ) -> RewardMetrics:
        """Compute reward for trajectory"""
        ...


# Validation utilities
class TypeValidator:
    """Type validation utilities"""

    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Validate that value matches expected type"""
        try:
            if hasattr(expected_type, "__origin__"):
                # Handle generic types
                origin = get_origin(expected_type)
                args = get_args(expected_type)

                if origin is Union:
                    return any(TypeValidator.validate_type(value, arg) for arg in args)
                elif (
                    str(origin).endswith("typing.Literal")
                    or origin.__name__ == "Literal"
                    if origin
                    else False
                ):
                    # typing.Literal support
                    return value in args
                elif origin is list:
                    if not isinstance(value, list):
                        return False
                    if args:
                        return all(
                            TypeValidator.validate_type(item, args[0]) for item in value
                        )
                    return True
                elif origin is dict:
                    if not isinstance(value, dict):
                        return False
                    if len(args) == 2:
                        key_type, value_type = args
                        return all(
                            TypeValidator.validate_type(k, key_type)
                            for k in value.keys()
                        ) and all(
                            TypeValidator.validate_type(v, value_type)
                            for v in value.values()
                        )
                    return True
                else:
                    return isinstance(value, origin)
            else:
                return isinstance(value, expected_type)
        except Exception:
            return False

    @staticmethod
    def validate_config(config: Dict[str, Any], config_type: type) -> List[str]:
        """Validate configuration against TypedDict"""
        errors = []

        if not hasattr(config_type, "__annotations__"):
            return errors

        annotations = get_type_hints(config_type)
        required_keys = getattr(config_type, "__required_keys__", set())

        # Check required keys
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        # Check types
        for key, value in config.items():
            if key in annotations:
                expected_type = annotations[key]
                if not TypeValidator.validate_type(value, expected_type):
                    errors.append(
                        f"Invalid type for {key}: expected {expected_type}, got {type(value)}"
                    )
            else:
                errors.append(f"Unexpected key: {key}")

        return errors

    @staticmethod
    def validate_function_signature(
        func: Callable, expected_signature: Callable
    ) -> bool:
        """Validate function signature matches expected"""
        try:
            sig = inspect.signature(func)
            expected_sig = inspect.signature(expected_signature)

            # Compare parameter names and types
            if len(sig.parameters) != len(expected_sig.parameters):
                return False

            for (name, param), (exp_name, exp_param) in zip(
                sig.parameters.items(), expected_sig.parameters.items()
            ):
                if name != exp_name:
                    return False
                if param.annotation != exp_param.annotation:
                    return False

            # Compare return type
            return sig.return_annotation == expected_sig.return_annotation
        except Exception:
            return False


class ConfigValidator:
    """Configuration validation with detailed error reporting"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        self.errors.clear()
        self.warnings.clear()

        # Validate against ModelConfig TypedDict
        type_errors = TypeValidator.validate_config(config, ModelConfig)
        self.errors.extend(type_errors)

        # Custom validation rules
        if "temperature" in config:
            temp = config["temperature"]
            if not 0.0 <= temp <= 2.0:
                self.errors.append(
                    f"Temperature must be between 0.0 and 2.0, got {temp}"
                )

        if "top_p" in config:
            top_p = config["top_p"]
            if not 0.0 <= top_p <= 1.0:
                self.errors.append(f"top_p must be between 0.0 and 1.0, got {top_p}")

        if "top_k" in config:
            top_k = config["top_k"]
            if top_k < 1:
                self.errors.append(f"top_k must be positive, got {top_k}")

        return len(self.errors) == 0

    def validate_training_config(self, config: Dict[str, Any]) -> bool:
        """Validate training configuration"""
        self.errors.clear()
        self.warnings.clear()

        # Validate against TrainingConfig TypedDict
        type_errors = TypeValidator.validate_config(config, TrainingConfig)
        self.errors.extend(type_errors)

        # Custom validation rules
        if "learning_rate" in config:
            lr = config["learning_rate"]
            if lr <= 0:
                self.errors.append("Learning rate must be positive")
            elif lr > 1e-2:
                self.warnings.append(
                    "Learning rate seems high, consider using smaller value"
                )

        if "batch_size" in config:
            batch_size = config["batch_size"]
            if batch_size < 1:
                self.errors.append("Batch size must be positive")
            elif batch_size > 128:
                self.warnings.append("Large batch size may cause memory issues")

        if "num_epochs" in config:
            epochs = config["num_epochs"]
            if epochs < 1:
                self.errors.append("Number of epochs must be positive")

        return len(self.errors) == 0

    def get_validation_report(self) -> Dict[str, List[str]]:
        """Get validation report"""
        return {
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "is_valid": len(self.errors) == 0,
        }


# Type-safe factory functions
def create_typed_config(config_type: type, **kwargs) -> Dict[str, Any]:
    """Create type-safe configuration"""
    config = kwargs.copy()

    # Validate configuration
    validator = ConfigValidator()
    if config_type == ModelConfig:
        if not validator.validate_model_config(config):
            raise ValueError(f"Invalid model config: {validator.errors}")
    elif config_type == TrainingConfig:
        if not validator.validate_training_config(config):
            raise ValueError(f"Invalid training config: {validator.errors}")

    return config


def ensure_type_safety(func: Callable) -> Callable:
    """Decorator to ensure type safety for function calls"""

    def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Validate argument types
        for param_name, value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                if not TypeValidator.validate_type(value, param.annotation):
                    raise TypeError(
                        f"Argument {param_name} expected {param.annotation}, "
                        f"got {type(value)}"
                    )

        return func(*args, **kwargs)

    return wrapper


async def ensure_async_type_safety(func: Callable) -> Callable:
    """Async version of type safety decorator"""

    async def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Validate argument types
        for param_name, value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                if not TypeValidator.validate_type(value, param.annotation):
                    raise TypeError(
                        f"Argument {param_name} expected {param.annotation}, "
                        f"got {type(value)}"
                    )

        result = await func(*args, **kwargs)

        # Validate return type
        if sig.return_annotation != inspect.Parameter.empty:
            # Handle Awaitable return types
            return_type = sig.return_annotation
            if (
                hasattr(return_type, "__origin__")
                and get_origin(return_type) is Awaitable
            ):
                return_type = get_args(return_type)[0]

            if not TypeValidator.validate_type(result, return_type):
                logger.warning(
                    f"Function {func.__name__} returned {type(result)}, "
                    f"expected {return_type}"
                )

        return result

    return wrapper


# Type-safe serialization
class TypeSafeSerializer:
    """Type-safe JSON serialization"""

    @staticmethod
    def serialize(obj: Any, expected_type: Optional[type] = None) -> str:
        """Serialize object with type information"""
        if expected_type and not TypeValidator.validate_type(obj, expected_type):
            raise TypeError(f"Object does not match expected type {expected_type}")

        # Add type information to serialized data
        data = {
            "__type__": obj.__class__.__name__,
            "__module__": obj.__class__.__module__,
            "data": obj,
        }

        return json.dumps(data, default=TypeSafeSerializer._default_serializer)

    @staticmethod
    def deserialize(json_str: str, expected_type: type) -> Any:
        """Deserialize object with type validation"""
        data = json.loads(json_str)

        if "__type__" in data and "__module__" in data:
            obj = data["data"]
            # Validate underlying object
            if not TypeValidator.validate_type(obj, expected_type):
                raise TypeError(
                    f"Deserialized object does not match expected type {expected_type}"
                )
            return data
        else:
            # Validate top-level
            if not TypeValidator.validate_type(data, expected_type):
                raise TypeError(
                    f"Deserialized object does not match expected type {expected_type}"
                )
            return data

    @staticmethod
    def _default_serializer(obj: Any) -> Any:
        """Default serializer for complex objects"""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)


# Export all types for convenient importing
__all__ = [
    # Type variables
    "T",
    "AgentType",
    "EnvironmentType",
    "RewardType",
    # Basic types
    "UserId",
    "ConversationId",
    "ModelName",
    "TokenCount",
    "Score",
    "Timestamp",
    # Enums
    "DeviceType",
    "ModelSize",
    "TrainingStage",
    # TypedDicts
    "ModelConfig",
    "TrainingConfig",
    "ConversationTurn",
    "TrajectoryData",
    "RewardMetrics",
    # Protocols
    "Configurable",
    "Trainable",
    "Evaluable",
    "Serializable",
    # Classes
    "TypedComponent",
    "Agent",
    "Environment",
    "RewardFunction",
    "TypeValidator",
    "ConfigValidator",
    "TypeSafeSerializer",
    # Functions
    "create_typed_config",
    "ensure_type_safety",
    "ensure_async_type_safety",
]
