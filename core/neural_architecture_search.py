"""
Neural Architecture Search for GRPO Agent Framework

This module provides automated neural architecture search capabilities to optimize
agent network architectures for specific tasks and performance requirements.
"""

import asyncio
import json
import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .advanced_monitoring import get_monitoring_service, monitor_async_function
from .error_handling import ErrorHandler, RetryConfig, retry_async
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Neural architecture search strategies"""

    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GRADIENT_BASED = "gradient_based"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    PROGRESSIVE = "progressive"


class LayerType(Enum):
    """Types of neural network layers"""

    LINEAR = "linear"
    ATTENTION = "attention"
    TRANSFORMER_BLOCK = "transformer_block"
    CONV1D = "conv1d"
    LSTM = "lstm"
    GRU = "gru"
    DROPOUT = "dropout"
    LAYER_NORM = "layer_norm"
    RESIDUAL = "residual"
    GATE = "gate"


class ActivationType(Enum):
    """Activation function types"""

    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    MISH = "mish"


@dataclass
class LayerConfig:
    """Configuration for a neural network layer"""

    layer_type: LayerType
    input_dim: int
    output_dim: int
    activation: Optional[ActivationType] = None
    dropout_rate: float = 0.0
    use_bias: bool = True
    layer_specific_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_type": self.layer_type.value,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation.value if self.activation else None,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "layer_specific_params": self.layer_specific_params,
        }


@dataclass
class ArchitectureConfig:
    """Complete neural architecture configuration"""

    layers: List[LayerConfig]
    total_parameters: int
    depth: int
    width: int  # Maximum layer width
    architecture_id: str
    performance_score: float = 0.0
    training_time: float = 0.0
    memory_usage: float = 0.0
    flops: int = 0  # Floating point operations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "total_parameters": self.total_parameters,
            "depth": self.depth,
            "width": self.width,
            "architecture_id": self.architecture_id,
            "performance_score": self.performance_score,
            "training_time": self.training_time,
            "memory_usage": self.memory_usage,
            "flops": self.flops,
        }


class ArchitectureSearchSpace:
    """Defines the search space for neural architectures"""

    def __init__(
        self,
        min_depth: int = 2,
        max_depth: int = 12,
        min_width: int = 64,
        max_width: int = 2048,
        allowed_layer_types: List[LayerType] = None,
        allowed_activations: List[ActivationType] = None,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_width = min_width
        self.max_width = max_width

        self.allowed_layer_types = allowed_layer_types or [
            LayerType.LINEAR,
            LayerType.ATTENTION,
            LayerType.TRANSFORMER_BLOCK,
            LayerType.DROPOUT,
            LayerType.LAYER_NORM,
            LayerType.RESIDUAL,
        ]

        self.allowed_activations = allowed_activations or [
            ActivationType.RELU,
            ActivationType.GELU,
            ActivationType.SWISH,
            ActivationType.TANH,
        ]

        # Width options (powers of 2 for efficiency)
        self.width_options = [2**i for i in range(6, 12)]  # 64 to 2048
        self.width_options = [
            w for w in self.width_options if self.min_width <= w <= self.max_width
        ]

    def sample_random_architecture(
        self, input_dim: int, output_dim: int
    ) -> ArchitectureConfig:
        """Sample a random architecture from the search space"""
        depth = random.randint(self.min_depth, self.max_depth)
        layers = []

        current_dim = input_dim
        max_width = 0

        for i in range(depth):
            # Choose layer type
            if i == depth - 1:  # Output layer
                layer_type = LayerType.LINEAR
                next_dim = output_dim
            else:
                layer_type = random.choice(self.allowed_layer_types)
                next_dim = random.choice(self.width_options)

            # Skip certain layers at input/output
            if i == 0 and layer_type in [LayerType.DROPOUT, LayerType.LAYER_NORM]:
                layer_type = LayerType.LINEAR

            # Create layer config
            layer_config = self._create_layer_config(layer_type, current_dim, next_dim)
            layers.append(layer_config)

            current_dim = next_dim
            max_width = max(max_width, current_dim)

        # Calculate total parameters
        total_params = self._calculate_parameters(layers)

        architecture_id = self._generate_architecture_id(layers)

        return ArchitectureConfig(
            layers=layers,
            total_parameters=total_params,
            depth=depth,
            width=max_width,
            architecture_id=architecture_id,
        )

    def _create_layer_config(
        self, layer_type: LayerType, input_dim: int, output_dim: int
    ) -> LayerConfig:
        """Create a layer configuration"""
        activation = (
            random.choice(self.allowed_activations)
            if layer_type != LayerType.DROPOUT
            else None
        )
        dropout_rate = (
            random.uniform(0.0, 0.3) if layer_type == LayerType.DROPOUT else 0.0
        )

        layer_specific_params = {}

        if layer_type == LayerType.ATTENTION:
            layer_specific_params.update(
                {
                    "num_heads": random.choice([4, 8, 12, 16]),
                    "head_dim": output_dim // random.choice([4, 8, 12, 16]),
                }
            )
        elif layer_type == LayerType.TRANSFORMER_BLOCK:
            layer_specific_params.update(
                {
                    "num_heads": random.choice([4, 8, 12, 16]),
                    "feedforward_dim": output_dim * random.choice([2, 4]),
                    "dropout": random.uniform(0.0, 0.2),
                }
            )
        elif layer_type == LayerType.CONV1D:
            layer_specific_params.update(
                {
                    "kernel_size": random.choice([3, 5, 7]),
                    "stride": random.choice([1, 2]),
                    "padding": "same",
                }
            )
        elif layer_type == LayerType.LSTM or layer_type == LayerType.GRU:
            layer_specific_params.update(
                {
                    "num_layers": random.choice([1, 2, 3]),
                    "bidirectional": random.choice([True, False]),
                }
            )

        return LayerConfig(
            layer_type=layer_type,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            layer_specific_params=layer_specific_params,
        )

    def _calculate_parameters(self, layers: List[LayerConfig]) -> int:
        """Calculate total number of parameters"""
        total_params = 0

        for layer in layers:
            if layer.layer_type == LayerType.LINEAR:
                params = layer.input_dim * layer.output_dim
                if layer.use_bias:
                    params += layer.output_dim
                total_params += params
            elif layer.layer_type == LayerType.ATTENTION:
                # Simplified attention parameter count
                num_heads = layer.layer_specific_params.get("num_heads", 8)
                head_dim = layer.layer_specific_params.get(
                    "head_dim", layer.output_dim // num_heads
                )
                params = layer.input_dim * (
                    3 * num_heads * head_dim
                )  # Q, K, V projections
                params += num_heads * head_dim * layer.output_dim  # Output projection
                total_params += params
            elif layer.layer_type == LayerType.TRANSFORMER_BLOCK:
                # Simplified transformer block parameter count
                attention_params = layer.input_dim * layer.output_dim * 4  # Attention
                ffn_dim = layer.layer_specific_params.get(
                    "feedforward_dim", layer.output_dim * 4
                )
                ffn_params = layer.output_dim * ffn_dim + ffn_dim * layer.output_dim
                layer_norm_params = layer.output_dim * 2  # Two layer norms
                total_params += attention_params + ffn_params + layer_norm_params
            # Other layer types have minimal parameters or are calculated differently

        return total_params

    def _generate_architecture_id(self, layers: List[LayerConfig]) -> str:
        """Generate a unique ID for the architecture"""
        layer_signature = []
        for layer in layers:
            sig = f"{layer.layer_type.value}_{layer.output_dim}"
            if layer.activation:
                sig += f"_{layer.activation.value}"
            layer_signature.append(sig)

        return "_".join(layer_signature)


class ArchitectureEvaluator:
    """Evaluates neural architectures"""

    def __init__(self):
        self.evaluation_cache: Dict[str, float] = {}
        self.error_handler = ErrorHandler()

    async def evaluate_architecture(
        self,
        architecture: ArchitectureConfig,
        train_fn: Callable,
        eval_fn: Callable,
        max_training_steps: int = 1000,
    ) -> float:
        """Evaluate an architecture's performance"""
        if architecture.architecture_id in self.evaluation_cache:
            return self.evaluation_cache[architecture.architecture_id]

        try:
            start_time = datetime.now()

            # Create and train the model
            model = self._build_model(architecture)
            performance_score = await self._train_and_evaluate(
                model, train_fn, eval_fn, max_training_steps
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # Update architecture with results
            architecture.performance_score = performance_score
            architecture.training_time = training_time

            # Cache the result
            self.evaluation_cache[architecture.architecture_id] = performance_score

            return performance_score

        except Exception as e:
            self.error_handler.handle_error(e, "architecture_evaluator", "evaluate")
            return 0.0

    def _build_model(self, architecture: ArchitectureConfig) -> nn.Module:
        """Build a PyTorch model from architecture config"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural architecture search")

        layers = []

        for layer_config in architecture.layers:
            layer = self._create_layer(layer_config)
            layers.append(layer)

        return nn.Sequential(*layers)

    def _create_layer(self, layer_config: LayerConfig) -> nn.Module:
        """Create a PyTorch layer from config"""
        if layer_config.layer_type == LayerType.LINEAR:
            layer = nn.Linear(
                layer_config.input_dim,
                layer_config.output_dim,
                bias=layer_config.use_bias,
            )
        elif layer_config.layer_type == LayerType.DROPOUT:
            layer = nn.Dropout(layer_config.dropout_rate)
        elif layer_config.layer_type == LayerType.LAYER_NORM:
            layer = nn.LayerNorm(layer_config.input_dim)
        elif layer_config.layer_type == LayerType.ATTENTION:
            layer = self._create_attention_layer(layer_config)
        elif layer_config.layer_type == LayerType.TRANSFORMER_BLOCK:
            layer = self._create_transformer_block(layer_config)
        else:
            # Default to linear layer
            layer = nn.Linear(layer_config.input_dim, layer_config.output_dim)

        # Add activation if specified
        if layer_config.activation:
            activation = self._get_activation(layer_config.activation)
            return nn.Sequential(layer, activation)

        return layer

    def _create_attention_layer(self, layer_config: LayerConfig) -> nn.Module:
        """Create multi-head attention layer"""
        num_heads = layer_config.layer_specific_params.get("num_heads", 8)
        return nn.MultiheadAttention(
            layer_config.input_dim,
            num_heads,
            dropout=layer_config.dropout_rate,
            batch_first=True,
        )

    def _create_transformer_block(self, layer_config: LayerConfig) -> nn.Module:
        """Create transformer encoder layer"""
        num_heads = layer_config.layer_specific_params.get("num_heads", 8)
        feedforward_dim = layer_config.layer_specific_params.get(
            "feedforward_dim", layer_config.output_dim * 4
        )
        dropout = layer_config.layer_specific_params.get("dropout", 0.1)

        return nn.TransformerEncoderLayer(
            d_model=layer_config.input_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )

    def _get_activation(self, activation_type: ActivationType) -> nn.Module:
        """Get activation function"""
        if activation_type == ActivationType.RELU:
            return nn.ReLU()
        elif activation_type == ActivationType.GELU:
            return nn.GELU()
        elif activation_type == ActivationType.SWISH:
            return nn.SiLU()  # SiLU is equivalent to Swish
        elif activation_type == ActivationType.TANH:
            return nn.Tanh()
        elif activation_type == ActivationType.SIGMOID:
            return nn.Sigmoid()
        elif activation_type == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU()
        else:
            return nn.ReLU()  # Default

    async def _train_and_evaluate(
        self, model: nn.Module, train_fn: Callable, eval_fn: Callable, max_steps: int
    ) -> float:
        """Train and evaluate the model"""
        # This is a simplified training loop
        # In practice, this would use the full GRPO training pipeline

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Quick training loop
        for step in range(min(max_steps, 100)):  # Reduced for NAS efficiency
            loss = await train_fn(model, optimizer)
            if step % 20 == 0:
                logger.debug(f"NAS training step {step}, loss: {loss:.4f}")

        # Evaluate performance
        performance = await eval_fn(model)
        return performance


class EvolutionarySearch:
    """Evolutionary search for neural architectures"""

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.6,
        elite_ratio: float = 0.2,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = int(elite_ratio * population_size)

        self.generation = 0
        self.best_architecture: Optional[ArchitectureConfig] = None
        self.best_score = float("-inf")

    async def search(
        self,
        search_space: ArchitectureSearchSpace,
        evaluator: ArchitectureEvaluator,
        train_fn: Callable,
        eval_fn: Callable,
        input_dim: int,
        output_dim: int,
        max_generations: int = 10,
    ) -> ArchitectureConfig:
        """Run evolutionary search"""

        # Initialize population
        population = [
            search_space.sample_random_architecture(input_dim, output_dim)
            for _ in range(self.population_size)
        ]

        for generation in range(max_generations):
            self.generation = generation
            logger.info(f"NAS Generation {generation + 1}/{max_generations}")

            # Evaluate population
            for arch in population:
                if arch.performance_score == 0.0:  # Not evaluated yet
                    score = await evaluator.evaluate_architecture(
                        arch, train_fn, eval_fn
                    )
                    arch.performance_score = score

                    if score > self.best_score:
                        self.best_score = score
                        self.best_architecture = arch
                        logger.info(f"New best architecture found: {score:.4f}")

            # Sort by performance
            population.sort(key=lambda x: x.performance_score, reverse=True)

            # Create next generation
            next_population = []

            # Keep elite
            next_population.extend(population[: self.elite_size])

            # Generate offspring
            while len(next_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1, parent2 = random.sample(
                        population[: self.population_size // 2], 2
                    )
                    child = self._crossover(
                        parent1, parent2, search_space, input_dim, output_dim
                    )
                else:
                    # Mutation only
                    parent = random.choice(population[: self.population_size // 2])
                    child = self._mutate(parent, search_space, input_dim, output_dim)

                next_population.append(child)

            population = next_population

        return self.best_architecture

    def _crossover(
        self,
        parent1: ArchitectureConfig,
        parent2: ArchitectureConfig,
        search_space: ArchitectureSearchSpace,
        input_dim: int,
        output_dim: int,
    ) -> ArchitectureConfig:
        """Create offspring through crossover"""
        # Simple crossover: take layers from both parents
        min_depth = min(len(parent1.layers), len(parent2.layers))
        crossover_point = random.randint(1, min_depth - 1)

        # Combine layers
        child_layers = (
            parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        )

        # Ensure dimensional consistency
        child_layers = self._fix_dimensions(child_layers, input_dim, output_dim)

        # Create new architecture
        total_params = search_space._calculate_parameters(child_layers)
        max_width = max(layer.output_dim for layer in child_layers)
        arch_id = search_space._generate_architecture_id(child_layers)

        return ArchitectureConfig(
            layers=child_layers,
            total_parameters=total_params,
            depth=len(child_layers),
            width=max_width,
            architecture_id=arch_id,
        )

    def _mutate(
        self,
        parent: ArchitectureConfig,
        search_space: ArchitectureSearchSpace,
        input_dim: int,
        output_dim: int,
    ) -> ArchitectureConfig:
        """Mutate an architecture"""
        child_layers = [layer for layer in parent.layers]  # Copy layers

        # Random mutations
        if random.random() < self.mutation_rate:
            # Add layer
            if len(child_layers) < search_space.max_depth:
                insert_pos = random.randint(0, len(child_layers) - 1)
                new_layer = search_space._create_layer_config(
                    random.choice(search_space.allowed_layer_types),
                    child_layers[insert_pos].input_dim,
                    child_layers[insert_pos].output_dim,
                )
                child_layers.insert(insert_pos, new_layer)

        if random.random() < self.mutation_rate:
            # Remove layer
            if len(child_layers) > search_space.min_depth:
                remove_pos = random.randint(
                    1, len(child_layers) - 2
                )  # Don't remove first/last
                child_layers.pop(remove_pos)

        if random.random() < self.mutation_rate:
            # Modify layer width
            layer_idx = random.randint(
                0, len(child_layers) - 2
            )  # Don't modify output layer
            new_width = random.choice(search_space.width_options)
            child_layers[layer_idx].output_dim = new_width

        # Fix dimensions
        child_layers = self._fix_dimensions(child_layers, input_dim, output_dim)

        # Create new architecture
        total_params = search_space._calculate_parameters(child_layers)
        max_width = max(layer.output_dim for layer in child_layers)
        arch_id = search_space._generate_architecture_id(child_layers)

        return ArchitectureConfig(
            layers=child_layers,
            total_parameters=total_params,
            depth=len(child_layers),
            width=max_width,
            architecture_id=arch_id,
        )

    def _fix_dimensions(
        self, layers: List[LayerConfig], input_dim: int, output_dim: int
    ) -> List[LayerConfig]:
        """Fix dimensional inconsistencies in layer sequence"""
        if not layers:
            return layers

        # Fix input dimension
        layers[0].input_dim = input_dim

        # Fix intermediate dimensions
        for i in range(1, len(layers)):
            layers[i].input_dim = layers[i - 1].output_dim

        # Fix output dimension
        if layers:
            layers[-1].output_dim = output_dim

        return layers


class NeuralArchitectureSearch:
    """Main Neural Architecture Search controller"""

    def __init__(
        self,
        strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY,
        search_space: Optional[ArchitectureSearchSpace] = None,
    ):
        self.strategy = strategy
        self.search_space = search_space or ArchitectureSearchSpace()
        self.evaluator = ArchitectureEvaluator()

        self.search_history: List[ArchitectureConfig] = []
        self.best_architectures: List[ArchitectureConfig] = []

        self.monitoring = get_monitoring_service()
        self.error_handler = ErrorHandler()

    @monitor_async_function("neural_architecture_search")
    async def search_optimal_architecture(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        input_dim: int,
        output_dim: int,
        max_search_time: int = 3600,  # 1 hour
        target_performance: Optional[float] = None,
    ) -> ArchitectureConfig:
        """Search for optimal neural architecture"""

        start_time = datetime.now()

        try:
            if self.strategy == SearchStrategy.EVOLUTIONARY:
                search_algorithm = EvolutionarySearch()
                best_arch = await search_algorithm.search(
                    self.search_space,
                    self.evaluator,
                    train_fn,
                    eval_fn,
                    input_dim,
                    output_dim,
                )

            elif self.strategy == SearchStrategy.RANDOM_SEARCH:
                best_arch = await self._random_search(
                    train_fn, eval_fn, input_dim, output_dim, max_search_time
                )

            else:
                # Default to random search
                best_arch = await self._random_search(
                    train_fn, eval_fn, input_dim, output_dim, max_search_time
                )

            # Log results
            search_time = (datetime.now() - start_time).total_seconds()
            await self._log_search_results(best_arch, search_time)

            return best_arch

        except Exception as e:
            self.error_handler.handle_error(e, "neural_architecture_search", "search")
            # Return a default architecture
            return self.search_space.sample_random_architecture(input_dim, output_dim)

    async def _random_search(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        input_dim: int,
        output_dim: int,
        max_time: int,
    ) -> ArchitectureConfig:
        """Random search for architectures"""

        start_time = datetime.now()
        best_arch = None
        best_score = float("-inf")

        search_count = 0
        while (datetime.now() - start_time).total_seconds() < max_time:
            # Sample random architecture
            arch = self.search_space.sample_random_architecture(input_dim, output_dim)

            # Evaluate
            score = await self.evaluator.evaluate_architecture(arch, train_fn, eval_fn)
            arch.performance_score = score

            self.search_history.append(arch)

            if score > best_score:
                best_score = score
                best_arch = arch
                logger.info(f"Random search found better architecture: {score:.4f}")

            search_count += 1

            # Early stopping if we've tried enough
            if search_count >= 50:
                break

        return best_arch

    async def _log_search_results(
        self, best_arch: ArchitectureConfig, search_time: float
    ):
        """Log search results to monitoring system"""
        metrics = {
            "best_performance": best_arch.performance_score,
            "best_architecture_params": best_arch.total_parameters,
            "best_architecture_depth": best_arch.depth,
            "best_architecture_width": best_arch.width,
            "search_time_seconds": search_time,
            "architectures_evaluated": len(self.search_history),
        }

        for metric_name, value in metrics.items():
            await self.monitoring.record_metric(f"nas.{metric_name}", value)

        logger.info(
            f"NAS completed: Best score {best_arch.performance_score:.4f}, "
            f"Params: {best_arch.total_parameters}, Time: {search_time:.2f}s"
        )

    def get_search_insights(self) -> Dict[str, Any]:
        """Get insights from the search process"""
        if not self.search_history:
            return {"message": "No search history available"}

        scores = [arch.performance_score for arch in self.search_history]
        param_counts = [arch.total_parameters for arch in self.search_history]

        return {
            "total_architectures_evaluated": len(self.search_history),
            "best_performance": max(scores),
            "average_performance": np.mean(scores),
            "performance_std": np.std(scores),
            "parameter_range": {
                "min": min(param_counts),
                "max": max(param_counts),
                "average": np.mean(param_counts),
            },
            "top_architectures": [
                arch.to_dict()
                for arch in sorted(
                    self.search_history, key=lambda x: x.performance_score, reverse=True
                )[:5]
            ],
        }


# Factory functions
def create_nas_controller(
    strategy: str = "evolutionary",
    min_depth: int = 2,
    max_depth: int = 12,
    min_width: int = 64,
    max_width: int = 2048,
) -> NeuralArchitectureSearch:
    """Create a Neural Architecture Search controller"""
    search_strategy = SearchStrategy(strategy)
    search_space = ArchitectureSearchSpace(
        min_depth=min_depth,
        max_depth=max_depth,
        min_width=min_width,
        max_width=max_width,
    )

    return NeuralArchitectureSearch(search_strategy, search_space)


def create_custom_search_space(
    layer_types: List[str], activations: List[str], **kwargs
) -> ArchitectureSearchSpace:
    """Create a custom search space"""
    layer_type_enums = [LayerType(lt) for lt in layer_types]
    activation_enums = [ActivationType(act) for act in activations]

    return ArchitectureSearchSpace(
        allowed_layer_types=layer_type_enums,
        allowed_activations=activation_enums,
        **kwargs,
    )
