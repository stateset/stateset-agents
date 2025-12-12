"""
Performance Optimization Module for GRPO Agent Framework

This module provides advanced performance optimization techniques including
memory management, computational efficiency, and resource utilization.
"""

import asyncio
import gc
import logging
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

if torch is not None:  # pragma: no cover - optional dependency
    try:
        from torch.cuda.amp import GradScaler, autocast  # type: ignore
    except Exception:  # pragma: no cover
        GradScaler = None  # type: ignore[assignment]
        autocast = None  # type: ignore[assignment]
else:  # pragma: no cover
    GradScaler = None  # type: ignore[assignment]
    autocast = None  # type: ignore[assignment]

# Lazy import for transformers to avoid torch/torchvision compatibility issues
PreTrainedModel = Any  # type: ignore[assignment]
_transformers_perf_loaded = False

def _load_pretrained_model() -> bool:
    """Lazily load PreTrainedModel to avoid import-time errors."""
    global _transformers_perf_loaded, PreTrainedModel
    if _transformers_perf_loaded:
        return True
    try:
        from transformers import PreTrainedModel as _PreTrainedModel  # type: ignore
        PreTrainedModel = _PreTrainedModel
        _transformers_perf_loaded = True
        return True
    except (ImportError, RuntimeError):  # pragma: no cover
        return False

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer
else:
    Tensor = Any
    Optimizer = Any

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels"""

    CONSERVATIVE = "conservative"  # Safe optimizations
    BALANCED = "balanced"  # Default optimizations
    AGGRESSIVE = "aggressive"  # Maximum optimizations


@dataclass
class MemoryConfig:
    """Memory optimization configuration"""

    enable_gradient_checkpointing: bool = True
    enable_cpu_offload: bool = False
    max_memory_usage_percent: float = 0.85
    clear_cache_frequency: int = 100  # Clear cache every N steps
    use_memory_mapping: bool = True
    enable_memory_profiling: bool = False


@dataclass
class ComputeConfig:
    """Computational optimization configuration"""

    use_mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    use_flash_attention: bool = True
    enable_tensor_parallel: bool = False
    fused_optimizer: bool = True
    use_channels_last: bool = False


@dataclass
class DataConfig:
    """Data loading optimization configuration"""

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    batch_size_scaling: bool = True
    dynamic_batching: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""

    timestamp: float = field(default_factory=time.time)
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    batch_size: int = 0
    sequence_length: int = 0


class MemoryMonitor:
    """Real-time memory monitoring and management"""

    def __init__(self, config: MemoryConfig):
        if psutil is None:
            raise ImportError(
                "psutil is required for MemoryMonitor. Install the 'training' extra: "
                "pip install stateset-agents[training]"
            )
        self.config = config
        self.peak_memory = 0.0
        self.memory_history: List[float] = []
        self.gc_stats = {"calls": 0, "freed_mb": 0.0}

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        # CPU memory
        assert psutil is not None  # Narrow type for type checkers
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024

        # GPU memory
        gpu_memory_mb = 0.0
        if torch and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.peak_memory = max(self.peak_memory, gpu_memory_mb)

        self.memory_history.append(gpu_memory_mb)
        if len(self.memory_history) > 1000:  # Keep last 1000 measurements
            self.memory_history.pop(0)

        return {
            "cpu_memory_mb": cpu_memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
            "peak_gpu_memory_mb": self.peak_memory,
            "memory_percent": (
                gpu_memory_mb
                / (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
                if torch and torch.cuda.is_available()
                else 0.0
            ),
        }

    def cleanup_memory(self) -> float:
        """Force memory cleanup and return freed memory"""
        initial_memory = self.get_memory_usage()["gpu_memory_mb"]

        # Clear PyTorch cache
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()
        self.gc_stats["calls"] += 1

        final_memory = self.get_memory_usage()["gpu_memory_mb"]
        freed_mb = initial_memory - final_memory
        self.gc_stats["freed_mb"] += freed_mb

        logger.debug(f"Memory cleanup freed {freed_mb:.2f}MB")
        return freed_mb

    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        memory_usage = self.get_memory_usage()
        return memory_usage["memory_percent"] > self.config.max_memory_usage_percent

    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for tracking memory usage of operations"""
        start_memory = self.get_memory_usage()
        start_time = time.time()

        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            end_time = time.time()

            memory_delta = end_memory["gpu_memory_mb"] - start_memory["gpu_memory_mb"]
            time_delta = end_time - start_time

            logger.debug(
                f"{operation_name}: "
                f"Memory delta: {memory_delta:.2f}MB, "
                f"Time: {time_delta:.3f}s"
            )


class ModelOptimizer:
    """Model-specific optimizations"""

    def __init__(self, config: ComputeConfig):
        if torch is None or nn is None:
            raise ImportError(
                "PyTorch is required for ModelOptimizer. Install the 'training' extra: "
                "pip install stateset-agents[training]"
            )
        self.config = config
        self.scaler = None
        if config.use_mixed_precision:
            if GradScaler is None:
                raise ImportError(
                    "torch.cuda.amp.GradScaler is unavailable. Upgrade PyTorch or "
                    "disable mixed precision."
                )
            self.scaler = GradScaler()

    def optimize_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply model optimizations"""

        # Enable gradient checkpointing
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

        # Compile model (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)
            logger.info("Compiled model with PyTorch 2.0")

        # Use channels last memory format
        if self.config.use_channels_last:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Converted model to channels_last memory format")

        # Apply mixed precision optimizations
        if self.config.use_mixed_precision:
            # Convert BatchNorm to FP32 for stability
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.float()

        return model

    def optimize_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Optimize optimizer settings"""

        # Use fused optimizer if available
        if self.config.fused_optimizer and hasattr(optimizer, "fused"):
            try:
                # Create fused version of optimizer
                optimizer_class = optimizer.__class__
                fused_optimizer = optimizer_class(
                    optimizer.param_groups[0]["params"],
                    **optimizer.defaults,
                    fused=True,
                )
                logger.info(f"Enabled fused {optimizer_class.__name__}")
                return fused_optimizer
            except Exception as e:
                logger.warning(f"Failed to enable fused optimizer: {e}")

        return optimizer

    @contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision operations"""
        if self.config.use_mixed_precision and self.scaler:
            if autocast is None:
                logger.warning(
                    "Requested mixed precision but torch.cuda.amp.autocast is unavailable. "
                    "Proceeding without autocast."
                )
                with nullcontext():
                    yield None
            else:
                with autocast():
                    yield self.scaler
        else:
            yield None


class BatchOptimizer:
    """Dynamic batch size optimization"""

    def __init__(self, config: DataConfig):
        if torch is None:
            raise ImportError(
                "PyTorch is required for BatchOptimizer. Install the 'training' extra: "
                "pip install stateset-agents[training]"
            )
        self.config = config
        self.optimal_batch_size = 1
        self.performance_history: List[PerformanceMetrics] = []
        self.last_oom_batch_size = None

    def find_optimal_batch_size(
        self,
        model: PreTrainedModel,
        sample_input: Tensor,
        max_batch_size: int = 64,
        growth_factor: float = 1.5,
    ) -> int:
        """Find optimal batch size through binary search"""

        logger.info("Finding optimal batch size...")

        # Start with small batch size
        current_batch_size = 1
        last_working_size = 1

        while current_batch_size <= max_batch_size:
            try:
                # Test batch size
                batch_input = sample_input.repeat(current_batch_size, 1)

                with torch.no_grad():
                    start_time = time.time()
                    output = model(batch_input)
                    end_time = time.time()

                # Calculate throughput
                throughput = current_batch_size / (end_time - start_time)

                logger.debug(
                    f"Batch size {current_batch_size}: {throughput:.2f} samples/sec"
                )

                last_working_size = current_batch_size
                current_batch_size = int(current_batch_size * growth_factor)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"OOM at batch size {current_batch_size}")
                    self.last_oom_batch_size = current_batch_size
                    break
                else:
                    raise

        self.optimal_batch_size = last_working_size
        logger.info(f"Optimal batch size: {self.optimal_batch_size}")

        return self.optimal_batch_size

    def adjust_batch_size(self, current_metrics: PerformanceMetrics) -> int:
        """Dynamically adjust batch size based on performance"""

        if not self.config.batch_size_scaling:
            return self.optimal_batch_size

        self.performance_history.append(current_metrics)

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Check for memory pressure
        if current_metrics.gpu_memory_usage_mb / 1024 > 0.9:  # > 90% GPU memory
            new_size = max(1, int(self.optimal_batch_size * 0.8))
            logger.info(f"Reducing batch size due to memory pressure: {new_size}")
            return new_size

        # Check for underutilization
        if (
            len(self.performance_history) >= 10
            and current_metrics.gpu_utilization_percent < 70.0
        ):
            new_size = min(
                self.last_oom_batch_size - 1
                if self.last_oom_batch_size
                else self.optimal_batch_size * 2,
                int(self.optimal_batch_size * 1.2),
            )
            logger.info(f"Increasing batch size due to underutilization: {new_size}")
            return new_size

        return self.optimal_batch_size


class PerformanceProfiler:
    """Performance profiling and analysis"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.profiling_active = False

    def start_profiling(self):
        """Start performance profiling"""
        self.profiling_active = True
        logger.info("Started performance profiling")

    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return analysis"""
        self.profiling_active = False
        logger.info("Stopped performance profiling")

        return self.analyze_performance()

    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        if self.profiling_active:
            self.metrics_history.append(metrics)

            # Limit history size
            if len(self.metrics_history) > 10000:
                self.metrics_history = self.metrics_history[-5000:]

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze recorded performance metrics"""
        if not self.metrics_history:
            return {"error": "No metrics recorded"}

        # Calculate statistics
        throughputs = [m.throughput_tokens_per_sec for m in self.metrics_history]
        latencies = [m.latency_ms for m in self.metrics_history]
        memory_usage = [m.gpu_memory_usage_mb for m in self.metrics_history]

        analysis = {
            "total_samples": len(self.metrics_history),
            "throughput": {
                "mean": sum(throughputs) / len(throughputs),
                "max": max(throughputs),
                "min": min(throughputs),
                "p95": sorted(throughputs)[int(len(throughputs) * 0.95)],
            },
            "latency": {
                "mean": sum(latencies) / len(latencies),
                "max": max(latencies),
                "min": min(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            },
            "memory": {
                "mean": sum(memory_usage) / len(memory_usage),
                "max": max(memory_usage),
                "peak_efficiency": min(memory_usage) / max(memory_usage)
                if max(memory_usage) > 0
                else 0,
            },
        }

        return analysis


class PerformanceOptimizer:
    """Main performance optimization coordinator"""

    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
        memory_config: Optional[MemoryConfig] = None,
        compute_config: Optional[ComputeConfig] = None,
        data_config: Optional[DataConfig] = None,
    ):
        self.optimization_level = optimization_level
        self.memory_config = memory_config or self._get_default_memory_config()
        self.compute_config = compute_config or self._get_default_compute_config()
        self.data_config = data_config or self._get_default_data_config()

        self.memory_monitor = MemoryMonitor(self.memory_config)
        self.model_optimizer = ModelOptimizer(self.compute_config)
        self.batch_optimizer = BatchOptimizer(self.data_config)
        self.profiler = PerformanceProfiler()

        self.optimization_step = 0

    def _get_default_memory_config(self) -> MemoryConfig:
        """Get default memory configuration based on optimization level"""
        if self.optimization_level == OptimizationLevel.CONSERVATIVE:
            return MemoryConfig(
                enable_gradient_checkpointing=True,
                max_memory_usage_percent=0.7,
                clear_cache_frequency=50,
            )
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return MemoryConfig(
                enable_gradient_checkpointing=True,
                enable_cpu_offload=True,
                max_memory_usage_percent=0.95,
                clear_cache_frequency=200,
            )
        else:  # BALANCED
            return MemoryConfig()

    def _get_default_compute_config(self) -> ComputeConfig:
        """Get default compute configuration based on optimization level"""
        if self.optimization_level == OptimizationLevel.CONSERVATIVE:
            return ComputeConfig(
                use_mixed_precision=False,
                compile_model=False,
                use_flash_attention=False,
            )
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return ComputeConfig(
                use_mixed_precision=True,
                compile_model=True,
                use_flash_attention=True,
                enable_tensor_parallel=True,
                fused_optimizer=True,
            )
        else:  # BALANCED
            return ComputeConfig()

    def _get_default_data_config(self) -> DataConfig:
        """Get default data configuration based on optimization level"""
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return DataConfig(
                num_workers=8, batch_size_scaling=True, dynamic_batching=True
            )
        else:
            return DataConfig()

    def optimize_training_step(self, model: PreTrainedModel) -> Dict[str, Any]:
        """Optimize a single training step"""

        self.optimization_step += 1

        # Monitor memory
        memory_stats = self.memory_monitor.get_memory_usage()

        # Cleanup memory if needed
        if (
            self.optimization_step % self.memory_config.clear_cache_frequency == 0
            or self.memory_monitor.should_cleanup()
        ):
            freed_memory = self.memory_monitor.cleanup_memory()
            logger.debug(f"Step {self.optimization_step}: Freed {freed_memory:.2f}MB")

        # Record metrics
        metrics = PerformanceMetrics(
            memory_usage_mb=memory_stats["cpu_memory_mb"],
            gpu_memory_usage_mb=memory_stats["gpu_memory_mb"],
            gpu_utilization_percent=memory_stats["memory_percent"] * 100,
        )

        self.profiler.record_metrics(metrics)

        return {
            "step": self.optimization_step,
            "memory_stats": memory_stats,
            "optimization_active": True,
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""

        memory_stats = self.memory_monitor.get_memory_usage()
        performance_analysis = self.profiler.analyze_performance()

        return {
            "optimization_level": self.optimization_level.value,
            "total_steps": self.optimization_step,
            "current_memory": memory_stats,
            "gc_stats": self.memory_monitor.gc_stats,
            "optimal_batch_size": self.batch_optimizer.optimal_batch_size,
            "performance_analysis": performance_analysis,
            "recommendations": self._generate_recommendations(
                memory_stats, performance_analysis
            ),
        }

    def _generate_recommendations(
        self, memory_stats: Dict[str, float], performance_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations"""

        recommendations = []

        # Memory recommendations
        if memory_stats["memory_percent"] > 0.9:
            recommendations.append(
                "Consider reducing batch size or enabling CPU offload"
            )

        if memory_stats["memory_percent"] < 0.5:
            recommendations.append(
                "Consider increasing batch size for better GPU utilization"
            )

        # Performance recommendations
        if performance_analysis.get("throughput", {}).get("mean", 0) < 100:
            recommendations.append("Consider enabling mixed precision training")

        if self.optimization_level == OptimizationLevel.CONSERVATIVE:
            recommendations.append(
                "Consider upgrading to 'balanced' optimization level"
            )

        return recommendations
