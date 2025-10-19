"""
Comprehensive Tests for Enhanced GRPO Framework Features

Tests for v0.3.0 improvements:
- Error handling and resilience
- Performance optimization
- Type safety and validation
- Async resource management
"""

import asyncio
import logging
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from stateset_agents.core.async_pool import (
    AsyncResourceFactory,
    AsyncResourcePool,
    AsyncTaskManager,
    HTTPSessionFactory,
    PooledResource,
    PoolStats,
)

# Import framework components
from stateset_agents.core.error_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    DataException,
    ErrorHandler,
    GRPOException,
    ModelException,
    NetworkException,
    ResourceException,
    RetryConfig,
    TrainingException,
    ValidationException,
    get_error_summary,
    handle_error,
    retry_async,
)
from stateset_agents.core.performance_optimizer import (
    BatchOptimizer,
    ComputeConfig,
    DataConfig,
    MemoryConfig,
    MemoryMonitor,
    ModelOptimizer,
    OptimizationLevel,
    PerformanceMetrics,
    PerformanceOptimizer,
)
from stateset_agents.core.type_system import (
    ConfigValidator,
    DeviceType,
    ModelConfig,
    ModelSize,
    TrainingConfig,
    TrainingStage,
    TypeSafeSerializer,
    TypeValidator,
    create_typed_config,
    ensure_type_safety,
)


class TestErrorHandling:
    """Test error handling and resilience features"""

    def test_grpo_exceptions(self):
        """Test custom exception hierarchy"""
        # Test base exception
        base_error = GRPOException("Base error")
        assert base_error.category.value == "system"
        assert base_error.severity.value == "medium"

        # Test specific exceptions
        training_error = TrainingException("Training failed")
        assert training_error.category.value == "training"

        model_error = ModelException("Model error")
        assert model_error.category.value == "model"

        data_error = DataException("Data error")
        assert data_error.category.value == "data"

        network_error = NetworkException("Network error")
        assert network_error.category.value == "network"

        resource_error = ResourceException("Resource error")
        assert resource_error.category.value == "resource"

        validation_error = ValidationException("Validation error")
        assert validation_error.category.value == "validation"

    def test_error_handler(self):
        """Test error handler functionality"""
        handler = ErrorHandler()

        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            context = handler.handle_error(e, "test_component", "test_operation")

            assert context.component == "test_component"
            assert context.operation == "test_operation"
            assert "test_error" in context.error_id
            assert len(handler.error_history) == 1

        # Test error summary
        summary = handler.get_error_summary()
        assert summary["total_errors"] >= 1
        assert "by_category" in summary
        assert "by_severity" in summary

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test async retry decorator"""
        call_count = 0

        @retry_async(RetryConfig(max_attempts=3, base_delay=0.01))
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkException("Simulated failure")
            return "success"

        result = await failing_function()
        assert result == "success"
        assert call_count == 3

    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        cb = CircuitBreaker(config)

        def failing_function():
            raise Exception("Always fails")

        # First failures should pass through
        with pytest.raises(Exception):
            cb.call(failing_function)

        with pytest.raises(Exception):
            cb.call(failing_function)

        # Circuit should now be open
        with pytest.raises(GRPOException, match="Circuit breaker is OPEN"):
            cb.call(failing_function)


class TestPerformanceOptimization:
    """Test performance optimization features"""

    def test_memory_monitor(self):
        """Test memory monitoring"""
        config = MemoryConfig(max_memory_usage_percent=0.8)
        monitor = MemoryMonitor(config)

        # Test memory usage tracking
        usage = monitor.get_memory_usage()
        assert "cpu_memory_mb" in usage
        assert "gpu_memory_mb" in usage
        assert "memory_percent" in usage

        # Test cleanup
        freed = monitor.cleanup_memory()
        assert isinstance(freed, float)
        assert freed >= 0

    def test_performance_optimizer_initialization(self):
        """Test performance optimizer initialization"""
        optimizer = PerformanceOptimizer(OptimizationLevel.BALANCED)

        assert optimizer.optimization_level == OptimizationLevel.BALANCED
        assert isinstance(optimizer.memory_monitor, MemoryMonitor)
        assert isinstance(optimizer.model_optimizer, ModelOptimizer)
        assert isinstance(optimizer.batch_optimizer, BatchOptimizer)

        # Test different optimization levels
        conservative = PerformanceOptimizer(OptimizationLevel.CONSERVATIVE)
        assert conservative.memory_config.max_memory_usage_percent == 0.7

        aggressive = PerformanceOptimizer(OptimizationLevel.AGGRESSIVE)
        assert aggressive.memory_config.max_memory_usage_percent == 0.95

    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        metrics = PerformanceMetrics(
            memory_usage_mb=100.0,
            gpu_memory_usage_mb=500.0,
            throughput_tokens_per_sec=150.0,
        )

        assert metrics.memory_usage_mb == 100.0
        assert metrics.gpu_memory_usage_mb == 500.0
        assert metrics.throughput_tokens_per_sec == 150.0
        assert metrics.timestamp > 0

    def test_model_optimizer(self):
        """Test model optimizer"""
        config = ComputeConfig(use_mixed_precision=True, compile_model=False)
        optimizer = ModelOptimizer(config)

        assert optimizer.config.use_mixed_precision
        assert not optimizer.config.compile_model
        assert optimizer.scaler is not None  # Should be initialized for mixed precision


class TestTypeSystem:
    """Test type safety and validation features"""

    def test_type_validator(self):
        """Test type validation"""
        validator = TypeValidator()

        # Test basic types
        assert validator.validate_type(42, int)
        assert validator.validate_type("hello", str)
        assert validator.validate_type([1, 2, 3], list)
        assert validator.validate_type({"a": 1}, dict)

        # Test complex types
        assert validator.validate_type([1, 2, 3], List[int])
        assert validator.validate_type({"key": "value"}, Dict[str, str])

        # Test invalid types
        assert not validator.validate_type("hello", int)
        assert not validator.validate_type([1, 2, "3"], List[int])

    def test_config_validator(self):
        """Test configuration validation"""
        validator = ConfigValidator()

        # Test valid model config
        valid_config = {
            "model_name": "gpt2",
            "device": DeviceType.AUTO,
            "torch_dtype": "float32",
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
        }

        assert validator.validate_model_config(valid_config)
        report = validator.get_validation_report()
        assert report["is_valid"]
        assert len(report["errors"]) == 0

        # Test invalid config
        invalid_config = {
            "model_name": "gpt2",
            "temperature": 5.0,  # Invalid temperature
            "top_p": 1.5,  # Invalid top_p
        }

        assert not validator.validate_model_config(invalid_config)
        report = validator.get_validation_report()
        assert not report["is_valid"]
        assert len(report["errors"]) > 0

    def test_create_typed_config(self):
        """Test type-safe config creation"""
        # Test valid config creation
        config = create_typed_config(
            ModelConfig,
            model_name="gpt2",
            device=DeviceType.CPU,
            torch_dtype="float32",
            max_length=256,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            do_sample=True,
        )

        assert config["model_name"] == "gpt2"
        assert config["device"] == DeviceType.CPU
        assert config["temperature"] == 0.8

        # Test invalid config creation
        with pytest.raises(ValueError):
            create_typed_config(
                ModelConfig, model_name="gpt2", temperature=5.0  # Invalid temperature
            )

    def test_type_safe_serialization(self):
        """Test type-safe serialization"""
        serializer = TypeSafeSerializer()

        # Test serialization
        data = {"key": "value", "number": 42}
        json_str = serializer.serialize(data, dict)

        # Test deserialization
        deserialized = serializer.deserialize(json_str, dict)
        assert deserialized["data"]["key"] == "value"
        assert deserialized["data"]["number"] == 42


class TestAsyncResourceManagement:
    """Test async resource management features"""

    class MockResourceFactory(AsyncResourceFactory):
        """Mock resource factory for testing"""

        def __init__(self):
            self.created_count = 0
            self.cleanup_count = 0

        async def create_resource(self, **kwargs):
            self.created_count += 1
            return f"resource_{self.created_count}"

        async def validate_resource(self, resource):
            return True

        async def cleanup_resource(self, resource):
            self.cleanup_count += 1

    @pytest.mark.asyncio
    async def test_async_resource_pool(self):
        """Test async resource pool"""
        factory = self.MockResourceFactory()
        pool = AsyncResourcePool(
            factory=factory, min_size=2, max_size=5, name="TestPool"
        )

        # Initialize pool
        await pool.initialize()
        assert factory.created_count == 2  # min_size

        # Test resource acquisition
        async with pool.acquire() as resource:
            assert resource.startswith("resource_")

        # Test pool stats
        stats = pool.get_stats()
        assert isinstance(stats, PoolStats)
        assert stats.total_connections >= 2

        # Cleanup
        await pool.close()
        assert factory.cleanup_count >= 2

    @pytest.mark.asyncio
    async def test_async_task_manager(self):
        """Test async task manager"""
        manager = AsyncTaskManager(max_concurrent_tasks=5)
        await manager.start()

        # Test single task submission
        async def sample_task(value):
            await asyncio.sleep(0.01)
            return value * 2

        result = await manager.submit_task(sample_task(5))
        assert result == 10

        # Test batch submission
        tasks = [sample_task(i) for i in range(5)]
        results = await manager.submit_batch(tasks)
        assert len(results) == 5
        assert all(isinstance(r, int) for r in results)

        # Test status
        status = manager.get_status()
        assert "active_tasks" in status
        assert "available_slots" in status

        # Cleanup
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_pooled_resource(self):
        """Test pooled resource wrapper"""
        factory = self.MockResourceFactory()
        pool = AsyncResourcePool(factory, min_size=1, max_size=1)

        resource = PooledResource("test_resource", "test_id", pool)

        # Test acquire/release cycle
        acquired = await resource.acquire()
        assert acquired == "test_resource"
        assert resource.info.use_count == 1

        await resource.release()
        assert resource.info.state.value == "idle"

        # Test failure marking
        await resource.mark_failed(Exception("Test error"))
        assert resource.info.state.value == "failed"
        assert "last_error" in resource.info.metadata


class TestIntegration:
    """Integration tests for all enhanced features"""

    @pytest.mark.asyncio
    async def test_error_handling_with_async_pool(self):
        """Test error handling integration with async pools"""

        class FailingFactory(AsyncResourceFactory):
            def __init__(self):
                self.attempt_count = 0

            async def create_resource(self, **kwargs):
                self.attempt_count += 1
                if self.attempt_count <= 2:
                    raise ResourceException("Simulated factory failure")
                return f"resource_{self.attempt_count}"

        factory = FailingFactory()
        error_handler = ErrorHandler()

        # Test error handling during pool initialization
        pool = AsyncResourcePool(factory, min_size=1, max_size=1)

        try:
            await pool.initialize()
        except Exception as e:
            error_context = error_handler.handle_error(e, "pool", "initialization")
            assert error_context.category.value == "resource"

    @pytest.mark.asyncio
    async def test_performance_optimization_with_type_safety(self):
        """Test performance optimization with type safety"""

        # Create type-safe performance config
        memory_config = MemoryConfig(
            enable_gradient_checkpointing=True, max_memory_usage_percent=0.8
        )

        compute_config = ComputeConfig(use_mixed_precision=True, compile_model=False)

        optimizer = PerformanceOptimizer(
            optimization_level=OptimizationLevel.BALANCED,
            memory_config=memory_config,
            compute_config=compute_config,
        )

        # Test type validation
        validator = TypeValidator()
        assert validator.validate_type(memory_config, MemoryConfig)
        assert validator.validate_type(compute_config, ComputeConfig)

        # Test performance report
        report = optimizer.get_performance_report()
        assert "optimization_level" in report
        assert "recommendations" in report

    def test_comprehensive_framework_integration(self):
        """Test overall framework integration"""

        # Test that all new components can be imported together
        from stateset_agents import (
            AsyncResourcePool,
            ErrorHandler,
            ModelConfig,
            OptimizationLevel,
            PerformanceOptimizer,
            TypeValidator,
        )

        # Test basic initialization
        optimizer = PerformanceOptimizer(OptimizationLevel.BALANCED)
        error_handler = ErrorHandler()
        type_validator = TypeValidator()

        assert all(
            [
                isinstance(optimizer, PerformanceOptimizer),
                isinstance(error_handler, ErrorHandler),
                isinstance(type_validator, TypeValidator),
            ]
        )


# Test fixtures and utilities
@pytest.fixture
def sample_model_config():
    """Fixture providing a valid model configuration"""
    return {
        "model_name": "gpt2",
        "device": DeviceType.CPU,
        "torch_dtype": "float32",
        "max_length": 256,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
    }


@pytest.fixture
def sample_training_config():
    """Fixture providing a valid training configuration"""
    return {
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "save_steps": 500,
        "eval_steps": 100,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "lr_scheduler_type": "linear",
        "output_dir": "./test_output",
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
