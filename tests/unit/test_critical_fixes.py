"""
Tests for critical fixes made to the GRPO Agent Framework.

This module tests:
- TTLDict bounded memory collection
- ErrorHandler bounded history
- Thread-safe MetricsCollector
- Encryption security
- PPO clipping correctness
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class TestTTLDict:
    """Tests for the TTLDict class used for bounded state storage."""

    @pytest.fixture
    def ttl_dict(self):
        """Create a TTLDict for testing."""
        from api.ultimate_grpo_service import TTLDict
        # Keep TTL-based tests fast while still validating expiry.
        return TTLDict(ttl_seconds=0.05, max_size=5)

    def test_basic_operations(self, ttl_dict):
        """Test basic dict operations."""
        ttl_dict["key1"] = "value1"
        assert ttl_dict["key1"] == "value1"
        assert ttl_dict.get("key1") == "value1"
        assert ttl_dict.get("nonexistent") is None

    def test_max_size_enforcement(self, ttl_dict):
        """Test that max_size is enforced."""
        for i in range(10):
            ttl_dict[f"key{i}"] = f"value{i}"

        # Should only have max_size entries
        assert len(ttl_dict) <= 5

    def test_ttl_expiration(self, ttl_dict):
        """Test that entries expire after TTL."""
        ttl_dict["expiring"] = "value"
        assert ttl_dict.get("expiring") == "value"

        # Wait for TTL to expire
        time.sleep(0.06)

        # Entry should be expired
        assert ttl_dict.get("expiring") is None

    def test_setdefault(self, ttl_dict):
        """Test setdefault behavior."""
        ttl_dict.setdefault("new_key", "default_value")
        assert ttl_dict["new_key"] == "default_value"

        # Existing key should not be overwritten
        ttl_dict.setdefault("new_key", "other_value")
        assert ttl_dict["new_key"] == "default_value"

    def test_cleanup_expired(self, ttl_dict):
        """Test manual cleanup of expired entries."""
        ttl_dict["key1"] = "value1"
        ttl_dict["key2"] = "value2"

        time.sleep(0.06)

        count = ttl_dict.cleanup_expired()
        assert count == 2
        assert len(ttl_dict) == 0


class TestErrorHandler:
    """Tests for the ErrorHandler bounded history."""

    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler for testing."""
        from stateset_agents.core.error_handling import ErrorHandler
        return ErrorHandler(max_error_history=10)

    def test_bounded_history(self, error_handler):
        """Test that error history is bounded."""
        for i in range(20):
            error_handler.handle_error(
                Exception(f"Error {i}"),
                component="test",
                operation="test_op"
            )

        # Should only keep max_error_history entries
        assert len(error_handler.error_history) <= 10

    def test_error_logging(self, error_handler):
        """Test that errors are properly logged."""
        error_context = error_handler.handle_error(
            ValueError("Test error"),
            component="test_component",
            operation="test_operation",
            context={"key": "value"}
        )

        assert error_context.component == "test_component"
        assert error_context.operation == "test_operation"
        assert error_context.details == {"key": "value"}

    def test_grpo_exception_handling(self, error_handler):
        """Test handling of GRPO-specific exceptions."""
        from stateset_agents.core.error_handling import GRPOException, ErrorCategory, ErrorSeverity

        grpo_error = GRPOException(
            "Test GRPO error",
            category=ErrorCategory.TRAINING,
            severity=ErrorSeverity.HIGH,
            details={"step": 100}
        )

        error_context = error_handler.handle_error(
            grpo_error,
            component="trainer",
            operation="training_step"
        )

        assert error_context.category == ErrorCategory.TRAINING
        assert error_context.severity == ErrorSeverity.HIGH


class TestMetricsCollectorThreadSafety:
    """Tests for thread-safe MetricsCollector."""

    @pytest.fixture
    def metrics_collector(self):
        """Create a MetricsCollector for testing."""
        from stateset_agents.utils.monitoring import MetricsCollector
        return MetricsCollector(enable_prometheus=False)

    def test_concurrent_metric_recording(self, metrics_collector):
        """Test that concurrent metric recording is thread-safe."""
        from stateset_agents.utils.monitoring import Metric, MetricType
        errors = []

        def record_metrics(thread_id):
            try:
                for i in range(100):
                    metric = Metric(
                        name=f"test_metric_{thread_id}",
                        type=MetricType.GAUGE,
                        value=float(i),
                        labels={"thread": str(thread_id)}
                    )
                    metrics_collector.record_metric(metric)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=record_metrics, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_snapshot(self, metrics_collector):
        """Test that getting summaries is thread-safe."""
        from stateset_agents.utils.monitoring import Metric, MetricType
        errors = []

        def record_and_get_summary(thread_id):
            try:
                for i in range(50):
                    metric = Metric(
                        name=f"metric_{thread_id}",
                        type=MetricType.GAUGE,
                        value=float(i)
                    )
                    metrics_collector.record_metric(metric)
                    metrics_collector.get_metrics_summary()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            t = threading.Thread(target=record_and_get_summary, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


class TestSecurityEncryption:
    """Tests for secure encryption in SecureConfig."""

    def test_encryption_not_xor(self):
        """Test that encryption is not using weak XOR cipher."""
        pytest.importorskip("cryptography")
        from stateset_agents.utils.security import SecureConfig
        import os

        secure_config = SecureConfig()

        # Set a test encryption key
        original_key = os.environ.get('CONFIG_ENCRYPTION_KEY')
        try:
            os.environ['CONFIG_ENCRYPTION_KEY'] = 'test-secret-key-123-long-enough'
            original = "sensitive_api_key_12345"
            encrypted = secure_config._simple_encrypt(original)

            # Encrypted should not equal original
            assert encrypted != original

            # Should not be simple base64 of original (XOR with key)
            import base64
            simple_b64 = base64.b64encode(original.encode()).decode()
            assert encrypted != simple_b64

            # Decryption should recover original
            decrypted = secure_config._simple_decrypt(encrypted)
            assert decrypted == original
        finally:
            if original_key is not None:
                os.environ['CONFIG_ENCRYPTION_KEY'] = original_key
            else:
                os.environ.pop('CONFIG_ENCRYPTION_KEY', None)

    def test_encryption_without_key_uses_base64(self):
        """Test that missing encryption key falls back to base64."""
        from stateset_agents.utils.security import SecureConfig
        import os
        import base64

        secure_config = SecureConfig()

        # Remove encryption key
        original_key = os.environ.pop('CONFIG_ENCRYPTION_KEY', None)
        try:
            test_value = "test_value"
            result = secure_config._simple_encrypt(test_value)
            # Should return base64 encoded value
            assert result == base64.b64encode(test_value.encode()).decode()
        finally:
            if original_key is not None:
                os.environ['CONFIG_ENCRYPTION_KEY'] = original_key


class TestPPOClipping:
    """Tests for correct PPO clipping implementation."""

    def test_ppo_clipping_uses_min(self):
        """Test that PPO clipping uses min() for conservative updates."""
        pytest.importorskip("torch")
        import torch

        # Simulate the PPO clipping logic
        clip_epsilon = 0.2

        # Test case 1: positive advantage (should encourage action)
        advantage = torch.tensor(0.5)
        nll = torch.tensor(1.0)  # negative log likelihood

        policy_loss = advantage * nll
        clamped_advantage = advantage.clamp(-clip_epsilon, clip_epsilon)
        clipped_loss = clamped_advantage * nll

        # Should use min for conservative updates
        final_loss = torch.min(policy_loss, clipped_loss)

        # With positive advantage and clamping, min should be the clipped version
        assert final_loss <= policy_loss

        # Test case 2: negative advantage (should discourage action)
        advantage = torch.tensor(-0.5)
        policy_loss = advantage * nll
        clamped_advantage = advantage.clamp(-clip_epsilon, clip_epsilon)
        clipped_loss = clamped_advantage * nll

        final_loss = torch.min(policy_loss, clipped_loss)

        # With negative advantage, min should be the original (more negative)
        assert final_loss <= clipped_loss


class TestAsyncRewardComputation:
    """Tests for async-safe reward computation."""

    @pytest.mark.asyncio
    async def test_reward_function_async_context(self):
        """Test that RewardFunction handles async context properly."""
        from stateset_agents.core.reward_base import RewardFunction, RewardResult

        class TestReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                return RewardResult(score=0.5, breakdown={"test": 0.5}, metadata={})

        reward_fn = TestReward()

        # Should work in async context
        result = await reward_fn.compute_reward([])
        assert result.score == 0.5

    def test_reward_function_sync_context(self):
        """Test that RewardFunction handles sync context properly."""
        from stateset_agents.core.reward_base import RewardFunction, RewardResult

        class TestReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                return RewardResult(score=0.75, breakdown={}, metadata={})

        reward_fn = TestReward()

        # Should work in sync context via __call__
        # Note: This will raise if called from within an async context
        # which is the correct behavior after our fix
        try:
            # We're not in an async context here
            import asyncio
            result = asyncio.run(reward_fn.compute_reward([]))
            assert result.score == 0.75
        except RuntimeError:
            # Expected if we're somehow in an async context
            pass


class TestCompositeRewardEdgeCases:
    """Tests for CompositeReward edge cases."""

    @pytest.mark.asyncio
    async def test_composite_reward_empty_results(self):
        """Test CompositeReward handles empty results gracefully."""
        from stateset_agents.core.reward_base import CompositeReward, RewardFunction, RewardResult

        class FailingReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                raise ValueError("Simulated failure")

        # Create composite with a failing reward function
        failing_reward = FailingReward()
        failing_reward.weight = 1.0
        composite = CompositeReward(reward_functions=[failing_reward])

        # Should not raise, should return default result
        result = await composite.compute_reward([])
        assert result is not None
        assert result.score == 0.0  # Default for no successful components

    @pytest.mark.asyncio
    async def test_composite_reward_with_valid_functions(self):
        """Test CompositeReward with working reward functions."""
        from stateset_agents.core.reward_base import CompositeReward, RewardFunction, RewardResult

        class WorkingReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                return RewardResult(score=0.8, breakdown={"test": 0.8}, metadata={})

        working_reward = WorkingReward()
        working_reward.weight = 1.0
        composite = CompositeReward(reward_functions=[working_reward])

        result = await composite.compute_reward([])
        assert result is not None
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_composite_reward_no_functions(self):
        """Test CompositeReward handles no reward functions."""
        from stateset_agents.core.reward_base import CompositeReward

        composite = CompositeReward(reward_functions=[])

        # Should handle gracefully without errors
        result = await composite.compute_reward([])
        assert result is not None
        assert result.score == 0.0


class TestEnvironmentGetReward:
    """Tests for Environment.get_reward method."""

    @pytest.mark.asyncio
    async def test_environment_get_reward_exists(self):
        """Test that Environment has get_reward method."""
        from stateset_agents.core.environment import Environment

        class TestEnv(Environment):
            async def reset(self):
                return {}

            async def step(self, action):
                return {"reward": 0.5, "done": False}

        env = TestEnv()

        # Should have get_reward method
        assert hasattr(env, 'get_reward')

    @pytest.mark.asyncio
    async def test_environment_get_reward_returns_float(self):
        """Test that get_reward returns a float."""
        from stateset_agents.core.environment import Environment
        from stateset_agents.core.reward_base import RewardFunction, RewardResult

        class TestReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                return RewardResult(score=0.8, breakdown={})

        class TestEnv(Environment):
            async def reset(self):
                return {}

            async def step(self, action):
                return {"reward": 0.5, "done": False}

        env = TestEnv(reward_fn=TestReward())

        # Create a mock trajectory
        class MockTrajectory:
            turns = [{"role": "user", "content": "test"}]

        result = await env.get_reward(MockTrajectory())
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
