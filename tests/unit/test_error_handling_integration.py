"""
Tests for error handling infrastructure.

Tests the CircuitBreaker state machine, retry logic, error hierarchy,
and error handler — using real objects, not mocks.
"""

import time

import pytest

from stateset_agents.core.error_handling import (
    CircuitBreaker,
    CircuitBreakerState,
    ErrorCategory,
    ErrorHandler,
    GRPOException,
    RetryConfig,
    TrainingException,
    ValidationException,
    retry_async,
)
from stateset_agents.core.errors import ModelError, StateSetError, wrap_exception


class TestErrorHierarchy:
    """Test the unified error hierarchy."""

    def test_grpo_exception_is_subclass_of_stateset_error(self):
        assert issubclass(GRPOException, StateSetError)

    def test_training_exception_caught_by_stateset_error(self):
        with pytest.raises(StateSetError):
            raise TrainingException("training failed")

    def test_validation_exception_caught_by_stateset_error(self):
        with pytest.raises(StateSetError):
            raise ValidationException("bad input")

    def test_grpo_exception_has_correct_error_code(self):
        exc = TrainingException("test")
        assert exc.error_code.value == "GRPO_E100"

    def test_grpo_exception_str_includes_code(self):
        exc = GRPOException("something broke")
        text = str(exc)
        assert "GRPO_E" in text
        assert "something broke" in text

    def test_wrap_exception_preserves_traceback(self):
        original = ValueError("original error")
        wrapped = wrap_exception(original, ModelError, "wrapped message")
        assert wrapped.cause is original
        assert "wrapped message" in str(wrapped)


class TestCircuitBreaker:
    """Test CircuitBreaker state transitions."""

    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert not cb.is_open

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(2):
            try:
                cb.call(self._failing_function)
            except ValueError:
                pass

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 2

    def test_opens_after_threshold_reached(self):
        cb = CircuitBreaker(failure_threshold=3, expected_exception=ValueError)

        for _ in range(3):
            try:
                cb.call(self._failing_function)
            except ValueError:
                pass

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.is_open

    def test_open_circuit_rejects_calls(self):
        cb = CircuitBreaker(failure_threshold=1, expected_exception=ValueError)

        try:
            cb.call(self._failing_function)
        except ValueError:
            pass

        assert cb.state == CircuitBreakerState.OPEN

        with pytest.raises(GRPOException, match="Circuit breaker is OPEN"):
            cb.call(self._succeeding_function)

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            expected_exception=ValueError,
        )

        try:
            cb.call(self._failing_function)
        except ValueError:
            pass

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Next call should transition to HALF_OPEN and succeed
        result = cb.call(self._succeeding_function)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    @staticmethod
    def _failing_function():
        raise ValueError("simulated failure")

    @staticmethod
    def _succeeding_function():
        return "success"


class TestRetryAsync:
    """Test the retry_async decorator."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        config = RetryConfig(max_attempts=3, retry_on=[ValueError])

        @retry_async(config)
        async def success():
            return "ok"

        result = await success()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_matching_exception(self):
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retry_on=[ValueError],
            jitter=False,
        )
        call_count = 0

        @retry_async(config)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "recovered"

        result = await flaky()
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        config = RetryConfig(
            max_attempts=2,
            base_delay=0.01,
            retry_on=[ValueError],
            jitter=False,
        )

        @retry_async(config)
        async def always_fails():
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await always_fails()

    @pytest.mark.asyncio
    async def test_non_matching_exception_not_retried(self):
        config = RetryConfig(
            max_attempts=3,
            retry_on=[ValueError],
        )
        call_count = 0

        @retry_async(config)
        async def wrong_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await wrong_error()

        assert call_count == 1  # No retries


class TestErrorHandler:
    """Test ErrorHandler."""

    def test_records_error_in_history(self):
        handler = ErrorHandler()
        exc = GRPOException("test error")
        ctx = handler.handle_error(exc, "test_component", "test_op")

        assert len(handler.error_history) == 1
        assert ctx.component == "test_component"
        assert ctx.operation == "test_op"

    def test_error_summary_counts_categories(self):
        handler = ErrorHandler()
        handler.handle_error(TrainingException("err1"), "trainer", "train")
        handler.handle_error(ValidationException("err2"), "validator", "validate")

        summary = handler.get_error_summary()
        assert summary["total_errors"] == 2
        assert "training" in summary["by_category"]
        assert "validation" in summary["by_category"]

    def test_error_history_bounded(self):
        handler = ErrorHandler(max_error_history=5)
        for i in range(10):
            handler.handle_error(GRPOException(f"error {i}"), "test", "op")
        assert len(handler.error_history) == 5

    def test_handles_non_grpo_exceptions(self):
        handler = ErrorHandler()
        ctx = handler.handle_error(RuntimeError("plain error"), "component", "op")
        assert ctx.category == ErrorCategory.SYSTEM
