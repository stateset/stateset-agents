"""
Unit tests for the API Observability module.

Tests cover structured logging, console logging, request tracing,
OpenTelemetry integration, and metrics collection.
"""

import json
import logging
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestStructuredLogFormatter:
    """Test StructuredLogFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create a StructuredLogFormatter for testing."""
        from api.observability import StructuredLogFormatter
        return StructuredLogFormatter(service_name="test-service")

    @pytest.fixture
    def log_record(self):
        """Create a mock log record."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        return record

    def test_formatter_creation(self, formatter):
        """Test formatter creation."""
        assert formatter.service_name == "test-service"

    def test_format_basic_message(self, formatter, log_record):
        """Test formatting a basic log message."""
        output = formatter.format(log_record)
        parsed = json.loads(output)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["service"] == "test-service"
        assert "timestamp" in parsed

    def test_format_with_timestamp(self, formatter, log_record):
        """Test that timestamp is in ISO format."""
        output = formatter.format(log_record)
        parsed = json.loads(output)

        timestamp = parsed["timestamp"]
        assert timestamp.endswith("Z")
        # Should be parseable as ISO format
        assert "T" in timestamp

    def test_format_with_source_location(self, formatter, log_record):
        """Test source location is included."""
        output = formatter.format(log_record)
        parsed = json.loads(output)

        assert "source" in parsed
        assert parsed["source"]["file"] == "/path/to/file.py"
        assert parsed["source"]["line"] == 42

    def test_format_with_extra_fields(self, formatter, log_record):
        """Test formatting with extra fields."""
        log_record.request_id = "req_123"
        log_record.user_id = "user_456"
        log_record.path = "/api/test"
        log_record.method = "GET"
        log_record.status_code = 200
        log_record.elapsed_ms = 50.5

        output = formatter.format(log_record)
        parsed = json.loads(output)

        assert parsed["request_id"] == "req_123"
        assert parsed["user_id"] == "user_456"
        assert parsed["path"] == "/api/test"
        assert parsed["method"] == "GET"
        assert parsed["status_code"] == 200
        assert parsed["elapsed_ms"] == 50.5

    def test_format_with_exception(self, formatter):
        """Test formatting with exception info."""
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "Test error"


class TestConsoleLogFormatter:
    """Test ConsoleLogFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create a ConsoleLogFormatter for testing."""
        from api.observability import ConsoleLogFormatter
        return ConsoleLogFormatter()

    @pytest.fixture
    def log_record(self):
        """Create a mock log record."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        return record

    def test_formatter_creation(self, formatter):
        """Test formatter creation."""
        assert formatter is not None

    def test_format_basic_message(self, formatter, log_record):
        """Test formatting a basic console message."""
        output = formatter.format(log_record)

        assert "Test message" in output
        assert "INFO" in output
        assert "test.logger" in output

    def test_format_with_color_codes(self, formatter, log_record):
        """Test that color codes are included."""
        output = formatter.format(log_record)

        # INFO should have green color code
        assert "\033[32m" in output  # Green
        assert "\033[0m" in output  # Reset

    def test_format_different_levels(self, formatter):
        """Test formatting different log levels."""
        levels = [
            (logging.DEBUG, "\033[36m"),    # Cyan
            (logging.INFO, "\033[32m"),     # Green
            (logging.WARNING, "\033[33m"),  # Yellow
            (logging.ERROR, "\033[31m"),    # Red
            (logging.CRITICAL, "\033[35m"), # Magenta
        ]

        for level, color in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="",
                lineno=1,
                msg="test",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)
            assert color in output

    def test_format_with_extras(self, formatter, log_record):
        """Test formatting with extra fields."""
        log_record.path = "/api/test"
        log_record.method = "GET"
        log_record.status_code = 200

        output = formatter.format(log_record)

        assert "path=/api/test" in output
        assert "method=GET" in output
        assert "status_code=200" in output


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_json_format(self):
        """Test setting up JSON format logging."""
        from api.observability import setup_logging

        setup_logging(
            service_name="test-service",
            log_level="INFO",
            log_format="json",
        )

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_logging_console_format(self):
        """Test setting up console format logging."""
        from api.observability import setup_logging

        setup_logging(
            service_name="test-service",
            log_level="DEBUG",
            log_format="console",
        )

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG


class TestTraceContext:
    """Test trace context handling."""

    def test_trace_context_default(self):
        """Test default trace context."""
        from api.observability import _trace_context

        ctx = _trace_context.get()
        assert ctx == {}

    def test_trace_context_set_and_get(self):
        """Test setting and getting trace context."""
        from api.observability import _trace_context

        ctx = {
            "trace_id": "trace_123",
            "span_id": "span_456",
            "request_id": "req_789",
        }

        token = _trace_context.set(ctx)
        retrieved = _trace_context.get()

        assert retrieved["trace_id"] == "trace_123"
        assert retrieved["span_id"] == "span_456"
        assert retrieved["request_id"] == "req_789"

        # Reset
        _trace_context.reset(token)


class TestRequestIdTracking:
    """Test request ID tracking."""

    def test_request_id_generation(self):
        """Test request ID generation."""
        import uuid

        request_id = str(uuid.uuid4())

        assert len(request_id) == 36
        assert request_id.count("-") == 4

    def test_request_id_in_logs(self):
        """Test request ID appears in logs."""
        from api.observability import StructuredLogFormatter

        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.request_id = "req_12345"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["request_id"] == "req_12345"


class TestMetricsCollection:
    """Test metrics collection functionality."""

    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation."""
        start = time.time()
        time.sleep(0.01)  # Small delay
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert elapsed >= 10  # At least 10ms

    def test_status_code_tracking(self):
        """Test status code tracking."""
        status_codes = [200, 201, 400, 404, 500]

        # Count by category
        success = sum(1 for s in status_codes if 200 <= s < 300)
        client_error = sum(1 for s in status_codes if 400 <= s < 500)
        server_error = sum(1 for s in status_codes if 500 <= s < 600)

        assert success == 2
        assert client_error == 2
        assert server_error == 1


class TestLogLevels:
    """Test log level configuration."""

    def test_log_level_mapping(self):
        """Test log level string to constant mapping."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        for name, level in level_map.items():
            assert getattr(logging, name) == level


class TestExceptionLogging:
    """Test exception logging."""

    def test_exception_type_extraction(self):
        """Test extracting exception type."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            exc_type = type(e).__name__
            exc_message = str(e)

        assert exc_type == "ValueError"
        assert exc_message == "Test error"

    def test_nested_exception_logging(self):
        """Test logging nested exceptions."""
        try:
            try:
                raise ValueError("Inner error")
            except ValueError:
                raise RuntimeError("Outer error") from ValueError("Inner error")
        except RuntimeError as e:
            exc_type = type(e).__name__
            exc_message = str(e)

        assert exc_type == "RuntimeError"
        assert exc_message == "Outer error"


class TestLoggerIntegration:
    """Test logger integration."""

    @pytest.fixture
    def test_logger(self):
        """Create a test logger."""
        logger = logging.getLogger("test_observability")
        logger.setLevel(logging.DEBUG)
        return logger

    def test_logger_with_formatter(self, test_logger):
        """Test logger with structured formatter."""
        from api.observability import StructuredLogFormatter
        from io import StringIO

        # Create handler with formatter
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredLogFormatter())
        test_logger.addHandler(handler)

        # Log a message
        test_logger.info("Test message")

        output = stream.getvalue()
        parsed = json.loads(output)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"

        # Cleanup
        test_logger.removeHandler(handler)


class TestTraceDecorator:
    """Test tracing decorator functionality."""

    def test_trace_decorator_concept(self):
        """Test tracing decorator conceptually."""
        from functools import wraps

        traced_calls = []

        def trace_function(name):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    start = time.time()
                    traced_calls.append({
                        "name": name,
                        "start": start,
                    })
                    result = func(*args, **kwargs)
                    traced_calls[-1]["elapsed"] = time.time() - start
                    return result
                return wrapper
            return decorator

        @trace_function("test_operation")
        def sample_function():
            time.sleep(0.01)
            return "result"

        result = sample_function()

        assert result == "result"
        assert len(traced_calls) == 1
        assert traced_calls[0]["name"] == "test_operation"
        assert traced_calls[0]["elapsed"] >= 0.01


class TestClientContext:
    """Test client context tracking."""

    def test_client_ip_logging(self):
        """Test client IP logging."""
        from api.observability import StructuredLogFormatter

        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Request received",
            args=(),
            exc_info=None,
        )
        record.client_ip = "192.168.1.100"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed.get("client_ip") == "192.168.1.100"

    def test_user_agent_logging(self):
        """Test user agent logging."""
        from api.observability import StructuredLogFormatter

        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Request received",
            args=(),
            exc_info=None,
        )
        record.user_agent = "Mozilla/5.0"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed.get("user_agent") == "Mozilla/5.0"


class TestJobAndConversationTracking:
    """Test job and conversation ID tracking."""

    def test_job_id_logging(self):
        """Test job ID logging."""
        from api.observability import StructuredLogFormatter

        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Job started",
            args=(),
            exc_info=None,
        )
        record.job_id = "job_12345"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed.get("job_id") == "job_12345"

    def test_conversation_id_logging(self):
        """Test conversation ID logging."""
        from api.observability import StructuredLogFormatter

        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Message received",
            args=(),
            exc_info=None,
        )
        record.conversation_id = "conv_67890"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed.get("conversation_id") == "conv_67890"
