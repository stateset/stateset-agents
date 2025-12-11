"""
Comprehensive Logging System for GRPO Agent Framework

This module provides advanced logging capabilities with structured logging,
metrics collection, and integration with monitoring systems.
"""

import csv
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional imports for enhanced functionality
try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False


class LogLevel(Enum):
    """Log levels for structured logging"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Categories for different types of logs"""

    SYSTEM = "system"
    API = "api"
    TRAINING = "training"
    CONVERSATION = "conversation"
    REWARD = "reward"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"


@dataclass
class LogEntry:
    """Structured log entry"""

    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["level"] = self.level.value
        data["category"] = self.category.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


class GRPOLogger:
    """Advanced logger for GRPO Agent Framework"""

    def __init__(
        self,
        name: str,
        log_level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_json: bool = True,
        enable_metrics: bool = True,
        enable_tracing: bool = False,
    ):
        self.name = name
        self.log_level = log_level
        self.enable_json = enable_json
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing

        # Initialize standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.value))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        if enable_console:
            self._setup_console_handler()

        if log_file:
            self._setup_file_handler(log_file)

        # Setup structured logging
        if HAS_STRUCTLOG and enable_json:
            self._setup_structlog()

        # Setup tracing
        if HAS_OPENTELEMETRY and enable_tracing:
            self._setup_tracing()

        # Metrics collection
        self.metrics = {
            "total_logs": 0,
            "logs_by_level": {},
            "logs_by_category": {},
            "errors": [],
            "performance_metrics": [],
        }

    def _setup_console_handler(self):
        """Setup console handler with formatting"""
        console_handler = logging.StreamHandler()

        if self.enable_json:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_file_handler(self, log_file: str):
        """Setup file handler with rotation"""
        from logging.handlers import RotatingFileHandler

        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )

        if self.enable_json:
            formatter = logging.Formatter("%(message)s")
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _setup_structlog(self):
        """Setup structured logging with structlog"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.struct_logger = structlog.get_logger(self.name)

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        trace.set_tracer_provider(TracerProvider())

        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )

        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        self.tracer = trace.get_tracer(self.name)

    def _create_log_entry(
        self, level: LogLevel, category: LogCategory, message: str, **kwargs
    ) -> LogEntry:
        """Create a structured log entry"""

        # Get trace information if available
        trace_id = None
        span_id = None

        if HAS_OPENTELEMETRY and self.enable_tracing:
            current_span = trace.get_current_span()
            if current_span:
                trace_id = format(current_span.get_span_context().trace_id, "032x")
                span_id = format(current_span.get_span_context().span_id, "016x")

        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            trace_id=trace_id,
            span_id=span_id,
            **kwargs,
        )

    def _log_entry(self, entry: LogEntry):
        """Log a structured entry"""
        # Update metrics
        if self.enable_metrics:
            self.metrics["total_logs"] += 1

            level_key = entry.level.value
            if level_key not in self.metrics["logs_by_level"]:
                self.metrics["logs_by_level"][level_key] = 0
            self.metrics["logs_by_level"][level_key] += 1

            category_key = entry.category.value
            if category_key not in self.metrics["logs_by_category"]:
                self.metrics["logs_by_category"][category_key] = 0
            self.metrics["logs_by_category"][category_key] += 1

            # Track errors
            if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self.metrics["errors"].append(
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "message": entry.message,
                        "metadata": entry.metadata,
                    }
                )

        # Log using appropriate method
        if self.enable_json and HAS_STRUCTLOG:
            self.struct_logger.info(
                entry.message,
                level=entry.level.value,
                category=entry.category.value,
                trace_id=entry.trace_id,
                span_id=entry.span_id,
                user_id=entry.user_id,
                session_id=entry.session_id,
                request_id=entry.request_id,
                component=entry.component,
                duration_ms=entry.duration_ms,
                **entry.metadata,
            )
        else:
            # Use standard logger
            log_method = getattr(self.logger, entry.level.value.lower())
            if self.enable_json:
                log_method(entry.to_json())
            else:
                log_method(f"[{entry.category.value}] {entry.message}")

    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log debug message"""
        entry = self._create_log_entry(LogLevel.DEBUG, category, message, **kwargs)
        self._log_entry(entry)

    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log info message"""
        entry = self._create_log_entry(LogLevel.INFO, category, message, **kwargs)
        self._log_entry(entry)

    def warning(
        self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs
    ):
        """Log warning message"""
        entry = self._create_log_entry(LogLevel.WARNING, category, message, **kwargs)
        self._log_entry(entry)

    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log error message"""
        entry = self._create_log_entry(LogLevel.ERROR, category, message, **kwargs)
        self._log_entry(entry)

    def critical(
        self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs
    ):
        """Log critical message"""
        entry = self._create_log_entry(LogLevel.CRITICAL, category, message, **kwargs)
        self._log_entry(entry)

    def exception(
        self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs
    ):
        """Log exception with traceback"""
        kwargs["exception"] = traceback.format_exc()
        self.error(message, category, **kwargs)

    @contextmanager
    def performance_log(
        self, operation: str, category: LogCategory = LogCategory.PERFORMANCE, **kwargs
    ):
        """Context manager for performance logging"""
        start_time = time.time()
        operation_id = str(uuid.uuid4())

        self.info(
            f"Starting operation: {operation}",
            category=category,
            operation_id=operation_id,
            **kwargs,
        )

        try:
            yield operation_id
            duration_ms = (time.time() - start_time) * 1000

            self.info(
                f"Completed operation: {operation}",
                category=category,
                operation_id=operation_id,
                duration_ms=duration_ms,
                **kwargs,
            )

            # Add to performance metrics
            if self.enable_metrics:
                self.metrics["performance_metrics"].append(
                    {
                        "operation": operation,
                        "duration_ms": duration_ms,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": kwargs,
                    }
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self.error(
                f"Failed operation: {operation}",
                category=category,
                operation_id=operation_id,
                duration_ms=duration_ms,
                error=str(e),
                **kwargs,
            )
            raise

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ):
        """Log API request"""
        level = LogLevel.INFO if status_code < 400 else LogLevel.ERROR

        entry = self._create_log_entry(
            level,
            LogCategory.API,
            f"{method} {path} - {status_code}",
            user_id=user_id,
            request_id=request_id,
            duration_ms=duration_ms,
            metadata={
                "method": method,
                "path": path,
                "status_code": status_code,
                **kwargs,
            },
        )

        self._log_entry(entry)

    def log_conversation_event(
        self,
        event_type: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        message_length: Optional[int] = None,
        quality_score: Optional[float] = None,
        **kwargs,
    ):
        """Log conversation event"""
        self.info(
            f"Conversation {event_type}: {conversation_id}",
            category=LogCategory.CONVERSATION,
            user_id=user_id,
            metadata={
                "event_type": event_type,
                "conversation_id": conversation_id,
                "message_length": message_length,
                "quality_score": quality_score,
                **kwargs,
            },
        )

    def log_training_event(
        self,
        event_type: str,
        iteration: int,
        loss: Optional[float] = None,
        reward: Optional[float] = None,
        **kwargs,
    ):
        """Log training event"""
        self.info(
            f"Training {event_type}: iteration {iteration}",
            category=LogCategory.TRAINING,
            metadata={
                "event_type": event_type,
                "iteration": iteration,
                "loss": loss,
                "reward": reward,
                **kwargs,
            },
        )

    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs,
    ):
        """Log security event"""
        level = LogLevel.WARNING if event_type.startswith("failed") else LogLevel.INFO

        entry = self._create_log_entry(
            level,
            LogCategory.SECURITY,
            f"Security event: {event_type}",
            user_id=user_id,
            metadata={"event_type": event_type, "ip_address": ip_address, **kwargs},
        )

        self._log_entry(entry)

    def get_metrics(self) -> Dict[str, Any]:
        """Get logging metrics"""
        return {
            "total_logs": self.metrics["total_logs"],
            "logs_by_level": self.metrics["logs_by_level"],
            "logs_by_category": self.metrics["logs_by_category"],
            "recent_errors": self.metrics["errors"][-10:],  # Last 10 errors
            "performance_summary": {
                "total_operations": len(self.metrics["performance_metrics"]),
                "average_duration_ms": sum(
                    m["duration_ms"] for m in self.metrics["performance_metrics"]
                )
                / len(self.metrics["performance_metrics"])
                if self.metrics["performance_metrics"]
                else 0,
            },
        }

    def export_logs(self, filepath: str, format: str = "json") -> str:
        """Export logs to file"""
        if format not in ["json", "csv"]:
            raise ValueError("Format must be 'json' or 'csv'")

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "logger_name": self.name,
            "metrics": self.get_metrics(),
            "recent_performance": self.metrics["performance_metrics"][
                -100:
            ],  # Last 100 operations
            "recent_errors": self.metrics["errors"][-50:],  # Last 50 errors
        }

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
        else:
            # Lightweight CSV export without extra dependencies.
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "value"])
                for key, value in export_data.items():
                    writer.writerow([key, json.dumps(value, default=str)])

        return filepath


# Global logger instance
_global_logger: Optional[GRPOLogger] = None


def get_logger(
    name: str = "grpo_framework", log_level: LogLevel = LogLevel.INFO, **kwargs
) -> GRPOLogger:
    """Get or create a logger instance"""
    global _global_logger

    if _global_logger is None:
        _global_logger = GRPOLogger(name, log_level, **kwargs)

    return _global_logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = True,
    enable_metrics: bool = True,
    enable_tracing: bool = False,
):
    """Setup global logging configuration"""
    global _global_logger

    # Convert string to LogLevel
    level = LogLevel(log_level.upper())

    # Default log file path
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "grpo_framework.log")

    _global_logger = GRPOLogger(
        name="grpo_framework",
        log_level=level,
        log_file=log_file,
        enable_json=enable_json,
        enable_metrics=enable_metrics,
        enable_tracing=enable_tracing,
    )

    return _global_logger


# Convenience functions
def debug(message: str, **kwargs):
    """Log debug message using global logger"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message using global logger"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message using global logger"""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message using global logger"""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message using global logger"""
    get_logger().critical(message, **kwargs)


def exception(message: str, **kwargs):
    """Log exception using global logger"""
    get_logger().exception(message, **kwargs)


# Decorator for automatic performance logging
def log_performance(
    operation: str,
    category: LogCategory = LogCategory.PERFORMANCE,
    log_args: bool = False,
    log_result: bool = False,
):
    """Decorator for automatic performance logging"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()

            # Prepare metadata
            metadata = {"function": func.__name__}
            if log_args:
                metadata["args"] = str(args)
                metadata["kwargs"] = str(kwargs)

            with logger.performance_log(operation, category, **metadata):
                result = func(*args, **kwargs)

                if log_result:
                    logger.debug(
                        f"Function {func.__name__} returned",
                        category=category,
                        result=str(result)[:200],  # Truncate long results
                    )

                return result

        return wrapper

    return decorator
