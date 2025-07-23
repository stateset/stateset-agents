"""
Enhanced Error Handling for GRPO Agent Framework

This module provides comprehensive error handling, retry mechanisms,
and recovery strategies for production-ready agent training and deployment.
"""

import asyncio
import logging
import traceback
from typing import Dict, Any, Optional, Callable, Type, Union, List
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path
import pickle


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better handling"""
    TRAINING = "training"
    MODEL = "model"
    DATA = "data"
    NETWORK = "network"
    RESOURCE = "resource"
    VALIDATION = "validation"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Rich error context for debugging"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    component: str
    operation: str
    details: Dict[str, Any]
    stack_trace: str
    recovery_actions: List[str]


class GRPOException(Exception):
    """Base exception for GRPO framework"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        recovery_actions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recovery_actions = recovery_actions or []
        self.timestamp = time.time()
        
    def to_context(self, component: str = "unknown", operation: str = "unknown") -> ErrorContext:
        """Convert to rich error context"""
        return ErrorContext(
            error_id=f"{self.category.value}_{int(self.timestamp)}",
            category=self.category,
            severity=self.severity,
            timestamp=self.timestamp,
            component=component,
            operation=operation,
            details=self.details,
            stack_trace=traceback.format_exc(),
            recovery_actions=self.recovery_actions
        )


class TrainingException(GRPOException):
    """Training-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.TRAINING
        super().__init__(message, **kwargs)


class ModelException(GRPOException):
    """Model-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.MODEL
        super().__init__(message, **kwargs)


class DataException(GRPOException):
    """Data processing errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.DATA
        super().__init__(message, **kwargs)


class NetworkException(GRPOException):
    """Network and API errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.NETWORK
        super().__init__(message, **kwargs)


class ResourceException(GRPOException):
    """Resource allocation errors (GPU, memory, etc.)"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.RESOURCE
        super().__init__(message, **kwargs)


class ValidationException(GRPOException):
    """Input validation errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.VALIDATION
        super().__init__(message, **kwargs)


class RetryConfig:
    """Configuration for retry mechanisms"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on or [NetworkException, ResourceException]


class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise GRPOException(
                    "Circuit breaker is OPEN",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.HIGH
                )
        
        try:
            result = func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
            return result
            
        except self.config.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                
            raise


class ErrorHandler:
    """Centralized error handling and recovery"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Handle and log error with rich context"""
        
        if isinstance(error, GRPOException):
            error_context = error.to_context(component, operation)
        else:
            error_context = ErrorContext(
                error_id=f"generic_{int(time.time())}",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                timestamp=time.time(),
                component=component,
                operation=operation,
                details=context or {},
                stack_trace=traceback.format_exc(),
                recovery_actions=[]
            )
        
        self.error_history.append(error_context)
        self._log_error(error_context)
        
        return error_context
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level"""
        log_data = {
            "error_id": error_context.error_id,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "component": error_context.component,
            "operation": error_context.operation,
            "details": error_context.details
        }
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {log_data}")
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR: {log_data}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR: {log_data}")
        else:
            self.logger.info(f"LOW SEVERITY ERROR: {log_data}")
    
    def register_circuit_breaker(
        self, 
        name: str, 
        config: CircuitBreakerConfig
    ):
        """Register a circuit breaker"""
        self.circuit_breakers[name] = CircuitBreaker(config)
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        recent_errors = [
            e for e in self.error_history 
            if time.time() - e.timestamp < 3600  # Last hour
        ]
        
        by_category = {}
        by_severity = {}
        
        for error in recent_errors:
            category = error.category.value
            severity = error.severity.value
            
            by_category[category] = by_category.get(category, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "by_category": by_category,
            "by_severity": by_severity,
            "circuit_breakers": {
                name: cb.state.value 
                for name, cb in self.circuit_breakers.items()
            }
        }


def retry_async(config: RetryConfig):
    """Async retry decorator with exponential backoff"""
    
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry this exception
                    if not any(isinstance(e, exc_type) for exc_type in config.retry_on):
                        raise
                    
                    # Don't wait after the last attempt
                    if attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    await asyncio.sleep(delay)
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def circuit_breaker(name: str, config: CircuitBreakerConfig, error_handler: ErrorHandler):
    """Circuit breaker decorator"""
    
    error_handler.register_circuit_breaker(name, config)
    
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            cb = error_handler.get_circuit_breaker(name)
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_global_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    return _global_error_handler


def set_global_error_handler(handler: ErrorHandler):
    """Set global error handler instance"""
    global _global_error_handler
    _global_error_handler = handler


# Convenience functions
def handle_error(
    error: Exception,
    component: str,
    operation: str,
    context: Optional[Dict[str, Any]] = None
) -> ErrorContext:
    """Handle error using global handler"""
    return _global_error_handler.handle_error(error, component, operation, context)


def get_error_summary() -> Dict[str, Any]:
    """Get error summary using global handler"""
    return _global_error_handler.get_error_summary()