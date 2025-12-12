"""
API Security Module

Comprehensive security utilities including input validation, prompt injection detection,
and security monitoring for the API layer.
"""

import hashlib
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    MAX_JSON_NESTING_DEPTH,
    MAX_MESSAGE_LENGTH,
    MAX_PROMPT_LENGTH,
    MAX_AUTH_FAILURES_BEFORE_LOCKOUT,
    AUTH_LOCKOUT_DURATION_SECONDS,
    PROMPT_INJECTION_PATTERNS,
    SUSPICIOUS_PATTERNS,
    ERROR_MSG_PROMPT_INJECTION,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Security Enums
# ============================================================================

class ThreatLevel(str, Enum):
    """Threat level classification."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(str, Enum):
    """Types of security events."""
    PROMPT_INJECTION_ATTEMPT = "prompt_injection_attempt"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALFORMED_REQUEST = "malformed_request"


# ============================================================================
# Security Event Tracking
# ============================================================================

@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: float
    client_ip: str
    path: str
    details: Dict[str, Any]
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    blocked: bool = False


@dataclass
class AuthFailureTracker:
    """Track authentication failures per IP/user."""
    failures: Dict[str, List[float]] = field(default_factory=dict)
    lockouts: Dict[str, float] = field(default_factory=dict)

    def record_failure(self, key: str) -> Tuple[bool, int]:
        """
        Record an auth failure and check if lockout should be applied.

        Returns:
            Tuple of (is_locked_out, remaining_attempts)
        """
        now = time.time()

        # Check if currently locked out
        if key in self.lockouts:
            lockout_until = self.lockouts[key]
            if now < lockout_until:
                return True, 0
            else:
                # Lockout expired, clear it
                del self.lockouts[key]
                self.failures.pop(key, None)

        # Clean old failures (older than lockout duration)
        if key in self.failures:
            self.failures[key] = [
                f for f in self.failures[key]
                if now - f < AUTH_LOCKOUT_DURATION_SECONDS
            ]
        else:
            self.failures[key] = []

        # Record new failure
        self.failures[key].append(now)
        failure_count = len(self.failures[key])

        # Check if lockout threshold reached
        if failure_count >= MAX_AUTH_FAILURES_BEFORE_LOCKOUT:
            self.lockouts[key] = now + AUTH_LOCKOUT_DURATION_SECONDS
            logger.warning(
                f"Auth lockout triggered for {key[:16]}..., "
                f"locked until {datetime.fromtimestamp(self.lockouts[key]).isoformat()}"
            )
            return True, 0

        remaining = MAX_AUTH_FAILURES_BEFORE_LOCKOUT - failure_count
        return False, remaining

    def is_locked_out(self, key: str) -> bool:
        """Check if a key is currently locked out."""
        if key not in self.lockouts:
            return False

        now = time.time()
        if now >= self.lockouts[key]:
            # Lockout expired
            del self.lockouts[key]
            self.failures.pop(key, None)
            return False

        return True

    def clear_failures(self, key: str) -> None:
        """Clear failures on successful auth."""
        self.failures.pop(key, None)
        self.lockouts.pop(key, None)


# ============================================================================
# Input Validation
# ============================================================================

class InputValidator:
    """
    Comprehensive input validation with prompt injection detection.

    This class provides security-focused validation for all user inputs,
    including detection of prompt injection attempts and other malicious patterns.
    """

    # Compiled regex patterns for efficiency
    _injection_patterns: List[re.Pattern] = []
    _suspicious_patterns: List[re.Pattern] = []
    _patterns_compiled: bool = False

    @classmethod
    def _compile_patterns(cls) -> None:
        """Compile regex patterns once for efficiency."""
        if cls._patterns_compiled:
            return

        cls._injection_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in PROMPT_INJECTION_PATTERNS
        ]
        cls._suspicious_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SUSPICIOUS_PATTERNS
        ]
        cls._patterns_compiled = True

    @classmethod
    def detect_prompt_injection(
        cls,
        text: str,
        strict: bool = True
    ) -> Tuple[bool, ThreatLevel, List[str]]:
        """
        Detect potential prompt injection attempts in text.

        Args:
            text: The text to analyze
            strict: If True, also flag suspicious patterns

        Returns:
            Tuple of (is_injection, threat_level, matched_patterns)
        """
        cls._compile_patterns()

        if not text:
            return False, ThreatLevel.NONE, []

        matched_patterns: List[str] = []
        threat_level = ThreatLevel.NONE

        # Check for definite injection patterns
        for pattern in cls._injection_patterns:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern)
                threat_level = ThreatLevel.HIGH

        # Check for suspicious patterns (lower threat)
        if strict:
            for pattern in cls._suspicious_patterns:
                if pattern.search(text):
                    matched_patterns.append(pattern.pattern)
                    if threat_level == ThreatLevel.NONE:
                        threat_level = ThreatLevel.LOW

        is_injection = threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)

        return is_injection, threat_level, matched_patterns

    @classmethod
    def validate_string(
        cls,
        value: str,
        max_length: int = MAX_MESSAGE_LENGTH,
        field_name: str = "input",
        allow_empty: bool = False,
        check_injection: bool = True,
    ) -> Tuple[str, Optional[SecurityEvent]]:
        """
        Validate a string input with security checks.

        Args:
            value: The string to validate
            max_length: Maximum allowed length
            field_name: Name of the field for error messages
            allow_empty: Whether empty strings are allowed
            check_injection: Whether to check for prompt injection

        Returns:
            Tuple of (cleaned_value, security_event_or_none)

        Raises:
            ValueError: If validation fails
        """
        # Handle None
        if value is None:
            if allow_empty:
                return "", None
            raise ValueError(f"{field_name} cannot be None")

        # Strip whitespace
        cleaned = value.strip()

        # Check empty
        if not cleaned and not allow_empty:
            raise ValueError(f"{field_name} cannot be empty")

        # Check length
        if len(cleaned) > max_length:
            raise ValueError(
                f"{field_name} exceeds maximum length of {max_length} characters "
                f"(got {len(cleaned)})"
            )

        # Check for prompt injection
        security_event = None
        if check_injection and cleaned:
            is_injection, threat_level, patterns = cls.detect_prompt_injection(cleaned)

            if is_injection:
                security_event = SecurityEvent(
                    event_type=SecurityEventType.PROMPT_INJECTION_ATTEMPT,
                    threat_level=threat_level,
                    timestamp=time.time(),
                    client_ip="unknown",  # Will be set by caller
                    path="unknown",  # Will be set by caller
                    details={
                        "field": field_name,
                        "matched_patterns": patterns,
                        "content_hash": hashlib.sha256(cleaned.encode()).hexdigest()[:16],
                    },
                    blocked=True,
                )
                raise ValueError(ERROR_MSG_PROMPT_INJECTION)

            elif threat_level != ThreatLevel.NONE:
                # Log suspicious but don't block
                security_event = SecurityEvent(
                    event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                    threat_level=threat_level,
                    timestamp=time.time(),
                    client_ip="unknown",
                    path="unknown",
                    details={
                        "field": field_name,
                        "matched_patterns": patterns,
                    },
                    blocked=False,
                )

        return cleaned, security_event

    @classmethod
    def validate_messages(
        cls,
        messages: List[Dict[str, Any]],
        max_messages: int = 100,
        max_content_length: int = MAX_MESSAGE_LENGTH,
    ) -> Tuple[List[Dict[str, Any]], List[SecurityEvent]]:
        """
        Validate a list of conversation messages.

        Args:
            messages: List of message dictionaries
            max_messages: Maximum number of messages allowed
            max_content_length: Maximum length for message content

        Returns:
            Tuple of (validated_messages, security_events)

        Raises:
            ValueError: If validation fails
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if len(messages) > max_messages:
            raise ValueError(f"Maximum {max_messages} messages allowed")

        security_events: List[SecurityEvent] = []
        validated: List[Dict[str, Any]] = []

        valid_roles = {"system", "user", "assistant"}

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i+1} must be a dictionary")

            if "role" not in msg:
                raise ValueError(f"Message {i+1} missing 'role' field")

            if "content" not in msg:
                raise ValueError(f"Message {i+1} missing 'content' field")

            role = msg["role"]
            if role not in valid_roles:
                raise ValueError(
                    f"Message {i+1} has invalid role '{role}'. "
                    f"Must be one of: {', '.join(valid_roles)}"
                )

            # Validate content with injection detection
            content, event = cls.validate_string(
                msg["content"],
                max_length=max_content_length,
                field_name=f"Message {i+1} content",
                check_injection=True,
            )

            if event:
                security_events.append(event)

            validated.append({
                "role": role,
                "content": content,
                **{k: v for k, v in msg.items() if k not in ("role", "content")}
            })

        return validated, security_events

    @classmethod
    def check_json_depth(cls, obj: Any, current_depth: int = 0) -> bool:
        """
        Check if a JSON object exceeds maximum nesting depth.

        Args:
            obj: The object to check
            current_depth: Current nesting level

        Returns:
            True if within limits, False if too deep
        """
        if current_depth > MAX_JSON_NESTING_DEPTH:
            return False

        if isinstance(obj, dict):
            for value in obj.values():
                if not cls.check_json_depth(value, current_depth + 1):
                    return False
        elif isinstance(obj, list):
            for item in obj:
                if not cls.check_json_depth(item, current_depth + 1):
                    return False

        return True


# ============================================================================
# CSRF Protection
# ============================================================================

class CSRFProtection:
    """
    CSRF token generation and validation.

    Uses double-submit cookie pattern for API protection.
    """

    TOKEN_LENGTH = 32
    TOKEN_HEADER = "X-CSRF-Token"
    TOKEN_COOKIE = "csrf_token"

    @classmethod
    def generate_token(cls) -> str:
        """Generate a cryptographically secure CSRF token."""
        return secrets.token_urlsafe(cls.TOKEN_LENGTH)

    @classmethod
    def validate_token(
        cls,
        header_token: Optional[str],
        cookie_token: Optional[str],
    ) -> bool:
        """
        Validate CSRF token using double-submit pattern.

        Args:
            header_token: Token from request header
            cookie_token: Token from cookie

        Returns:
            True if tokens match and are valid
        """
        if not header_token or not cookie_token:
            return False

        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(header_token, cookie_token)


# ============================================================================
# Security Monitor (API-specific)
# ============================================================================

class APISecurityMonitor:
    """
    API-specific security monitoring and event logging.

    Tracks security events, suspicious patterns, and provides
    real-time threat assessment.
    """

    def __init__(self, max_events: int = 10000):
        self.events: List[SecurityEvent] = []
        self.max_events = max_events
        self.auth_tracker = AuthFailureTracker()
        self._event_counts: Dict[SecurityEventType, int] = {
            event_type: 0 for event_type in SecurityEventType
        }

    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        self.events.append(event)
        self._event_counts[event.event_type] += 1

        # Trim old events if needed
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Log based on threat level
        log_msg = (
            f"Security event: {event.event_type.value} "
            f"[{event.threat_level.value}] "
            f"from {event.client_ip} at {event.path}"
        )

        if event.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            logger.warning(log_msg, extra={"event": event.details})
        elif event.threat_level == ThreatLevel.MEDIUM:
            logger.info(log_msg, extra={"event": event.details})
        else:
            logger.debug(log_msg, extra={"event": event.details})

    def log_prompt_injection_attempt(
        self,
        client_ip: str,
        path: str,
        content_preview: str,
        patterns: List[str],
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Log a prompt injection attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.PROMPT_INJECTION_ATTEMPT,
            threat_level=ThreatLevel.HIGH,
            timestamp=time.time(),
            client_ip=client_ip,
            path=path,
            details={
                "content_preview": content_preview[:100] + "..." if len(content_preview) > 100 else content_preview,
                "matched_patterns": patterns,
            },
            request_id=request_id,
            user_id=user_id,
            blocked=True,
        )
        self.log_event(event)

    def log_auth_failure(
        self,
        client_ip: str,
        path: str,
        reason: str,
        request_id: Optional[str] = None,
    ) -> Tuple[bool, int]:
        """
        Log an authentication failure and check for lockout.

        Returns:
            Tuple of (is_locked_out, remaining_attempts)
        """
        is_locked, remaining = self.auth_tracker.record_failure(client_ip)

        event = SecurityEvent(
            event_type=SecurityEventType.AUTH_FAILURE,
            threat_level=ThreatLevel.MEDIUM if not is_locked else ThreatLevel.HIGH,
            timestamp=time.time(),
            client_ip=client_ip,
            path=path,
            details={
                "reason": reason,
                "locked_out": is_locked,
                "remaining_attempts": remaining,
            },
            request_id=request_id,
            blocked=is_locked,
        )
        self.log_event(event)

        return is_locked, remaining

    def get_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        now = time.time()
        last_hour = [e for e in self.events if now - e.timestamp < 3600]

        return {
            "total_events": len(self.events),
            "events_last_hour": len(last_hour),
            "events_by_type": dict(self._event_counts),
            "blocked_events": sum(1 for e in self.events if e.blocked),
            "active_lockouts": len(self.auth_tracker.lockouts),
            "high_threat_events": sum(
                1 for e in last_hour
                if e.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)
            ),
        }


# Global security monitor instance
_security_monitor: Optional[APISecurityMonitor] = None


def get_api_security_monitor() -> APISecurityMonitor:
    """Get the global API security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = APISecurityMonitor()
    return _security_monitor
