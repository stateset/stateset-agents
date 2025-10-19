"""
Security utilities for the StateSet Agents framework.

This module provides security-related utilities including input validation,
authentication helpers, secure configuration management, and security monitoring.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation utilities."""

    # Common regex patterns (compiled for performance)
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    URL_PATTERN = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
    ALPHANUMERIC_PATTERN = re.compile(r"^[a-zA-Z0-9]+$")
    SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")

    # Dangerous patterns to block
    SQL_INJECTION_PATTERNS = [
        re.compile(
            r";\s*(?:select|insert|update|delete|drop|create|alter)", re.IGNORECASE
        ),
        re.compile(r"union\s+select", re.IGNORECASE),
        re.compile(r"--\s*$"),
        re.compile(r"/\*.*\*/", re.DOTALL),
    ]

    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*="),
    ]

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        if not isinstance(email, str) or len(email) > 254:
            return False
        return bool(InputValidator.EMAIL_PATTERN.match(email))

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        if not isinstance(url, str) or len(url) > 2048:
            return False
        return bool(InputValidator.URL_PATTERN.match(url))

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")

        # Remove potentially dangerous characters
        sanitized = re.sub(r"[<>]", "", input_str)
        return sanitized[:max_length]

    @staticmethod
    def validate_filename(
        filename: str, allowed_extensions: Optional[Set[str]] = None
    ) -> bool:
        """Validate filename for security."""
        if not isinstance(filename, str) or not filename:
            return False

        # Check for directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return False

        # Check filename pattern
        if not InputValidator.SAFE_FILENAME_PATTERN.match(filename):
            return False

        # Check file extension if specified
        if allowed_extensions:
            extension = Path(filename).suffix.lower()
            if extension not in allowed_extensions:
                return False

        return True

    @staticmethod
    def detect_injection_attempts(input_str: str) -> List[str]:
        """Detect potential injection attacks."""
        threats = []

        # Check for SQL injection patterns
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if pattern.search(input_str):
                threats.append("sql_injection")
                break

        # Check for XSS patterns
        for pattern in InputValidator.XSS_PATTERNS:
            if pattern.search(input_str):
                threats.append("xss")
                break

        return threats

    @staticmethod
    def validate_api_key(api_key: str, min_length: int = 32) -> bool:
        """Validate API key format."""
        if not isinstance(api_key, str):
            return False

        if len(api_key) < min_length:
            return False

        # Check for basic format (alphanumeric + some special chars)
        if not re.match(r"^[a-zA-Z0-9_-]+$", api_key):
            return False

        return True


class SecureConfig:
    """Secure configuration management."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file) if config_file else None
        self._config: Dict[str, Any] = {}
        self._secrets: Dict[str, str] = {}

        # Load configuration if file provided
        if self.config_file and self.config_file.exists():
            self.load_config()

    def set_secret(self, key: str, value: str, encrypt: bool = True):
        """Securely store a secret."""
        if encrypt:
            # In production, use proper encryption
            # This is a simplified example
            encrypted = self._simple_encrypt(value)
            self._secrets[key] = encrypted
        else:
            self._secrets[key] = value

    def get_secret(self, key: str, decrypt: bool = True) -> Optional[str]:
        """Retrieve a secret."""
        if key not in self._secrets:
            return None

        value = self._secrets[key]
        if decrypt:
            return self._simple_decrypt(value)
        return value

    def _simple_encrypt(self, value: str) -> str:
        """Simple encryption for demonstration (use proper encryption in production)."""
        # WARNING: This is NOT secure! Use proper encryption libraries
        key = os.environ.get("CONFIG_ENCRYPTION_KEY", "default_key")
        return "".join(
            chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(value)
        )

    def _simple_decrypt(self, value: str) -> str:
        """Simple decryption for demonstration."""
        return self._simple_encrypt(value)  # XOR is symmetric

    def save_config(self):
        """Save configuration to file."""
        if not self.config_file:
            raise ValueError("No config file specified")

        # Don't save secrets in plain config file
        config_data = {
            "config": self._config,
            "secrets_keys": list(self._secrets.keys()),  # Just keys, not values
        }

        with open(self.config_file, "w") as f:
            json.dump(config_data, f, indent=2)

    def load_config(self):
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return

        with open(self.config_file, "r") as f:
            data = json.load(f)
            self._config = data.get("config", {})


class SecurityMonitor:
    """Monitor security events and anomalies."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__ + ".SecurityMonitor")

    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "info",
        source_ip: Optional[str] = None,
    ):
        """Log a security event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "severity": severity,
            "details": details,
            "source_ip": source_ip,
        }

        self.events.append(event)

        # Log based on severity
        log_method = getattr(self.logger, severity, self.logger.info)
        log_method(f"Security event: {event_type} - {details}")

    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent security events."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        return [
            event
            for event in self.events
            if datetime.fromisoformat(event["timestamp"]) > cutoff
        ]

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect security anomalies."""
        anomalies = []
        recent_events = self.get_recent_events(hours=1)

        # Simple anomaly detection
        event_counts = {}
        for event in recent_events:
            event_type = event["type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Flag high-frequency events
        for event_type, count in event_counts.items():
            if count > 10:  # Threshold for anomaly
                anomalies.append(
                    {
                        "type": "high_frequency",
                        "event_type": event_type,
                        "count": count,
                        "threshold": 10,
                    }
                )

        return anomalies


class AuthService:
    """Authentication and authorization service."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.monitor = SecurityMonitor()

    def register_user(
        self, username: str, password: str, roles: List[str] = None
    ) -> bool:
        """Register a new user."""
        if username in self.users:
            return False

        # Hash password securely
        password_hash = self._hash_password(password)

        self.users[username] = {
            "password_hash": password_hash,
            "roles": roles or ["user"],
            "created_at": datetime.utcnow().isoformat(),
            "failed_attempts": 0,
            "locked_until": None,
        }

        return True

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        if username not in self.users:
            self.monitor.log_security_event(
                "failed_login", {"username": username, "reason": "user_not_found"}
            )
            return None

        user = self.users[username]

        # Check if account is locked
        if user.get("locked_until"):
            lock_time = datetime.fromisoformat(user["locked_until"])
            if datetime.utcnow() < lock_time:
                self.monitor.log_security_event(
                    "failed_login", {"username": username, "reason": "account_locked"}
                )
                return None

        # Verify password
        if not self._verify_password(password, user["password_hash"]):
            user["failed_attempts"] += 1

            # Lock account after 5 failed attempts
            if user["failed_attempts"] >= 5:
                user["locked_until"] = (
                    datetime.utcnow() + timedelta(minutes=30)
                ).isoformat()
                self.monitor.log_security_event(
                    "account_locked",
                    {"username": username, "failed_attempts": user["failed_attempts"]},
                )
            else:
                self.monitor.log_security_event(
                    "failed_login", {"username": username, "reason": "wrong_password"}
                )
            return None

        # Reset failed attempts on successful login
        user["failed_attempts"] = 0

        # Create session
        session_token = self._generate_session_token()
        self.sessions[session_token] = {
            "username": username,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
        }

        self.monitor.log_security_event("successful_login", {"username": username})

        return session_token

    def authorize(self, session_token: str, required_role: str) -> bool:
        """Check if session has required role."""
        if session_token not in self.sessions:
            return False

        session = self.sessions[session_token]
        username = session["username"]
        user = self.users.get(username)

        if not user:
            return False

        return required_role in user["roles"]

    def _hash_password(self, password: str) -> str:
        """Hash password securely."""
        import bcrypt

        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode(), salt).decode()

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        import bcrypt

        return bcrypt.checkpw(password.encode(), hashed.encode())

    def _generate_session_token(self) -> str:
        """Generate secure session token."""
        return secrets.token_urlsafe(32)


# Global security monitor instance
_security_monitor = SecurityMonitor()


def log_security_event(
    event_type: str,
    details: Dict[str, Any],
    severity: str = "info",
    source_ip: Optional[str] = None,
):
    """Log a security event globally."""
    _security_monitor.log_security_event(event_type, details, severity, source_ip)


def get_security_events(hours: int = 24) -> List[Dict[str, Any]]:
    """Get recent security events."""
    return _security_monitor.get_recent_events(hours)


def detect_security_anomalies() -> List[Dict[str, Any]]:
    """Detect security anomalies."""
    return _security_monitor.detect_anomalies()
