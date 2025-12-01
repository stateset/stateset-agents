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
        """Encrypt value using Fernet symmetric encryption.

        Requires CONFIG_ENCRYPTION_KEY environment variable (32+ character key).
        Falls back to a warning and base64 encoding if cryptography is not installed.
        """
        key_str = os.environ.get("CONFIG_ENCRYPTION_KEY")

        if not key_str:
            logger.warning(
                "CONFIG_ENCRYPTION_KEY not set. Secrets will be stored in plaintext. "
                "Set a 32+ character key for production use."
            )
            import base64
            return base64.b64encode(value.encode()).decode()

        try:
            from cryptography.fernet import Fernet
            import base64

            # Derive a valid Fernet key from the provided key
            # Fernet requires 32 url-safe base64-encoded bytes
            key_bytes = hashlib.sha256(key_str.encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            cipher = Fernet(fernet_key)
            return cipher.encrypt(value.encode()).decode()
        except ImportError:
            logger.warning(
                "cryptography package not installed. Using base64 encoding only. "
                "Install cryptography for secure secret storage: pip install cryptography"
            )
            import base64
            return base64.b64encode(value.encode()).decode()

    def _simple_decrypt(self, value: str) -> str:
        """Decrypt value using Fernet symmetric encryption."""
        key_str = os.environ.get("CONFIG_ENCRYPTION_KEY")

        if not key_str:
            # Assume base64 encoded if no key
            import base64
            try:
                return base64.b64decode(value.encode()).decode()
            except Exception:
                return value

        try:
            from cryptography.fernet import Fernet
            import base64

            # Derive the same Fernet key
            key_bytes = hashlib.sha256(key_str.encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            cipher = Fernet(fernet_key)
            return cipher.decrypt(value.encode()).decode()
        except ImportError:
            import base64
            try:
                return base64.b64decode(value.encode()).decode()
            except Exception:
                return value
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise ValueError("Failed to decrypt secret. Check CONFIG_ENCRYPTION_KEY.")

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
