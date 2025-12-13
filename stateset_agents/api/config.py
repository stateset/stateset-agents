"""
API Configuration Module

Centralized, validated configuration for all API services.
Loads from environment variables with proper validation and defaults.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


def _get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with validation."""
    value = os.getenv(key, default)
    if required and not value:
        raise ConfigurationError(f"Required environment variable '{key}' is not set")
    return value


def _get_int(key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Get integer environment variable with range validation."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        if min_val is not None and value < min_val:
            logger.warning(f"{key}={value} is below minimum {min_val}, using {min_val}")
            return min_val
        if max_val is not None and value > max_val:
            logger.warning(f"{key}={value} is above maximum {max_val}, using {max_val}")
            return max_val
        return value
    except ValueError:
        logger.warning(f"Invalid integer value for {key}: {raw}, using default {default}")
        return default


def _get_float(key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Get float environment variable with range validation."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
        if min_val is not None and value < min_val:
            logger.warning(f"{key}={value} is below minimum {min_val}, using {min_val}")
            return min_val
        if max_val is not None and value > max_val:
            logger.warning(f"{key}={value} is above maximum {max_val}, using {max_val}")
            return max_val
        return value
    except ValueError:
        logger.warning(f"Invalid float value for {key}: {raw}, using default {default}")
        return default


def _get_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in ("true", "1", "yes", "on")


def _get_list(key: str, default: List[str], separator: str = ",") -> List[str]:
    """Get list environment variable."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return [item.strip() for item in raw.split(separator) if item.strip()]


def _parse_api_keys(raw: str) -> Dict[str, List[str]]:
    """
    Parse API keys from environment variable.

    Format: key1:role1|role2,key2:role3|role4
    Each key must be at least 32 characters.
    """
    parsed: Dict[str, List[str]] = {}

    if not raw:
        return parsed

    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue

        key, _, roles_raw = item.partition(":")
        key = key.strip()

        # Validate key length
        if len(key) < 32:
            logger.warning(f"API key is shorter than 32 characters, skipping (key prefix: {key[:8]}...)")
            continue

        # Validate key format
        if not re.match(r"^[a-zA-Z0-9_-]+$", key):
            logger.warning(f"API key contains invalid characters, skipping (key prefix: {key[:8]}...)")
            continue

        roles = [role.strip() for role in roles_raw.split("|") if role.strip()]
        parsed[key] = roles or ["user"]

    return parsed


@dataclass
class CORSConfig:
    """CORS configuration."""
    allowed_origins: List[str] = field(default_factory=list)
    allow_credentials: bool = False
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    allowed_headers: List[str] = field(default_factory=lambda: ["Authorization", "Content-Type", "X-Request-ID", "X-API-Key"])
    max_age: int = 600  # 10 minutes

    @classmethod
    def from_env(cls) -> "CORSConfig":
        """Load CORS config from environment."""
        origins_raw = os.getenv("API_CORS_ORIGINS", "")

        # Parse origins - default to empty (no CORS) in production
        if origins_raw == "*":
            # Only allow wildcard in development
            env = os.getenv("API_ENVIRONMENT", "development").lower()
            if env == "development":
                allowed_origins = ["*"]
                logger.warning("CORS wildcard (*) enabled - only use in development!")
            else:
                raise ConfigurationError(
                    "CORS wildcard (*) is not allowed in production. "
                    "Set API_CORS_ORIGINS to specific origins or API_ENVIRONMENT=development"
                )
        else:
            allowed_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]

        return cls(
            allowed_origins=allowed_origins,
            allow_credentials=_get_bool("API_CORS_CREDENTIALS", False),
            allowed_methods=_get_list("API_CORS_METHODS", ["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
            allowed_headers=_get_list("API_CORS_HEADERS", ["Authorization", "Content-Type", "X-Request-ID", "X-API-Key"]),
            max_age=_get_int("API_CORS_MAX_AGE", 600, min_val=0, max_val=86400),
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    burst_size: int = 10
    window_seconds: int = 60
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Load rate limit config from environment."""
        return cls(
            requests_per_minute=_get_int("API_RATE_LIMIT_PER_MIN", 60, min_val=1, max_val=10000),
            burst_size=_get_int("API_RATE_LIMIT_BURST", 10, min_val=1, max_val=100),
            window_seconds=_get_int("API_RATE_LIMIT_WINDOW", 60, min_val=1, max_val=3600),
            enabled=_get_bool("API_RATE_LIMIT_ENABLED", True),
        )


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_keys: Dict[str, List[str]] = field(default_factory=dict)
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    require_auth: bool = True

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load security config from environment."""
        # JWT secret is required in production
        jwt_secret = os.getenv("API_JWT_SECRET")
        env = os.getenv("API_ENVIRONMENT", "development").lower()

        if not jwt_secret and env == "production":
            raise ConfigurationError(
                "API_JWT_SECRET is required in production. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )
        elif not jwt_secret:
            # Use a random secret for development (won't persist across restarts)
            jwt_secret = os.urandom(64).hex()
            logger.warning("Using random JWT secret - tokens won't persist across restarts")

        return cls(
            api_keys=_parse_api_keys(os.getenv("API_KEYS", "")),
            jwt_secret=jwt_secret,
            jwt_algorithm=os.getenv("API_JWT_ALGORITHM", "HS256"),
            jwt_expiry_hours=_get_int("API_JWT_EXPIRY_HOURS", 24, min_val=1, max_val=720),
            require_auth=_get_bool("API_REQUIRE_AUTH", True),
        )


@dataclass
class ValidationConfig:
    """Input validation configuration."""
    max_prompts: int = 8
    max_prompt_length: int = 4000
    max_message_length: int = 4000
    max_iterations: int = 1000
    max_request_size_mb: int = 10

    @classmethod
    def from_env(cls) -> "ValidationConfig":
        """Load validation config from environment."""
        return cls(
            max_prompts=_get_int("API_MAX_PROMPTS", 8, min_val=1, max_val=100),
            max_prompt_length=_get_int("API_MAX_PROMPT_LENGTH", 4000, min_val=100, max_val=100000),
            max_message_length=_get_int("API_MAX_MESSAGE_LENGTH", 4000, min_val=100, max_val=100000),
            max_iterations=_get_int("API_MAX_ITERATIONS", 1000, min_val=1, max_val=1000),
            max_request_size_mb=_get_int("API_MAX_REQUEST_SIZE_MB", 10, min_val=1, max_val=100),
        )


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    enable_prometheus: bool = False
    enable_tracing: bool = False
    tracing_endpoint: Optional[str] = None
    log_level: str = "INFO"
    log_format: str = "json"

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Load observability config from environment."""
        return cls(
            enable_prometheus=_get_bool("API_ENABLE_PROMETHEUS", False),
            enable_tracing=_get_bool("API_ENABLE_TRACING", False),
            tracing_endpoint=os.getenv("API_TRACING_ENDPOINT"),
            log_level=os.getenv("API_LOG_LEVEL", "INFO").upper(),
            log_format=os.getenv("API_LOG_FORMAT", "json").lower(),
        )


@dataclass
class APIConfig:
    """Main API configuration."""
    environment: Environment = Environment.DEVELOPMENT
    host: str = "0.0.0.0"
    port: int = 8000
    api_version: str = "v1"
    title: str = "StateSet Agents API"

    cors: CORSConfig = field(default_factory=CORSConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load full configuration from environment."""
        env_str = os.getenv("API_ENVIRONMENT", "development").lower()
        try:
            environment = Environment(env_str)
        except ValueError:
            logger.warning(f"Unknown environment '{env_str}', defaulting to development")
            environment = Environment.DEVELOPMENT

        return cls(
            environment=environment,
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=_get_int("API_PORT", 8000, min_val=1, max_val=65535),
            api_version=os.getenv("API_VERSION", "v1"),
            title=os.getenv("API_TITLE", "StateSet Agents API"),
            cors=CORSConfig.from_env(),
            rate_limit=RateLimitConfig.from_env(),
            security=SecurityConfig.from_env(),
            validation=ValidationConfig.from_env(),
            observability=ObservabilityConfig.from_env(),
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.environment == Environment.PRODUCTION:
            if not self.security.api_keys and self.security.require_auth:
                warnings.append("No API keys configured but authentication is required")

            if not self.cors.allowed_origins:
                warnings.append("No CORS origins configured - API won't be accessible from browsers")

            if not self.security.jwt_secret:
                warnings.append("JWT secret not configured")

        return warnings

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION


# Singleton instance
_config: Optional[APIConfig] = None


def get_config() -> APIConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = APIConfig.from_env()
    return _config


def reload_config() -> APIConfig:
    """Reload configuration from environment."""
    global _config
    _config = APIConfig.from_env()
    return _config
