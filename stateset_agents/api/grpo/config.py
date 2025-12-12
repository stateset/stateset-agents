"""
GRPO Configuration Module

Centralized configuration for the GRPO service.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


def _get_int_from_env(env_var: str, default: int) -> int:
    """Parse an integer environment variable safely."""
    try:
        return int(os.getenv(env_var, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s, using default %s", env_var, default)
        return default


def _get_bool_from_env(env_var: str, default: bool = False) -> bool:
    """Parse a boolean environment variable safely."""
    value = os.getenv(env_var, str(default)).lower()
    return value in {"1", "true", "yes", "on"}


@dataclass
class GRPOConfig:
    """Runtime configuration for the GRPO API."""

    # Authentication
    api_keys: Dict[str, List[str]]
    allow_anonymous: bool

    # Rate limiting
    rate_limit_per_minute: int
    rate_limit_window_seconds: int

    # Request limits
    max_prompts: int
    max_prompt_length: int
    max_message_length: int
    max_iterations: int

    # Engine settings
    use_lightweight_engine: bool
    enable_prometheus: bool

    # CORS settings
    cors_origins: List[str]
    cors_allow_credentials: bool

    @classmethod
    def from_env(cls) -> "GRPOConfig":
        """Create configuration from environment variables."""
        # Parse API keys: format "key1:role1|role2,key2:role3"
        raw_keys = os.getenv("GRPO_API_KEYS", "")
        parsed_keys: Dict[str, List[str]] = {}

        for item in raw_keys.split(","):
            cleaned = item.strip()
            if not cleaned:
                continue

            key, _, roles_raw = cleaned.partition(":")
            roles = [role.strip() for role in roles_raw.split("|") if role.strip()]
            parsed_keys[key] = roles or ["user"]

        # Parse CORS origins
        cors_origins_env = os.getenv("GRPO_CORS_ORIGINS", "")
        if cors_origins_env:
            cors_origins = [
                origin.strip()
                for origin in cors_origins_env.split(",")
                if origin.strip()
            ]
        else:
            # Default: permissive for development, restrictive for production
            env = os.getenv("GRPO_ENV", "development")
            cors_origins = ["*"] if env == "development" else []

        # Credentials only allowed when origins are explicitly specified
        cors_allow_credentials = cors_origins and cors_origins != ["*"]

        return cls(
            api_keys=parsed_keys,
            allow_anonymous=_get_bool_from_env("GRPO_ALLOW_ANONYMOUS", False),
            rate_limit_per_minute=_get_int_from_env("GRPO_RATE_LIMIT_PER_MIN", 60),
            rate_limit_window_seconds=_get_int_from_env("GRPO_RATE_LIMIT_WINDOW", 60),
            max_prompts=_get_int_from_env("GRPO_MAX_PROMPTS", 8),
            max_prompt_length=_get_int_from_env("GRPO_MAX_PROMPT_CHARS", 4000),
            max_message_length=_get_int_from_env("GRPO_MAX_MESSAGE_CHARS", 4000),
            max_iterations=_get_int_from_env("GRPO_MAX_ITERATIONS", 50),
            use_lightweight_engine=_get_bool_from_env("GRPO_API_LIGHT_ENGINE", True),
            enable_prometheus=_get_bool_from_env("GRPO_ENABLE_PROMETHEUS", False),
            cors_origins=cors_origins,
            cors_allow_credentials=cors_allow_credentials,
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.rate_limit_per_minute < 1:
            issues.append("rate_limit_per_minute must be at least 1")

        if self.max_prompts < 1:
            issues.append("max_prompts must be at least 1")

        if self.max_prompt_length < 1:
            issues.append("max_prompt_length must be at least 1")

        if self.max_iterations < 1:
            issues.append("max_iterations must be at least 1")

        if not self.api_keys and not self.allow_anonymous:
            issues.append(
                "Either GRPO_API_KEYS must be set or GRPO_ALLOW_ANONYMOUS=true"
            )

        return issues


# Global singleton
_config: GRPOConfig = None


def get_grpo_config() -> GRPOConfig:
    """Get the global GRPO configuration instance."""
    global _config
    if _config is None:
        _config = GRPOConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset configuration (for testing)."""
    global _config
    _config = None
