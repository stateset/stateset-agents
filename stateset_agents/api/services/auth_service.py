"""Authentication service provider.

This module is imported by FastAPI dependency wiring. It must not eagerly
load environment configuration (which may require secrets) at import time.
"""

from __future__ import annotations

from typing import Optional

from stateset_agents.utils.security import AuthService

from ..config import get_config

_auth_service: Optional[AuthService] = None


async def get_auth_service() -> AuthService:
    """Return a cached AuthService instance.

    This dependency is async to avoid FastAPI running it in a threadpool, which
    can deadlock under certain middleware/test transports.
    """
    global _auth_service
    if _auth_service is None:
        config = get_config()
        secret_key = config.security.jwt_secret or "temporary-secret-for-dev"
        _auth_service = AuthService(secret_key)
    return _auth_service
